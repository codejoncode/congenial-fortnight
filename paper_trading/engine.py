"""
Paper Trading Engine
Simulates MetaTrader order execution without real money
Tracks positions, calculates P&L, manages risk
"""
import logging
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from django.utils import timezone
from django.db.models import Sum, Avg, Q

from .models import PaperTrade, PerformanceMetrics
from .us_forex_rules import USForexRules

logger = logging.getLogger(__name__)


class PaperTradingEngine:
    """
    Simulates forex trading without real money
    Manages positions, risk, and performance tracking
    """
    
    def __init__(self, initial_balance=10000.0, user=None):
        # Handle both calling patterns:
        # 1. PaperTradingEngine(user) - old pattern from tests
        # 2. PaperTradingEngine(initial_balance=10000.0, user=user) - new pattern
        if user is None and hasattr(initial_balance, 'username'):
            # If first arg is a user object, swap them
            user = initial_balance
            initial_balance = 10000.0
        
        self.initial_balance = Decimal(str(initial_balance))
        self.current_balance = self.initial_balance
        self.user = user
        self.us_rules = USForexRules(self)
        
    def execute_order(
        self,
        pair: str = None,
        order_type: str = None,
        entry_price: float = None,
        stop_loss: float = None,
        take_profit_1: float = None,
        take_profit_2: float = None,
        take_profit_3: float = None,
        lot_size: float = 0.01,
        signal_id: str = None,
        signal_type: str = None,
        signal_source: str = None,
        notes: str = None,
        # Support old parameter names from tests
        symbol: str = None,
        **kwargs
    ) -> PaperTrade:
        """
        Execute a paper trade order
        
        Args:
            pair/symbol: Currency pair (e.g., 'EURUSD')
            order_type/signal_type: 'buy'/'sell' or 'BUY'/'SELL'
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit_1: First take profit level
            take_profit_2: Second take profit level (optional)
            take_profit_3: Third take profit level (optional)
            lot_size: Position size in lots
            signal_id: Associated signal ID
            signal_type: Type of signal that generated trade (also used as order_type)
            signal_source: Source model/system
            notes: Additional notes
            
        Returns:
            PaperTrade object
        """
        # Handle parameter aliasing for backward compatibility
        if symbol is not None and pair is None:
            pair = symbol
        
        if signal_type is not None and order_type is None:
            order_type = signal_type
        
        # Normalize order_type to lowercase
        if order_type:
            order_type = order_type.lower()
        
        # Validate order
        if order_type not in ['buy', 'sell']:
            raise ValueError(f"Invalid order_type: {order_type}")
        
        # Calculate risk
        risk_reward_ratio = self._calculate_risk_reward(
            entry_price, stop_loss, take_profit_1, order_type
        )
        
        # Create trade record
        trade = PaperTrade.objects.create(
            user=self.user,
            pair=pair,
            order_type=order_type,
            entry_price=Decimal(str(entry_price)),
            stop_loss=Decimal(str(stop_loss)),
            take_profit_1=Decimal(str(take_profit_1)) if take_profit_1 else None,
            take_profit_2=Decimal(str(take_profit_2)) if take_profit_2 else None,
            take_profit_3=Decimal(str(take_profit_3)) if take_profit_3 else None,
            lot_size=Decimal(str(lot_size)),
            signal_id=signal_id,
            signal_type=signal_type,
            signal_source=signal_source,
            risk_reward_ratio=Decimal(str(risk_reward_ratio)) if risk_reward_ratio else None,
            status='open',
            entry_time=timezone.now(),
            notes=notes
        )
        
        logger.info(
            f"📊 Paper trade executed: {pair} {order_type.upper()} @ {entry_price} "
            f"(SL: {stop_loss}, TP: {take_profit_1}, R:R: {risk_reward_ratio:.2f}:1)"
        )
        
        return trade
    
    def update_positions(self, current_prices: Dict[str, float]) -> List[Dict]:
        """
        Update all open positions with current prices
        Check for SL/TP hits
        
        Args:
            current_prices: Dict of {pair: current_price}
            
        Returns:
            List of closed trades (if any hit SL/TP)
        """
        open_trades = PaperTrade.objects.filter(status='open')
        closed_trades = []
        
        for trade in open_trades:
            if trade.pair not in current_prices:
                continue
            
            current_price = current_prices[trade.pair]
            
            # Check if SL or TP hit
            hit, level = self._check_sl_tp_hit(trade, current_price)
            
            if hit:
                # Determine exit price based on level
                if level == 'sl_hit':
                    exit_price = float(trade.stop_loss)
                elif level == 'tp1_hit':
                    exit_price = float(trade.take_profit_1)
                elif level == 'tp2_hit':
                    exit_price = float(trade.take_profit_2)
                elif level == 'tp3_hit':
                    exit_price = float(trade.take_profit_3)
                else:
                    exit_price = current_price
                
                self.close_position(trade.id, exit_price=exit_price, reason=level)
                closed_trades.append({
                    'trade_id': trade.id,
                    'pair': trade.pair,
                    'exit_reason': level,
                    'exit_price': exit_price,
                    'pips': trade.pips_gained,
                    'pnl': trade.profit_loss
                })
                
                logger.info(
                    f"✅ Trade closed: {trade.pair} {level} hit @ {exit_price} "
                    f"(Pips: {trade.pips_gained}, P&L: ${trade.profit_loss})"
                )
        
        return closed_trades
    
    def close_position(
        self,
        position_id: int = None,
        exit_price: float = None,
        reason: str = None,
        exit_time: datetime = None,
        trade_id: int = None,  # Alias for position_id
        exit_reason: str = None,  # Alias for reason
        **kwargs
    ) -> Optional[PaperTrade]:
        """
        Close a position manually or via SL/TP
        
        Args:
            position_id: Trade ID to close (can also use trade_id)
            exit_price: Exit price
            reason: Reason for closing (can also use exit_reason)
            exit_time: Exit timestamp (default: now)
            
        Returns:
            Updated PaperTrade object or None if not found
        """
        # Handle parameter aliasing
        if trade_id is not None and position_id is None:
            position_id = trade_id
        if exit_reason is not None and reason is None:
            reason = exit_reason
            
        try:
            trade = PaperTrade.objects.get(id=position_id, status='open')
        except PaperTrade.DoesNotExist:
            logger.warning(f"Trade {position_id} not found or already closed")
            return None
        
        # Set exit reason if provided
        if reason:
            trade.exit_reason = reason
        
        pnl = trade.close_trade(exit_price, exit_time)
        
        # Update current balance
        self.current_balance += Decimal(str(pnl))
        
        # Update daily performance metrics
        self._update_daily_metrics(trade)
        
        return trade
    
    def get_open_positions(self) -> List[PaperTrade]:
        """Get all open positions"""
        return list(PaperTrade.objects.filter(status='open').order_by('-entry_time'))
    
    def get_pending_orders(self) -> List[PaperTrade]:
        """Get all pending orders"""
        return list(PaperTrade.objects.filter(status='pending').order_by('-created_at'))
    
    def get_trade_history(
        self,
        pair: str = None,
        days: int = 30,
        limit: int = 100
    ) -> List[PaperTrade]:
        """
        Get trade history
        
        Args:
            pair: Filter by currency pair (optional)
            days: Number of days to look back
            limit: Max number of trades to return
            
        Returns:
            List of PaperTrade objects
        """
        cutoff_date = timezone.now() - timedelta(days=days)
        query = PaperTrade.objects.filter(
            status='closed',
            entry_time__gte=cutoff_date
        )
        
        if pair:
            query = query.filter(pair=pair)
        
        return list(query.order_by('-entry_time')[:limit])
    
    def get_performance_summary(self, days: int = 30) -> Dict:
        """
        Get overall performance summary
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dict with performance metrics
        """
        cutoff_date = timezone.now() - timedelta(days=days)
        trades = PaperTrade.objects.filter(
            status='closed',
            entry_time__gte=cutoff_date
        )
        
        total_trades = trades.count()
        
        if total_trades == 0:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_pips': 0.0,
                'total_pnl': 0.0,
                'avg_rr': 0.0,
                'best_trade': None,
                'worst_trade': None,
                'current_balance': float(self.current_balance)
            }
        
        winning_trades = trades.filter(pips_gained__gt=0).count()
        
        agg = trades.aggregate(
            total_pips=Sum('pips_gained'),
            total_pnl=Sum('profit_loss'),
            avg_rr=Avg('risk_reward_ratio')
        )
        
        best_trade = trades.order_by('-pips_gained').first()
        worst_trade = trades.order_by('pips_gained').first()
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': round((winning_trades / total_trades) * 100, 2),
            'total_pips': float(agg['total_pips'] or 0),
            'total_pnl': float(agg['total_pnl'] or 0),
            'avg_rr': float(agg['avg_rr'] or 0),
            'best_trade': {
                'pair': best_trade.pair,
                'pips': float(best_trade.pips_gained),
                'pnl': float(best_trade.profit_loss)
            } if best_trade else None,
            'worst_trade': {
                'pair': worst_trade.pair,
                'pips': float(worst_trade.pips_gained),
                'pnl': float(worst_trade.profit_loss)
            } if worst_trade else None,
            'current_balance': float(self.current_balance)
        }
    
    def get_equity_curve(self, days: int = 30) -> List[Dict]:
        """
        Get equity curve data for charting
        
        Args:
            days: Number of days to retrieve (or initial_balance - legacy param ignored)
            
        Returns:
            List of {date, equity} dicts
        """
        # Handle legacy test calling with initial_balance instead of days
        from decimal import Decimal
        if isinstance(days, Decimal) or (isinstance(days, (int, float)) and days > 1000):
            days = 30  # Default to 30 days if passed initial_balance
        
        cutoff_date = timezone.now() - timedelta(days=int(days))
        trades = PaperTrade.objects.filter(
            status='closed',
            exit_time__gte=cutoff_date
        ).order_by('exit_time')
        
        equity_curve = []
        running_balance = float(self.initial_balance)
        
        for trade in trades:
            running_balance += float(trade.profit_loss or 0)
            equity_curve.append({
                'date': trade.exit_time.isoformat(),
                'balance': running_balance,  # Tests expect 'balance', not 'equity'
                'pnl': float(trade.profit_loss or 0),
                'pair': trade.pair
            })
        
        return equity_curve
    
    def _calculate_pips(
        self,
        symbol: str,
        entry_price: Decimal,
        exit_price: Decimal
    ) -> Decimal:
        """
        Calculate pips gained/lost
        
        Args:
            symbol: Currency pair or instrument
            entry_price: Entry price
            exit_price: Exit price
            
        Returns:
            Pips as Decimal
        """
        # Convert to Decimal if needed
        if not isinstance(entry_price, Decimal):
            entry_price = Decimal(str(entry_price))
        if not isinstance(exit_price, Decimal):
            exit_price = Decimal(str(exit_price))
        
        # Calculate price difference
        diff = exit_price - entry_price
        
        # Determine pip multiplier based on instrument
        if 'JPY' in symbol.upper():
            # JPY pairs use 2 decimal places (0.01 = 1 pip)
            multiplier = Decimal('100')
        elif 'XAU' in symbol.upper() or 'GOLD' in symbol.upper():
            # Gold uses 0.10 = 1 pip
            multiplier = Decimal('100')
        else:
            # Standard forex pairs use 4 decimal places (0.0001 = 1 pip)
            multiplier = Decimal('10000')
        
        pips = diff * multiplier
        return pips.quantize(Decimal('0.1'))
    
    def _calculate_risk_reward(
        self,
        entry: float,
        stop_loss: float,
        take_profit: float,
        order_type: str
    ) -> Optional[float]:
        """Calculate risk:reward ratio"""
        if not take_profit:
            return None
        
        if order_type == 'buy':
            risk = entry - stop_loss
            reward = take_profit - entry
        else:  # sell
            risk = stop_loss - entry
            reward = entry - take_profit
        
        if risk <= 0:
            return None
        
        return round(reward / risk, 2)
    
    def _check_sl_tp_hit(
        self,
        trade: PaperTrade,
        current_price: float
    ) -> Tuple:
        """
        Check if stop loss or take profit was hit
        
        Returns:
            Tuple of (hit_boolean, level_name) for tests
            or (level_name, exit_price) for production code
        """
        # Get order type from either field, handle None values
        order_type = getattr(trade, 'order_type', None)
        if not order_type:
            order_type = getattr(trade, 'signal_type', 'buy')
        if order_type:
            order_type = order_type.lower()
        
        # Convert to Decimal for precise comparison
        from decimal import Decimal
        if not isinstance(current_price, Decimal):
            current_price = Decimal(str(current_price))
        
        if order_type == 'buy':
            # Check stop loss
            if current_price <= trade.stop_loss:
                return (True, 'sl_hit')
            
            # Check take profits (prioritize closest)
            if trade.take_profit_1 and current_price >= trade.take_profit_1:
                return (True, 'tp1_hit')
            if trade.take_profit_2 and current_price >= trade.take_profit_2:
                return (True, 'tp2_hit')
            if trade.take_profit_3 and current_price >= trade.take_profit_3:
                return (True, 'tp3_hit')
        
        else:  # sell
            # Check stop loss
            if current_price >= trade.stop_loss:
                return (True, 'sl_hit')
            
            # Check take profits (prioritize closest)
            if trade.take_profit_1 and current_price <= trade.take_profit_1:
                return (True, 'tp1_hit')
            if trade.take_profit_2 and current_price <= trade.take_profit_2:
                return (True, 'tp2_hit')
            if trade.take_profit_3 and current_price <= trade.take_profit_3:
                return (True, 'tp3_hit')
        
        return (False, None)
    
    def _update_daily_metrics(self, trade: PaperTrade):
        """Update or create daily performance metrics"""
        trade_date = trade.entry_time.date()
        
        metrics, created = PerformanceMetrics.objects.get_or_create(
            date=trade_date,
            pair=trade.pair,
            defaults={'starting_equity': self.initial_balance}
        )
        
        metrics.update_metrics()
    
    # ========== US Forex Rules Delegation Methods ==========
    # These methods delegate to the USForexRules class for NFA compliance
    
    def check_hedging_violation(self, symbol: str, signal_type: str) -> bool:
        """Check if opening position would violate no-hedging rule"""
        return self.us_rules.check_hedging_violation(symbol, signal_type)
    
    def close_position_with_fifo(self, symbol: str, exit_price: Decimal, exit_reason: str = 'fifo') -> Dict:
        """Close oldest position on symbol (FIFO compliance)"""
        return self.us_rules.close_position_with_fifo(symbol, exit_price, exit_reason)
    
    def close_position_fifo_compliant(self, symbol: str, exit_price: Decimal) -> Dict:
        """Alias for close_position_with_fifo"""
        return self.us_rules.close_position_fifo_compliant(symbol, exit_price)
    
    def close_all_positions(self, symbol: str, exit_price: Decimal) -> Dict:
        """Close all positions on a symbol (FIFO order)"""
        return self.us_rules.close_all_positions(symbol, exit_price)
    
    def close_partial_position(self, symbol: str, lot_size: Decimal, exit_price: Decimal) -> Dict:
        """Partially close position (FIFO compliant)"""
        return self.us_rules.close_partial_position(symbol, lot_size, exit_price)
    
    def calculate_max_position_size(self, symbol: str, account_balance: Decimal, leverage: Optional[int] = None) -> Dict:
        """Calculate maximum position size based on leverage limits"""
        return self.us_rules.calculate_max_position_size(symbol, account_balance, leverage)
    
    def calculate_position_size(self, account_balance: Decimal, risk_percent: Decimal, stop_loss_pips: int, symbol: str = 'EURUSD') -> Dict:
        """Calculate optimal position size based on risk management"""
        return self.us_rules.calculate_position_size(account_balance, risk_percent, stop_loss_pips, symbol)
    
    def calculate_margin_requirement(self, symbol: str, lot_size: Decimal, entry_price: Decimal, leverage: Optional[int] = None) -> Dict:
        """Calculate margin requirement for a position"""
        return self.us_rules.calculate_margin_requirement(symbol, lot_size, entry_price, leverage)
    
    def calculate_margin_level(self, account_equity: Decimal, used_margin: Decimal) -> Decimal:
        """Calculate margin level percentage"""
        return self.us_rules.calculate_margin_level(account_equity, used_margin)
    
    def calculate_pip_value(self, symbol: str, lot_size: Decimal) -> Decimal:
        """Calculate pip value for a position"""
        return self.us_rules.calculate_pip_value(symbol, lot_size)
