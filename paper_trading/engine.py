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

logger = logging.getLogger(__name__)


class PaperTradingEngine:
    """
    Simulates forex trading without real money
    Manages positions, risk, and performance tracking
    """
    
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = Decimal(str(initial_balance))
        self.current_balance = self.initial_balance
        
    def execute_order(
        self,
        pair: str,
        order_type: str,
        entry_price: float,
        stop_loss: float,
        take_profit_1: float = None,
        take_profit_2: float = None,
        take_profit_3: float = None,
        lot_size: float = 0.01,
        signal_id: str = None,
        signal_type: str = None,
        signal_source: str = None,
        notes: str = None
    ) -> PaperTrade:
        """
        Execute a paper trade order
        
        Args:
            pair: Currency pair (e.g., 'EURUSD')
            order_type: 'buy' or 'sell'
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit_1: First take profit level
            take_profit_2: Second take profit level (optional)
            take_profit_3: Third take profit level (optional)
            lot_size: Position size in lots
            signal_id: Associated signal ID
            signal_type: Type of signal that generated trade
            signal_source: Source model/system
            notes: Additional notes
            
        Returns:
            PaperTrade object
        """
        # Validate order
        if order_type not in ['buy', 'sell']:
            raise ValueError(f"Invalid order_type: {order_type}")
        
        # Calculate risk
        risk_reward_ratio = self._calculate_risk_reward(
            entry_price, stop_loss, take_profit_1, order_type
        )
        
        # Create trade record
        trade = PaperTrade.objects.create(
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
            hit_level, exit_price = self._check_sl_tp_hit(trade, current_price)
            
            if hit_level:
                self.close_position(trade.id, exit_price)
                closed_trades.append({
                    'trade_id': trade.id,
                    'pair': trade.pair,
                    'exit_reason': hit_level,
                    'exit_price': exit_price,
                    'pips': trade.pips_gained,
                    'pnl': trade.profit_loss
                })
                
                logger.info(
                    f"✅ Trade closed: {trade.pair} {hit_level} hit @ {exit_price} "
                    f"(Pips: {trade.pips_gained}, P&L: ${trade.profit_loss})"
                )
        
        return closed_trades
    
    def close_position(
        self,
        position_id: int,
        exit_price: float,
        exit_time: datetime = None
    ) -> Optional[PaperTrade]:
        """
        Close a position manually or via SL/TP
        
        Args:
            position_id: Trade ID to close
            exit_price: Exit price
            exit_time: Exit timestamp (default: now)
            
        Returns:
            Updated PaperTrade object or None if not found
        """
        try:
            trade = PaperTrade.objects.get(id=position_id, status='open')
        except PaperTrade.DoesNotExist:
            logger.warning(f"Trade {position_id} not found or already closed")
            return None
        
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
            days: Number of days to retrieve
            
        Returns:
            List of {date, equity} dicts
        """
        cutoff_date = timezone.now() - timedelta(days=days)
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
                'equity': running_balance,
                'pnl': float(trade.profit_loss or 0),
                'pair': trade.pair
            })
        
        return equity_curve
    
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
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        Check if stop loss or take profit was hit
        
        Returns:
            Tuple of (level_hit, exit_price) or (None, None)
        """
        if trade.order_type == 'buy':
            # Check stop loss
            if current_price <= float(trade.stop_loss):
                return ('STOP_LOSS', float(trade.stop_loss))
            
            # Check take profits (prioritize closest)
            if trade.take_profit_1 and current_price >= float(trade.take_profit_1):
                return ('TAKE_PROFIT_1', float(trade.take_profit_1))
            if trade.take_profit_2 and current_price >= float(trade.take_profit_2):
                return ('TAKE_PROFIT_2', float(trade.take_profit_2))
            if trade.take_profit_3 and current_price >= float(trade.take_profit_3):
                return ('TAKE_PROFIT_3', float(trade.take_profit_3))
        
        else:  # sell
            # Check stop loss
            if current_price >= float(trade.stop_loss):
                return ('STOP_LOSS', float(trade.stop_loss))
            
            # Check take profits (prioritize closest)
            if trade.take_profit_1 and current_price <= float(trade.take_profit_1):
                return ('TAKE_PROFIT_1', float(trade.take_profit_1))
            if trade.take_profit_2 and current_price <= float(trade.take_profit_2):
                return ('TAKE_PROFIT_2', float(trade.take_profit_2))
            if trade.take_profit_3 and current_price <= float(trade.take_profit_3):
                return ('TAKE_PROFIT_3', float(trade.take_profit_3))
        
        return (None, None)
    
    def _update_daily_metrics(self, trade: PaperTrade):
        """Update or create daily performance metrics"""
        trade_date = trade.entry_time.date()
        
        metrics, created = PerformanceMetrics.objects.get_or_create(
            date=trade_date,
            pair=trade.pair,
            defaults={'starting_equity': self.initial_balance}
        )
        
        metrics.update_metrics()
