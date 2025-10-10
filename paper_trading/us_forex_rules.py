"""
US Forex Rules Extension for Paper Trading Engine
Implements FIFO, no-hedging, leverage limits, and position sizing
"""
from decimal import Decimal
from typing import Dict, List, Optional
from django.db.models import Q
from django.utils import timezone
import logging

logger = logging.getLogger(__name__)


class USForexRules:
    """
    US NFA Forex Trading Rules Implementation
    - FIFO (First In, First Out)
    - No Hedging
    - Leverage Limits (50:1 major, 20:1 minor)
    - Position Sizing
    """
    
    # Leverage limits by pair type
    MAJOR_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD']
    MAJOR_PAIR_LEVERAGE = 50
    MINOR_PAIR_LEVERAGE = 20
    
    # Pip values
    JPY_PAIRS = ['USDJPY', 'EURJPY', 'GBPJPY', 'AUDJPY', 'CHFJPY', 'CADJPY', 'NZDJPY']
    
    def __init__(self, engine):
        self.engine = engine
    
    def check_hedging_violation(self, symbol: str, signal_type: str) -> bool:
        """
        Check if opening this position would violate no-hedging rule
        
        Args:
            symbol: Trading pair
            signal_type: 'BUY' or 'SELL'
            
        Returns:
            True if would violate (opposing position exists), False if OK
        """
        from paper_trading.models import PaperTrade
        
        opposite_type = 'SELL' if signal_type == 'BUY' else 'BUY'
        
        # Check for existing positions in opposite direction
        opposing_positions = PaperTrade.objects.filter(
            user=self.engine.user,
            pair=symbol,
            order_type=opposite_type.lower(),
            status='open'
        ).exists()
        
        if opposing_positions:
            logger.warning(
                f"❌ Hedging violation: Cannot open {signal_type} position on {symbol} "
                f"- opposing {opposite_type} position(s) exist"
            )
            return True
        
        return False
    
    def close_position_with_fifo(
        self,
        symbol: str,
        exit_price: Decimal,
        exit_reason: str = 'fifo'
    ) -> Dict:
        """
        Close oldest position on symbol (FIFO compliance)
        
        Args:
            symbol: Trading pair
            exit_price: Exit price
            exit_reason: Reason for exit
            
        Returns:
            Dict with closed trade info
        """
        from paper_trading.models import PaperTrade
        
        # Get oldest open position for this symbol
        oldest_position = PaperTrade.objects.filter(
            user=self.engine.user,
            pair=symbol,
            status='open'
        ).order_by('entry_time').first()
        
        if not oldest_position:
            return {
                'success': False,
                'error': f'No open positions found for {symbol}'
            }
        
        # Close the oldest position
        close_result = self.engine.close_position(
            trade_id=oldest_position.id,
            exit_price=exit_price,
            exit_reason=exit_reason
        )
        
        logger.info(
            f"✅ FIFO Close: {symbol} position #{oldest_position.id} "
            f"(opened {oldest_position.entry_time})"
        )
        
        return close_result
    
    def close_position_fifo_compliant(
        self,
        symbol: str,
        exit_price: Decimal
    ) -> Dict:
        """
        Alias for close_position_with_fifo for clearer intent
        """
        return self.close_position_with_fifo(symbol, exit_price, 'manual_fifo')
    
    def close_all_positions(
        self,
        symbol: str,
        exit_price: Decimal
    ) -> Dict:
        """
        Close all positions on a symbol (oldest first - FIFO)
        
        Args:
            symbol: Trading pair
            exit_price: Exit price
            
        Returns:
            Dict with summary
        """
        from paper_trading.models import PaperTrade
        
        positions = PaperTrade.objects.filter(
            user=self.engine.user,
            pair=symbol,
            status='open'
        ).order_by('entry_time')  # FIFO order
        
        closed_count = 0
        total_pips = Decimal('0')
        total_profit = Decimal('0')
        
        for position in positions:
            result = self.engine.close_position(
                trade_id=position.id,
                exit_price=exit_price,
                exit_reason='close_all'
            )
            
            if result['success']:
                closed_count += 1
                total_pips += result['pips_gained']
                total_profit += result['profit_loss']
        
        logger.info(
            f"✅ Closed all {closed_count} {symbol} positions: "
            f"{total_pips} pips, ${total_profit}"
        )
        
        return {
            'success': True,
            'closed_count': closed_count,
            'total_pips': total_pips,
            'total_profit': total_profit,
            'symbol': symbol
        }
    
    def close_partial_position(
        self,
        symbol: str,
        lot_size: Decimal,
        exit_price: Decimal
    ) -> Dict:
        """
        Partially close position (reduce lot size) - FIFO compliant
        
        Args:
            symbol: Trading pair
            lot_size: Amount to close
            exit_price: Exit price
            
        Returns:
            Dict with result
        """
        from paper_trading.models import PaperTrade
        
        # Get oldest position
        oldest_position = PaperTrade.objects.filter(
            user=self.engine.user,
            pair=symbol,
            status='open'
        ).order_by('entry_time').first()
        
        if not oldest_position:
            return {'success': False, 'error': 'No open positions'}
        
        if lot_size >= oldest_position.lot_size:
            # Close entire position
            return self.close_position_with_fifo(symbol, exit_price, 'partial_full')
        
        # Calculate partial profit
        pips_gained = self.engine._calculate_pips(
            symbol=symbol,
            entry=oldest_position.entry_price,
            exit=exit_price
        )
        
        if oldest_position.signal_type == 'SELL':
            pips_gained = -pips_gained
        
        pip_value = self._calculate_pip_value(symbol, lot_size)
        profit = pips_gained * pip_value
        
        # Reduce position size
        oldest_position.lot_size -= lot_size
        oldest_position.save()
        
        logger.info(
            f"✅ Partial close: {symbol} reduced by {lot_size} lots, "
            f"{pips_gained} pips, ${profit}"
        )
        
        return {
            'success': True,
            'partial_close': True,
            'lot_size_closed': lot_size,
            'remaining_lot_size': oldest_position.lot_size,
            'pips_gained': pips_gained,
            'profit': profit
        }
    
    def calculate_max_position_size(
        self,
        symbol: str,
        account_balance: Decimal,
        leverage: Optional[int] = None
    ) -> Dict:
        """
        Calculate maximum position size based on leverage limits
        
        Args:
            symbol: Trading pair
            account_balance: Current account balance
            leverage: Override leverage (or use default for pair type)
            
        Returns:
            Dict with max lot size and leverage info
        """
        # Determine leverage limit
        if leverage is None:
            leverage = (self.MAJOR_PAIR_LEVERAGE if symbol in self.MAJOR_PAIRS 
                       else self.MINOR_PAIR_LEVERAGE)
        
        # Max position value with leverage
        max_position_value = account_balance * Decimal(leverage)
        
        # Convert to lot size (1 lot = 100,000 units)
        max_lot_size = max_position_value / Decimal('100000')
        
        return {
            'max_lot_size': max_lot_size.quantize(Decimal('0.01')),
            'leverage_used': leverage,
            'max_position_value': max_position_value,
            'account_balance': account_balance
        }
    
    def calculate_position_size(
        self,
        account_balance: Decimal,
        risk_percent: Decimal,
        stop_loss_pips: int,
        symbol: str = 'EURUSD'
    ) -> Dict:
        """
        Calculate optimal position size based on risk management
        
        Args:
            account_balance: Current account balance
            risk_percent: Percentage of account to risk (e.g., 1.0 or 2.0)
            stop_loss_pips: Stop loss distance in pips
            symbol: Trading pair
            
        Returns:
            Dict with lot size, risk amount, and other info
        """
        # Calculate risk amount
        risk_amount = account_balance * (risk_percent / Decimal('100'))
        
        # Calculate pip value for position sizing
        # Standard lot (1.0) = $10/pip for major pairs
        pip_value_per_lot = Decimal('10.0') if symbol not in self.JPY_PAIRS else Decimal('9.0')
        
        # Calculate lot size
        # Risk amount = stop loss pips * pip value * lot size
        # lot_size = risk_amount / (stop_loss_pips * pip_value_per_lot)
        lot_size = risk_amount / (Decimal(stop_loss_pips) * pip_value_per_lot)
        
        # Round to 3 decimal places to support micro lots (0.001)
        lot_size = lot_size.quantize(Decimal('0.001'))
        
        return {
            'lot_size': lot_size,
            'risk_amount': risk_amount,
            'risk_percent': risk_percent,
            'stop_loss_pips': stop_loss_pips,
            'pip_value': pip_value_per_lot * lot_size,
            'account_balance': account_balance
        }
    
    def calculate_margin_requirement(
        self,
        symbol: str,
        lot_size: Decimal,
        entry_price: Decimal,
        leverage: Optional[int] = None
    ) -> Dict:
        """
        Calculate margin requirement for a position
        
        Args:
            symbol: Trading pair
            lot_size: Position size in lots
            entry_price: Entry price
            leverage: Leverage (or default for pair type)
            
        Returns:
            Dict with margin info
        """
        if leverage is None:
            leverage = (self.MAJOR_PAIR_LEVERAGE if symbol in self.MAJOR_PAIRS 
                       else self.MINOR_PAIR_LEVERAGE)
        
        # Position value = lot size * contract size * entry price
        position_value = lot_size * Decimal('100000') * entry_price
        
        # Margin required = position value / leverage
        margin_required = position_value / Decimal(leverage)
        
        return {
            'position_value': position_value.quantize(Decimal('0.01')),
            'margin_required': margin_required.quantize(Decimal('0.01')),
            'leverage_used': leverage,
            'lot_size': lot_size,
            'entry_price': entry_price
        }
    
    def calculate_margin_level(
        self,
        account_equity: Decimal,
        used_margin: Decimal
    ) -> Decimal:
        """
        Calculate margin level percentage
        
        Args:
            account_equity: Current account equity (balance + floating P&L)
            used_margin: Total margin used by open positions
            
        Returns:
            Margin level as percentage
        """
        if used_margin == 0:
            return Decimal('999999.99')  # Infinity (no positions)
        
        margin_level = (account_equity / used_margin) * Decimal('100')
        return margin_level.quantize(Decimal('0.01'))
    
    def calculate_pip_value(
        self,
        symbol: str,
        lot_size: Decimal
    ) -> Decimal:
        """
        Calculate pip value for a position
        
        Args:
            symbol: Trading pair
            lot_size: Position size in lots
            
        Returns:
            Pip value in account currency (USD)
        """
        # Standard lot = 100,000 units
        # For EURUSD: 1 pip = 0.0001
        # Pip value = lot_size * 100,000 * 0.0001 = lot_size * 10
        
        if symbol in self.JPY_PAIRS:
            # JPY pairs: 1 pip = 0.01
            pip_value = lot_size * Decimal('1000') * Decimal('0.01')
        else:
            # Major pairs: 1 pip = 0.0001
            pip_value = lot_size * Decimal('100000') * Decimal('0.0001')
        
        return pip_value.quantize(Decimal('0.01'))
    
    def _calculate_pip_value(self, symbol: str, lot_size: Decimal) -> Decimal:
        """Alias for calculate_pip_value"""
        return self.calculate_pip_value(symbol, lot_size)


def add_us_forex_rules_to_engine(engine_class):
    """
    Add US forex rules methods to PaperTradingEngine
    
    Usage:
        from paper_trading.engine import PaperTradingEngine
        from paper_trading.us_forex_rules import add_us_forex_rules_to_engine
        
        add_us_forex_rules_to_engine(PaperTradingEngine)
    """
    
    def _init_us_rules(self):
        if not hasattr(self, '_us_rules'):
            self._us_rules = USForexRules(self)
        return self._us_rules
    
    # Add methods to engine class
    engine_class.check_hedging_violation = lambda self, symbol, signal_type: _init_us_rules(self).check_hedging_violation(symbol, signal_type)
    engine_class.close_position_with_fifo = lambda self, symbol, exit_price, exit_reason='fifo': _init_us_rules(self).close_position_with_fifo(symbol, exit_price, exit_reason)
    engine_class.close_position_fifo_compliant = lambda self, symbol, exit_price: _init_us_rules(self).close_position_fifo_compliant(symbol, exit_price)
    engine_class.close_all_positions = lambda self, symbol, exit_price: _init_us_rules(self).close_all_positions(symbol, exit_price)
    engine_class.close_partial_position = lambda self, symbol, lot_size, exit_price: _init_us_rules(self).close_partial_position(symbol, lot_size, exit_price)
    engine_class.calculate_max_position_size = lambda self, symbol, account_balance, leverage=None: _init_us_rules(self).calculate_max_position_size(symbol, account_balance, leverage)
    engine_class.calculate_position_size = lambda self, account_balance, risk_percent, stop_loss_pips, symbol='EURUSD': _init_us_rules(self).calculate_position_size(account_balance, risk_percent, stop_loss_pips, symbol)
    engine_class.calculate_margin_requirement = lambda self, symbol, lot_size, entry_price, leverage=None: _init_us_rules(self).calculate_margin_requirement(symbol, lot_size, entry_price, leverage)
    engine_class.calculate_margin_level = lambda self, account_equity, used_margin: _init_us_rules(self).calculate_margin_level(account_equity, used_margin)
    engine_class.calculate_pip_value = lambda self, symbol, lot_size: _init_us_rules(self).calculate_pip_value(symbol, lot_size)
    
    logger.info("✅ US Forex Rules extension added to PaperTradingEngine")
