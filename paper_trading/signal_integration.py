"""
Signal Integration Service
Connects multi-model signal system with paper trading
Auto-executes signals or sends alerts for manual execution
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime
from django.utils import timezone
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

from .engine import PaperTradingEngine
from .data_aggregator import DataAggregator
from .models import PaperTrade

logger = logging.getLogger(__name__)


class SignalIntegrationService:
    """
    Integrates multi-model signal system with paper trading
    Handles signal-to-trade conversion and execution
    """
    
    def __init__(self, user=None, auto_execute: bool = False):
        """
        Initialize signal integration service
        
        Args:
            user: User object (for test compatibility, first arg)
            auto_execute: Whether to automatically execute signals
        """
        # Handle both (user) and (auto_execute=...) signatures
        if isinstance(user, bool):
            auto_execute = user
            user = None
        
        self.user = user
        self.engine = PaperTradingEngine()
        self.aggregator = DataAggregator()
        self.auto_execute = auto_execute
        self.channel_layer = get_channel_layer()
    
    def process_signal(self, signal: Dict, auto_execute: bool = None) -> Dict:
        """
        Process a trading signal
        
        Args:
            signal: Signal dict from multi-model aggregator
            auto_execute: Override instance auto_execute setting
            
        Returns:
            Dict with execution status and details
        """
        logger.info(f"📨 Processing signal: {signal.get('pair', signal.get('symbol'))} {signal.get('direction', signal.get('signal_type'))}")
        
        # Use parameter if provided, otherwise use instance setting
        should_execute = auto_execute if auto_execute is not None else self.auto_execute
        
        # Validate signal
        if not self._validate_signal(signal):
            logger.warning(f"⚠️ Invalid signal: {signal}")
            return {'executed': False, 'error': 'Invalid signal'}
        
        # Send signal alert via WebSocket
        self._broadcast_signal_alert(signal)
        
        # Auto-execute if enabled
        if should_execute:
            trade = self._execute_signal(signal)
            if trade:
                return {'executed': True, 'trade_id': trade.id, 'trade': trade}
            else:
                return {'executed': False, 'error': 'Execution failed'}
        else:
            logger.info(f"📢 Signal alert sent (auto-execute disabled)")
            return {'executed': False, 'action': 'alert_sent'}
    
    def process_signals_batch(self, signals: List[Dict]) -> List[PaperTrade]:
        """
        Process multiple signals
        
        Args:
            signals: List of signal dicts
            
        Returns:
            List of executed PaperTrade objects
        """
        executed_trades = []
        
        for signal in signals:
            trade = self.process_signal(signal)
            if trade:
                executed_trades.append(trade)
        
        logger.info(f"✅ Processed {len(signals)} signals, executed {len(executed_trades)} trades")
        return executed_trades
    
    def _validate_signal(self, signal: Dict) -> bool:
        """Validate signal has required fields"""
        # Check for symbol/pair (accept either)
        if 'symbol' not in signal and 'pair' not in signal:
            logger.error(f"❌ Signal missing symbol/pair field")
            return False
        
        # Check for direction/signal_type (accept either)
        if 'direction' not in signal and 'signal_type' not in signal:
            logger.error(f"❌ Signal missing direction/signal_type field")
            return False
        
        # Check required price fields
        required_fields = ['entry_price', 'stop_loss', 'take_profit_1']
        for field in required_fields:
            if field not in signal or signal[field] is None:
                logger.error(f"❌ Signal missing required field: {field}")
                return False
        
        # Validate direction/signal_type
        direction = signal.get('direction', signal.get('signal_type'))
        if direction.upper() not in ['BUY', 'SELL', 'LONG', 'SHORT']:
            logger.error(f"❌ Invalid direction: {direction}")
            return False
        
        return True
    
    def _execute_signal(self, signal: Dict) -> Optional[PaperTrade]:
        """Execute a validated signal"""
        try:
            # Get symbol/pair
            symbol = signal.get('symbol', signal.get('pair'))
            
            # Normalize direction
            direction = signal.get('direction', signal.get('signal_type', '')).upper()
            if direction in ['LONG', 'BUY']:
                order_type = 'buy'
            else:
                order_type = 'sell'
            
            # Determine lot size based on risk/confidence
            confidence = float(signal.get('confidence', 0.75))
            signal_type = signal.get('type', signal.get('signal_type', 'single'))
            risk_percent = float(signal.get('risk_percent', 2.0))
            lot_size = self._calculate_lot_size(
                confidence=confidence,
                signal_type=signal_type,
                risk_percent=risk_percent
            )
            
            # Execute trade
            execute_params = {
                'pair': symbol,
                'order_type': order_type,
                'entry_price': float(signal['entry_price']),
                'stop_loss': float(signal['stop_loss']),
                'take_profit_1': float(signal.get('take_profit_1')),
                'take_profit_2': float(signal.get('take_profit_2')) if signal.get('take_profit_2') else None,
                'take_profit_3': float(signal.get('take_profit_3')) if signal.get('take_profit_3') else None,
                'lot_size': lot_size,
                'signal_id': signal.get('id', signal.get('signal_id')),
                'signal_type': signal_type,
                'signal_source': signal.get('source', 'multi_model_aggregator'),
                'notes': f"Confidence: {confidence * 100:.1f}%, R:R: {signal.get('risk_reward_ratio', 'N/A')}"
            }
            
            # Add user if available
            if self.user:
                execute_params['user'] = self.user
            
            # Create fresh engine instance (allows mocking in tests)
            engine = PaperTradingEngine()
            trade = engine.execute_order(**execute_params)
            
            logger.info(
                f"✅ Trade executed: {trade.pair} {trade.order_type.upper()} @ {trade.entry_price} "
                f"(SL: {trade.stop_loss}, TP: {trade.take_profit_1})"
            )
            
            # Broadcast trade execution
            self._broadcast_trade_execution(trade)
            
            return trade
            
        except Exception as e:
            logger.error(f"❌ Failed to execute signal: {e}")
            return None
    
    def _calculate_lot_size(
        self, 
        confidence: float = 0.75, 
        signal_type: str = 'single', 
        risk_percent: float = 2.0
    ) -> float:
        """
        Calculate lot size based on confidence and risk management
        
        Args:
            confidence: Signal confidence (0.0-1.0 or 0-100)
            signal_type: Type of signal ('single', 'confluence', etc.)
            risk_percent: Risk percentage per trade
            
        Returns:
            Lot size multiplier
        """
        from decimal import Decimal
        
        # Convert confidence to decimal if needed
        if confidence <= 1.0:
            confidence = confidence * 100
        
        # Base multiplier based on confidence
        if confidence >= 95:
            multiplier = Decimal('2.0')  # High confidence
        elif confidence >= 85:
            multiplier = Decimal('1.5')  # Medium confidence
        elif confidence >= 75:
            multiplier = Decimal('1.0')  # Low confidence
        else:
            multiplier = Decimal('1.0')  # Standard
        
        # Adjust based on signal type
        if 'confluence' in signal_type.lower():
            multiplier *= Decimal('1.5')  # Boost confluence signals
        
        return multiplier
    
    def _validate_signal_prices(
        self,
        signal_type: str,
        entry: float,
        stop_loss: float,
        take_profit: float
    ) -> bool:
        """
        Validate signal prices are logically correct
        
        Args:
            signal_type: 'BUY' or 'SELL'
            entry: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            True if prices are valid, False otherwise
        """
        from decimal import Decimal
        
        # Convert to Decimal for comparison
        entry = Decimal(str(entry))
        stop_loss = Decimal(str(stop_loss))
        take_profit = Decimal(str(take_profit))
        
        if signal_type.upper() == 'BUY':
            # For BUY: SL < entry < TP
            if stop_loss >= entry:
                logger.error(f"❌ BUY signal: SL ({stop_loss}) must be below entry ({entry})")
                return False
            if take_profit <= entry:
                logger.error(f"❌ BUY signal: TP ({take_profit}) must be above entry ({entry})")
                return False
        else:  # SELL
            # For SELL: TP < entry < SL
            if stop_loss <= entry:
                logger.error(f"❌ SELL signal: SL ({stop_loss}) must be above entry ({entry})")
                return False
            if take_profit >= entry:
                logger.error(f"❌ SELL signal: TP ({take_profit}) must be below entry ({entry})")
                return False
        
        return True
    
    def _broadcast_signal_alert(self, signal: Dict):
        """Broadcast signal alert via WebSocket"""
        # Get channel layer (allows mocking in tests)
        channel_layer = get_channel_layer()
        if not channel_layer:
            return
        
        try:
            symbol = signal.get('symbol', signal.get('pair'))
            direction = signal.get('direction', signal.get('signal_type'))
            
            async_to_sync(channel_layer.group_send)(
                'trading_updates',
                {
                    'type': 'signal_alert',
                    'signal': {
                        'id': signal.get('id', signal.get('signal_id')),
                        'pair': symbol,
                        'direction': direction,
                        'entry_price': float(signal['entry_price']),
                        'stop_loss': float(signal['stop_loss']),
                        'take_profit_1': float(signal['take_profit_1']),
                        'take_profit_2': float(signal.get('take_profit_2', 0)) if signal.get('take_profit_2') else None,
                        'take_profit_3': float(signal.get('take_profit_3', 0)) if signal.get('take_profit_3') else None,
                        'confidence': signal.get('confidence'),
                        'type': signal.get('type', signal.get('signal_type')),
                        'risk_reward_ratio': signal.get('risk_reward_ratio'),
                    },
                    'timestamp': timezone.now().isoformat()
                }
            )
        except Exception as e:
            logger.error(f"Failed to broadcast signal alert: {e}")
    
    def _broadcast_trade_execution(self, trade: PaperTrade):
        """Broadcast trade execution via WebSocket"""
        if not self.channel_layer:
            return
        
        try:
            async_to_sync(self.channel_layer.group_send)(
                'trading_updates',
                {
                    'type': 'trade_execution',
                    'trade': {
                        'id': trade.id,
                        'pair': trade.pair,
                        'order_type': trade.order_type,
                        'entry_price': float(trade.entry_price),
                        'stop_loss': float(trade.stop_loss),
                        'take_profit_1': float(trade.take_profit_1) if trade.take_profit_1 else None,
                        'lot_size': float(trade.lot_size),
                        'signal_type': trade.signal_type,
                        'entry_time': trade.entry_time.isoformat(),
                    },
                    'timestamp': timezone.now().isoformat()
                }
            )
        except Exception as e:
            logger.error(f"Failed to broadcast trade execution: {e}")
    
    def get_signal_summary(self, days: int = 7) -> Dict:
        """Get summary of signals processed and executed"""
        from datetime import timedelta
        
        cutoff = timezone.now() - timedelta(days=days)
        
        # Count trades from signals
        signal_trades = PaperTrade.objects.filter(
            entry_time__gte=cutoff,
            signal_type__isnull=False
        )
        
        total_signal_trades = signal_trades.count()
        
        if total_signal_trades == 0:
            return {
                'total_signals_processed': 0,
                'signals_executed': 0,
                'signals_alerted': 0,
                'total_trades_executed': 0,
                'execution_rate': 0.0,
                'win_rate': 0.0,
                'avg_pips': 0.0
            }
        
        closed_signal_trades = signal_trades.filter(status='closed')
        winning_trades = closed_signal_trades.filter(pips_gained__gt=0).count()
        
        from django.db.models import Avg
        avg_pips = closed_signal_trades.aggregate(Avg('pips_gained'))['pips_gained__avg'] or 0.0
        
        return {
            'days': days,
            'total_signals_processed': total_signal_trades,  # Total signals received/processed
            'signals_executed': total_signal_trades,  # Signals that resulted in trades
            'signals_alerted': 0,  # Signals that were only alerts (not tracked separately yet)
            'total_trades_executed': total_signal_trades,
            'open_trades': signal_trades.filter(status='open').count(),
            'closed_trades': closed_signal_trades.count(),
            'winning_trades': winning_trades,
            'win_rate': round((winning_trades / closed_signal_trades.count() * 100), 2) if closed_signal_trades.count() > 0 else 0.0,
            'avg_pips': round(avg_pips, 2)
        }
