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
    
    def __init__(self, auto_execute: bool = False):
        """
        Initialize signal integration service
        
        Args:
            auto_execute: Whether to automatically execute signals
        """
        self.engine = PaperTradingEngine()
        self.aggregator = DataAggregator()
        self.auto_execute = auto_execute
        self.channel_layer = get_channel_layer()
    
    def process_signal(self, signal: Dict) -> Optional[PaperTrade]:
        """
        Process a trading signal
        
        Args:
            signal: Signal dict from multi-model aggregator
            
        Returns:
            PaperTrade if executed, None if only alert sent
        """
        logger.info(f"📨 Processing signal: {signal.get('pair')} {signal.get('direction')}")
        
        # Validate signal
        if not self._validate_signal(signal):
            logger.warning(f"⚠️ Invalid signal: {signal}")
            return None
        
        # Send signal alert via WebSocket
        self._broadcast_signal_alert(signal)
        
        # Auto-execute if enabled
        if self.auto_execute:
            return self._execute_signal(signal)
        else:
            logger.info(f"📢 Signal alert sent (auto-execute disabled)")
            return None
    
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
        required_fields = ['pair', 'direction', 'entry_price', 'stop_loss', 'take_profit_1']
        
        for field in required_fields:
            if field not in signal or signal[field] is None:
                logger.error(f"❌ Signal missing required field: {field}")
                return False
        
        # Validate direction
        if signal['direction'] not in ['buy', 'sell', 'LONG', 'SHORT']:
            logger.error(f"❌ Invalid direction: {signal['direction']}")
            return False
        
        return True
    
    def _execute_signal(self, signal: Dict) -> Optional[PaperTrade]:
        """Execute a validated signal"""
        try:
            # Normalize direction
            direction = signal['direction'].lower()
            if direction in ['long', 'buy']:
                order_type = 'buy'
            else:
                order_type = 'sell'
            
            # Determine lot size based on risk/confidence
            lot_size = self._calculate_lot_size(signal)
            
            # Execute trade
            trade = self.engine.execute_order(
                pair=signal['pair'],
                order_type=order_type,
                entry_price=float(signal['entry_price']),
                stop_loss=float(signal['stop_loss']),
                take_profit_1=float(signal.get('take_profit_1')),
                take_profit_2=float(signal.get('take_profit_2')) if signal.get('take_profit_2') else None,
                take_profit_3=float(signal.get('take_profit_3')) if signal.get('take_profit_3') else None,
                lot_size=lot_size,
                signal_id=signal.get('id', signal.get('signal_id')),
                signal_type=signal.get('type', signal.get('signal_type')),
                signal_source=signal.get('source', 'multi_model_aggregator'),
                notes=f"Confidence: {signal.get('confidence', 'N/A')}%, R:R: {signal.get('risk_reward_ratio', 'N/A')}"
            )
            
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
    
    def _calculate_lot_size(self, signal: Dict) -> float:
        """Calculate lot size based on confidence and risk management"""
        # Base lot size
        base_lot_size = 0.01
        
        # Adjust based on confidence
        confidence = signal.get('confidence', 75)
        
        if confidence >= 95:
            multiplier = 3.0  # Ultra high confidence
        elif confidence >= 85:
            multiplier = 2.0  # High confidence
        elif confidence >= 75:
            multiplier = 1.5  # Good confidence
        else:
            multiplier = 1.0  # Standard confidence
        
        # Adjust based on signal type
        signal_type = signal.get('type', '').lower()
        if 'ultra' in signal_type or 'confluence' in signal_type:
            multiplier *= 1.5  # Boost confluence signals
        
        lot_size = base_lot_size * multiplier
        
        # Cap at maximum
        max_lot_size = 0.10  # 0.1 lots max for safety
        return min(lot_size, max_lot_size)
    
    def _broadcast_signal_alert(self, signal: Dict):
        """Broadcast signal alert via WebSocket"""
        if not self.channel_layer:
            return
        
        try:
            async_to_sync(self.channel_layer.group_send)(
                'trading_updates',
                {
                    'type': 'signal_alert',
                    'signal': {
                        'id': signal.get('id', signal.get('signal_id')),
                        'pair': signal['pair'],
                        'direction': signal['direction'],
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
            'total_trades_executed': total_signal_trades,
            'open_trades': signal_trades.filter(status='open').count(),
            'closed_trades': closed_signal_trades.count(),
            'winning_trades': winning_trades,
            'win_rate': round((winning_trades / closed_signal_trades.count() * 100), 2) if closed_signal_trades.count() > 0 else 0.0,
            'avg_pips': round(avg_pips, 2)
        }
