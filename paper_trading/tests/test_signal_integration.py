"""
Tests for Signal Integration Service
"""
from decimal import Decimal
from django.test import TestCase
from django.contrib.auth import get_user_model
from django.utils import timezone
from unittest.mock import Mock, patch, MagicMock
from paper_trading.signal_integration import SignalIntegrationService
from paper_trading.models import PaperTrade

User = get_user_model()


class SignalIntegrationServiceTest(TestCase):
    """Test Signal Integration Service"""
    
    def setUp(self):
        """Set up test user and service"""
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        self.service = SignalIntegrationService(self.user)
    
    def test_calculate_lot_size_low_confidence(self):
        """Test lot size calculation for low confidence signal"""
        lot_size = self.service._calculate_lot_size(
            confidence=0.75,
            signal_type='single',
            risk_percent=2.0
        )
        
        # Low confidence (75%) should use 1x multiplier
        self.assertEqual(lot_size, Decimal('1.0'))
    
    def test_calculate_lot_size_medium_confidence(self):
        """Test lot size calculation for medium confidence signal"""
        lot_size = self.service._calculate_lot_size(
            confidence=0.85,
            signal_type='single',
            risk_percent=2.0
        )
        
        # Medium confidence (85%) should use 1.5x multiplier
        self.assertEqual(lot_size, Decimal('1.5'))
    
    def test_calculate_lot_size_high_confidence(self):
        """Test lot size calculation for high confidence signal"""
        lot_size = self.service._calculate_lot_size(
            confidence=0.95,
            signal_type='single',
            risk_percent=2.0
        )
        
        # High confidence (95%) should use 2x multiplier
        self.assertEqual(lot_size, Decimal('2.0'))
    
    def test_calculate_lot_size_confluence_signal(self):
        """Test lot size calculation for confluence signal"""
        lot_size = self.service._calculate_lot_size(
            confidence=0.85,
            signal_type='confluence',
            risk_percent=2.0
        )
        
        # Confluence gets 1.5x boost: 1.5 * 1.5 = 2.25
        self.assertEqual(lot_size, Decimal('2.25'))
    
    @patch('paper_trading.signal_integration.PaperTradingEngine')
    def test_process_signal_execute_trade(self, mock_engine_class):
        """Test processing signal and executing trade"""
        # Mock the engine
        mock_engine = MagicMock()
        mock_trade = Mock()
        mock_trade.id = 1
        mock_trade.symbol = 'EURUSD'
        mock_engine.execute_order.return_value = mock_trade
        mock_engine_class.return_value = mock_engine
        
        signal_data = {
            'symbol': 'EURUSD',
            'signal_type': 'BUY',
            'entry_price': Decimal('1.1000'),
            'stop_loss': Decimal('1.0950'),
            'take_profit_1': Decimal('1.1100'),
            'confidence': 0.85,
            'signal_source': 'ML Model'
        }
        
        result = self.service.process_signal(
            signal_data,
            auto_execute=True
        )
        
        self.assertTrue(result['executed'])
        self.assertEqual(result['trade_id'], 1)
        mock_engine.execute_order.assert_called_once()
    
    @patch('paper_trading.signal_integration.PaperTradingEngine')
    def test_process_signal_alert_only(self, mock_engine_class):
        """Test processing signal with alert only (no execution)"""
        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine
        
        signal_data = {
            'symbol': 'EURUSD',
            'signal_type': 'BUY',
            'entry_price': Decimal('1.1000'),
            'stop_loss': Decimal('1.0950'),
            'take_profit_1': Decimal('1.1100'),
            'confidence': 0.85,
            'signal_source': 'Harmonic Pattern'
        }
        
        result = self.service.process_signal(
            signal_data,
            auto_execute=False
        )
        
        self.assertFalse(result['executed'])
        self.assertEqual(result['action'], 'alert_sent')
        mock_engine.execute_order.assert_not_called()
    
    def test_process_signal_validation_missing_fields(self):
        """Test signal validation with missing required fields"""
        signal_data = {
            'symbol': 'EURUSD',
            'signal_type': 'BUY'
            # Missing entry_price, stop_loss, take_profit_1
        }
        
        result = self.service.process_signal(signal_data)
        
        self.assertFalse(result['executed'])
        self.assertIn('error', result)
    
    def test_process_signal_validation_invalid_type(self):
        """Test signal validation with invalid signal type"""
        signal_data = {
            'symbol': 'EURUSD',
            'signal_type': 'INVALID',
            'entry_price': Decimal('1.1000'),
            'stop_loss': Decimal('1.0950'),
            'take_profit_1': Decimal('1.1100'),
            'confidence': 0.85
        }
        
        result = self.service.process_signal(signal_data)
        
        self.assertFalse(result['executed'])
        self.assertIn('error', result)
    
    @patch('paper_trading.signal_integration.PaperTradingEngine')
    def test_execute_signal_with_multiple_tps(self, mock_engine_class):
        """Test executing signal with multiple take profits"""
        mock_engine = MagicMock()
        mock_trade = Mock()
        mock_trade.id = 1
        mock_engine.execute_order.return_value = mock_trade
        mock_engine_class.return_value = mock_engine
        
        signal_data = {
            'symbol': 'EURUSD',
            'signal_type': 'BUY',
            'entry_price': Decimal('1.1000'),
            'stop_loss': Decimal('1.0950'),
            'take_profit_1': Decimal('1.1100'),
            'take_profit_2': Decimal('1.1150'),
            'take_profit_3': Decimal('1.1200'),
            'confidence': 0.85
        }
        
        trade = self.service._execute_signal(signal_data)
        
        self.assertIsNotNone(trade)
        # Verify execute_order was called with all TPs
        call_args = mock_engine.execute_order.call_args
        self.assertIn('take_profit_2', call_args.kwargs)
        self.assertIn('take_profit_3', call_args.kwargs)
    
    def test_get_signal_summary(self):
        """Test getting signal summary"""
        # Process some signals
        signal_data = {
            'symbol': 'EURUSD',
            'signal_type': 'BUY',
            'entry_price': Decimal('1.1000'),
            'stop_loss': Decimal('1.0950'),
            'take_profit_1': Decimal('1.1100'),
            'confidence': 0.85
        }
        
        with patch('paper_trading.signal_integration.PaperTradingEngine'):
            self.service.process_signal(signal_data, auto_execute=True)
            self.service.process_signal(signal_data, auto_execute=False)
        
        summary = self.service.get_signal_summary()
        
        self.assertIn('total_signals_processed', summary)
        self.assertIn('signals_executed', summary)
        self.assertIn('signals_alerted', summary)
    
    @patch('paper_trading.signal_integration.async_to_sync')
    @patch('paper_trading.signal_integration.get_channel_layer')
    def test_broadcast_signal_alert(self, mock_channel_layer, mock_async):
        """Test broadcasting signal alert via WebSocket"""
        mock_layer = MagicMock()
        mock_channel_layer.return_value = mock_layer
        
        signal_data = {
            'symbol': 'EURUSD',
            'signal_type': 'BUY',
            'entry_price': Decimal('1.1000'),
            'stop_loss': Decimal('1.0950'),
            'take_profit_1': Decimal('1.1100'),
            'confidence': 0.85
        }
        
        self.service._broadcast_signal_alert(signal_data)
        
        # Verify channel layer was called
        mock_channel_layer.assert_called_once()
    
    def test_validate_signal_buy_prices(self):
        """Test price validation for BUY signal"""
        # Valid BUY signal: entry > SL, TP > entry
        valid = self.service._validate_signal_prices(
            signal_type='BUY',
            entry=Decimal('1.1000'),
            stop_loss=Decimal('1.0950'),
            take_profit=Decimal('1.1100')
        )
        self.assertTrue(valid)
        
        # Invalid: SL above entry
        invalid = self.service._validate_signal_prices(
            signal_type='BUY',
            entry=Decimal('1.1000'),
            stop_loss=Decimal('1.1050'),
            take_profit=Decimal('1.1100')
        )
        self.assertFalse(invalid)
    
    def test_validate_signal_sell_prices(self):
        """Test price validation for SELL signal"""
        # Valid SELL signal: entry < SL, TP < entry
        valid = self.service._validate_signal_prices(
            signal_type='SELL',
            entry=Decimal('1.1000'),
            stop_loss=Decimal('1.1050'),
            take_profit=Decimal('1.0900')
        )
        self.assertTrue(valid)
        
        # Invalid: SL below entry
        invalid = self.service._validate_signal_prices(
            signal_type='SELL',
            entry=Decimal('1.1000'),
            stop_loss=Decimal('1.0950'),
            take_profit=Decimal('1.0900')
        )
        self.assertFalse(invalid)
