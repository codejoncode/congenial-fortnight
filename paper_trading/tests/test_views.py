"""
Tests for Paper Trading API Views
"""
from decimal import Decimal
from django.test import TestCase
from django.contrib.auth import get_user_model
from django.utils import timezone
from django.urls import reverse
from rest_framework.test import APIClient
from rest_framework import status
from paper_trading.models import PaperTrade

User = get_user_model()


class PaperTradeViewSetTest(TestCase):
    """Test Paper Trade API ViewSet"""
    
    def setUp(self):
        """Set up test client and user"""
        self.client = APIClient()
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        self.client.force_authenticate(user=self.user)
    
    def test_list_trades(self):
        """Test listing all trades"""
        # Create some trades
        PaperTrade.objects.create(
            user=self.user,
            symbol='EURUSD',
            signal_type='BUY',
            entry_price=Decimal('1.1000'),
            stop_loss=Decimal('1.0950'),
            take_profit_1=Decimal('1.1100'),
            lot_size=Decimal('1.0'),
            entry_time=timezone.now(),
            status='open'
        )
        
        url = reverse('papertrade-list')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 1)
    
    def test_create_trade_not_allowed(self):
        """Test that direct trade creation is not allowed"""
        url = reverse('papertrade-list')
        data = {
            'symbol': 'EURUSD',
            'signal_type': 'BUY',
            'entry_price': '1.1000',
            'stop_loss': '1.0950',
            'take_profit_1': '1.1100',
            'lot_size': '1.0'
        }
        
        response = self.client.post(url, data)
        
        # Should not allow direct creation
        self.assertNotEqual(response.status_code, status.HTTP_201_CREATED)
    
    def test_execute_trade_endpoint(self):
        """Test execute trade endpoint"""
        url = reverse('papertrade-execute')
        data = {
            'symbol': 'EURUSD',
            'signal_type': 'BUY',
            'entry_price': '1.1000',
            'stop_loss': '1.0950',
            'take_profit_1': '1.1100',
            'lot_size': '1.0'
        }
        
        response = self.client.post(url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['symbol'], 'EURUSD')
        self.assertEqual(response.data['status'], 'open')
    
    def test_execute_trade_missing_required_fields(self):
        """Test execute trade with missing fields"""
        url = reverse('papertrade-execute')
        data = {
            'symbol': 'EURUSD',
            'signal_type': 'BUY'
            # Missing entry_price, stop_loss, etc.
        }
        
        response = self.client.post(url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
    
    def test_close_position_endpoint(self):
        """Test close position endpoint"""
        # Create a trade
        trade = PaperTrade.objects.create(
            user=self.user,
            symbol='EURUSD',
            signal_type='BUY',
            entry_price=Decimal('1.1000'),
            stop_loss=Decimal('1.0950'),
            take_profit_1=Decimal('1.1100'),
            lot_size=Decimal('1.0'),
            entry_time=timezone.now(),
            status='open'
        )
        
        url = reverse('papertrade-close', kwargs={'pk': trade.id})
        data = {
            'exit_price': '1.1050',
            'reason': 'manual'
        }
        
        response = self.client.post(url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['status'], 'closed')
        self.assertEqual(response.data['exit_reason'], 'manual')
    
    def test_get_open_positions(self):
        """Test get open positions endpoint"""
        # Create open and closed trades
        PaperTrade.objects.create(
            user=self.user,
            symbol='EURUSD',
            signal_type='BUY',
            entry_price=Decimal('1.1000'),
            stop_loss=Decimal('1.0950'),
            take_profit_1=Decimal('1.1100'),
            lot_size=Decimal('1.0'),
            entry_time=timezone.now(),
            status='open'
        )
        
        PaperTrade.objects.create(
            user=self.user,
            symbol='XAUUSD',
            signal_type='SELL',
            entry_price=Decimal('1950.00'),
            stop_loss=Decimal('1955.00'),
            take_profit_1=Decimal('1940.00'),
            lot_size=Decimal('0.5'),
            entry_time=timezone.now(),
            exit_time=timezone.now(),
            status='closed'
        )
        
        url = reverse('papertrade-open-positions')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0]['status'], 'open')
    
    def test_get_performance_summary(self):
        """Test get performance summary endpoint"""
        # Create and close a trade
        trade = PaperTrade.objects.create(
            user=self.user,
            symbol='EURUSD',
            signal_type='BUY',
            entry_price=Decimal('1.1000'),
            stop_loss=Decimal('1.0950'),
            take_profit_1=Decimal('1.1100'),
            lot_size=Decimal('1.0'),
            entry_time=timezone.now(),
            exit_time=timezone.now(),
            exit_price=Decimal('1.1050'),
            pips_gained=Decimal('50.0'),
            profit_loss=Decimal('500.00'),
            status='closed'
        )
        
        url = reverse('papertrade-performance')
        response = self.client.get(url, {'days': 30})
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('total_trades', response.data)
        self.assertIn('win_rate', response.data)
    
    def test_get_equity_curve(self):
        """Test get equity curve endpoint"""
        url = reverse('papertrade-equity-curve')
        response = self.client.get(url, {'initial_balance': 10000})
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIsInstance(response.data, list)
    
    def test_update_positions_endpoint(self):
        """Test update positions endpoint"""
        # Create an open trade
        PaperTrade.objects.create(
            user=self.user,
            symbol='EURUSD',
            signal_type='BUY',
            entry_price=Decimal('1.1000'),
            stop_loss=Decimal('1.0950'),
            take_profit_1=Decimal('1.1100'),
            lot_size=Decimal('1.0'),
            entry_time=timezone.now(),
            status='open'
        )
        
        url = reverse('papertrade-update-positions')
        response = self.client.post(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('updated', response.data)
    
    def test_authentication_required(self):
        """Test that authentication is required"""
        self.client.force_authenticate(user=None)
        
        url = reverse('papertrade-list')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
    
    def test_user_can_only_see_own_trades(self):
        """Test that users can only see their own trades"""
        # Create another user with trades
        other_user = User.objects.create_user(
            username='otheruser',
            email='other@example.com',
            password='testpass123'
        )
        
        PaperTrade.objects.create(
            user=other_user,
            symbol='EURUSD',
            signal_type='BUY',
            entry_price=Decimal('1.1000'),
            stop_loss=Decimal('1.0950'),
            take_profit_1=Decimal('1.1100'),
            lot_size=Decimal('1.0'),
            entry_time=timezone.now(),
            status='open'
        )
        
        # Our user should not see other user's trades
        url = reverse('papertrade-list')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 0)
    
    def test_filter_trades_by_pair(self):
        """Test filtering trades by currency pair"""
        PaperTrade.objects.create(
            user=self.user,
            symbol='EURUSD',
            signal_type='BUY',
            entry_price=Decimal('1.1000'),
            stop_loss=Decimal('1.0950'),
            take_profit_1=Decimal('1.1100'),
            lot_size=Decimal('1.0'),
            entry_time=timezone.now(),
            status='open'
        )
        
        PaperTrade.objects.create(
            user=self.user,
            symbol='GBPUSD',
            signal_type='SELL',
            entry_price=Decimal('1.2500'),
            stop_loss=Decimal('1.2550'),
            take_profit_1=Decimal('1.2400'),
            lot_size=Decimal('0.5'),
            entry_time=timezone.now(),
            status='open'
        )
        
        url = reverse('papertrade-list') + '?pair=EURUSD'
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0]['symbol'], 'EURUSD')
    
    def test_filter_trades_by_status(self):
        """Test filtering trades by status"""
        PaperTrade.objects.create(
            user=self.user,
            symbol='EURUSD',
            signal_type='BUY',
            entry_price=Decimal('1.1000'),
            stop_loss=Decimal('1.0950'),
            take_profit_1=Decimal('1.1100'),
            lot_size=Decimal('1.0'),
            entry_time=timezone.now(),
            status='open'
        )
        
        PaperTrade.objects.create(
            user=self.user,
            symbol='GBPUSD',
            signal_type='SELL',
            entry_price=Decimal('1.2500'),
            stop_loss=Decimal('1.2550'),
            take_profit_1=Decimal('1.2400'),
            lot_size=Decimal('0.5'),
            entry_time=timezone.now(),
            exit_time=timezone.now(),
            status='closed'
        )
        
        url = reverse('papertrade-list') + '?status=open'
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0]['status'], 'open')
    
    def test_filter_trades_by_signal_type(self):
        """Test filtering trades by signal type"""
        PaperTrade.objects.create(
            user=self.user,
            symbol='EURUSD',
            signal_type='BUY',
            entry_price=Decimal('1.1000'),
            stop_loss=Decimal('1.0950'),
            take_profit_1=Decimal('1.1100'),
            lot_size=Decimal('1.0'),
            entry_time=timezone.now(),
            status='open'
        )
        
        PaperTrade.objects.create(
            user=self.user,
            symbol='GBPUSD',
            signal_type='SELL',
            entry_price=Decimal('1.2500'),
            stop_loss=Decimal('1.2550'),
            take_profit_1=Decimal('1.2400'),
            lot_size=Decimal('0.5'),
            entry_time=timezone.now(),
            status='open'
        )
        
        url = reverse('papertrade-list') + '?signal_type=BUY'
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0]['signal_type'], 'BUY')
    
    def test_filter_trades_by_days(self):
        """Test filtering trades by days"""
        from datetime import timedelta
        
        # Create old trade
        old_time = timezone.now() - timedelta(days=40)
        PaperTrade.objects.create(
            user=self.user,
            symbol='EURUSD',
            signal_type='BUY',
            entry_price=Decimal('1.1000'),
            stop_loss=Decimal('1.0950'),
            take_profit_1=Decimal('1.1100'),
            lot_size=Decimal('1.0'),
            entry_time=old_time,
            status='closed'
        )
        
        # Create recent trade
        PaperTrade.objects.create(
            user=self.user,
            symbol='GBPUSD',
            signal_type='SELL',
            entry_price=Decimal('1.2500'),
            stop_loss=Decimal('1.2550'),
            take_profit_1=Decimal('1.2400'),
            lot_size=Decimal('0.5'),
            entry_time=timezone.now(),
            status='open'
        )
        
        url = reverse('papertrade-list') + '?days=30'
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0]['symbol'], 'GBPUSD')
    
    def test_close_already_closed_trade(self):
        """Test closing an already closed trade returns error"""
        trade = PaperTrade.objects.create(
            user=self.user,
            symbol='EURUSD',
            signal_type='BUY',
            entry_price=Decimal('1.1000'),
            stop_loss=Decimal('1.0950'),
            take_profit_1=Decimal('1.1100'),
            lot_size=Decimal('1.0'),
            entry_time=timezone.now(),
            exit_time=timezone.now(),
            status='closed'
        )
        
        url = reverse('papertrade-close', kwargs={'pk': trade.id})
        data = {'exit_price': '1.1050'}
        
        response = self.client.post(url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('error', response.data)
        self.assertIn('not open', response.data['error'])
    
    def test_close_trade_missing_exit_price(self):
        """Test closing trade without exit price returns error"""
        trade = PaperTrade.objects.create(
            user=self.user,
            symbol='EURUSD',
            signal_type='BUY',
            entry_price=Decimal('1.1000'),
            stop_loss=Decimal('1.0950'),
            take_profit_1=Decimal('1.1100'),
            lot_size=Decimal('1.0'),
            entry_time=timezone.now(),
            status='open'
        )
        
        url = reverse('papertrade-close', kwargs={'pk': trade.id})
        data = {}  # No exit_price
        
        response = self.client.post(url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('error', response.data)
        self.assertIn('exit_price', response.data['error'])
    
    def test_performance_with_custom_days(self):
        """Test performance endpoint with custom days parameter"""
        url = reverse('papertrade-performance') + '?days=7'
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('total_trades', response.data)
    
    def test_equity_curve_default_parameters(self):
        """Test equity curve with default parameters"""
        url = reverse('papertrade-equity-curve')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIsInstance(response.data, list)
    
    def test_update_positions_with_empty_prices(self):
        """Test update positions with empty prices dict"""
        url = reverse('papertrade-update-positions')
        response = self.client.post(url, {}, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['updated'], 0)
        self.assertIn('No prices provided', response.data['message'])
    
    def test_update_positions_with_valid_prices(self):
        """Test update positions with valid price data"""
        # Create open trade
        PaperTrade.objects.create(
            user=self.user,
            symbol='EURUSD',
            signal_type='BUY',
            entry_price=Decimal('1.1000'),
            stop_loss=Decimal('1.0950'),
            take_profit_1=Decimal('1.1100'),
            lot_size=Decimal('1.0'),
            entry_time=timezone.now(),
            status='open'
        )
        
        url = reverse('papertrade-update-positions')
        data = {
            'prices': {
                'EURUSD': 1.1050
            }
        }
        
        response = self.client.post(url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('success', response.data)
        self.assertTrue(response.data['success'])


class PriceAPITest(TestCase):
    """Test Price-related API endpoints"""
    
    def setUp(self):
        """Set up test client"""
        self.client = APIClient()
    
    def test_get_realtime_price_no_symbol(self):
        """Test realtime price without symbol parameter"""
        url = reverse('realtime-price')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('error', response.data)
        self.assertIn('symbol parameter', response.data['error'])
    
    def test_get_historical_ohlc_no_symbol(self):
        """Test historical OHLC without symbol parameter"""
        url = reverse('historical-ohlc')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('error', response.data)
        self.assertIn('symbol parameter', response.data['error'])
    
    def test_get_historical_ohlc_with_parameters(self):
        """Test historical OHLC with all parameters"""
        url = reverse('historical-ohlc') + '?symbol=EURUSD&interval=1h&limit=50'
        response = self.client.get(url)
        
        # May succeed or fail depending on data availability
        # Just verify it doesn't crash
        self.assertIn(response.status_code, [
            status.HTTP_200_OK,
            status.HTTP_404_NOT_FOUND,
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ])


class MT5BridgeAPITest(TestCase):
    """Test MT5 Bridge API endpoints"""
    
    def setUp(self):
        """Set up test client"""
        self.client = APIClient()
    
    def test_mt_account_info(self):
        """Test MT5 account info endpoint"""
        url = reverse('mt-account')
        response = self.client.get(url)
        
        # Should return account info or error
        self.assertIn(response.status_code, [
            status.HTTP_200_OK,
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ])
    
    def test_mt_positions(self):
        """Test MT5 positions endpoint"""
        url = reverse('mt-positions')
        response = self.client.get(url)
        
        # Should return positions or error
        self.assertIn(response.status_code, [
            status.HTTP_200_OK,
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ])


class PerformanceMetricsViewSetTest(TestCase):
    """Test Performance Metrics ViewSet"""
    
    def setUp(self):
        """Set up test client"""
        self.client = APIClient()
        from paper_trading.models import PerformanceMetrics
        
        # Create sample metrics
        PerformanceMetrics.objects.create(
            pair='EURUSD',
            date=timezone.now().date(),
            total_trades=10,
            winning_trades=7,
            losing_trades=3,
            win_rate=Decimal('70.0'),
            total_pnl=Decimal('500.00')
        )
        
        PerformanceMetrics.objects.create(
            pair='GBPUSD',
            date=timezone.now().date(),
            total_trades=5,
            winning_trades=3,
            losing_trades=2,
            win_rate=Decimal('60.0'),
            total_pnl=Decimal('250.00')
        )
    
    def test_list_performance_metrics(self):
        """Test listing all performance metrics"""
        url = reverse('performancemetrics-list')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 2)
    
    def test_filter_metrics_by_pair(self):
        """Test filtering metrics by currency pair"""
        url = reverse('performancemetrics-list') + '?pair=EURUSD'
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0]['pair'], 'EURUSD')
    
    def test_filter_metrics_by_days(self):
        """Test filtering metrics by days"""
        from datetime import timedelta
        
        # Create old metric
        old_date = timezone.now().date() - timedelta(days=40)
        from paper_trading.models import PerformanceMetrics
        PerformanceMetrics.objects.create(
            pair='XAUUSD',
            date=old_date,
            total_trades=3,
            winning_trades=2,
            losing_trades=1,
            win_rate=Decimal('66.67'),
            total_pnl=Decimal('100.00')
        )
        
        url = reverse('performancemetrics-list') + '?days=30'
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        # Should only show recent metrics (EURUSD and GBPUSD)
        self.assertEqual(len(response.data), 2)
    
    def test_retrieve_single_metric(self):
        """Test retrieving a single metric"""
        from paper_trading.models import PerformanceMetrics
        metric = PerformanceMetrics.objects.first()
        
        url = reverse('performancemetrics-detail', kwargs={'pk': metric.id})
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['pair'], metric.pair)
