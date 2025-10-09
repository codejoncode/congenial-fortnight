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
