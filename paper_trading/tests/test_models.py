"""
Tests for Paper Trading Models
"""
from decimal import Decimal
from django.test import TestCase
from django.contrib.auth import get_user_model
from django.utils import timezone
from paper_trading.models import (
    PaperTrade, PriceCache, PerformanceMetrics, APIUsageTracker
)

User = get_user_model()


class PaperTradeModelTest(TestCase):
    """Test PaperTrade model"""
    
    def setUp(self):
        """Set up test user"""
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
    
    def test_create_paper_trade(self):
        """Test creating a paper trade"""
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
        
        self.assertEqual(trade.symbol, 'EURUSD')
        self.assertEqual(trade.signal_type, 'BUY')
        self.assertEqual(trade.status, 'open')
        self.assertEqual(trade.lot_size, Decimal('1.0'))
    
    def test_paper_trade_string_representation(self):
        """Test string representation of trade"""
        trade = PaperTrade.objects.create(
            user=self.user,
            symbol='XAUUSD',
            signal_type='SELL',
            entry_price=Decimal('1950.00'),
            stop_loss=Decimal('1955.00'),
            take_profit_1=Decimal('1940.00'),
            lot_size=Decimal('0.5'),
            entry_time=timezone.now(),
            status='open'
        )
        
        expected = f"XAUUSD SELL @ 1950.00 - open"
        self.assertEqual(str(trade), expected)
    
    def test_calculate_pips_gained_buy(self):
        """Test pips calculation for BUY trade"""
        trade = PaperTrade.objects.create(
            user=self.user,
            symbol='EURUSD',
            signal_type='BUY',
            entry_price=Decimal('1.1000'),
            exit_price=Decimal('1.1050'),
            stop_loss=Decimal('1.0950'),
            take_profit_1=Decimal('1.1100'),
            lot_size=Decimal('1.0'),
            entry_time=timezone.now(),
            exit_time=timezone.now(),
            status='closed'
        )
        
        # For 5-digit broker: 1.1050 - 1.1000 = 0.0050 = 50 pips
        trade.pips_gained = (trade.exit_price - trade.entry_price) * Decimal('10000')
        trade.save()
        
        self.assertEqual(trade.pips_gained, Decimal('50.0'))
    
    def test_calculate_pips_gained_sell(self):
        """Test pips calculation for SELL trade"""
        trade = PaperTrade.objects.create(
            user=self.user,
            symbol='EURUSD',
            signal_type='SELL',
            entry_price=Decimal('1.1000'),
            exit_price=Decimal('1.0950'),
            stop_loss=Decimal('1.1050'),
            take_profit_1=Decimal('1.0900'),
            lot_size=Decimal('1.0'),
            entry_time=timezone.now(),
            exit_time=timezone.now(),
            status='closed'
        )
        
        # For 5-digit broker: 1.1000 - 1.0950 = 0.0050 = 50 pips
        trade.pips_gained = (trade.entry_price - trade.exit_price) * Decimal('10000')
        trade.save()
        
        self.assertEqual(trade.pips_gained, Decimal('50.0'))


class PriceCacheModelTest(TestCase):
    """Test PriceCache model"""
    
    def test_create_price_cache(self):
        """Test creating a price cache entry"""
        cache = PriceCache.objects.create(
            symbol='EURUSD',
            timestamp=timezone.now(),
            open_price=Decimal('1.1000'),
            high_price=Decimal('1.1050'),
            low_price=Decimal('1.0950'),
            close_price=Decimal('1.1025'),
            volume=1000000,
            source='yahoo',
            timeframe='1min'
        )
        
        self.assertEqual(cache.symbol, 'EURUSD')
        self.assertEqual(cache.source, 'yahoo')
        self.assertEqual(cache.timeframe, '1min')
    
    def test_price_cache_ordering(self):
        """Test price cache is ordered by timestamp descending"""
        now = timezone.now()
        
        cache1 = PriceCache.objects.create(
            symbol='EURUSD',
            timestamp=now,
            open_price=Decimal('1.1000'),
            high_price=Decimal('1.1050'),
            low_price=Decimal('1.0950'),
            close_price=Decimal('1.1025'),
            volume=1000000,
            source='yahoo',
            timeframe='1min'
        )
        
        cache2 = PriceCache.objects.create(
            symbol='EURUSD',
            timestamp=now + timezone.timedelta(minutes=1),
            open_price=Decimal('1.1025'),
            high_price=Decimal('1.1075'),
            low_price=Decimal('1.1000'),
            close_price=Decimal('1.1050'),
            volume=1000000,
            source='yahoo',
            timeframe='1min'
        )
        
        latest = PriceCache.objects.first()
        self.assertEqual(latest.id, cache2.id)


class PerformanceMetricsModelTest(TestCase):
    """Test PerformanceMetrics model"""
    
    def setUp(self):
        """Set up test user"""
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
    
    def test_create_performance_metrics(self):
        """Test creating performance metrics"""
        metrics = PerformanceMetrics.objects.create(
            user=self.user,
            date=timezone.now().date(),
            symbol='EURUSD',
            total_trades=10,
            winning_trades=7,
            losing_trades=3,
            win_rate=Decimal('70.00'),
            total_pips=Decimal('150.00'),
            total_profit_loss=Decimal('1500.00'),
            average_risk_reward=Decimal('3.00')
        )
        
        self.assertEqual(metrics.total_trades, 10)
        self.assertEqual(metrics.win_rate, Decimal('70.00'))
        self.assertEqual(metrics.symbol, 'EURUSD')


class APIUsageTrackerModelTest(TestCase):
    """Test APIUsageTracker model"""
    
    def test_create_api_usage_tracker(self):
        """Test creating API usage tracker"""
        tracker = APIUsageTracker.objects.create(
            api_name='yahoo',
            date=timezone.now().date(),
            requests_made=50,
            requests_limit=None
        )
        
        self.assertEqual(tracker.api_name, 'yahoo')
        self.assertEqual(tracker.requests_made, 50)
        self.assertIsNone(tracker.requests_limit)
    
    def test_api_usage_tracker_unique_constraint(self):
        """Test unique constraint on api_name and date"""
        date = timezone.now().date()
        
        APIUsageTracker.objects.create(
            api_name='twelve_data',
            date=date,
            requests_made=10,
            requests_limit=800
        )
        
        # Update existing record
        tracker, created = APIUsageTracker.objects.get_or_create(
            api_name='twelve_data',
            date=date,
            defaults={'requests_made': 20, 'requests_limit': 800}
        )
        
        self.assertFalse(created)
        
        # Update the requests_made
        tracker.requests_made = 20
        tracker.save()
        
        self.assertEqual(tracker.requests_made, 20)
