"""
Tests for Data Aggregator
"""
from decimal import Decimal
from django.test import TestCase
from django.core.cache import cache
from django.utils import timezone
from unittest.mock import Mock, patch, MagicMock
from paper_trading.data_aggregator import DataAggregator
from paper_trading.models import PriceCache, APIUsageTracker


class DataAggregatorTest(TestCase):
    """Test Data Aggregator"""
    
    def setUp(self):
        """Set up test environment"""
        self.aggregator = DataAggregator()
        cache.clear()
        APIUsageTracker.objects.all().delete()
    
    def tearDown(self):
        """Clean up after tests"""
        cache.clear()
    
    def test_can_use_api_no_limit(self):
        """Test API with no limit (like Yahoo)"""
        can_use = self.aggregator._can_use_api('yahoo')
        self.assertTrue(can_use)
    
    def test_can_use_api_within_limit(self):
        """Test API within daily limit"""
        # Create tracker with usage below limit
        APIUsageTracker.objects.create(
            api_name='twelve_data',
            date=timezone.now().date(),
            requests_made=100,
            requests_limit=800
        )
        
        can_use = self.aggregator._can_use_api('twelve_data')
        self.assertTrue(can_use)
    
    def test_can_use_api_exceeds_limit(self):
        """Test API that exceeds daily limit"""
        # Create tracker with usage at limit
        APIUsageTracker.objects.create(
            api_name='alpha_vantage',
            date=timezone.now().date(),
            requests_made=25,
            requests_limit=25
        )
        
        can_use = self.aggregator._can_use_api('alpha_vantage')
        self.assertFalse(can_use)
    
    def test_track_api_usage_new_entry(self):
        """Test tracking API usage for new entry"""
        self.aggregator._track_api_usage('yahoo')
        
        tracker = APIUsageTracker.objects.get(
            api_name='yahoo',
            date=timezone.now().date()
        )
        
        self.assertEqual(tracker.requests_made, 1)
    
    def test_track_api_usage_existing_entry(self):
        """Test tracking API usage for existing entry"""
        # Create initial tracker
        APIUsageTracker.objects.create(
            api_name='twelve_data',
            date=timezone.now().date(),
            requests_made=5,
            requests_limit=800
        )
        
        self.aggregator._track_api_usage('twelve_data')
        
        tracker = APIUsageTracker.objects.get(
            api_name='twelve_data',
            date=timezone.now().date()
        )
        
        self.assertEqual(tracker.requests_made, 6)
    
    def test_convert_symbol_yahoo(self):
        """Test symbol conversion for Yahoo Finance"""
        symbol = self.aggregator._convert_symbol('EURUSD', 'yahoo')
        self.assertEqual(symbol, 'EURUSD=X')
        
        symbol = self.aggregator._convert_symbol('XAUUSD', 'yahoo')
        self.assertEqual(symbol, 'GC=F')
    
    def test_convert_symbol_twelve_data(self):
        """Test symbol conversion for Twelve Data"""
        symbol = self.aggregator._convert_symbol('EURUSD', 'twelve_data')
        self.assertEqual(symbol, 'EUR/USD')
    
    @patch('paper_trading.data_aggregator.yf.Ticker')
    def test_fetch_from_yahoo_success(self, mock_ticker):
        """Test fetching data from Yahoo Finance successfully"""
        # Mock Yahoo Finance response
        mock_info = {
            'bid': 1.1000,
            'ask': 1.1002,
            'regularMarketTime': timezone.now().timestamp()
        }
        mock_ticker.return_value.info = mock_info
        
        data = self.aggregator._fetch_from_yahoo('EURUSD')
        
        self.assertIsNotNone(data)
        self.assertEqual(data['bid'], Decimal('1.1000'))
        self.assertEqual(data['ask'], Decimal('1.1002'))
    
    @patch('paper_trading.data_aggregator.yf.Ticker')
    def test_fetch_from_yahoo_failure(self, mock_ticker):
        """Test Yahoo Finance failure handling"""
        # Mock Yahoo Finance error
        mock_ticker.side_effect = Exception("API Error")
        
        data = self.aggregator._fetch_from_yahoo('EURUSD')
        
        self.assertIsNone(data)
    
    @patch('paper_trading.data_aggregator.requests.get')
    def test_fetch_from_twelve_data_success(self, mock_get):
        """Test fetching from Twelve Data successfully"""
        # Mock Twelve Data response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'price': '1.1000',
            'timestamp': int(timezone.now().timestamp())
        }
        mock_get.return_value = mock_response
        
        data = self.aggregator._fetch_from_twelve_data('EURUSD')
        
        self.assertIsNotNone(data)
    
    def test_cache_price_redis(self):
        """Test caching price in Redis"""
        price_data = {
            'bid': Decimal('1.1000'),
            'ask': Decimal('1.1002'),
            'timestamp': timezone.now()
        }
        
        self.aggregator._cache_price('EURUSD', price_data, 'realtime')
        
        # Check Redis cache
        cached = cache.get('price_cache_EURUSD_realtime')
        self.assertIsNotNone(cached)
    
    def test_cache_ohlc_database(self):
        """Test caching OHLC data in database"""
        ohlc_data = [{
            'timestamp': timezone.now(),
            'open': Decimal('1.1000'),
            'high': Decimal('1.1050'),
            'low': Decimal('1.0950'),
            'close': Decimal('1.1025'),
            'volume': 1000000
        }]
        
        self.aggregator._cache_ohlc('EURUSD', ohlc_data, '1hour', 'yahoo')
        
        # Check database cache
        cached = PriceCache.objects.filter(
            symbol='EURUSD',
            timeframe='1hour'
        )
        
        self.assertEqual(cached.count(), 1)
    
    @patch.object(DataAggregator, '_fetch_from_yahoo')
    def test_get_realtime_price_from_cache(self, mock_fetch):
        """Test getting price from cache"""
        # Pre-populate cache
        price_data = {
            'bid': Decimal('1.1000'),
            'ask': Decimal('1.1002'),
            'timestamp': timezone.now()
        }
        cache.set('price_cache_EURUSD_realtime', price_data, 60)
        
        # Get price
        result = self.aggregator.get_realtime_price('EURUSD')
        
        # Should not call API
        mock_fetch.assert_not_called()
        self.assertEqual(result['bid'], Decimal('1.1000'))
    
    @patch.object(DataAggregator, '_fetch_from_yahoo')
    def test_get_realtime_price_api_rotation(self, mock_yahoo):
        """Test API rotation when fetching price"""
        # Mock Yahoo to return None (failure)
        mock_yahoo.return_value = None
        
        # Mock other APIs to return None as well
        with patch.object(self.aggregator, '_fetch_from_twelve_data', return_value=None):
            with patch.object(self.aggregator, '_fetch_from_finnhub', return_value=None):
                with patch.object(self.aggregator, '_fetch_from_alpha_vantage', return_value=None):
                    result = self.aggregator.get_realtime_price('EURUSD')
        
        # Should try Yahoo first
        mock_yahoo.assert_called_once()
        
        # Result should be None if all fail
        self.assertIsNone(result)
    
    def test_get_historical_ohlc_from_database(self):
        """Test getting OHLC from database cache"""
        # Pre-populate database cache
        now = timezone.now()
        PriceCache.objects.create(
            symbol='EURUSD',
            timestamp=now,
            open_price=Decimal('1.1000'),
            high_price=Decimal('1.1050'),
            low_price=Decimal('1.0950'),
            close_price=Decimal('1.1025'),
            volume=1000000,
            source='yahoo',
            timeframe='1hour'
        )
        
        # Get OHLC
        result = self.aggregator.get_historical_ohlc('EURUSD', '1hour', limit=1)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['open'], Decimal('1.1000'))
    
    def test_symbol_mapping(self):
        """Test symbol mappings for different APIs"""
        # Test major pairs
        self.assertIn('EURUSD', self.aggregator.symbol_mappings)
        self.assertIn('XAUUSD', self.aggregator.symbol_mappings)
        self.assertIn('GBPUSD', self.aggregator.symbol_mappings)
        
        # Test Yahoo mapping
        yahoo_mapping = self.aggregator.symbol_mappings['EURUSD']['yahoo']
        self.assertEqual(yahoo_mapping, 'EURUSD=X')
