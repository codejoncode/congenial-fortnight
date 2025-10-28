"""
Tests for Paper Trading Engine
"""
from decimal import Decimal
from django.test import TestCase
from django.contrib.auth import get_user_model
from django.utils import timezone
from unittest.mock import Mock, patch
from paper_trading.engine import PaperTradingEngine
from paper_trading.models import PaperTrade

User = get_user_model()


class PaperTradingEngineTest(TestCase):
    """Test Paper Trading Engine"""
    
    def setUp(self):
        """Set up test user and engine"""
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        self.engine = PaperTradingEngine(self.user)
    
    def test_execute_order_buy(self):
        """Test executing a BUY order"""
        trade = self.engine.execute_order(
            symbol='EURUSD',
            signal_type='BUY',
            entry_price=Decimal('1.1000'),
            stop_loss=Decimal('1.0950'),
            take_profit_1=Decimal('1.1100'),
            lot_size=Decimal('1.0')
        )
        
        self.assertIsNotNone(trade)
        self.assertEqual(trade.symbol, 'EURUSD')
        self.assertEqual(trade.signal_type, 'BUY')
        self.assertEqual(trade.status, 'open')
        self.assertEqual(trade.entry_price, Decimal('1.1000'))
    
    def test_execute_order_sell(self):
        """Test executing a SELL order"""
        trade = self.engine.execute_order(
            symbol='XAUUSD',
            signal_type='SELL',
            entry_price=Decimal('1950.00'),
            stop_loss=Decimal('1955.00'),
            take_profit_1=Decimal('1940.00'),
            lot_size=Decimal('0.5')
        )
        
        self.assertIsNotNone(trade)
        self.assertEqual(trade.signal_type, 'SELL')
        self.assertEqual(trade.symbol, 'XAUUSD')
    
    def test_execute_order_with_multiple_tps(self):
        """Test executing order with multiple take profits"""
        trade = self.engine.execute_order(
            symbol='EURUSD',
            signal_type='BUY',
            entry_price=Decimal('1.1000'),
            stop_loss=Decimal('1.0950'),
            take_profit_1=Decimal('1.1100'),
            take_profit_2=Decimal('1.1150'),
            take_profit_3=Decimal('1.1200'),
            lot_size=Decimal('1.0')
        )
        
        self.assertIsNotNone(trade.take_profit_2)
        self.assertIsNotNone(trade.take_profit_3)
    
    def test_get_open_positions(self):
        """Test getting open positions"""
        # Create multiple trades
        self.engine.execute_order(
            symbol='EURUSD',
            signal_type='BUY',
            entry_price=Decimal('1.1000'),
            stop_loss=Decimal('1.0950'),
            take_profit_1=Decimal('1.1100'),
            lot_size=Decimal('1.0')
        )
        
        self.engine.execute_order(
            symbol='XAUUSD',
            signal_type='SELL',
            entry_price=Decimal('1950.00'),
            stop_loss=Decimal('1955.00'),
            take_profit_1=Decimal('1940.00'),
            lot_size=Decimal('0.5')
        )
        
        positions = self.engine.get_open_positions()
        self.assertEqual(len(positions), 2)
    
    def test_close_position(self):
        """Test closing a position"""
        # Create a trade
        trade = self.engine.execute_order(
            symbol='EURUSD',
            signal_type='BUY',
            entry_price=Decimal('1.1000'),
            stop_loss=Decimal('1.0950'),
            take_profit_1=Decimal('1.1100'),
            lot_size=Decimal('1.0')
        )
        
        # Close the trade
        closed_trade = self.engine.close_position(
            trade.id,
            exit_price=Decimal('1.1050'),
            reason='manual'
        )
        
        self.assertEqual(closed_trade.status, 'closed')
        self.assertEqual(closed_trade.exit_price, Decimal('1.1050'))
        self.assertEqual(closed_trade.exit_reason, 'manual')
        self.assertIsNotNone(closed_trade.exit_time)
        self.assertIsNotNone(closed_trade.pips_gained)
        self.assertIsNotNone(closed_trade.profit_loss)
    
    def test_calculate_pips_forex(self):
        """Test pip calculation for forex pairs"""
        pips = self.engine._calculate_pips(
            'EURUSD',
            Decimal('1.1000'),
            Decimal('1.1050')
        )
        # 50 pips difference
        self.assertEqual(pips, Decimal('50.0'))
    
    def test_calculate_pips_jpy(self):
        """Test pip calculation for JPY pairs"""
        pips = self.engine._calculate_pips(
            'USDJPY',
            Decimal('110.00'),
            Decimal('110.50')
        )
        # 50 pips difference (JPY uses 2 decimal places)
        self.assertEqual(pips, Decimal('50.0'))
    
    def test_calculate_pips_gold(self):
        """Test pip calculation for gold"""
        pips = self.engine._calculate_pips(
            'XAUUSD',
            Decimal('1950.00'),
            Decimal('1951.00')
        )
        # 100 pips difference
        self.assertEqual(pips, Decimal('100.0'))
    
    def test_check_sl_hit_buy(self):
        """Test stop loss detection for BUY trade"""
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
        
        # Current price hits SL
        current_price = Decimal('1.0950')
        hit, level = self.engine._check_sl_tp_hit(trade, current_price)
        
        self.assertTrue(hit)
        self.assertEqual(level, 'sl_hit')
    
    def test_check_tp_hit_buy(self):
        """Test take profit detection for BUY trade"""
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
        
        # Current price hits TP
        current_price = Decimal('1.1100')
        hit, level = self.engine._check_sl_tp_hit(trade, current_price)
        
        self.assertTrue(hit)
        self.assertEqual(level, 'tp1_hit')
    
    def test_check_sl_hit_sell(self):
        """Test stop loss detection for SELL trade"""
        trade = PaperTrade.objects.create(
            user=self.user,
            symbol='EURUSD',
            signal_type='SELL',
            entry_price=Decimal('1.1000'),
            stop_loss=Decimal('1.1050'),
            take_profit_1=Decimal('1.0900'),
            lot_size=Decimal('1.0'),
            entry_time=timezone.now(),
            status='open'
        )
        
        # Current price hits SL
        current_price = Decimal('1.1050')
        hit, level = self.engine._check_sl_tp_hit(trade, current_price)
        
        self.assertTrue(hit)
        self.assertEqual(level, 'sl_hit')
    
    def test_get_performance_summary(self):
        """Test getting performance summary"""
        # Create and close some trades
        trade1 = self.engine.execute_order(
            symbol='EURUSD',
            signal_type='BUY',
            entry_price=Decimal('1.1000'),
            stop_loss=Decimal('1.0950'),
            take_profit_1=Decimal('1.1100'),
            lot_size=Decimal('1.0')
        )
        self.engine.close_position(trade1.id, Decimal('1.1050'), 'manual')
        
        trade2 = self.engine.execute_order(
            symbol='EURUSD',
            signal_type='BUY',
            entry_price=Decimal('1.1000'),
            stop_loss=Decimal('1.0950'),
            take_profit_1=Decimal('1.1100'),
            lot_size=Decimal('1.0')
        )
        self.engine.close_position(trade2.id, Decimal('1.0970'), 'manual')
        
        summary = self.engine.get_performance_summary()
        
        self.assertEqual(summary['total_trades'], 2)
        self.assertEqual(summary['winning_trades'], 1)
        self.assertEqual(summary['losing_trades'], 1)
        self.assertEqual(summary['win_rate'], 50.0)
    
    def test_get_equity_curve(self):
        """Test getting equity curve"""
        initial_balance = Decimal('10000.00')
        
        # Create and close a trade
        trade = self.engine.execute_order(
            symbol='EURUSD',
            signal_type='BUY',
            entry_price=Decimal('1.1000'),
            stop_loss=Decimal('1.0950'),
            take_profit_1=Decimal('1.1100'),
            lot_size=Decimal('1.0')
        )
        self.engine.close_position(trade.id, Decimal('1.1050'), 'manual')
        
        equity_curve = self.engine.get_equity_curve(initial_balance)
        
        self.assertGreater(len(equity_curve), 0)
        self.assertIn('date', equity_curve[0])
        self.assertIn('balance', equity_curve[0])
