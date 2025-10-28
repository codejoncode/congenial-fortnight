"""
Test US NFA Forex Trading Rules
Tests FIFO, no-hedging, leverage limits, position sizing, and margin calculations
"""
import pytest
from decimal import Decimal
from django.contrib.auth.models import User
from paper_trading.engine import PaperTradingEngine


@pytest.fixture
def user_with_balance(db):
    """Create test user with initial balance"""
    user = User.objects.create_user(
        username='test_trader',
        password='testpass123'
    )
    # Set initial balance metadata
    user.account_balance = Decimal('10000.00')
    user.save()
    return user


@pytest.fixture
def engine(user_with_balance):
    """Create engine with specific balance"""
    return PaperTradingEngine(initial_balance=Decimal('10000.00'), user=user_with_balance)


@pytest.mark.django_db
class TestUSForexCalculations:
    """Test US Forex rules calculation methods"""
    
    def test_position_sizing_small_account(self, engine):
        """Test position sizing for $100 account"""
        result = engine.calculate_position_size(
            account_balance=Decimal('100'),
            risk_percent=Decimal('2'),
            stop_loss_pips=50,
            symbol='EURUSD'
        )
        
        # $100 * 2% = $2 risk
        # $2 / (50 pips * $10/pip per lot) = 0.004 lots
        assert result['risk_amount'] == Decimal('2.00')
        assert abs(result['lot_size'] - Decimal('0.004')) < Decimal('0.001')
        
    def test_position_sizing_medium_account(self, engine):
        """Test position sizing for $10,000 account"""
        result = engine.calculate_position_size(
            account_balance=Decimal('10000'),
            risk_percent=Decimal('2'),
            stop_loss_pips=50,
            symbol='EURUSD'
        )
        
        # $10,000 * 2% = $200 risk
        # $200 / (50 pips * $10/pip) = 0.40 lots
        assert result['risk_amount'] == Decimal('200.00')
        assert abs(result['lot_size'] - Decimal('0.40')) < Decimal('0.01')
    
    def test_position_sizing_large_account(self, engine):
        """Test position sizing for $100,000 account"""
        result = engine.calculate_position_size(
            account_balance=Decimal('100000'),
            risk_percent=Decimal('2'),
            stop_loss_pips=50,
            symbol='EURUSD'
        )
        
        # $100,000 * 2% = $2,000 risk
        # $2,000 / (50 pips * $10/pip) = 4.00 lots
        assert result['risk_amount'] == Decimal('2000.00')
        assert abs(result['lot_size'] - Decimal('4.00')) < Decimal('0.01')
    
    def test_position_sizing_institutional_account(self, engine):
        """Test position sizing for $100M account"""
        result = engine.calculate_position_size(
            account_balance=Decimal('100000000'),
            risk_percent=Decimal('2'),
            stop_loss_pips=50,
            symbol='EURUSD'
        )
        
        # $100M * 2% = $2M risk
        # $2M / (50 pips * $10/pip) = 4,000 lots
        assert result['risk_amount'] == Decimal('2000000.00')
        assert abs(result['lot_size'] - Decimal('4000.00')) < Decimal('10.00')
    
    def test_pip_value_standard_lot(self, engine):
        """Test pip value for 1.0 standard lot"""
        pip_value = engine.calculate_pip_value('EURUSD', Decimal('1.0'))
        assert pip_value == Decimal('10.00')
    
    def test_pip_value_mini_lot(self, engine):
        """Test pip value for 0.1 mini lot"""
        pip_value = engine.calculate_pip_value('EURUSD', Decimal('0.1'))
        assert pip_value == Decimal('1.00')
    
    def test_pip_value_micro_lot(self, engine):
        """Test pip value for 0.01 micro lot"""
        pip_value = engine.calculate_pip_value('EURUSD', Decimal('0.01'))
        assert pip_value == Decimal('0.10')
    
    def test_pip_value_jpy_pair(self, engine):
        """Test pip value for JPY pair (different pip size)"""
        pip_value = engine.calculate_pip_value('USDJPY', Decimal('1.0'))
        # JPY pairs: 1 pip = 0.01, so value is slightly different
        assert pip_value == Decimal('10.00')
    
    def test_margin_requirement_major_pair(self, engine):
        """Test margin requirement for major pair with 50:1 leverage"""
        result = engine.calculate_margin_requirement(
            symbol='EURUSD',
            lot_size=Decimal('1.0'),
            entry_price=Decimal('1.1000'),
            leverage=50
        )
        
        # Position value = 1.0 * 100,000 * 1.1000 = $110,000
        # Margin = $110,000 / 50 = $2,200
        assert result['position_value'] == Decimal('110000.00')
        assert result['margin_required'] == Decimal('2200.00')
        assert result['leverage_used'] == 50
    
    def test_margin_requirement_minor_pair(self, engine):
        """Test margin requirement for minor pair with 20:1 leverage"""
        result = engine.calculate_margin_requirement(
            symbol='EURGBP',
            lot_size=Decimal('1.0'),
            entry_price=Decimal('0.8500'),
            leverage=20
        )
        
        # Position value = 1.0 * 100,000 * 0.8500 = $85,000
        # Margin = $85,000 / 20 = $4,250
        assert result['position_value'] == Decimal('85000.00')
        assert result['margin_required'] == Decimal('4250.00')
        assert result['leverage_used'] == 20
    
    def test_margin_level_safe(self, engine):
        """Test margin level calculation - safe position"""
        margin_level = engine.calculate_margin_level(
            account_equity=Decimal('10000'),
            used_margin=Decimal('2000')
        )
        
        # Margin level = (10,000 / 2,000) * 100 = 500%
        assert margin_level == Decimal('500.00')
    
    def test_margin_level_warning(self, engine):
        """Test margin level calculation - margin call territory"""
        margin_level = engine.calculate_margin_level(
            account_equity=Decimal('10000'),
            used_margin=Decimal('8000')
        )
        
        # Margin level = (10,000 / 8,000) * 100 = 125%
        assert margin_level == Decimal('125.00')
    
    def test_margin_level_no_positions(self, engine):
        """Test margin level with no open positions"""
        margin_level = engine.calculate_margin_level(
            account_equity=Decimal('10000'),
            used_margin=Decimal('0')
        )
        
        # Should return very high number (infinity)
        assert margin_level > Decimal('100000')
    
    def test_max_position_size_major_pair(self, engine):
        """Test maximum position size for major pair (50:1 leverage)"""
        result = engine.calculate_max_position_size(
            symbol='EURUSD',
            account_balance=Decimal('10000'),
            leverage=50
        )
        
        # Max position value = $10,000 * 50 = $500,000
        # Max lots = $500,000 / $100,000 = 5.00 lots
        assert result['max_position_value'] == Decimal('500000')
        assert result['max_lot_size'] == Decimal('5.00')
        assert result['leverage_used'] == 50
    
    def test_max_position_size_minor_pair(self, engine):
        """Test maximum position size for minor pair (20:1 leverage)"""
        result = engine.calculate_max_position_size(
            symbol='EURGBP',
            account_balance=Decimal('10000'),
            leverage=None  # Should default to 20:1
        )
        
        # Max position value = $10,000 * 20 = $200,000
        # Max lots = $200,000 / $100,000 = 2.00 lots
        assert result['max_position_value'] == Decimal('200000')
        assert result['max_lot_size'] == Decimal('2.00')
        assert result['leverage_used'] == 20


@pytest.mark.django_db
class TestRiskManagementParametrized:
    """Parametrized tests for position sizing across account sizes"""
    
    @pytest.mark.parametrize("account_balance,risk_percent,expected_lot_size", [
        (Decimal('100'), Decimal('2'), Decimal('0.004')),
        (Decimal('1000'), Decimal('2'), Decimal('0.04')),
        (Decimal('10000'), Decimal('2'), Decimal('0.40')),
        (Decimal('100000'), Decimal('2'), Decimal('4.00')),
        (Decimal('1000000'), Decimal('2'), Decimal('40.00')),
        (Decimal('10000000'), Decimal('2'), Decimal('400.00')),
        (Decimal('100000000'), Decimal('2'), Decimal('4000.00')),
    ])
    def test_position_sizing_scales_correctly(self, engine, account_balance, risk_percent, expected_lot_size):
        """Test that position sizing scales correctly across account sizes"""
        result = engine.calculate_position_size(
            account_balance=account_balance,
            risk_percent=risk_percent,
            stop_loss_pips=50,
            symbol='EURUSD'
        )
        
        # Verify risk amount is exactly risk_percent of account
        expected_risk = account_balance * (risk_percent / Decimal('100'))
        assert result['risk_amount'] == expected_risk
        
        # Verify lot size is within 1% of expected (allow small rounding)
        tolerance = expected_lot_size * Decimal('0.01')
        assert abs(result['lot_size'] - expected_lot_size) <= tolerance


@pytest.mark.django_db
class TestHedgingViolation:
    """Test no-hedging rule enforcement"""
    
    def test_check_hedging_no_positions(self, engine):
        """Test hedging check when no positions exist"""
        violation = engine.check_hedging_violation('EURUSD', 'BUY')
        assert violation == False
    
    def test_check_hedging_same_direction(self, engine):
        """Test hedging check when positions in same direction exist"""
        from paper_trading.models import PaperTrade
        from django.utils import timezone
        
        # Create existing BUY position
        PaperTrade.objects.create(
            user=engine.user,
            symbol='EURUSD',
            signal_type='BUY',
            entry_price=Decimal('1.1000'),
            stop_loss=Decimal('1.0950'),
            take_profit_1=Decimal('1.1100'),
            lot_size=Decimal('1.0'),
            entry_time=timezone.now(),
            status='open'
        )
        
        # Another BUY should be allowed (no hedging)
        violation = engine.check_hedging_violation('EURUSD', 'BUY')
        assert violation == False
    
    def test_check_hedging_opposite_direction(self, engine):
        """Test hedging check detects opposing positions"""
        from paper_trading.models import PaperTrade
        from django.utils import timezone
        
        # Create existing BUY position
        PaperTrade.objects.create(
            user=engine.user,
            symbol='EURUSD',
            signal_type='BUY',
            entry_price=Decimal('1.1000'),
            stop_loss=Decimal('1.0950'),
            take_profit_1=Decimal('1.1100'),
            lot_size=Decimal('1.0'),
            entry_time=timezone.now(),
            status='open'
        )
        
        # SELL should be blocked (hedging violation)
        violation = engine.check_hedging_violation('EURUSD', 'SELL')
        assert violation == True
    
    def test_check_hedging_closed_position_ok(self, engine):
        """Test that closed positions don't affect hedging check"""
        from paper_trading.models import PaperTrade
        from django.utils import timezone
        
        # Create closed BUY position
        PaperTrade.objects.create(
            user=engine.user,
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
        
        # SELL should be allowed (previous position closed)
        violation = engine.check_hedging_violation('EURUSD', 'SELL')
        assert violation == False


@pytest.mark.django_db
class TestFIFOPositionClosing:
    """Test FIFO position closing rules"""
    
    def test_close_position_with_fifo_no_positions(self, engine):
        """Test FIFO close when no positions exist"""
        result = engine.close_position_with_fifo('EURUSD', Decimal('1.1000'))
        
        assert result['success'] == False
        assert 'No open positions' in result['error']
    
    def test_close_position_with_fifo_single_position(self, engine):
        """Test FIFO close with single position"""
        from paper_trading.models import PaperTrade
        from django.utils import timezone
        
        trade = PaperTrade.objects.create(
            user=engine.user,
            symbol='EURUSD',
            signal_type='BUY',
            entry_price=Decimal('1.1000'),
            stop_loss=Decimal('1.0950'),
            take_profit_1=Decimal('1.1100'),
            lot_size=Decimal('1.0'),
            entry_time=timezone.now(),
            status='open'
        )
        
        result = engine.close_position_with_fifo('EURUSD', Decimal('1.1050'))
        
        assert result['success'] == True
        # Verify trade was closed
        trade.refresh_from_db()
        assert trade.status == 'closed'
    
    def test_close_all_positions_empty(self, engine):
        """Test closing all positions when none exist"""
        result = engine.close_all_positions('EURUSD', Decimal('1.1000'))
        
        assert result['success'] == True
        assert result['closed_count'] == 0
    
    def test_close_all_positions_multiple(self, engine):
        """Test closing multiple positions in FIFO order"""
        from paper_trading.models import PaperTrade
        from django.utils import timezone
        from datetime import timedelta
        
        # Create three positions at different times
        now = timezone.now()
        
        PaperTrade.objects.create(
            user=engine.user,
            symbol='EURUSD',
            signal_type='BUY',
            entry_price=Decimal('1.1000'),
            stop_loss=Decimal('1.0950'),
            take_profit_1=Decimal('1.1100'),
            lot_size=Decimal('1.0'),
            entry_time=now - timedelta(hours=3),
            status='open'
        )
        
        PaperTrade.objects.create(
            user=engine.user,
            symbol='EURUSD',
            signal_type='BUY',
            entry_price=Decimal('1.1010'),
            stop_loss=Decimal('1.0960'),
            take_profit_1=Decimal('1.1110'),
            lot_size=Decimal('0.5'),
            entry_time=now - timedelta(hours=2),
            status='open'
        )
        
        PaperTrade.objects.create(
            user=engine.user,
            symbol='EURUSD',
            signal_type='BUY',
            entry_price=Decimal('1.1020'),
            stop_loss=Decimal('1.0970'),
            take_profit_1=Decimal('1.1120'),
            lot_size=Decimal('0.3'),
            entry_time=now - timedelta(hours=1),
            status='open'
        )
        
        result = engine.close_all_positions('EURUSD', Decimal('1.1050'))
        
        assert result['success'] == True
        assert result['closed_count'] == 3
        assert result['symbol'] == 'EURUSD'


@pytest.mark.django_db
class TestPartialPositionClose:
    """Test partial position closing"""
    
    def test_close_partial_no_positions(self, engine):
        """Test partial close when no positions exist"""
        result = engine.close_partial_position('EURUSD', Decimal('0.5'), Decimal('1.1050'))
        
        assert result['success'] == False
        assert 'No open positions' in result['error']
    
    def test_close_partial_less_than_position_size(self, engine):
        """Test partial close of less than full position"""
        from paper_trading.models import PaperTrade
        from django.utils import timezone
        
        trade = PaperTrade.objects.create(
            user=engine.user,
            symbol='EURUSD',
            signal_type='BUY',
            entry_price=Decimal('1.1000'),
            stop_loss=Decimal('1.0950'),
            take_profit_1=Decimal('1.1100'),
            lot_size=Decimal('1.0'),
            entry_time=timezone.now(),
            status='open'
        )
        
        result = engine.close_partial_position('EURUSD', Decimal('0.5'), Decimal('1.1050'))
        
        assert result['success'] == True
        assert result['partial_close'] == True
        assert result['lot_size_closed'] == Decimal('0.5')
        assert result['remaining_lot_size'] == Decimal('0.5')
        
        # Verify position was reduced
        trade.refresh_from_db()
        assert trade.lot_size == Decimal('0.5')
        assert trade.status == 'open'
    
    def test_close_partial_equal_to_position_size(self, engine):
        """Test partial close equal to position size closes entire position"""
        from paper_trading.models import PaperTrade
        from django.utils import timezone
        
        trade = PaperTrade.objects.create(
            user=engine.user,
            symbol='EURUSD',
            signal_type='BUY',
            entry_price=Decimal('1.1000'),
            stop_loss=Decimal('1.0950'),
            take_profit_1=Decimal('1.1100'),
            lot_size=Decimal('1.0'),
            entry_time=timezone.now(),
            status='open'
        )
        
        result = engine.close_partial_position('EURUSD', Decimal('1.0'), Decimal('1.1050'))
        
        assert result['success'] == True
        
        # Verify trade was fully closed
        trade.refresh_from_db()
        assert trade.status == 'closed'


@pytest.mark.django_db
class TestPositionSizingEdgeCases:
    """Test position sizing edge cases and validation"""
    
    def test_position_sizing_zero_risk(self, engine):
        """Test position sizing with zero risk percent"""
        result = engine.calculate_position_size(
            account_balance=Decimal('10000'),
            risk_percent=Decimal('0'),
            stop_loss_pips=50,
            symbol='EURUSD'
        )
        
        assert result['risk_amount'] == Decimal('0.00')
        assert result['lot_size'] == Decimal('0.000')
    
    def test_position_sizing_high_risk(self, engine):
        """Test position sizing with high risk percent"""
        result = engine.calculate_position_size(
            account_balance=Decimal('10000'),
            risk_percent=Decimal('10'),  # 10% risk
            stop_loss_pips=50,
            symbol='EURUSD'
        )
        
        # $10,000 * 10% = $1,000 risk
        # $1,000 / (50 pips * $10/pip) = 2.00 lots
        assert result['risk_amount'] == Decimal('1000.00')
        assert abs(result['lot_size'] - Decimal('2.00')) < Decimal('0.01')
    
    def test_position_sizing_tight_stop_loss(self, engine):
        """Test position sizing with very tight stop loss"""
        result = engine.calculate_position_size(
            account_balance=Decimal('10000'),
            risk_percent=Decimal('2'),
            stop_loss_pips=10,  # Very tight
            symbol='EURUSD'
        )
        
        # $10,000 * 2% = $200 risk
        # $200 / (10 pips * $10/pip) = 2.00 lots
        assert result['risk_amount'] == Decimal('200.00')
        assert abs(result['lot_size'] - Decimal('2.00')) < Decimal('0.01')
    
    def test_position_sizing_wide_stop_loss(self, engine):
        """Test position sizing with very wide stop loss"""
        result = engine.calculate_position_size(
            account_balance=Decimal('10000'),
            risk_percent=Decimal('2'),
            stop_loss_pips=500,  # Very wide
            symbol='EURUSD'
        )
        
        # $10,000 * 2% = $200 risk
        # $200 / (500 pips * $10/pip) = 0.04 lots
        assert result['risk_amount'] == Decimal('200.00')
        assert abs(result['lot_size'] - Decimal('0.04')) < Decimal('0.01')
    
    def test_position_sizing_jpy_pair(self, engine):
        """Test position sizing for JPY pair"""
        result = engine.calculate_position_size(
            account_balance=Decimal('10000'),
            risk_percent=Decimal('2'),
            stop_loss_pips=50,
            symbol='USDJPY'
        )
        
        # JPY pairs have slightly different pip value
        assert result['risk_amount'] == Decimal('200.00')
        # Should still calculate lot size properly
        assert result['lot_size'] > Decimal('0')


@pytest.mark.django_db
class TestMarginCalculations:
    """Test margin-related calculations"""
    
    def test_margin_requirement_micro_lot(self, engine):
        """Test margin requirement for micro lot"""
        result = engine.calculate_margin_requirement(
            symbol='EURUSD',
            lot_size=Decimal('0.01'),
            entry_price=Decimal('1.1000'),
            leverage=50
        )
        
        # Position value = 0.01 * 100,000 * 1.1000 = $1,100
        # Margin = $1,100 / 50 = $22
        assert result['position_value'] == Decimal('1100.00')
        assert result['margin_required'] == Decimal('22.00')
    
    def test_margin_requirement_high_leverage(self, engine):
        """Test margin with very high leverage"""
        result = engine.calculate_margin_requirement(
            symbol='EURUSD',
            lot_size=Decimal('1.0'),
            entry_price=Decimal('1.1000'),
            leverage=100
        )
        
        # Position value = 1.0 * 100,000 * 1.1000 = $110,000
        # Margin = $110,000 / 100 = $1,100
        assert result['margin_required'] == Decimal('1100.00')
    
    def test_margin_requirement_low_leverage(self, engine):
        """Test margin with low leverage"""
        result = engine.calculate_margin_requirement(
            symbol='EURUSD',
            lot_size=Decimal('1.0'),
            entry_price=Decimal('1.1000'),
            leverage=10
        )
        
        # Position value = 1.0 * 100,000 * 1.1000 = $110,000
        # Margin = $110,000 / 10 = $11,000
        assert result['margin_required'] == Decimal('11000.00')
    
    def test_margin_level_critical(self, engine):
        """Test margin level at margin call level"""
        margin_level = engine.calculate_margin_level(
            account_equity=Decimal('10000'),
            used_margin=Decimal('10000')
        )
        
        # Margin level = (10,000 / 10,000) * 100 = 100%
        assert margin_level == Decimal('100.00')
    
    def test_margin_level_stop_out(self, engine):
        """Test margin level at stop out level"""
        margin_level = engine.calculate_margin_level(
            account_equity=Decimal('5000'),
            used_margin=Decimal('10000')
        )
        
        # Margin level = (5,000 / 10,000) * 100 = 50%
        assert margin_level == Decimal('50.00')


@pytest.mark.django_db
class TestPipValueCalculations:
    """Test pip value calculations for different pairs and lot sizes"""
    
    def test_pip_value_various_lot_sizes(self, engine):
        """Test pip values for various lot sizes"""
        test_cases = [
            (Decimal('10.0'), Decimal('100.00')),   # 10 lots
            (Decimal('5.0'), Decimal('50.00')),     # 5 lots
            (Decimal('1.0'), Decimal('10.00')),     # Standard lot
            (Decimal('0.5'), Decimal('5.00')),      # Half lot
            (Decimal('0.1'), Decimal('1.00')),      # Mini lot
            (Decimal('0.01'), Decimal('0.10')),     # Micro lot
            (Decimal('0.001'), Decimal('0.01')),    # Nano lot
        ]
        
        for lot_size, expected_pip_value in test_cases:
            pip_value = engine.calculate_pip_value('EURUSD', lot_size)
            assert pip_value == expected_pip_value
    
    def test_pip_value_jpy_pairs(self, engine):
        """Test pip value for various JPY pairs"""
        jpy_pairs = ['USDJPY', 'EURJPY', 'GBPJPY', 'AUDJPY']
        
        for pair in jpy_pairs:
            pip_value = engine.calculate_pip_value(pair, Decimal('1.0'))
            # All JPY pairs should have same pip value
            assert pip_value == Decimal('10.00')


@pytest.mark.django_db
class TestMaxPositionSize:
    """Test maximum position size calculations"""
    
    def test_max_position_size_auto_detect_major(self, engine):
        """Test auto-detection of major pair for leverage"""
        result = engine.calculate_max_position_size(
            symbol='GBPUSD',  # Major pair
            account_balance=Decimal('10000'),
            leverage=None  # Should auto-detect 50:1
        )
        
        assert result['leverage_used'] == 50
        assert result['max_lot_size'] == Decimal('5.00')
    
    def test_max_position_size_auto_detect_minor(self, engine):
        """Test auto-detection of minor pair for leverage"""
        result = engine.calculate_max_position_size(
            symbol='EURCHF',  # Minor pair (not in major list)
            account_balance=Decimal('10000'),
            leverage=None  # Should auto-detect 20:1
        )
        
        assert result['leverage_used'] == 20
        assert result['max_lot_size'] == Decimal('2.00')
    
    def test_max_position_size_small_account(self, engine):
        """Test max position size for very small account"""
        result = engine.calculate_max_position_size(
            symbol='EURUSD',
            account_balance=Decimal('100'),
            leverage=50
        )
        
        # Max position value = $100 * 50 = $5,000
        # Max lots = $5,000 / $100,000 = 0.05 lots
        assert result['max_lot_size'] == Decimal('0.05')
    
    def test_max_position_size_large_account(self, engine):
        """Test max position size for large account"""
        result = engine.calculate_max_position_size(
            symbol='EURUSD',
            account_balance=Decimal('1000000'),
            leverage=50
        )
        
        # Max position value = $1,000,000 * 50 = $50,000,000
        # Max lots = $50,000,000 / $100,000 = 500.00 lots
        assert result['max_lot_size'] == Decimal('500.00')


"""
US NFA Forex Trading Rules Reference
=====================================

1. FIFO (First In, First Out)
   - When multiple positions exist on the same currency pair
   - Must close the oldest position first
   - Cannot selectively close newer positions

2. No Hedging
   - Cannot have opposing positions (long & short) on same pair simultaneously
   - Must close existing position before opening opposite direction

3. Leverage Limits
   - Major pairs (EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, USDCAD, NZDUSD): 50:1 max
   - Minor/exotic pairs: 20:1 max

4. Position Sizing Formula
   Lot Size = (Account Balance × Risk %) / (Stop Loss Pips × Pip Value)
   
   Where:
   - Standard lot (1.0) = $10/pip for major pairs
   - Mini lot (0.1) = $1/pip
   - Micro lot (0.01) = $0.10/pip

5. Margin Calculations
   - Required Margin = (Lot Size × Contract Size × Price) / Leverage
   - Contract Size = 100,000 units
   - Margin Level = (Equity / Used Margin) × 100
   - Margin Call typically at 100% margin level
   - Stop Out typically at 50% margin level

6. Account Size Examples
   - Micro: $100 - $1,000 (trade 0.001 - 0.01 lots)
   - Mini: $1,000 - $10,000 (trade 0.01 - 0.1 lots)
   - Standard: $10,000 - $100,000 (trade 0.1 - 10 lots)
   - Institutional: $1M+ (trade 10+ lots)
"""
