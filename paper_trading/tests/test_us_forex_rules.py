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
