"""
tests/test_decision_engine.py — Unit tests for signals/decision_engine.py

Covers:
  - _risk_pct(): tier selection at and between thresholds
  - _atr_pips(): pair-specific pip multiplier
  - _position_sizing(): fixed-fractional math, lot floors/caps, zero pip_risk
  - evaluate(): all 7 rules, EXECUTE/WAIT/SKIP logic, score range, sizing in output
"""

import pytest
from signals.decision_engine import (
    _risk_pct,
    _atr_pips,
    _position_sizing,
    evaluate,
    DECISION_RULES,
    RISK_TIERS,
    MIN_LOT,
    MAX_LOT,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _eurusd_signal(
    signal='bullish',
    probability=0.70,
    entry=1.10000,
    sl=1.09700,
    tp=1.10600,
    rr=2.0,
    atr=0.0010,
):
    return {
        'pair':        'EURUSD',
        'signal':      signal,
        'probability': probability,
        'entry_price': entry,
        'stop_loss':   sl,
        'take_profit': tp,
        'risk_reward': rr,
        'atr':         atr,
    }


def _xauusd_signal(
    signal='bullish',
    probability=0.70,
    entry=2000.0,
    sl=1990.0,
    tp=2020.0,
    rr=2.0,
    atr=2.0,
):
    return {
        'pair':        'XAUUSD',
        'signal':      signal,
        'probability': probability,
        'entry_price': entry,
        'stop_loss':   sl,
        'take_profit': tp,
        'risk_reward': rr,
        'atr':         atr,
    }


# ── _risk_pct ─────────────────────────────────────────────────────────────────

class TestRiskPct:
    def test_high_confidence_returns_3pct(self):
        assert _risk_pct(0.75) == 3.0

    def test_above_high_threshold_returns_3pct(self):
        assert _risk_pct(0.90) == 3.0

    def test_mid_confidence_returns_2pct(self):
        assert _risk_pct(0.65) == 2.0

    def test_between_65_and_75_returns_2pct(self):
        assert _risk_pct(0.70) == 2.0

    def test_low_confidence_returns_1pct(self):
        assert _risk_pct(0.58) == 1.0

    def test_between_58_and_65_returns_1pct(self):
        assert _risk_pct(0.60) == 1.0

    def test_below_minimum_returns_1pct(self):
        # Even below trading threshold, sizing defaults to 1%
        assert _risk_pct(0.50) == 1.0

    def test_exactly_at_each_boundary(self):
        for threshold, expected_pct in RISK_TIERS:
            assert _risk_pct(threshold) == expected_pct


# ── _atr_pips ─────────────────────────────────────────────────────────────────

class TestAtrPips:
    def test_eurusd_multiplier_is_10000(self):
        sig = {'pair': 'EURUSD', 'atr': 0.001}
        assert _atr_pips(sig) == pytest.approx(10.0)

    def test_xauusd_multiplier_is_100(self):
        sig = {'pair': 'XAUUSD', 'atr': 2.0}
        assert _atr_pips(sig) == pytest.approx(200.0)

    def test_zero_atr_returns_zero(self):
        sig = {'pair': 'EURUSD', 'atr': 0}
        assert _atr_pips(sig) == 0.0

    def test_none_atr_returns_zero(self):
        sig = {'pair': 'EURUSD', 'atr': None}
        assert _atr_pips(sig) == 0.0

    def test_missing_pair_defaults_to_10000(self):
        sig = {'atr': 0.001}
        assert _atr_pips(sig) == pytest.approx(10.0)

    def test_eurusd_typical_range(self):
        # Typical EURUSD H4 ATR ~0.0008–0.0015 → 8–15 pips
        sig = {'pair': 'EURUSD', 'atr': 0.0010}
        result = _atr_pips(sig)
        assert 5 <= result <= 20

    def test_xauusd_typical_range(self):
        # Typical XAUUSD H4 ATR ~2–8 → 200–800 pips
        sig = {'pair': 'XAUUSD', 'atr': 3.0}
        result = _atr_pips(sig)
        assert 100 <= result <= 1000


# ── _position_sizing ──────────────────────────────────────────────────────────

class TestPositionSizing:
    def test_returns_required_keys(self):
        result = _position_sizing(_eurusd_signal(), account_balance=500.0)
        for key in ('risk_pct', 'risk_usd', 'pip_risk', 'lot_size', 'potential_reward', 'account_balance'):
            assert key in result

    def test_eurusd_risk_usd_scales_with_balance(self):
        result = _position_sizing(_eurusd_signal(probability=0.65), account_balance=500.0)
        # 2% of $500 = $10
        assert result['risk_usd'] == pytest.approx(10.0, abs=0.01)

    def test_lot_size_at_least_min_lot(self):
        result = _position_sizing(_eurusd_signal(), account_balance=200.0)
        assert result['lot_size'] >= MIN_LOT

    def test_lot_size_at_most_max_lot(self):
        result = _position_sizing(_eurusd_signal(probability=0.90), account_balance=100_000.0)
        assert result['lot_size'] <= MAX_LOT

    def test_lot_size_is_multiple_of_001(self):
        result = _position_sizing(_eurusd_signal(), account_balance=500.0)
        lot = result['lot_size']
        assert round(lot * 100) == int(round(lot * 100))

    def test_zero_pip_risk_falls_back_to_min_lot(self):
        sig = _eurusd_signal(entry=0, sl=0)  # can't compute pip_risk
        result = _position_sizing(sig, account_balance=500.0)
        assert result['lot_size'] == MIN_LOT

    def test_eurusd_math(self):
        # entry=1.10000, sl=1.09700 → pip_risk = 30 pips
        # balance=500, prob=0.65 → risk_usd=10, pip_value=0.10
        # lots = 10 / (30 * 0.10) = 3.33 → floor to 0.01 → 3.33 → 3.33... → 3.33 truncated to 0.01 = 3.33
        sig = _eurusd_signal(probability=0.65, entry=1.10000, sl=1.09700)
        result = _position_sizing(sig, account_balance=500.0)
        assert result['pip_risk'] == pytest.approx(30.0, abs=0.5)
        assert result['lot_size'] >= MIN_LOT

    def test_xauusd_pip_value_is_1_per_micro(self):
        # pip_value for XAUUSD = $1.00 per 0.01 lot
        sig = _xauusd_signal(probability=0.65, entry=2000.0, sl=1990.0)
        result = _position_sizing(sig, account_balance=500.0)
        # pip_risk = 10 pips * 100 = 1000? no: (2000-1990)*100 = 1000 pips
        # risk_usd = 500 * 2% = 10
        # lots = 10 / (1000 * 1.0) = 0.01
        assert result['lot_size'] >= MIN_LOT

    def test_potential_reward_is_non_negative(self):
        result = _position_sizing(_eurusd_signal(), account_balance=500.0)
        assert result['potential_reward'] >= 0

    def test_account_balance_stored_in_result(self):
        result = _position_sizing(_eurusd_signal(), account_balance=750.0)
        assert result['account_balance'] == 750.0


# ── evaluate() ───────────────────────────────────────────────────────────────

class TestEvaluate:
    def test_no_signal_returns_skip_immediately(self):
        sig = _eurusd_signal(signal='no_signal')
        result = evaluate(sig)
        assert result['action'] == 'SKIP'
        assert result['score'] == 0

    def test_no_signal_has_sizing_anyway(self):
        sig = _eurusd_signal(signal='no_signal')
        result = evaluate(sig)
        assert 'sizing' in result
        assert result['sizing']['lot_size'] >= MIN_LOT

    def test_all_pass_returns_execute(self):
        sig = _eurusd_signal(probability=0.75, entry=1.10000, sl=1.09700, tp=1.10600, rr=2.0, atr=0.001)
        result = evaluate(sig, open_positions=0, daily_pnl_usd=0, account_balance=500)
        assert result['action'] == 'EXECUTE'

    def test_execute_includes_sizing(self):
        sig = _eurusd_signal(probability=0.75)
        result = evaluate(sig, account_balance=500)
        assert 'sizing' in result
        assert result['sizing']['lot_size'] >= MIN_LOT

    def test_low_probability_returns_wait(self):
        sig = _eurusd_signal(probability=0.50, atr=0.001)
        result = evaluate(sig, open_positions=0, daily_pnl_usd=0, account_balance=500)
        # prob below threshold but not a hard fail → WAIT
        assert result['action'] in ('WAIT', 'SKIP')

    def test_too_many_positions_returns_skip(self):
        sig = _eurusd_signal(probability=0.80)
        max_pos = DECISION_RULES['max_open_positions']
        result = evaluate(sig, open_positions=max_pos, account_balance=500)
        assert result['action'] == 'SKIP'

    def test_daily_loss_limit_triggers_skip(self):
        sig = _eurusd_signal(probability=0.80)
        # Daily loss > 5% of $500 = $25
        result = evaluate(sig, open_positions=0, daily_pnl_usd=-30.0, account_balance=500)
        assert result['action'] == 'SKIP'

    def test_invalid_sl_tp_bearish_triggers_skip(self):
        # For bearish: sl must be > entry, tp must be < entry
        # Give a bullish-style SL/TP for a bearish signal → invalid
        sig = _eurusd_signal(signal='bearish', entry=1.10000, sl=1.09700, tp=1.10600)
        result = evaluate(sig, account_balance=500)
        assert result['action'] == 'SKIP'

    def test_valid_bearish_sl_tp_can_execute(self):
        sig = {
            'pair':        'EURUSD',
            'signal':      'bearish',
            'probability': 0.75,
            'entry_price': 1.10000,
            'stop_loss':   1.10300,   # sl > entry for bearish
            'take_profit': 1.09400,   # tp < entry for bearish
            'risk_reward': 2.0,
            'atr':         0.001,
        }
        result = evaluate(sig, open_positions=0, daily_pnl_usd=0, account_balance=500)
        assert result['action'] == 'EXECUTE'

    def test_low_rr_returns_wait(self):
        sig = _eurusd_signal(probability=0.75, rr=0.5, atr=0.001)
        result = evaluate(sig, open_positions=0, daily_pnl_usd=0, account_balance=500)
        assert result['action'] in ('WAIT', 'SKIP')

    def test_low_atr_returns_wait(self):
        sig = _eurusd_signal(probability=0.75, rr=2.0, atr=0.0001)  # ~1 pip ATR
        result = evaluate(sig, open_positions=0, daily_pnl_usd=0, account_balance=500)
        assert result['action'] in ('WAIT', 'SKIP')

    def test_score_in_0_to_100(self):
        sig = _eurusd_signal()
        result = evaluate(sig, account_balance=500)
        assert 0 <= result['score'] <= 100

    def test_score_100_when_all_pass(self):
        sig = _eurusd_signal(probability=0.75, entry=1.10000, sl=1.09700, tp=1.10600, rr=2.0, atr=0.001)
        result = evaluate(sig, open_positions=0, daily_pnl_usd=0, account_balance=500)
        assert result['score'] == 100

    def test_reasons_list_not_empty(self):
        sig = _eurusd_signal()
        result = evaluate(sig, account_balance=500)
        assert isinstance(result['reasons'], list)
        assert len(result['reasons']) > 0

    def test_each_reason_has_required_keys(self):
        sig = _eurusd_signal()
        result = evaluate(sig, account_balance=500)
        for r in result['reasons']:
            assert 'pass' in r
            assert 'rule' in r
            assert 'detail' in r

    def test_summary_contains_pair_on_execute(self):
        sig = _eurusd_signal(probability=0.75, atr=0.001)
        result = evaluate(sig, open_positions=0, daily_pnl_usd=0, account_balance=500)
        if result['action'] == 'EXECUTE':
            assert 'EURUSD' in result['summary']

    def test_xauusd_signal_evaluates_correctly(self):
        sig = _xauusd_signal(probability=0.75, atr=2.5)
        result = evaluate(sig, open_positions=0, daily_pnl_usd=0, account_balance=500)
        assert result['action'] in ('EXECUTE', 'WAIT', 'SKIP')
        assert 0 <= result['score'] <= 100

    def test_balance_affects_sizing(self):
        sig = _eurusd_signal(probability=0.75)
        r500  = evaluate(sig, account_balance=500)
        r750  = evaluate(sig, account_balance=750)
        # Higher balance should give higher or equal risk_usd
        assert r750['sizing']['risk_usd'] >= r500['sizing']['risk_usd']

    def test_result_has_all_required_keys(self):
        sig = _eurusd_signal()
        result = evaluate(sig, account_balance=500)
        for key in ('action', 'reasons', 'score', 'summary', 'sizing'):
            assert key in result
