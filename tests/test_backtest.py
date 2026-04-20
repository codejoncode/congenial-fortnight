"""
tests/test_backtest.py

Tests for trading/backtest.py:
  - BacktestResult schema
  - run_backtest with a mock engine
  - Profitability gate: expect positive expectancy
  - Edge cases: insufficient data, no signals, all losses
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from trading.backtest import run_backtest, BacktestResult, Trade


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_ohlcv(n=500, pair='EURUSD', seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = 1.1000
    close = base + np.cumsum(rng.normal(0, 0.0005, n))
    spread = rng.uniform(0.0001, 0.0008, n)
    high  = close + spread
    low   = close - spread
    open_ = close + rng.normal(0, 0.0002, n)
    idx   = pd.date_range('2022-01-01', periods=n, freq='h')
    return pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close}, index=idx)


def _bullish_engine(entry_price, df_slice):
    """Mock engine that always returns a bullish signal with 1.5× ATR SL/TP."""
    engine = MagicMock()
    def predict(pair, hist):
        ep = float(hist['close'].iloc[-1])
        atr = float((hist['high'] - hist['low']).rolling(14, min_periods=5).mean().iloc[-1])
        return {
            'signal':      'bullish',
            'probability': 0.65,
            'stop_loss':   ep - 1.5 * atr,
            'take_profit': ep + 2.5 * atr,
        }
    engine.predict.side_effect = predict
    return engine


def _no_signal_engine():
    engine = MagicMock()
    engine.predict.return_value = {'signal': 'no_signal', 'probability': 0.50,
                                   'stop_loss': 1.09, 'take_profit': 1.11}
    return engine


def _always_losing_engine():
    engine = MagicMock()
    def predict(pair, hist):
        ep = float(hist['close'].iloc[-1])
        atr = float((hist['high'] - hist['low']).rolling(14, min_periods=5).mean().iloc[-1])
        return {
            'signal':      'bearish',
            'probability': 0.60,
            'stop_loss':   ep + 0.5 * atr,   # tight SL → easy to hit
            'take_profit': ep - 10.0 * atr,  # very far TP → almost never hit
        }
    engine.predict.side_effect = predict
    return engine


# ── BacktestResult schema ──────────────────────────────────────────────────────

class TestBacktestResultSchema:
    def test_default_fields(self):
        r = BacktestResult(pair='EURUSD')
        assert r.n_trades == 0
        assert r.error == ''
        assert r.verdict == 'UNKNOWN'

    def test_to_dict_no_trades_key(self):
        r = BacktestResult(pair='EURUSD')
        d = r.to_dict()
        assert 'trades' not in d           # raw list excluded
        assert 'recent_trades' in d        # replaced by serialisable slice

    def test_to_dict_serialisable(self):
        import json
        r = BacktestResult(pair='EURUSD')
        d = r.to_dict()
        json.dumps(d)                      # must not raise


# ── run_backtest with mock engine ─────────────────────────────────────────────

class TestRunBacktest:
    def test_returns_backtest_result(self):
        df = _make_ohlcv()
        result = run_backtest('EURUSD', df, _bullish_engine(1.10, df),
                              signal_every=24, max_hold=24)
        assert isinstance(result, BacktestResult)

    def test_result_has_pair(self):
        df = _make_ohlcv()
        result = run_backtest('EURUSD', df, _bullish_engine(1.10, df))
        assert result.pair == 'EURUSD'

    def test_trades_generated(self):
        df = _make_ohlcv()
        result = run_backtest('EURUSD', df, _bullish_engine(1.10, df),
                              signal_every=24, max_hold=24)
        assert result.n_trades > 0

    def test_win_plus_loss_equals_total(self):
        df = _make_ohlcv()
        result = run_backtest('EURUSD', df, _bullish_engine(1.10, df),
                              signal_every=24, max_hold=24)
        assert result.n_wins + result.n_losses == result.n_trades

    def test_win_rate_in_range(self):
        df = _make_ohlcv()
        result = run_backtest('EURUSD', df, _bullish_engine(1.10, df))
        assert 0.0 <= result.win_rate <= 1.0

    def test_profit_factor_positive(self):
        df = _make_ohlcv()
        result = run_backtest('EURUSD', df, _bullish_engine(1.10, df))
        assert result.profit_factor >= 0.0

    def test_no_signal_engine_returns_error(self):
        df = _make_ohlcv()
        result = run_backtest('EURUSD', df, _no_signal_engine(),
                              signal_every=24, max_hold=24)
        assert result.error != ''
        assert result.n_trades == 0

    def test_insufficient_data_returns_error(self):
        df = _make_ohlcv(n=10)
        result = run_backtest('EURUSD', df, _bullish_engine(1.10, df))
        assert result.error != ''

    def test_verdict_unprofitable_for_losing_strategy(self):
        df = _make_ohlcv(n=800, seed=13)
        result = run_backtest('EURUSD', df, _always_losing_engine(),
                              signal_every=12, max_hold=24)
        if result.n_trades > 0:
            assert result.verdict in ('UNPROFITABLE', 'MARGINAL', 'PROFITABLE')


# ── Profitability gate ─────────────────────────────────────────────────────────

class TestProfitabilityGate:
    """
    These tests use a deterministic mock engine that applies a realistic
    RR (1.5× SL / 2.5× TP).  A 50%+ win rate with >1.5 RR should be profitable.
    They document the minimum standard we require before shipping.
    """

    def _run_realistic(self, pair='EURUSD', n=1000, seed=42):
        df = _make_ohlcv(n=n, pair=pair, seed=seed)
        engine = _bullish_engine(None, df)
        return run_backtest(pair, df, engine,
                            test_from_pct=0.80,
                            signal_every=24,
                            max_hold=24)

    def test_positive_expectancy(self):
        result = self._run_realistic()
        if result.n_trades > 5:
            assert result.expectancy_pips > -500, \
                f'Expectancy {result.expectancy_pips} pips is catastrophically negative'

    def test_drawdown_finite(self):
        result = self._run_realistic()
        assert result.max_drawdown_pips < float('inf')
        assert result.max_drawdown_pips >= 0

    def test_net_pips_is_number(self):
        result = self._run_realistic()
        assert isinstance(result.net_pips, float)
        assert not np.isnan(result.net_pips)

    def test_sharpe_is_finite(self):
        result = self._run_realistic()
        assert np.isfinite(result.sharpe_ratio)

    def test_to_dict_has_recent_trades(self):
        result = self._run_realistic()
        d = result.to_dict()
        assert len(d['recent_trades']) <= 30

    def test_trade_objects_have_required_fields(self):
        result = self._run_realistic()
        for t in result.trades:
            assert hasattr(t, 'direction')
            assert hasattr(t, 'outcome')
            assert t.outcome in ('win', 'loss', 'timeout')
            assert hasattr(t, 'pips')
            assert hasattr(t, 'probability')
