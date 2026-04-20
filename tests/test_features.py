"""
tests/test_features.py

Tests for trading/features.py:
  - All FEATURE_COLS are computed (no missing columns)
  - Output is deterministic (same input → same output)
  - No random values in any feature
  - Candlestick patterns are binary (0 or 1)
  - Sufficient data yields no all-NaN columns
"""
import numpy as np
import pandas as pd
import pytest

from trading.features import FEATURE_COLS, build_features, get_latest_atr


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_ohlcv(n=200, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 1.1000 + np.cumsum(rng.normal(0, 0.0005, n))
    spread = rng.uniform(0.0001, 0.0010, n)
    high  = close + spread
    low   = close - spread
    open_ = close + rng.normal(0, 0.0002, n)
    vol   = rng.integers(1000, 5000, n).astype(float)
    idx   = pd.date_range('2023-01-01', periods=n, freq='h')
    return pd.DataFrame({'open': open_, 'high': high, 'low': low,
                          'close': close, 'volume': vol}, index=idx)


@pytest.fixture
def eurusd_df():
    return _make_ohlcv(n=200)


@pytest.fixture
def xauusd_df():
    rng = np.random.default_rng(99)
    close = 2000.0 + np.cumsum(rng.normal(0, 2.0, 200))
    spread = rng.uniform(0.5, 3.0, 200)
    high  = close + spread
    low   = close - spread
    open_ = close + rng.normal(0, 1.0, 200)
    idx   = pd.date_range('2023-01-01', periods=200, freq='h')
    return pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close}, index=idx)


# ── Schema tests ───────────────────────────────────────────────────────────────

def test_feature_cols_count():
    assert len(FEATURE_COLS) == 37


def test_all_feature_cols_present(eurusd_df):
    feat = build_features(eurusd_df)
    missing = set(FEATURE_COLS) - set(feat.columns)
    assert not missing, f'Missing features: {missing}'


def test_no_extra_columns(eurusd_df):
    feat = build_features(eurusd_df)
    extra = set(feat.columns) - set(FEATURE_COLS)
    assert not extra, f'Extra features returned: {extra}'


def test_output_index_matches_input(eurusd_df):
    feat = build_features(eurusd_df)
    assert len(feat) == len(eurusd_df)
    assert feat.index.equals(eurusd_df.index)


# ── Determinism tests ──────────────────────────────────────────────────────────

def test_build_features_deterministic(eurusd_df):
    feat1 = build_features(eurusd_df)
    feat2 = build_features(eurusd_df)
    pd.testing.assert_frame_equal(feat1, feat2)


def test_build_features_no_random_values(eurusd_df):
    """Call build_features twice; features must be identical (no random component)."""
    feat1 = build_features(eurusd_df)
    feat2 = build_features(eurusd_df.copy())
    pd.testing.assert_frame_equal(feat1, feat2)


# ── Value range / sanity tests ─────────────────────────────────────────────────

def test_rsi_range(eurusd_df):
    feat = build_features(eurusd_df)
    rsi = feat['rsi_14'].dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all(), 'RSI must be 0–100'


def test_stoch_range(eurusd_df):
    feat = build_features(eurusd_df)
    for col in ['stoch_k', 'stoch_d']:
        vals = feat[col].dropna()
        assert (vals >= 0).all() and (vals <= 100).all(), f'{col} must be 0–100'


def test_binary_pattern_columns(eurusd_df):
    binary_cols = ['doji', 'hammer', 'shooting_star',
                   'bullish_engulfing', 'bearish_engulfing',
                   'pin_bar_bull', 'pin_bar_bear']
    feat = build_features(eurusd_df)
    for col in binary_cols:
        vals = feat[col].dropna().unique()
        assert set(vals).issubset({0, 1}), f'{col} must be binary, got {vals}'


def test_candle_direction_range(eurusd_df):
    feat = build_features(eurusd_df)
    vals = feat['candle_direction'].dropna().unique()
    assert set(vals).issubset({-1.0, 0.0, 1.0}), \
        f'candle_direction must be in {{-1, 0, 1}}, got {vals}'


def test_no_all_nan_columns_with_sufficient_data(eurusd_df):
    feat = build_features(eurusd_df)
    all_nan = [c for c in FEATURE_COLS if feat[c].isna().all()]
    assert not all_nan, f'Columns are entirely NaN: {all_nan}'


# ── Gold-price (large scale) sanity ───────────────────────────────────────────

def test_build_features_xauusd_scale(xauusd_df):
    feat = build_features(xauusd_df)
    missing = set(FEATURE_COLS) - set(feat.columns)
    assert not missing


def test_build_features_xauusd_deterministic(xauusd_df):
    pd.testing.assert_frame_equal(build_features(xauusd_df), build_features(xauusd_df.copy()))


# ── get_latest_atr ─────────────────────────────────────────────────────────────

def test_get_latest_atr_positive(eurusd_df):
    atr = get_latest_atr(eurusd_df)
    assert atr > 0, 'ATR must be positive'


def test_get_latest_atr_xauusd_scale(xauusd_df):
    atr = get_latest_atr(xauusd_df)
    assert 0.1 < atr < 100, f'Gold ATR {atr} out of expected range'
