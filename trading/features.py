"""
trading/features.py — Canonical feature engineering for the signal engine.

Single source of truth used by BOTH model training and live inference.
NO random values. All features are deterministic given price data.

Features: ~37 TA + candlestick + 12 Fibonacci harmonic pattern features.
"""

import numpy as np
import pandas as pd
from trading.harmonic import HARMONIC_COLS, add_harmonic_features

# ── Canonical feature list ────────────────────────────────────────────────────
# Any change here requires retraining. The SignalEngine saves this list in
# meta.json so training and inference are always consistent.
FEATURE_COLS = [
    # Log returns
    'ret_1', 'ret_5', 'ret_10',
    # MA relationships (scale-invariant ratios)
    'close_vs_sma5', 'close_vs_sma10', 'close_vs_sma20', 'close_vs_sma50',
    'ema5_vs_ema20',
    'sma5_slope', 'sma20_slope',
    # Momentum oscillators
    'rsi_14',
    'stoch_k', 'stoch_d',
    # MACD (normalized by close)
    'macd_norm', 'macd_signal_norm', 'macd_hist_norm',
    # Bollinger Bands
    'bb_pct', 'bb_width_norm',
    # Volatility
    'atr_pct', 'vol_20',
    # Candlestick geometry (normalized by ATR, so scale-invariant)
    'body_pct', 'upper_shadow_pct', 'lower_shadow_pct', 'candle_direction',
    # Real candlestick patterns (binary — deterministic, no random)
    'doji', 'hammer', 'shooting_star',
    'bullish_engulfing', 'bearish_engulfing',
    'pin_bar_bull', 'pin_bar_bear',
    # Momentum / Rate of change
    'roc_5', 'roc_10',
    # Support / resistance proximity
    'dist_from_high20', 'dist_from_low20',
    # Market regime
    'trend_strength',      # (close - sma50) / sma50 * 100
    'vol_regime',          # vol_20 > rolling median → 1 else 0
    # Fibonacci harmonic patterns (Gartley, Bat, Butterfly, Crab, Shark, ABCD)
    *HARMONIC_COLS,
]


# ── Private helpers ───────────────────────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).clip(0, 100)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _stoch(
    high: pd.Series, low: pd.Series, close: pd.Series,
    k_period: int = 14, d_period: int = 3
):
    low_min = low.rolling(k_period, min_periods=k_period).min()
    high_max = high.rolling(k_period, min_periods=k_period).max()
    denom = (high_max - low_min).replace(0, np.nan)
    k = (100 * (close - low_min) / denom).clip(0, 100)
    d = k.rolling(d_period, min_periods=1).mean()
    return k, d


# ── Public API ────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all FEATURE_COLS for an OHLCV DataFrame.

    Required columns: open, high, low, close
    Optional:        volume

    Returns a DataFrame indexed the same as df, with columns = FEATURE_COLS.
    NaN rows (insufficient lookback) are NOT dropped here — caller decides.
    Caller should do:
        feat = build_features(df).dropna()
    """
    o = df['open'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)

    feat = pd.DataFrame(index=df.index)

    # ── Returns ───────────────────────────────────────────────────────────────
    feat['ret_1']  = c.pct_change(1)
    feat['ret_5']  = c.pct_change(5)
    feat['ret_10'] = c.pct_change(10)

    # ── Moving averages ───────────────────────────────────────────────────────
    sma5  = c.rolling(5,  min_periods=5 ).mean()
    sma10 = c.rolling(10, min_periods=10).mean()
    sma20 = c.rolling(20, min_periods=20).mean()
    sma50 = c.rolling(50, min_periods=50).mean()
    ema5  = _ema(c, 5)
    ema20 = _ema(c, 20)

    feat['close_vs_sma5']  = (c - sma5)  / sma5
    feat['close_vs_sma10'] = (c - sma10) / sma10
    feat['close_vs_sma20'] = (c - sma20) / sma20
    feat['close_vs_sma50'] = (c - sma50) / sma50
    feat['ema5_vs_ema20']  = (ema5 - ema20) / ema20
    feat['sma5_slope']     = sma5.pct_change(1)
    feat['sma20_slope']    = sma20.pct_change(1)

    # ── Oscillators ───────────────────────────────────────────────────────────
    feat['rsi_14'] = _rsi(c, 14)
    feat['stoch_k'], feat['stoch_d'] = _stoch(h, l, c)

    # ── MACD ──────────────────────────────────────────────────────────────────
    ema12 = _ema(c, 12)
    ema26 = _ema(c, 26)
    macd  = ema12 - ema26
    macd_sig = _ema(macd, 9)
    macd_h   = macd - macd_sig
    c_safe = c.replace(0, np.nan)
    feat['macd_norm']        = macd    / c_safe
    feat['macd_signal_norm'] = macd_sig / c_safe
    feat['macd_hist_norm']   = macd_h  / c_safe

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb_mid   = sma20
    bb_std   = c.rolling(20, min_periods=20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = bb_upper - bb_lower
    feat['bb_pct']        = (c - bb_lower) / (bb_width + 1e-10)
    feat['bb_width_norm'] = bb_width / c_safe

    # ── ATR / Volatility ──────────────────────────────────────────────────────
    atr = _atr(h, l, c, 14)
    atr_safe = atr.replace(0, np.nan)
    feat['atr_pct'] = atr / c_safe
    feat['vol_20']  = c.pct_change().rolling(20, min_periods=10).std()

    # ── Candlestick geometry ──────────────────────────────────────────────────
    body         = (c - o).abs()
    upper_shadow = h - pd.concat([c, o], axis=1).max(axis=1)
    lower_shadow = pd.concat([c, o], axis=1).min(axis=1) - l

    feat['body_pct']         = body         / atr_safe
    feat['upper_shadow_pct'] = upper_shadow / atr_safe
    feat['lower_shadow_pct'] = lower_shadow / atr_safe
    feat['candle_direction']  = np.sign(c - o)   # 1=bull, -1=bear, 0=doji

    # ── Real candlestick patterns (DETERMINISTIC — no random) ─────────────────
    bar_range = (h - l).replace(0, np.nan)

    # Doji: body < 15% of ATR
    feat['doji'] = (body < 0.15 * atr_safe).astype(int)

    # Hammer: long lower wick, short upper wick, appearing in downtrend
    feat['hammer'] = (
        (lower_shadow > 2.0 * body) &
        (upper_shadow < 0.5 * body + 1e-10) &
        (feat['ret_5'] < 0)
    ).astype(int)

    # Shooting star: long upper wick, short lower wick, appearing in uptrend
    feat['shooting_star'] = (
        (upper_shadow > 2.0 * body) &
        (lower_shadow < 0.5 * body + 1e-10) &
        (feat['ret_5'] > 0)
    ).astype(int)

    # Bullish engulfing (2-bar): prev bearish, curr bullish, curr body engulfs prev
    prev_o = o.shift(1)
    prev_c = c.shift(1)
    feat['bullish_engulfing'] = (
        (prev_c < prev_o) &              # prev bar bearish
        (c > o) &                        # curr bar bullish
        (o <= prev_c) &                  # opens at or below prev close
        (c >= prev_o) &                  # closes at or above prev open
        ((c - o) > (prev_o - prev_c))    # body engulfs
    ).astype(int)

    # Bearish engulfing (2-bar): prev bullish, curr bearish, curr body engulfs prev
    feat['bearish_engulfing'] = (
        (prev_c > prev_o) &
        (c < o) &
        (o >= prev_c) &
        (c <= prev_o) &
        ((o - c) > (prev_c - prev_o))
    ).astype(int)

    # Pin bar bullish: lower wick > 60% of bar range, body < 25%, close in upper 60%
    feat['pin_bar_bull'] = (
        (lower_shadow > 0.60 * bar_range) &
        (body < 0.25 * bar_range) &
        ((c - l) / bar_range > 0.60)
    ).astype(int)

    # Pin bar bearish: upper wick > 60% of bar range, body < 25%, close in lower 40%
    feat['pin_bar_bear'] = (
        (upper_shadow > 0.60 * bar_range) &
        (body < 0.25 * bar_range) &
        ((c - l) / bar_range < 0.40)
    ).astype(int)

    # ── Momentum ──────────────────────────────────────────────────────────────
    feat['roc_5']  = c.pct_change(5)
    feat['roc_10'] = c.pct_change(10)

    # ── Support / resistance ──────────────────────────────────────────────────
    high20 = h.rolling(20, min_periods=10).max()
    low20  = l.rolling(20, min_periods=10).min()
    feat['dist_from_high20'] = (high20 - c) / c_safe   # 0 = AT resistance
    feat['dist_from_low20']  = (c - low20)  / c_safe   # 0 = AT support

    # ── Market regime ─────────────────────────────────────────────────────────
    feat['trend_strength'] = ((c - sma50) / sma50 * 100)

    vol_median = feat['vol_20'].rolling(50, min_periods=20).median()
    feat['vol_regime'] = (feat['vol_20'] > vol_median).astype(int)

    # ── Harmonic patterns ─────────────────────────────────────────────────────
    # add_harmonic_features adds HARMONIC_COLS columns directly to df-aligned frame
    harm_df = add_harmonic_features(df[['high', 'low']].copy())
    for col in HARMONIC_COLS:
        feat[col] = harm_df[col].values

    return feat[FEATURE_COLS]


def get_latest_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Return the ATR of the last bar (used for SL/TP calculation)."""
    return float(_atr(
        df['high'].astype(float),
        df['low'].astype(float),
        df['close'].astype(float),
        period,
    ).iloc[-1])
