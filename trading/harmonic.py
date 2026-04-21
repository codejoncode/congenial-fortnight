"""
trading/harmonic.py — Fibonacci harmonic pattern detection for OHLCV DataFrames.

Patterns detected (XABCD swing structure, alternating high/low pivots):
  Gartley   — "222" pattern, tightest reversal
  Bat        — deep B retracement, D at 0.886 XA
  Butterfly  — extended reversal, D beyond X
  Crab       — extreme extension, D at 1.618 XA
  Shark      — newer pattern, BC extension
  ABCD       — simpler 4-point pattern (no X required)

Usage:
    from trading.harmonic import add_harmonic_features
    feat_df = add_harmonic_features(df)   # adds 11 columns to df

Integration:
    Called from trading/features.py::build_features() so these features
    flow automatically into SignalEngine training and inference.
"""

import numpy as np
import pandas as pd

# ── Fibonacci ratios per pattern (min, target, max with tolerance) ─────────────
# Format: (ratio_name, ideal_value, tolerance)
_TOL = 0.08   # ±8% tolerance — balances signal frequency vs false positives

def _in(value: float, lo: float, hi: float) -> bool:
    return lo <= value <= hi

def _near(value: float, target: float, tol: float = _TOL) -> bool:
    return abs(value - target) <= tol


# ── Pivot detection ────────────────────────────────────────────────────────────

def _find_pivots(high: np.ndarray, low: np.ndarray, strength: int = 3):
    """
    Find swing highs and swing lows.
    strength: bars required on EACH side (3 = 7-bar confirmation window).
    Returns (pivot_high_idx, pivot_low_idx) as arrays of integer indices.
    """
    n = len(high)
    ph, pl = [], []
    for i in range(strength, n - strength):
        lo_win = low[i - strength: i]
        hi_win = high[i - strength: i]
        lo_win2 = low[i + 1: i + strength + 1]
        hi_win2 = high[i + 1: i + strength + 1]

        if high[i] > hi_win.max() and high[i] > hi_win2.max():
            ph.append(i)
        if low[i] < lo_win.min() and low[i] < lo_win2.min():
            pl.append(i)
    return np.array(ph, dtype=int), np.array(pl, dtype=int)


# ── Pattern checkers ───────────────────────────────────────────────────────────

def _ratios(X: float, A: float, B: float, C: float, D: float):
    """Return key Fibonacci ratios for an XABCD leg set."""
    XA = abs(A - X)
    AB = abs(B - A)
    BC = abs(C - B)
    CD = abs(D - C)
    if XA < 1e-10:
        return None
    return {
        'AB_XA': AB / XA,
        'BC_AB': BC / AB if AB > 1e-10 else 0,
        'CD_BC': CD / BC if BC > 1e-10 else 0,
        'AD_XA': abs(D - A) / XA,
    }


def _check_patterns(X: float, A: float, B: float, C: float, D: float,
                    bullish: bool) -> dict:
    """
    Given XABCD leg prices (alternating hi/lo pivots), check all patterns.
    For bullish: X=high, A=low, B=high, C=low, D=current bar low (potential reversal up)
    For bearish: X=low, A=high, B=low, C=high, D=current bar high

    Returns dict of pattern_name -> 1/0.
    """
    r = _ratios(X, A, B, C, D)
    if r is None:
        return {}

    AB_XA = r['AB_XA']
    BC_AB = r['BC_AB']
    CD_BC = r['CD_BC']
    AD_XA = r['AD_XA']

    results = {}

    # Gartley: AB≈0.618 XA, BC=0.382-0.886 AB, CD=1.13-1.618 BC, AD≈0.786 XA
    results['gartley'] = int(
        _near(AB_XA, 0.618) and
        _in(BC_AB, 0.382, 0.886) and
        _in(CD_BC, 1.13, 1.618) and
        _near(AD_XA, 0.786)
    )

    # Bat: AB=0.382-0.50 XA, BC=0.382-0.886 AB, CD=1.618-2.618 BC, AD≈0.886 XA
    results['bat'] = int(
        _in(AB_XA, 0.382, 0.500) and
        _in(BC_AB, 0.382, 0.886) and
        _in(CD_BC, 1.618, 2.618) and
        _near(AD_XA, 0.886)
    )

    # Butterfly: AB≈0.786 XA, BC=0.382-0.886 AB, CD=1.618-2.618 BC, AD=1.27-1.618 XA
    results['butterfly'] = int(
        _near(AB_XA, 0.786) and
        _in(BC_AB, 0.382, 0.886) and
        _in(CD_BC, 1.618, 2.618) and
        _in(AD_XA, 1.27 - _TOL, 1.618 + _TOL)
    )

    # Crab: AB=0.382-0.618 XA, BC=0.382-0.886 AB, CD=2.618-3.618 BC, AD≈1.618 XA
    results['crab'] = int(
        _in(AB_XA, 0.382, 0.618) and
        _in(BC_AB, 0.382, 0.886) and
        _in(CD_BC, 2.618, 3.618) and
        _near(AD_XA, 1.618, 0.10)
    )

    # Shark: AB=0.382-0.618 XA, BC=1.13-1.618 AB, AD=0.886-1.13 XA
    results['shark'] = int(
        _in(AB_XA, 0.382, 0.618) and
        _in(BC_AB, 1.13, 1.618) and
        _in(AD_XA, 0.886, 1.13)
    )

    return results


def _check_abcd(A: float, B: float, C: float, D: float) -> int:
    """
    ABCD pattern: BC retraces 0.618 or 0.786 of AB, CD ≈ AB in length.
    Returns 1 if pattern detected.
    """
    AB = abs(B - A)
    BC = abs(C - B)
    CD = abs(D - C)
    if AB < 1e-10:
        return 0
    BC_AB = BC / AB
    CD_AB = CD / AB
    return int(
        (_near(BC_AB, 0.618, 0.08) or _near(BC_AB, 0.786, 0.08)) and
        _near(CD_AB, 1.0, 0.10)
    )


# ── Public API ────────────────────────────────────────────────────────────────

HARMONIC_COLS = [
    'harm_gartley_bull', 'harm_gartley_bear',
    'harm_bat_bull',     'harm_bat_bear',
    'harm_butterfly_bull','harm_butterfly_bear',
    'harm_crab_bull',    'harm_crab_bear',
    'harm_shark_bull',   'harm_shark_bear',
    'harm_abcd',
    'harm_score',        # net bullish - bearish patterns at current bar
]


def add_harmonic_features(df: pd.DataFrame, strength: int = 2) -> pd.DataFrame:
    """
    Detect harmonic patterns in df and add HARMONIC_COLS as new columns.

    df must have: high, low, close columns (lowercase).
    Returns df with harmonic columns added (NaN rows stay NaN).

    strength: pivot confirmation bars each side (default=3, i.e. 7-bar window)
    """
    h = df['high'].astype(float).values
    l = df['low'].astype(float).values
    n = len(df)

    # Initialise output arrays
    out = {col: np.zeros(n, dtype=np.float32) for col in HARMONIC_COLS}
    out['harm_score'] = np.zeros(n, dtype=np.float32)

    ph, pl = _find_pivots(h, l, strength=strength)

    # Need at least 2 highs + 2 lows for XABCD
    if len(ph) < 2 or len(pl) < 2:
        for col in HARMONIC_COLS:
            df[col] = out[col]
        return df

    # For each bar i, search recent pivots for valid XABCD sequences.
    # A valid bullish XABCD: X=pivot_high, A=pivot_low, B=pivot_high,
    #                        C=pivot_low, D=current_low — all in chronological order.
    for i in range(strength, n):
        ph_before = ph[ph < i]
        pl_before = pl[pl < i]

        if len(ph_before) < 2 or len(pl_before) < 2:
            continue

        cur_h = h[i]
        cur_l = l[i]
        bullish_score = 0
        bearish_score = 0

        # Try the 3 most recent pivot pairs to find alternating sequences
        for hi in range(min(3, len(ph_before))):
            for li in range(min(3, len(pl_before))):
                H1_idx = ph_before[-(hi + 1)]
                L1_idx = pl_before[-(li + 1)]

                # ── Bullish XABCD: need two highs and two lows in order ───────
                if hi + 1 < len(ph_before) and li + 1 < len(pl_before):
                    H2_idx = ph_before[-(hi + 2)]
                    L2_idx = pl_before[-(li + 2)]
                    # Chronological: H2, L2, H1, L1 (alternating hi-lo-hi-lo)
                    if H2_idx < L2_idx < H1_idx < L1_idx:
                        X, A, B, C = h[H2_idx], l[L2_idx], h[H1_idx], l[L1_idx]
                        res = _check_patterns(X, A, B, C, cur_l, bullish=True)
                        for pat in ['gartley', 'bat', 'butterfly', 'crab', 'shark']:
                            v = res.get(pat, 0)
                            if v and not out[f'harm_{pat}_bull'][i]:
                                out[f'harm_{pat}_bull'][i] = v
                                bullish_score += v

                    # Bearish XABCD: L2, H2, L1, H1 (alternating lo-hi-lo-hi)
                    if L2_idx < H2_idx < L1_idx < H1_idx:
                        X, A, B, C = l[L2_idx], h[H2_idx], l[L1_idx], h[H1_idx]
                        res = _check_patterns(X, A, B, C, cur_h, bullish=False)
                        for pat in ['gartley', 'bat', 'butterfly', 'crab', 'shark']:
                            v = res.get(pat, 0)
                            if v and not out[f'harm_{pat}_bear'][i]:
                                out[f'harm_{pat}_bear'][i] = v
                                bearish_score += v

                # ── ABCD (3-point, no X) — A=high, B=low, C=high, D=cur_l ──────
                if H1_idx < L1_idx:    # high then low → bullish ABCD
                    if hi + 1 < len(ph_before):
                        H2_idx = ph_before[-(hi + 2)]
                        if H2_idx < H1_idx < L1_idx:
                            abcd = _check_abcd(h[H2_idx], l[H1_idx], h[L1_idx], cur_l)
                            if abcd:
                                out['harm_abcd'][i] = 1

        out['harm_score'][i] = float(bullish_score - bearish_score)

    for col in HARMONIC_COLS:
        df[col] = out[col]

    return df
