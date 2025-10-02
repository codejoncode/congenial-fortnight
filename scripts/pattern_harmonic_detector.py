#!/usr/bin/env python3
"""Pattern and harmonic detector utilities for advanced feature engineering.

This module provides clustering-inspired chart pattern detection and harmonic
pattern recognition to enrich feature sets for the signal generation system.
The detectors operate on rolling windows and expose numeric flags and scores
that can be consumed by downstream models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class PatternConfig:
    pattern_window: int = 60
    pivot_lookback: int = 5
    min_pivots: int = 5
    flag_trend_window: int = 20
    flag_consolidation_window: int = 10
    flag_trend_threshold: float = 0.02
    flag_slope_threshold: float = 0.0005
    flag_vol_ratio: float = 0.75
    pennant_vol_ratio: float = 0.55
    triangle_tolerance: float = 0.002
    slope_floor: float = 0.0003
    harmonic_tolerance: float = 0.12


class PatternHarmonicDetector:
    """Detect classical chart patterns and harmonic structures."""

    pattern_feature_names: Tuple[str, ...] = (
        "pattern_sym_triangle",
        "pattern_sym_triangle_score",
        "pattern_ascending_triangle",
        "pattern_descending_triangle",
        "pattern_bull_flag",
        "pattern_bear_flag",
        "pattern_bull_pennant",
        "pattern_bear_pennant",
        "pattern_consolidation_ratio",
    )

    harmonic_feature_names: Tuple[str, ...] = (
        "harmonic_abcd_bull",
        "harmonic_abcd_bear",
        "harmonic_abcd_score",
        "harmonic_gartley_bull",
        "harmonic_gartley_bear",
        "harmonic_gartley_score",
        "harmonic_butterfly_bull",
        "harmonic_butterfly_bear",
        "harmonic_butterfly_score",
    )

    def __init__(self, config: PatternConfig | None = None):
        self.config = config or PatternConfig()

    def augment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Augment a price dataframe with pattern and harmonic features."""
        if df.empty:
            for name in (*self.pattern_feature_names, *self.harmonic_feature_names):
                df[name] = 0.0
            return df

        augmented = df.copy()
        patterns = self._compute_pattern_matrix(augmented)
        harmonics = self._compute_harmonic_matrix(augmented)

        for name, series in {**patterns, **harmonics}.items():
            augmented[name] = series

        return augmented

    # ------------------------------------------------------------------
    # Pattern detection helpers
    # ------------------------------------------------------------------
    def _compute_pattern_matrix(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        matrix = {
            name: pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
            for name in self.pattern_feature_names
        }

        min_idx = max(self.config.pattern_window, self.config.flag_trend_window + self.config.flag_consolidation_window)

        for i in range(len(df)):
            if i < min_idx:
                continue

            window = df.iloc[i - self.config.pattern_window : i + 1]
            features = self._evaluate_pattern_window(window)
            for name, value in features.items():
                matrix[name].iat[i] = value

        return matrix

    def _evaluate_pattern_window(self, window: pd.DataFrame) -> Dict[str, float]:
        features = {name: 0.0 for name in self.pattern_feature_names}

        if len(window) < self.config.pattern_window:
            return features

        pivots = self._identify_pivots(window)
        if pivots:
            features.update(self._analyze_triangle_patterns(window, pivots))

        features.update(self._analyze_flag_patterns(window))
        return features

    def _identify_pivots(self, window: pd.DataFrame) -> List[Dict[str, object]]:
        pivots: List[Dict[str, object]] = []
        highs = window['High'].values
        lows = window['Low'].values
        idxs = window.index

        for i in range(self.config.pivot_lookback, len(window) - self.config.pivot_lookback):
            high_slice = highs[i - self.config.pivot_lookback : i + self.config.pivot_lookback + 1]
            low_slice = lows[i - self.config.pivot_lookback : i + self.config.pivot_lookback + 1]

            if highs[i] == np.max(high_slice):
                pivots.append({'type': 'high', 'idx': idxs[i], 'price': highs[i], 'position': i})

            if lows[i] == np.min(low_slice):
                pivots.append({'type': 'low', 'idx': idxs[i], 'price': lows[i], 'position': i})

        pivots.sort(key=lambda x: x['position'])
        return pivots

    def _analyze_triangle_patterns(self, window: pd.DataFrame, pivots: List[Dict[str, object]]) -> Dict[str, float]:
        results = {
            "pattern_sym_triangle": 0.0,
            "pattern_sym_triangle_score": 0.0,
            "pattern_ascending_triangle": 0.0,
            "pattern_descending_triangle": 0.0,
        }

        high_pivots = [p for p in pivots if p['type'] == 'high'][-self.config.min_pivots :]
        low_pivots = [p for p in pivots if p['type'] == 'low'][-self.config.min_pivots :]

        if len(high_pivots) < 3 or len(low_pivots) < 3:
            return results

        try:
            upper_x = np.arange(len(high_pivots))
            upper_y = np.array([p['price'] for p in high_pivots])
            lower_x = np.arange(len(low_pivots))
            lower_y = np.array([p['price'] for p in low_pivots])

            slope_high = np.polyfit(upper_x, upper_y, 1)[0]
            slope_low = np.polyfit(lower_x, lower_y, 1)[0]
            price_scale = window['Close'].iloc[-1]
            slope_high_rel = slope_high / price_scale
            slope_low_rel = slope_low / price_scale

            slope_diff = abs(slope_high_rel - slope_low_rel)
            score = max(0.0, 1.0 - slope_diff / self.config.triangle_tolerance)

            if slope_high_rel < -self.config.slope_floor and slope_low_rel > self.config.slope_floor and slope_diff < self.config.triangle_tolerance:
                results["pattern_sym_triangle"] = 1.0
                results["pattern_sym_triangle_score"] = score

            if abs(slope_high_rel) < self.config.slope_floor and slope_low_rel > self.config.slope_floor:
                results["pattern_ascending_triangle"] = score

            if slope_high_rel < -self.config.slope_floor and abs(slope_low_rel) < self.config.slope_floor:
                results["pattern_descending_triangle"] = score

        except np.linalg.LinAlgError:
            # Degenerate pivot set; ignore
            pass

        return results

    def _analyze_flag_patterns(self, window: pd.DataFrame) -> Dict[str, float]:
        results = {
            "pattern_bull_flag": 0.0,
            "pattern_bear_flag": 0.0,
            "pattern_bull_pennant": 0.0,
            "pattern_bear_pennant": 0.0,
            "pattern_consolidation_ratio": 0.0,
        }

        close = window['Close']
        if len(close) < self.config.flag_trend_window + self.config.flag_consolidation_window + 2:
            return results

        trend_slice = close.iloc[-(self.config.flag_consolidation_window + self.config.flag_trend_window) : -self.config.flag_consolidation_window]
        consolidation_slice = close.iloc[-self.config.flag_consolidation_window :]

        if len(trend_slice) < 2 or len(consolidation_slice) < 2:
            return results

        trend_return = (trend_slice.iloc[-1] / trend_slice.iloc[0]) - 1
        consolidation_slope = np.polyfit(np.arange(len(consolidation_slice)), consolidation_slice.values, 1)[0] / consolidation_slice.iloc[-1]
        consolidation_range = (consolidation_slice.max() - consolidation_slice.min()) / consolidation_slice.iloc[-1]
        results["pattern_consolidation_ratio"] = float(consolidation_range)

        if trend_return > self.config.flag_trend_threshold and abs(consolidation_slope) < self.config.flag_slope_threshold:
            results["pattern_bull_flag"] = float(1.0 - min(1.0, consolidation_range / self.config.flag_vol_ratio))

        if trend_return < -self.config.flag_trend_threshold and abs(consolidation_slope) < self.config.flag_slope_threshold:
            results["pattern_bear_flag"] = float(1.0 - min(1.0, consolidation_range / self.config.flag_vol_ratio))

        # Pennant focus on volatility contraction relative to trend move size
        if trend_return > self.config.flag_trend_threshold:
            if consolidation_range < self.config.pennant_vol_ratio * abs(trend_return):
                results["pattern_bull_pennant"] = float(1.0 - min(1.0, consolidation_range / (abs(trend_return) + 1e-6)))

        if trend_return < -self.config.flag_trend_threshold:
            if consolidation_range < self.config.pennant_vol_ratio * abs(trend_return):
                results["pattern_bear_pennant"] = float(1.0 - min(1.0, consolidation_range / (abs(trend_return) + 1e-6)))

        return results

    # ------------------------------------------------------------------
    # Harmonic detection helpers
    # ------------------------------------------------------------------
    def _compute_harmonic_matrix(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        matrix = {
            name: pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
            for name in self.harmonic_feature_names
        }

        for i in range(len(df)):
            if i < self.config.pattern_window:
                continue

            window = df.iloc[i - self.config.pattern_window : i + 1]
            pivots = self._identify_pivots(window)
            features = self._analyze_harmonic_patterns(pivots)
            for name, value in features.items():
                matrix[name].iat[i] = value

        return matrix

    def _analyze_harmonic_patterns(self, pivots: List[Dict[str, object]]) -> Dict[str, float]:
        results = {name: 0.0 for name in self.harmonic_feature_names}
        if len(pivots) < 5:
            return results

        last_five = pivots[-5:]
        X, A, B, C, D = last_five

        XA = A['price'] - X['price']
        AB = B['price'] - A['price']
        BC = C['price'] - B['price']
        CD = D['price'] - C['price']
        AD = D['price'] - A['price']

        def ratio(value: float, base: float) -> float:
            return abs(value / base) if base != 0 else np.nan

        # AB=CD pattern
        abcd_ratio = ratio(CD, AB)
        bc_ratio = ratio(BC, AB)
        if not np.isnan(abcd_ratio) and not np.isnan(bc_ratio):
            if abs(abcd_ratio - 1.0) < self.config.harmonic_tolerance and 0.382 <= bc_ratio <= 0.886:
                score = 1.0 - (abs(abcd_ratio - 1.0) / self.config.harmonic_tolerance)
                score *= 1.0 - min(1.0, abs(bc_ratio - 0.618) / 0.5)
                score = max(0.0, min(1.0, score))
                if CD < 0:
                    results['harmonic_abcd_bull'] = 1.0
                else:
                    results['harmonic_abcd_bear'] = 1.0
                results['harmonic_abcd_score'] = score

        # Gartley pattern
        ab_retr = ratio(AB, XA)
        bc_retr = ratio(BC, AB)
        cd_ext = ratio(CD, BC)
        ad_retr = ratio(AD, XA)

        if all(not np.isnan(v) for v in [ab_retr, bc_retr, cd_ext, ad_retr]):
            if abs(ab_retr - 0.618) < self.config.harmonic_tolerance and 0.382 <= bc_retr <= 0.886 and abs(ad_retr - 0.786) < self.config.harmonic_tolerance:
                if 1.27 <= cd_ext <= 1.618:
                    score = 1.0 - (abs(ab_retr - 0.618) + abs(ad_retr - 0.786)) / (2 * self.config.harmonic_tolerance)
                    score *= 1.0 - min(1.0, abs(cd_ext - 1.27) / 1.0)
                    score = max(0.0, min(1.0, score))
                    if CD < 0:
                        results['harmonic_gartley_bull'] = 1.0
                    else:
                        results['harmonic_gartley_bear'] = 1.0
                    results['harmonic_gartley_score'] = score

        # Butterfly pattern
        if all(not np.isnan(v) for v in [ab_retr, bc_retr, cd_ext, ad_retr]):
            if abs(ab_retr - 0.786) < self.config.harmonic_tolerance and 0.382 <= bc_retr <= 0.886:
                if 1.618 <= cd_ext <= 2.24 and abs(ad_retr - 1.27) < self.config.harmonic_tolerance:
                    score = 1.0 - (abs(ab_retr - 0.786) + abs(ad_retr - 1.27)) / (2 * self.config.harmonic_tolerance)
                    score *= 1.0 - min(1.0, abs(cd_ext - 1.618) / 1.0)
                    score = max(0.0, min(1.0, score))
                    if CD < 0:
                        results['harmonic_butterfly_bull'] = 1.0
                    else:
                        results['harmonic_butterfly_bear'] = 1.0
                    results['harmonic_butterfly_score'] = score

        return results
