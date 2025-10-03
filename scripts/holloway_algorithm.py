"""Complete Holloway Algorithm - exact PineScript translation.

This module provides a direct port of the better-Holloway-Algorithm.txt
PineScript indicator into Python. It includes weighted historical scoring,
multi-average smoothing (SMA/EMA/HMA/RMA), trend diagnostics, critical level
tracking, and double-failure pattern detection.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import warnings
import re

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


def _weighted_average(values: np.ndarray) -> float:
    """Weighted moving average helper used for HMA calculations."""
    if values.size == 0:
        return np.nan
    mask = ~np.isnan(values)
    if not mask.any():
        return np.nan
    filtered = values[mask]
    weights = np.arange(1, filtered.size + 1, dtype=float)
    return np.average(filtered, weights=weights)


class CompleteHollowayAlgorithm:
    """Exact PineScript translation of the Holloway Algorithm."""

    def __init__(self, data_dir: str = "data") -> None:
        self.data_dir = data_dir
        self.ensure_data_dir()

        # Weighted scoring ladder from PineScript implementation
        self.historical_weights = [3.0, 2.7, 2.4, 2.1, 1.8, 1.5, 1.2, 0.9, 0.6]

        # Critical level thresholds for regime monitoring
        self.resistance_level = 95.0
        self.support_level = 12.0

        print("🎯 Complete Holloway Algorithm initialized with:")
        print(f"   📊 Historical weighting: {self.historical_weights}")
        print(f"   📈 Resistance level: {self.resistance_level}")
        print(f"   📉 Support level: {self.support_level}")

    def ensure_data_dir(self) -> None:
        """Create the data directory if it does not yet exist."""
        if self.data_dir and not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"✅ Created data directory: {self.data_dir}")

    def calculate_parabolic_sar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Parabolic SAR exactly as defined in PineScript."""
        df = df.copy()
        if df.empty:
            return df

        start = 0.02
        inc = 0.02
        max_af = 0.2  # PineScript reference uses 0.2

        sar = np.zeros(len(df))
        ep = np.zeros(len(df))
        af = np.zeros(len(df))
        direction = np.ones(len(df))

        sar[0] = df["low"].iat[0]
        ep[0] = df["high"].iat[0]
        af[0] = start
        direction[0] = 1

        for i in range(1, len(df)):
            high = df["high"].iat[i]
            low = df["low"].iat[i]
            prev_high = df["high"].iat[i - 1]
            prev_low = df["low"].iat[i - 1]

            prev_sar = sar[i - 1]
            prev_ep = ep[i - 1]
            prev_af = af[i - 1]
            prev_dir = direction[i - 1]

            if prev_dir == 1:
                new_sar = prev_sar + prev_af * (prev_ep - prev_sar)
                if low <= new_sar:
                    direction[i] = -1
                    sar[i] = max(high, prev_ep)
                    ep[i] = low
                    af[i] = start
                else:
                    direction[i] = 1
                    sar[i] = max(new_sar, prev_low, low)
                    if high > prev_ep:
                        ep[i] = high
                        af[i] = min(max_af, prev_af + inc)
                    else:
                        ep[i] = prev_ep
                        af[i] = prev_af
            else:
                new_sar = prev_sar - prev_af * (prev_sar - prev_ep)
                if high >= new_sar:
                    direction[i] = 1
                    sar[i] = min(low, prev_ep)
                    ep[i] = high
                    af[i] = start
                else:
                    direction[i] = -1
                    sar[i] = min(new_sar, prev_high, high)
                    if low < prev_ep:
                        ep[i] = low
                        af[i] = min(max_af, prev_af + inc)
                    else:
                        ep[i] = prev_ep
                        af[i] = prev_af

        df["sar"] = sar
        # Ensure boolean dtype for SAR comparisons (avoid float/NaN which break bitwise ops)
        df["bull_sar"] = (df["close"] > df["sar"]).fillna(False).astype(bool)
        df["bear_sar"] = (df["close"] < df["sar"]).fillna(False).astype(bool)

        return df

    def calculate_pattern_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate pattern-based signals (engulfing, inside bars, SAR touches)."""
        df = df.copy()
        if df.empty:
            return df

        df["key_sma"] = df["close"].rolling(window=7, min_periods=1).mean()
        df["sma22"] = df["close"].rolling(window=22, min_periods=1).mean()
        df["sma3"] = df["close"].rolling(window=3, min_periods=1).mean()

        df["engulf"] = (df["high"] > df["high"].shift(1)) & (df["low"] < df["low"].shift(1))
        df["inside"] = (df["high"].shift(1) > df["high"]) & (df["low"].shift(1) < df["low"])

        df["confirmed_bull_engulf"] = df["engulf"].shift(1) & (df["high"] > df["high"].shift(1))
        df["confirmed_bear_engulf"] = df["engulf"].shift(1) & (df["low"] < df["low"].shift(1))
        df["confirmed_bull_inside"] = df["inside"].shift(1) & (df["high"] > df["high"].shift(2))
        df["confirmed_bear_inside"] = df["inside"].shift(1) & (df["low"] < df["low"].shift(2))

        df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

        df["bullish_signal"] = (
            (df["bull_sar"] & (~df["bull_sar"].shift(1).fillna(False)))
            & (
                df["inside"] | df["inside"].shift(1) | df["inside"].shift(2)
                | df["engulf"] | df["engulf"].shift(1) | df["engulf"].shift(2)
            )
            & (df["high"] > df["ema50"])
        )
        df["bearish_signal"] = (
            (df["bear_sar"] & (~df["bear_sar"].shift(1).fillna(False)))
            & (
                df["inside"] | df["inside"].shift(1) | df["inside"].shift(2)
                | df["engulf"] | df["engulf"].shift(1) | df["engulf"].shift(2)
            )
            & (df["low"] < df["ema50"])
        )

        consecutive_min = 2
        consecutive_max = 10

        bars_above = np.zeros(len(df), dtype=int)
        bars_below = np.zeros(len(df), dtype=int)
        close = df["close"].to_numpy()
        key_sma = df["key_sma"].to_numpy()

        for i in range(1, len(df)):
            if np.isnan(close[i]) or np.isnan(key_sma[i]):
                continue

            prev_close = close[i - 1]
            prev_key = key_sma[i - 1]

            if (
                not np.isnan(prev_close)
                and not np.isnan(prev_key)
                and prev_close > prev_key
                and close[i] > key_sma[i]
            ):
                bars_above[i] = bars_above[i - 1] + 1
            elif close[i] > key_sma[i]:
                bars_above[i] = 1
            else:
                bars_above[i] = 0

            if (
                not np.isnan(prev_close)
                and not np.isnan(prev_key)
                and prev_close < prev_key
                and close[i] < key_sma[i]
            ):
                bars_below[i] = bars_below[i - 1] + 1
            elif close[i] < key_sma[i]:
                bars_below[i] = 1
            else:
                bars_below[i] = 0

        df["bars_above_key_sma"] = bars_above
        df["bars_below_key_sma"] = bars_below

        df["bull_close_indicator"] = (
            (df["bars_above_key_sma"] >= consecutive_min)
            & (df["bars_above_key_sma"] < consecutive_max)
        )
        df["bear_close_indicator"] = (
            (df["bars_below_key_sma"] >= consecutive_min)
            & (df["bars_below_key_sma"] < consecutive_max)
        )

        df["new_cross_bull"] = (
            (df["key_sma"] > df["sma22"])
            & (df["key_sma"].shift(1) <= df["sma22"].shift(1))
        )
        df["new_cross_bear"] = (
            (df["key_sma"] < df["sma22"])
            & (df["key_sma"].shift(1) >= df["sma22"].shift(1))
        )

        df["bull_signal"] = (
            (df["bars_above_key_sma"] == consecutive_min)
            & (df["sma3"] > df["key_sma"])
            & (df["close"] > df["sma22"])
        )
        df["bear_signal"] = (
            (df["bars_below_key_sma"] == consecutive_min)
            & (df["sma3"] < df["key_sma"])
            & (df["close"] < df["sma22"])
        )

        df["bull_sar2"] = (
            (df["close"] > df["sar"])
            & (df["close"].shift(1) < df["sar"].shift(1))
            & (df["close"] > df["key_sma"])
            & (df["close"].shift(1) < df["key_sma"].shift(1))
        )
        df["bear_sar2"] = (
            (df["close"] < df["sar"])
            & (df["close"].shift(1) > df["sar"].shift(1))
            & (df["close"] < df["key_sma"])
            & (df["close"].shift(1) > df["key_sma"].shift(1))
        )

        lookback = 10
        df["max15"] = df["high"].rolling(window=lookback, min_periods=1).max()
        df["min15"] = df["low"].rolling(window=lookback, min_periods=1).min()

        df["bull_sig"] = df["high"] == df["max15"]
        df["bear_sig"] = df["low"] == df["min15"]

        df["bear_touch"] = (
            (df["high"] >= df["sma22"])
            & (df["close"] < df["key_sma"])
            & (df["key_sma"] < df["sma22"])
        )
        df["bull_touch"] = (
            (df["low"] <= df["sma22"])
            & (df["close"] > df["key_sma"])
            & (df["key_sma"] > df["sma22"])
        )

        df["new_bear"] = df["bear_touch"] & ~(
            df["bear_touch"].shift(1) | df["bear_touch"].shift(2) | df["bear_touch"].shift(3)
        )
        df["new_bull"] = df["bull_touch"] & ~(
            df["bull_touch"].shift(1) | df["bull_touch"].shift(2) | df["bull_touch"].shift(3)
        )

        return df

    def calculate_bull_count(self, df: pd.DataFrame) -> pd.Series:
        """Weighted bull condition sum using PineScript scoring ladder."""
        bull_count = np.zeros(len(df), dtype=float)

        conditions = [
            "bull_sar",
            "engulf",
            "inside",
            "confirmed_bull_engulf",
            "confirmed_bull_inside",
            "bullish_signal",
            "bull_close_indicator",
            "new_cross_bull",
            "bull_signal",
            "bull_sar2",
            "bull_sig",
            "new_bull",
        ]

        price_conditions = [
            ("close", "key_sma", ">"),
            ("close", "sma22", ">"),
        ]

        alignment_conditions = [
            (
                "bull_close_sig",
                lambda frame, idx: idx > 0
                and pd.notna(frame["close"].iat[idx])
                and pd.notna(frame["key_sma"].iat[idx])
                and pd.notna(frame["close"].iat[idx - 1])
                and pd.notna(frame["key_sma"].iat[idx - 1])
                and frame["close"].iat[idx] > frame["key_sma"].iat[idx]
                and frame["close"].iat[idx - 1] < frame["key_sma"].iat[idx - 1],
            ),
            (
                "bull_aligned",
                lambda frame, idx: pd.notna(frame["close"].iat[idx])
                and pd.notna(frame["key_sma"].iat[idx])
                and pd.notna(frame["sma22"].iat[idx])
                and frame["close"].iat[idx] > frame["key_sma"].iat[idx]
                and frame["key_sma"].iat[idx] > frame["sma22"].iat[idx],
            ),
        ]

        for i in range(len(df)):
            total = 0.0

            for condition in conditions:
                if condition not in df.columns:
                    continue
                for j, weight in enumerate(self.historical_weights):
                    idx = i - j
                    if idx < 0:
                        break
                    value = df[condition].iat[idx]
                    if pd.notna(value) and bool(value):
                        total += weight

            for price_col, ma_col, operator in price_conditions:
                if price_col not in df.columns or ma_col not in df.columns:
                    continue
                for j, weight in enumerate(self.historical_weights):
                    idx = i - j
                    if idx < 0:
                        break
                    price_val = df[price_col].iat[idx]
                    ma_val = df[ma_col].iat[idx]
                    if pd.isna(price_val) or pd.isna(ma_val):
                        continue
                    if operator == ">" and price_val > ma_val:
                        total += weight
                    elif operator == "<" and price_val < ma_val:
                        total += weight

            for _, condition_func in alignment_conditions:
                for j, weight in enumerate(self.historical_weights):
                    idx = i - j
                    if idx < 0:
                        break
                    try:
                        if condition_func(df, idx):
                            total += weight
                    except (IndexError, KeyError):
                        continue

            bull_count[i] = total

        return pd.Series(bull_count, index=df.index, name="bull_count")

    def calculate_bear_count(self, df: pd.DataFrame) -> pd.Series:
        """Weighted bear condition sum using PineScript scoring ladder."""
        bear_count = np.zeros(len(df), dtype=float)

        conditions = [
            "bear_sar",
            "engulf",
            "inside",
            "confirmed_bear_engulf",
            "confirmed_bear_inside",
            "bearish_signal",
            "bear_close_indicator",
            "new_cross_bear",
            "bear_signal",
            "bear_sar2",
            "bear_sig",
            "new_bear",
        ]

        price_conditions = [
            ("close", "key_sma", "<"),
            ("close", "sma22", "<"),
        ]

        alignment_conditions = [
            (
                "bear_close_sig",
                lambda frame, idx: idx > 0
                and pd.notna(frame["close"].iat[idx])
                and pd.notna(frame["key_sma"].iat[idx])
                and pd.notna(frame["close"].iat[idx - 1])
                and pd.notna(frame["key_sma"].iat[idx - 1])
                and frame["close"].iat[idx] < frame["key_sma"].iat[idx]
                and frame["close"].iat[idx - 1] > frame["key_sma"].iat[idx - 1],
            ),
            (
                "bear_aligned",
                lambda frame, idx: pd.notna(frame["close"].iat[idx])
                and pd.notna(frame["key_sma"].iat[idx])
                and pd.notna(frame["sma22"].iat[idx])
                and frame["close"].iat[idx] < frame["key_sma"].iat[idx]
                and frame["key_sma"].iat[idx] < frame["sma22"].iat[idx],
            ),
        ]

        for i in range(len(df)):
            total = 0.0

            for condition in conditions:
                if condition not in df.columns:
                    continue
                for j, weight in enumerate(self.historical_weights):
                    idx = i - j
                    if idx < 0:
                        break
                    value = df[condition].iat[idx]
                    if pd.notna(value) and bool(value):
                        total += weight

            for price_col, ma_col, operator in price_conditions:
                if price_col not in df.columns or ma_col not in df.columns:
                    continue
                for j, weight in enumerate(self.historical_weights):
                    idx = i - j
                    if idx < 0:
                        break
                    price_val = df[price_col].iat[idx]
                    ma_val = df[ma_col].iat[idx]
                    if pd.isna(price_val) or pd.isna(ma_val):
                        continue
                    if operator == ">" and price_val > ma_val:
                        total += weight
                    elif operator == "<" and price_val < ma_val:
                        total += weight

            for _, condition_func in alignment_conditions:
                for j, weight in enumerate(self.historical_weights):
                    idx = i - j
                    if idx < 0:
                        break
                    try:
                        if condition_func(df, idx):
                            total += weight
                    except (IndexError, KeyError):
                        continue

            bear_count[i] = total

        return pd.Series(bear_count, index=df.index, name="bear_count")

    def calculate_multi_average_smoothing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create SMA/EMA/RMA/HMA smoothed counts for bull and bear totals."""
        df = df.copy()
        if df.empty:
            return df

        period = 27
        half_window = max(1, period // 2)
        sqrt_window = max(1, int(np.sqrt(period)))

        df["sma_bull_count"] = df["bull_count"].rolling(window=period, min_periods=period).mean()
        df["ema_bull_count"] = df["bull_count"].ewm(span=period, adjust=False).mean()
        df["rma_bull_count"] = df["bull_count"].ewm(alpha=1 / period, adjust=False).mean()

        wma_half = df["bull_count"].rolling(window=half_window, min_periods=half_window).apply(
            _weighted_average, raw=True
        )
        wma_full = df["bull_count"].rolling(window=period, min_periods=period).apply(
            _weighted_average, raw=True
        )
        hma_input = 2 * wma_half - wma_full
        df["hma_bull_count"] = hma_input.rolling(window=sqrt_window, min_periods=sqrt_window).apply(
            _weighted_average, raw=True
        )

        df["sma_bear_count"] = df["bear_count"].rolling(window=period, min_periods=period).mean()
        df["ema_bear_count"] = df["bear_count"].ewm(span=period, adjust=False).mean()
        df["rma_bear_count"] = df["bear_count"].ewm(alpha=1 / period, adjust=False).mean()

        wma_half_bear = df["bear_count"].rolling(window=half_window, min_periods=half_window).apply(
            _weighted_average, raw=True
        )
        wma_full_bear = df["bear_count"].rolling(window=period, min_periods=period).apply(
            _weighted_average, raw=True
        )
        hma_input_bear = 2 * wma_half_bear - wma_full_bear
        df["hma_bear_count"] = hma_input_bear.rolling(window=sqrt_window, min_periods=sqrt_window).apply(
            _weighted_average, raw=True
        )

        df["bully"] = (
            df["sma_bull_count"]
            + df["ema_bull_count"]
            + df["hma_bull_count"]
            + df["rma_bull_count"]
        ) / 4
        df["beary"] = (
            df["sma_bear_count"]
            + df["ema_bear_count"]
            + df["hma_bear_count"]
            + df["rma_bear_count"]
        ) / 4

        return df

    def integrate_rsi(self, df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
        """Compute RSI(14) with overbought/oversold state tracking."""
        df = df.copy()
        if df.empty:
            return df

        delta = df[price_col].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=14, min_periods=14).mean()
        avg_loss = loss.rolling(window=14, min_periods=14).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi_14"] = 100 - (100 / (1 + rs))
        df["rsi_overbought"] = df["rsi_14"] > 70
        df["rsi_oversold"] = df["rsi_14"] < 30
        df["rsi_midline_cross_up"] = (df["rsi_14"] > 51) & (df["rsi_14"].shift(1) <= 51)
        df["rsi_midline_cross_down"] = (df["rsi_14"] < 49) & (df["rsi_14"].shift(1) >= 49)
        return df

    def detect_momentum_changes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify slowing momentum and average cross confirmations."""
        df = df.copy()
        if df.empty:
            return df

        df["bear_slowing"] = (
            (df["beary"] < df["beary"].shift(1))
            & (df["beary"].shift(1) < df["beary"].shift(2))
            & (df["beary"].shift(2) < df["beary"].shift(3))
            & (df["beary"] > df["bully"])
        ).fillna(False)

        df["bull_slowing"] = (
            (df["bully"] < df["bully"].shift(1))
            & (df["bully"].shift(1) < df["bully"].shift(2))
            & (df["bully"].shift(2) < df["bully"].shift(3))
            & (df["bully"] > df["beary"])
        ).fillna(False)

        df["bull_strength_increasing"] = (df["bull_count"] > df["bull_count"].shift(1)).fillna(False)
        df["bear_strength_increasing"] = (df["bear_count"] > df["bear_count"].shift(1)).fillna(False)

        df["bully_over_beary"] = (df["bully"] > df["beary"]).fillna(False)
        df["beary_over_bully"] = (df["beary"] > df["bully"]).fillna(False)

        df["bull_over_average"] = (df["bull_count"] > df["bully"]).fillna(False)
        df["bull_under_average"] = (df["bull_count"] < df["bully"]).fillna(False)
        df["bear_over_average"] = (df["bear_count"] > df["beary"]).fillna(False)
        df["bear_under_average"] = (df["bear_count"] < df["beary"]).fillna(False)

        return df

    def calculate_days_count_tracking(self, df: pd.DataFrame) -> pd.DataFrame:
        """Track consecutive periods counts stay above or below their averages."""
        df = df.copy()
        if df.empty:
            return df

        df["days_bull_count_over_average"] = 0
        df["days_bull_count_under_average"] = 0
        df["days_bear_count_over_average"] = 0
        df["days_bear_count_under_average"] = 0

        bull_over_idx = df.columns.get_loc("days_bull_count_over_average")
        bull_under_idx = df.columns.get_loc("days_bull_count_under_average")
        bear_over_idx = df.columns.get_loc("days_bear_count_over_average")
        bear_under_idx = df.columns.get_loc("days_bear_count_under_average")

        for i in range(1, len(df)):
            bull_count = df["bull_count"].iat[i]
            bull_avg = df["bully"].iat[i]
            prev_bull_count = df["bull_count"].iat[i - 1]
            prev_bull_avg = df["bully"].iat[i - 1]

            bear_count = df["bear_count"].iat[i]
            bear_avg = df["beary"].iat[i]
            prev_bear_count = df["bear_count"].iat[i - 1]
            prev_bear_avg = df["beary"].iat[i - 1]

            if pd.notna(bull_count) and pd.notna(bull_avg) and bull_count > bull_avg:
                if pd.notna(prev_bull_count) and pd.notna(prev_bull_avg) and prev_bull_count > prev_bull_avg:
                    df.iat[i, bull_over_idx] = df.iat[i - 1, bull_over_idx] + 1
                else:
                    df.iat[i, bull_over_idx] = 1

            if pd.notna(bull_count) and pd.notna(bull_avg) and bull_count < bull_avg:
                if pd.notna(prev_bull_count) and pd.notna(prev_bull_avg) and prev_bull_count < prev_bull_avg:
                    df.iat[i, bull_under_idx] = df.iat[i - 1, bull_under_idx] + 1
                else:
                    df.iat[i, bull_under_idx] = 1

            if pd.notna(bear_count) and pd.notna(bear_avg) and bear_count > bear_avg:
                if pd.notna(prev_bear_count) and pd.notna(prev_bear_avg) and prev_bear_count > prev_bear_avg:
                    df.iat[i, bear_over_idx] = df.iat[i - 1, bear_over_idx] + 1
                else:
                    df.iat[i, bear_over_idx] = 1

            if pd.notna(bear_count) and pd.notna(bear_avg) and bear_count < bear_avg:
                if pd.notna(prev_bear_count) and pd.notna(prev_bear_avg) and prev_bear_count < prev_bear_avg:
                    df.iat[i, bear_under_idx] = df.iat[i - 1, bear_under_idx] + 1
                else:
                    df.iat[i, bear_under_idx] = 1

        return df

    def detect_critical_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Monitor 95+ resistance and <12 support streaks."""
        df = df.copy()
        if df.empty:
            return df

        df["bully_at_resistance"] = (df["bully"] >= self.resistance_level).fillna(False)
        df["beary_at_resistance"] = (df["beary"] >= self.resistance_level).fillna(False)
        df["bully_at_support"] = (df["bully"] <= self.support_level).fillna(False)
        df["beary_at_support"] = (df["beary"] <= self.support_level).fillna(False)

        bully_res = np.zeros(len(df), dtype=int)
        beary_res = np.zeros(len(df), dtype=int)
        bully_sup = np.zeros(len(df), dtype=int)
        beary_sup = np.zeros(len(df), dtype=int)

        for i in range(1, len(df)):
            if df["bully_at_resistance"].iat[i]:
                bully_res[i] = bully_res[i - 1] + 1 if df["bully_at_resistance"].iat[i - 1] else 1
            if df["beary_at_resistance"].iat[i]:
                beary_res[i] = beary_res[i - 1] + 1 if df["beary_at_resistance"].iat[i - 1] else 1
            if df["bully_at_support"].iat[i]:
                bully_sup[i] = bully_sup[i - 1] + 1 if df["bully_at_support"].iat[i - 1] else 1
            if df["beary_at_support"].iat[i]:
                beary_sup[i] = beary_sup[i - 1] + 1 if df["beary_at_support"].iat[i - 1] else 1

        df["bully_resistance_periods"] = bully_res
        df["beary_resistance_periods"] = beary_res
        df["bully_support_periods"] = bully_sup
        df["beary_support_periods"] = beary_sup

        df["resistance_reversal_signal"] = (
            (df["bully_at_resistance"].shift(1).fillna(False) & (~df["bully_at_resistance"].fillna(False)))
            | (df["beary_at_resistance"].shift(1).fillna(False) & (~df["beary_at_resistance"].fillna(False)))
        ).fillna(False)
        df["support_reversal_signal"] = (
            (df["bully_at_support"].shift(1).fillna(False) & (~df["bully_at_support"].fillna(False)))
            | (df["beary_at_support"].shift(1).fillna(False) & (~df["beary_at_support"].fillna(False)))
        ).fillna(False)

        return df

    def detect_double_failure_pattern(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect consecutive lower peaks (double failure) for bull/bear counts."""
        df = df.copy()
        if df.empty:
            return df

        df["bully_peak"] = (df["bully"] > df["bully"].shift(1)) & (df["bully"] > df["bully"].shift(-1))
        df["beary_peak"] = (df["beary"] > df["beary"].shift(1)) & (df["beary"] > df["beary"].shift(-1))

        df["bull_double_failure"] = False
        df["bear_double_failure"] = False

        bull_col = df.columns.get_loc("bull_double_failure")
        bear_col = df.columns.get_loc("bear_double_failure")

        bull_peaks: List[float] = []
        bear_peaks: List[float] = []

        for i in range(len(df)):
            if df["bully_peak"].iat[i]:
                bull_peaks.append(df["bully"].iat[i])
                if len(bull_peaks) >= 3 and bull_peaks[-1] < bull_peaks[-2] < bull_peaks[-3]:
                    df.iat[i, bull_col] = True
                elif len(bull_peaks) >= 2 and bull_peaks[-1] < bull_peaks[-2]:
                    df.iat[i, bull_col] = True

            if df["beary_peak"].iat[i]:
                bear_peaks.append(df["beary"].iat[i])
                if len(bear_peaks) >= 3 and bear_peaks[-1] < bear_peaks[-2] < bear_peaks[-3]:
                    df.iat[i, bear_col] = True
                elif len(bear_peaks) >= 2 and bear_peaks[-1] < bear_peaks[-2]:
                    df.iat[i, bear_col] = True

        return df

    def calculate_complete_holloway_algorithm(
        self, df: pd.DataFrame, price_col: str = "close", verbose: bool = False
    ) -> pd.DataFrame:
        """Run the full Holloway feature pipeline."""
        if df.empty:
            logger.warning("Empty dataframe supplied to Holloway calculation.")
            return df.copy()

        printer = print if verbose else (lambda *args, **kwargs: None)

        printer("🚀 Starting Complete Holloway Algorithm calculation...")
        work_df = df.copy()

        printer("  📊 Calculating Parabolic SAR...")
        work_df = self.calculate_parabolic_sar(work_df)

        printer("  🎯 Calculating pattern signals...")
        work_df = self.calculate_pattern_signals(work_df)

        printer("  📈 Calculating bull count with weighted history...")
        work_df["bull_count"] = self.calculate_bull_count(work_df)

        printer("  📉 Calculating bear count with weighted history...")
        work_df["bear_count"] = self.calculate_bear_count(work_df)

        printer("  🌊 Calculating multi-average smoothing...")
        work_df = self.calculate_multi_average_smoothing(work_df)

        printer("  ♻️ Integrating RSI analysis...")
        work_df = self.integrate_rsi(work_df, price_col)

        printer("  ⚡ Detecting momentum changes...")
        work_df = self.detect_momentum_changes(work_df)

        printer("  📆 Tracking regime durations...")
        work_df = self.calculate_days_count_tracking(work_df)

        printer("  📊 Detecting critical levels (95+ / <12)...")
        work_df = self.detect_critical_levels(work_df)

        printer("  ⚠️ Detecting double failure patterns...")
        work_df = self.detect_double_failure_pattern(work_df)

        printer("  🎯 Computing final Holloway signals...")

        work_df["holloway_bull_count"] = work_df["bull_count"]
        work_df["holloway_bear_count"] = work_df["bear_count"]
        work_df["holloway_bull_avg"] = work_df["bully"]
        work_df["holloway_bear_avg"] = work_df["beary"]

        work_df["holloway_count_diff"] = work_df["bull_count"] - work_df["bear_count"]
        work_df["holloway_count_ratio"] = work_df["bull_count"] / (work_df["bear_count"] + 1)
        work_df["holloway_bull_max_20"] = work_df["bull_count"].rolling(window=20, min_periods=1).max()
        work_df["holloway_bull_min_20"] = work_df["bull_count"].rolling(window=20, min_periods=1).min()

        work_df["holloway_bull_cross_up"] = (
            (work_df["holloway_bull_count"] > work_df["holloway_bull_avg"])
            & (work_df["holloway_bull_count"].shift(1) <= work_df["holloway_bull_avg"].shift(1))
        ).fillna(False)
        work_df["holloway_bull_cross_down"] = (
            (work_df["holloway_bull_count"] < work_df["holloway_bull_avg"])
            & (work_df["holloway_bull_count"].shift(1) >= work_df["holloway_bull_avg"].shift(1))
        ).fillna(False)
        work_df["holloway_bear_cross_up"] = (
            (work_df["holloway_bear_count"] > work_df["holloway_bear_avg"])
            & (work_df["holloway_bear_count"].shift(1) <= work_df["holloway_bear_avg"].shift(1))
        ).fillna(False)
        work_df["holloway_bear_cross_down"] = (
            (work_df["holloway_bear_count"] < work_df["holloway_bear_avg"])
            & (work_df["holloway_bear_count"].shift(1) >= work_df["holloway_bear_avg"].shift(1))
        ).fillna(False)

        work_df["bull_rise_signal"] = work_df["holloway_bull_cross_up"]
        work_df["bear_rise_signal"] = work_df["holloway_bear_cross_up"]

        work_df["holloway_bull_signal"] = work_df["holloway_bull_cross_up"] & ~work_df["rsi_overbought"]
        work_df["holloway_bear_signal"] = work_df["holloway_bear_cross_up"] & ~work_df["rsi_oversold"]

        work_df["bull_strength_signal"] = (
            work_df["bull_over_average"] & work_df["bull_strength_increasing"] & ~work_df["bully_at_resistance"]
        )
        work_df["bear_strength_signal"] = (
            work_df["bear_over_average"] & work_df["bear_strength_increasing"] & ~work_df["beary_at_resistance"]
        )

        work_df["reversal_signal"] = (
            work_df["resistance_reversal_signal"]
            | work_df["support_reversal_signal"]
            | work_df["bull_double_failure"]
            | work_df["bear_double_failure"]
        )

        work_df["weakness_signal"] = work_df["bull_slowing"] | work_df["bear_slowing"]

        work_df["holloway_bull_signal_raw"] = work_df["bull_rise_signal"]
        work_df["holloway_bear_signal_raw"] = work_df["bear_rise_signal"]
        work_df["holloway_days_bull_over_avg"] = work_df["days_bull_count_over_average"]
        work_df["holloway_days_bull_under_avg"] = work_df["days_bull_count_under_average"]
        work_df["holloway_days_bear_over_avg"] = work_df["days_bear_count_over_average"]
        work_df["holloway_days_bear_under_avg"] = work_df["days_bear_count_under_average"]

        printer("✅ Complete Holloway Algorithm calculation finished!")
        return work_df

    def calculate_complete_holloway_system(
        self, df: pd.DataFrame, price_col: str = "close", verbose: bool = False
    ) -> pd.DataFrame:
        """Backward compatible alias for external scripts."""
        return self.calculate_complete_holloway_algorithm(df, price_col=price_col, verbose=verbose)

    def get_holloway_summary(self, df: pd.DataFrame) -> Dict:
        """Return summary statistics for the latest Holloway run."""
        if "bull_count" not in df.columns:
            return {"error": "Holloway Algorithm not calculated yet"}

        latest = df.iloc[-1]

        summary = {
            "current_state": {
                "bull_count": float(latest["bull_count"]),
                "bear_count": float(latest["bear_count"]),
                "bully": float(latest["bully"]),
                "beary": float(latest["beary"]),
                "trend_direction": "BULLISH" if latest["bully"] > latest["beary"] else "BEARISH",
            },
            "signals": {
                "holloway_bull_signals": int(df["holloway_bull_signal"].sum()),
                "holloway_bear_signals": int(df["holloway_bear_signal"].sum()),
                "strength_signals": int(df["bull_strength_signal"].sum() + df["bear_strength_signal"].sum()),
                "reversal_signals": int(df["reversal_signal"].sum()),
                "weakness_signals": int(df["weakness_signal"].sum()),
            },
            "critical_levels": {
                "at_resistance": bool(latest["bully_at_resistance"] or latest["beary_at_resistance"]),
                "at_support": bool(latest["bully_at_support"] or latest["beary_at_support"]),
                "resistance_periods": int(latest["bully_resistance_periods"] + latest["beary_resistance_periods"]),
                "support_periods": int(latest["bully_support_periods"] + latest["beary_support_periods"]),
            },
            "momentum": {
                "bull_slowing": bool(latest["bull_slowing"]),
                "bear_slowing": bool(latest["bear_slowing"]),
                "bull_over_beary": bool(latest["bully_over_beary"]),
                "double_failure": bool(latest["bull_double_failure"] or latest["bear_double_failure"]),
            },
            "statistics": {
                "total_periods": len(df),
                "avg_bull_count": float(df["bull_count"].mean()),
                "avg_bear_count": float(df["bear_count"].mean()),
                "max_bull_count": float(df["bull_count"].max()),
                "max_bear_count": float(df["bear_count"].max()),
            },
        }

        return summary

    def save_holloway_results(self, df: pd.DataFrame, pair: str, timeframe: str) -> str:
        """Persist Holloway output columns to CSV for further analysis."""
        filename = f"{pair}_{timeframe}_complete_holloway.csv"
        filepath = os.path.join(self.data_dir, filename)

        output_columns = [
            "open",
            "high",
            "low",
            "close",
            "bull_count",
            "bear_count",
            "bully",
            "beary",
            "holloway_bull_signal",
            "holloway_bear_signal",
            "bull_strength_signal",
            "bear_strength_signal",
            "reversal_signal",
            "weakness_signal",
            "bull_slowing",
            "bear_slowing",
            "bully_over_beary",
            "bully_at_resistance",
            "beary_at_resistance",
            "bully_at_support",
            "beary_at_support",
            "bull_double_failure",
            "bear_double_failure",
            "holloway_bull_signal_raw",
            "holloway_bear_signal_raw",
            "holloway_days_bull_over_avg",
            "holloway_days_bull_under_avg",
            "holloway_days_bear_over_avg",
            "holloway_days_bear_under_avg",
        ]

        available_columns = [col for col in output_columns if col in df.columns]
        df_to_save = df[available_columns].copy()

        # Add provenance: source filename used (if present as an attribute)
        provenance_file = getattr(df, '_source_file', None)
        if provenance_file:
            df_to_save['source_file'] = provenance_file

        # If index is datetime-like, reset it to a 'date' column for portability
        if isinstance(df_to_save.index, pd.DatetimeIndex):
            df_to_save = df_to_save.reset_index()
            # rename index column to date if not already
            if df_to_save.columns[0].lower() != 'date':
                df_to_save.rename(columns={df_to_save.columns[0]: 'date'}, inplace=True)

        df_to_save.to_csv(filepath, index=False)

        print(f"✅ Holloway results saved: {filepath}")
        return filepath

    def save_results(self, df: pd.DataFrame, pair: str, timeframe: str) -> str:
        """Alias for legacy scripts."""
        return self.save_holloway_results(df, pair, timeframe)


def load_data_file(pair: str, timeframe: str = "daily", data_dir: str = "data") -> pd.DataFrame:
    """Load OHLC data for the specified pair/timeframe."""
    # Normalize search terms
    pair_lower = pair.lower()
    timeframe_lower = timeframe.lower()

    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return pd.DataFrame()

    # List files and look for case-insensitive matches containing both pair and timeframe
    candidates: List[str] = []
    for fname in os.listdir(data_dir):
        if not fname.lower().endswith('.csv'):
            continue
        fl = fname.lower()
        if pair_lower in fl and timeframe_lower in fl:
            candidates.append(fname)

    # If no exact timeframe match, accept any file with the pair name (e.g. EURUSD_Daily vs EURUSD_daily)
    if not candidates:
        for fname in os.listdir(data_dir):
            if not fname.lower().endswith('.csv'):
                continue
            if pair_lower in fname.lower():
                candidates.append(fname)

    if not candidates:
        print(f"❌ Could not find data for {pair} {timeframe} in {data_dir}")
        return pd.DataFrame()

    # Prefer candidates that explicitly contain timeframe (case-insensitive)
    candidates = sorted(candidates, key=lambda x: (timeframe_lower in x.lower(), x), reverse=True)

    for name in candidates:
        filepath = os.path.join(data_dir, name)
        try:
            # Try a few separator options to handle comma, tab, semicolon, or whitespace-separated files
            read_attempts = [
                {"sep": ","},
                {"sep": "\t"},
                {"sep": ";"},
                {"sep": None, "engine": "python"},
            ]

            df = None
            for opts in read_attempts:
                try:
                    df = pd.read_csv(filepath, **opts)
                    # if it parsed into a single column with tabs preserved, try next
                    if df.shape[1] == 1 and df.iloc[0].astype(str).str.contains('\t').any():
                        continue
                    break
                except Exception:
                    df = None
                    continue

            if df is None:
                print(f"❌ Error parsing {filepath} with known separators")
                continue

            # Normalize column names: strip non-alphanumeric (e.g. '<' '>'), trim and lowercase
            cleaned = []
            for c in df.columns:
                if not isinstance(c, str):
                    c = str(c)
                s = re.sub(r"[^0-9a-zA-Z_]", "", c).lower()
                cleaned.append(s)
            df.columns = cleaned

            # If a date-like column exists (e.g. date, timestamp) set it as index
            if 'date' in df.columns:
                try:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                except Exception:
                    pass

            # At this point index may be set; ensure it's datetime if possible
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception:
                    pass

            required = {"open", "high", "low", "close"}
            if not required.issubset(set(df.columns)):
                print(f"⚠️ Missing required OHLC columns in {filepath}")
                continue

            # attach provenance file name on the dataframe for downstream saving
            try:
                df._source_file = name
            except Exception:
                pass

            print(f"✅ Loaded {pair} {timeframe} from {name}: {len(df)} records")
            return df
        except Exception as exc:
            print(f"❌ Error loading {filepath}: {exc}")

    print(f"❌ Could not load any candidate files for {pair} {timeframe}")
    return pd.DataFrame()


def run_complete_holloway_analysis() -> Dict[str, Dict[str, Dict]]:
    """Standalone runner used for manual diagnostics."""
    print("🚀 STARTING COMPLETE HOLLOWAY ALGORITHM ANALYSIS")
    print("=" * 70)

    holloway = CompleteHollowayAlgorithm()

    pairs = ["EURUSD", "XAUUSD"]
    # Supported timeframe suffixes in data/ (do not resample, keep as-is)
    timeframes = ["daily", "h4", "h1", "weekly", "monthly"]
    results: Dict[str, Dict[str, Dict]] = {}

    for pair in pairs:
        results[pair] = {}
        # For each pair compute Holloway for all available timeframes and merge latest features
        pair_features = {}
        for timeframe in timeframes:
            print(f"\n📊 Processing {pair} {timeframe}...")
            df = load_data_file(pair, timeframe)
            if df.empty:
                print(f"⚠️ No data found for {pair} {timeframe}")
                continue
            # attach provenance and compute
            # load_data_file does not currently set a field, so pass filename via attribute if known
            # but load_data_file returns df without file info; attempt to infer from candidate names
            df_holloway = holloway.calculate_complete_holloway_algorithm(df, verbose=False)
            # attempt to attach provenance if load_data_file set _source_file
            provenance = getattr(df, '_source_file', None)
            if provenance is not None:
                df_holloway._source_file = provenance

            filepath = holloway.save_holloway_results(df_holloway, pair, timeframe)
            summary = holloway.get_holloway_summary(df_holloway)

            results[pair][timeframe] = {
                "summary": summary,
                "filepath": filepath,
                "data_points": len(df_holloway),
            }

            # Print compact per-timeframe summary
            if "current_state" in summary:
                current = summary["current_state"]
                signals = summary["signals"]
                print(
                    f"  {timeframe.upper()}: {len(df_holloway)} periods, Trend: {current['trend_direction']}, "
                    f"Bull Signals: {signals['holloway_bull_signals']}, Bear Signals: {signals['holloway_bear_signals']}"
                )

            # store latest row features (if any) with suffix
            if len(df_holloway) > 0:
                latest = df_holloway.iloc[-1].to_dict()
                # include provenance for this timeframe
                if hasattr(df_holloway, '_source_file') and df_holloway._source_file:
                    latest['source_file'] = df_holloway._source_file

                suffixed = {f"{k}_{timeframe}": v for k, v in latest.items()}
                pair_features[timeframe] = suffixed

        # Merge latest-per-timeframe features into a single dataframe and save
        if pair_features:
            merged = {}
            for tf, feat in pair_features.items():
                merged.update(feat)

            merged_df = pd.DataFrame([merged])
            merged_path = os.path.join(holloway.data_dir, f"{pair}_latest_multi_timeframe_features.csv")
            merged_df.to_csv(merged_path, index=False)
            print(f"\n✅ Saved merged latest features for {pair}: {merged_path}")
            results[pair]["latest_features"] = {"filepath": merged_path, "rows": len(merged_df)}

    print("\n🎉 COMPLETE HOLLOWAY ANALYSIS FINISHED!")
    print("=" * 70)
    return results


if __name__ == "__main__":
    analysis_results = run_complete_holloway_analysis()

    print("\n📋 FINAL SUMMARY:")
    for pair, timeframes in analysis_results.items():
        print(f"\n{pair}:")
        for timeframe, result in timeframes.items():
            summary = result.get("summary", {})
            current = summary.get("current_state", {})
            signals = summary.get("signals", {})
            print(
                f"  {timeframe}: {result.get('data_points', 0)} periods, "
                f"Trend: {current.get('trend_direction', 'N/A')}, "
                f"Signals: {signals.get('holloway_bull_signals', 0)} bull / "
                f"{signals.get('holloway_bear_signals', 0)} bear"
            )
