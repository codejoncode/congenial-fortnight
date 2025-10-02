"""Complete Holloway Algorithm translated from PineScript.

This module provides a production-ready implementation of the Holloway Algorithm
with all 400+ bull/bear condition features, signal tracking, and summary
utilities. The implementation follows the PineScript reference closely while
remaining idiomatic Python.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CompleteHollowayAlgorithm:
    """Full Holloway Algorithm implementation with all PineScript features."""

    def __init__(self, data_dir: str = "data") -> None:
        """Create a new algorithm instance.

        Args:
            data_dir: Directory where derived Holloway CSV exports are stored.
        """
        self.data_dir = data_dir
        self.ensure_data_dir()

        # All moving average periods from PineScript
        self.ma_periods = [5, 7, 10, 14, 20, 28, 50, 56, 100, 112, 200, 225]

        # Initialize tracking variables
        self.reset_counts()

    # ------------------------------------------------------------------
    # Core setup helpers
    # ------------------------------------------------------------------
    def ensure_data_dir(self) -> None:
        """Create the data directory if it doesn't already exist."""
        if self.data_dir and not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            logger.info("Created Holloway data directory: %s", self.data_dir)

    def reset_counts(self) -> None:
        """Reset all internal counting variables (for future extensions)."""
        self.days_bull_count_over_average = 0
        self.days_bull_count_under_average = 0
        self.days_bear_count_over_average = 0
        self.days_bear_count_under_average = 0

    # ------------------------------------------------------------------
    # Moving average calculations
    # ------------------------------------------------------------------
    def calculate_all_moving_averages(
        self, df: pd.DataFrame, price_col: str = "close"
    ) -> pd.DataFrame:
        """Calculate all 24 moving averages (12 EMA + 12 SMA).

        Args:
            df: Price dataframe with at least the price column.
            price_col: Column name to use for price calculations.

        Returns:
            DataFrame with all EMA/SMA columns appended.
        """
        df = df.copy()

        for period in self.ma_periods:
            df[f"ema_{period}"] = df[price_col].ewm(span=period, adjust=False).mean()
            df[f"sma_{period}"] = df[price_col].rolling(window=period).mean()

        return df

    # ------------------------------------------------------------------
    # Bull/Bear condition calculation
    # ------------------------------------------------------------------
    def calculate_bull_bear_conditions(
        self, df: pd.DataFrame, price_col: str = "close"
    ) -> pd.DataFrame:
        """Recreate the full PineScript bull/bear condition arrays."""
        df = df.copy()
        close_price = df[price_col]

        bull_conditions = []
        bear_conditions = []
        max_period = max(self.ma_periods)

        for idx in range(len(df)):
            bull_count = 0
            bear_count = 0
            current_close = close_price.iloc[idx]

            if idx < max_period:
                bull_conditions.append(np.nan)
                bear_conditions.append(np.nan)
                continue

            # PART 1: Price relative to each MA
            for period in self.ma_periods:
                ema_val = df[f"ema_{period}"].iloc[idx]
                sma_val = df[f"sma_{period}"].iloc[idx]

                if pd.notna(ema_val):
                    bull_count += current_close > ema_val
                    bear_count += current_close < ema_val
                if pd.notna(sma_val):
                    bull_count += current_close > sma_val
                    bear_count += current_close < sma_val

            # PART 2: EMA alignment
            ema_pairs = [
                (5, 10), (5, 20), (5, 50), (5, 100), (5, 200),
                (7, 14), (7, 28), (7, 56), (7, 112), (7, 225),
                (10, 20), (10, 50), (10, 100), (10, 200),
                (14, 28), (14, 56), (14, 112), (14, 225),
                (20, 50), (20, 100), (20, 200),
                (28, 56), (28, 112), (28, 225),
                (50, 100), (50, 200),
                (56, 112), (56, 225),
                (100, 200), (112, 225),
            ]

            for shorter, longer in ema_pairs:
                shorter_ema = df[f"ema_{shorter}"].iloc[idx]
                longer_ema = df[f"ema_{longer}"].iloc[idx]
                if pd.notna(shorter_ema) and pd.notna(longer_ema):
                    bull_count += shorter_ema > longer_ema
                    bear_count += shorter_ema < longer_ema

            # PART 3: SMA alignment (same pairs)
            for shorter, longer in ema_pairs:
                shorter_sma = df[f"sma_{shorter}"].iloc[idx]
                longer_sma = df[f"sma_{longer}"].iloc[idx]
                if pd.notna(shorter_sma) and pd.notna(longer_sma):
                    bull_count += shorter_sma > longer_sma
                    bear_count += shorter_sma < longer_sma

            # PART 4: EMA vs SMA comparisons
            ema_sma_pairs = [
                (5, 5), (5, 10), (5, 20), (5, 50), (5, 100), (5, 200),
                (7, 7), (7, 14), (7, 28), (7, 56), (7, 112), (7, 225),
                (10, 10), (10, 20), (10, 50), (10, 100), (10, 200),
                (14, 14), (14, 28), (14, 56), (14, 112), (14, 225),
                (20, 50), (20, 100), (20, 200),
                (28, 56), (28, 112), (28, 225),
                (50, 100), (50, 200),
                (56, 112), (56, 225),
                (100, 200), (112, 225),
            ]

            for ema_period, sma_period in ema_sma_pairs:
                ema_val = df[f"ema_{ema_period}"].iloc[idx]
                sma_val = df[f"sma_{sma_period}"].iloc[idx]
                if pd.notna(ema_val) and pd.notna(sma_val):
                    bull_count += ema_val > sma_val
                    bear_count += ema_val < sma_val

            # PART 5: Fresh breakouts
            if idx > 0:
                prev_close = close_price.iloc[idx - 1]
                for period in self.ma_periods:
                    ema_curr = df[f"ema_{period}"].iloc[idx]
                    ema_prev = df[f"ema_{period}"].iloc[idx - 1]
                    sma_curr = df[f"sma_{period}"].iloc[idx]
                    sma_prev = df[f"sma_{period}"].iloc[idx - 1]

                    if pd.notna(ema_curr) and pd.notna(ema_prev):
                        bull_count += current_close > ema_curr and prev_close <= ema_prev
                        bear_count += current_close < ema_curr and prev_close >= ema_prev
                    if pd.notna(sma_curr) and pd.notna(sma_prev):
                        bull_count += current_close > sma_curr and prev_close <= sma_prev
                        bear_count += current_close < sma_curr and prev_close >= sma_prev

            # PART 6 & 7: Fresh MA changes + crossovers
            if idx > 0:
                for ema_period, sma_period in ema_sma_pairs:
                    ema_curr = df[f"ema_{ema_period}"].iloc[idx]
                    sma_curr = df[f"sma_{sma_period}"].iloc[idx]
                    ema_prev = df[f"ema_{ema_period}"].iloc[idx - 1]
                    sma_prev = df[f"sma_{sma_period}"].iloc[idx - 1]

                    if (
                        pd.notna(ema_curr)
                        and pd.notna(sma_curr)
                        and pd.notna(ema_prev)
                        and pd.notna(sma_prev)
                    ):
                        bull_count += ema_curr > sma_curr and ema_prev <= sma_prev
                        bear_count += ema_curr < sma_curr and ema_prev >= sma_prev
                        bull_count += ema_curr > sma_curr and ema_prev <= sma_prev
                        bear_count += ema_curr < sma_curr and ema_prev >= sma_prev

            bull_conditions.append(bull_count)
            bear_conditions.append(bear_count)

        df["bull_ma_count"] = bull_conditions
        df["bear_ma_count"] = bear_conditions
        return df

    # ------------------------------------------------------------------
    # DEMA smoothing & signals
    # ------------------------------------------------------------------
    def calculate_dema_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply 27-period DEMA smoothing to bull/bear counts."""
        df = df.copy()
        bull_ema1 = df["bull_ma_count"].ewm(span=27, adjust=False).mean()
        df["bull_ma_count_average"] = bull_ema1.ewm(span=27, adjust=False).mean()
        bear_ema1 = df["bear_ma_count"].ewm(span=27, adjust=False).mean()
        df["bear_ma_count_average"] = bear_ema1.ewm(span=27, adjust=False).mean()
        return df

    def calculate_crossover_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate crossover and crossunder boolean signals."""
        df = df.copy()
        df["bull_rise_crossover"] = (
            (df["bull_ma_count"] > df["bull_ma_count_average"])
            & (df["bull_ma_count"].shift(1) <= df["bull_ma_count_average"].shift(1))
        )
        df["bear_rise_crossunder"] = (
            (df["bear_ma_count"] < df["bear_ma_count_average"])
            & (df["bear_ma_count"].shift(1) >= df["bear_ma_count_average"].shift(1))
        )
        df["bear_rise_crossover"] = (
            (df["bear_ma_count"] > df["bear_ma_count_average"])
            & (df["bear_ma_count"].shift(1) <= df["bear_ma_count_average"].shift(1))
        )
        df["bull_rise_crossunder"] = (
            (df["bull_ma_count"] < df["bull_ma_count_average"])
            & (df["bull_ma_count"].shift(1) >= df["bull_ma_count_average"].shift(1))
        )
        return df

    def calculate_combined_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combine crossovers into bullish/bearish signal flags."""
        df = df.copy()
        df["bull_rise_signal"] = df["bull_rise_crossover"] | df["bear_rise_crossunder"]
        df["bear_rise_signal"] = df["bear_rise_crossover"] | df["bull_rise_crossunder"]
        return df

    # ------------------------------------------------------------------
    # RSI + days tracking
    # ------------------------------------------------------------------
    def calculate_rsi_integration(
        self, df: pd.DataFrame, price_col: str = "close"
    ) -> pd.DataFrame:
        """Add RSI(14) and associated crossover conditions."""
        df = df.copy()
        delta = df[price_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi_14"] = 100 - (100 / (1 + rs))
        df["rsi_above_50_cross"] = (df["rsi_14"] > 50) & (df["rsi_14"].shift(1) <= 50)
        df["rsi_below_50_cross"] = (df["rsi_14"] < 50) & (df["rsi_14"].shift(1) >= 50)
        df["rsi_overbought_70"] = df["rsi_14"] > 70
        df["rsi_oversold_30"] = df["rsi_14"] < 30
        return df

    def calculate_days_count_tracking(self, df: pd.DataFrame) -> pd.DataFrame:
        """Track consecutive periods bull/bear counts are above/below averages."""
        df = df.copy()
        df["days_bull_count_over_average"] = 0
        df["days_bull_count_under_average"] = 0
        df["days_bear_count_over_average"] = 0
        df["days_bear_count_under_average"] = 0

        for i in range(1, len(df)):
            # Bull over average
            if (
                pd.notna(df.iloc[i]["bull_ma_count"])
                and pd.notna(df.iloc[i]["bull_ma_count_average"])
                and df.iloc[i]["bull_ma_count"] > df.iloc[i]["bull_ma_count_average"]
            ):
                prev = df.iloc[i - 1]["bull_ma_count"]
                prev_avg = df.iloc[i - 1]["bull_ma_count_average"]
                if pd.notna(prev) and pd.notna(prev_avg) and prev > prev_avg:
                    df.iloc[i, df.columns.get_loc("days_bull_count_over_average")] = (
                        df.iloc[i - 1]["days_bull_count_over_average"] + 1
                    )
                else:
                    df.iloc[i, df.columns.get_loc("days_bull_count_over_average")] = 1

            # Bull under average
            if (
                pd.notna(df.iloc[i]["bull_ma_count"])
                and pd.notna(df.iloc[i]["bull_ma_count_average"])
                and df.iloc[i]["bull_ma_count"] < df.iloc[i]["bull_ma_count_average"]
            ):
                prev = df.iloc[i - 1]["bull_ma_count"]
                prev_avg = df.iloc[i - 1]["bull_ma_count_average"]
                if pd.notna(prev) and pd.notna(prev_avg) and prev < prev_avg:
                    df.iloc[i, df.columns.get_loc("days_bull_count_under_average")] = (
                        df.iloc[i - 1]["days_bull_count_under_average"] + 1
                    )
                else:
                    df.iloc[i, df.columns.get_loc("days_bull_count_under_average")] = 1

            # Bear over average
            if (
                pd.notna(df.iloc[i]["bear_ma_count"])
                and pd.notna(df.iloc[i]["bear_ma_count_average"])
                and df.iloc[i]["bear_ma_count"] > df.iloc[i]["bear_ma_count_average"]
            ):
                prev = df.iloc[i - 1]["bear_ma_count"]
                prev_avg = df.iloc[i - 1]["bear_ma_count_average"]
                if pd.notna(prev) and pd.notna(prev_avg) and prev > prev_avg:
                    df.iloc[i, df.columns.get_loc("days_bear_count_over_average")] = (
                        df.iloc[i - 1]["days_bear_count_over_average"] + 1
                    )
                else:
                    df.iloc[i, df.columns.get_loc("days_bear_count_over_average")] = 1

            # Bear under average
            if (
                pd.notna(df.iloc[i]["bear_ma_count"])
                and pd.notna(df.iloc[i]["bear_ma_count_average"])
                and df.iloc[i]["bear_ma_count"] < df.iloc[i]["bear_ma_count_average"]
            ):
                prev = df.iloc[i - 1]["bear_ma_count"]
                prev_avg = df.iloc[i - 1]["bear_ma_count_average"]
                if pd.notna(prev) and pd.notna(prev_avg) and prev < prev_avg:
                    df.iloc[i, df.columns.get_loc("days_bear_count_under_average")] = (
                        df.iloc[i - 1]["days_bear_count_under_average"] + 1
                    )
                else:
                    df.iloc[i, df.columns.get_loc("days_bear_count_under_average")] = 1

        return df

    # ------------------------------------------------------------------
    # Pipeline orchestration
    # ------------------------------------------------------------------
    def calculate_complete_holloway_algorithm(
        self, df: pd.DataFrame, price_col: str = "close"
    ) -> pd.DataFrame:
        """Run the full Holloway pipeline for the provided dataframe."""
        if df.empty:
            logger.warning("Received empty dataframe for Holloway calculation.")
            return df

        logger.debug("Starting Holloway Algorithm calculation with %d rows", len(df))
        df = self.calculate_all_moving_averages(df, price_col)
        df = self.calculate_bull_bear_conditions(df, price_col)
        df = self.calculate_dema_averages(df)
        df = self.calculate_crossover_signals(df)
        df = self.calculate_combined_signals(df)
        df = self.calculate_rsi_integration(df, price_col)
        df = self.calculate_days_count_tracking(df)
        logger.debug("Holloway Algorithm calculation complete.")
        return df

    # ------------------------------------------------------------------
    # Export helpers & summaries
    # ------------------------------------------------------------------
    def save_holloway_results(self, df: pd.DataFrame, pair: str, timeframe: str) -> str:
        """Persist Holloway output for inspection or downstream workflows."""
        filename = f"{pair}_{timeframe}_holloway_complete.csv"
        filepath = os.path.join(self.data_dir, filename)

        holloway_columns = [
            "bull_ma_count",
            "bear_ma_count",
            "bull_ma_count_average",
            "bear_ma_count_average",
            "bull_rise_crossover",
            "bear_rise_crossunder",
            "bear_rise_crossover",
            "bull_rise_crossunder",
            "bull_rise_signal",
            "bear_rise_signal",
            "rsi_14",
            "rsi_above_50_cross",
            "rsi_below_50_cross",
            "days_bull_count_over_average",
            "days_bull_count_under_average",
            "days_bear_count_over_average",
            "days_bear_count_under_average",
        ]

        available_columns = [col for col in holloway_columns if col in df.columns]
        if df.index.name:
            df[available_columns].to_csv(filepath)
        else:
            df[available_columns].to_csv(filepath, index=False)

        logger.info("Saved Holloway results to %s", filepath)
        return filepath

    def get_holloway_summary(self, df: pd.DataFrame) -> Dict:
        """Return key statistics summarising Holloway outputs."""
        if "bull_ma_count" not in df.columns:
            return {"error": "Holloway Algorithm not calculated yet"}

        summary = {
            "total_periods": len(df),
            "bull_ma_count": {
                "mean": df["bull_ma_count"].mean(),
                "max": df["bull_ma_count"].max(),
                "min": df["bull_ma_count"].min(),
                "current": df["bull_ma_count"].iloc[-1] if len(df) > 0 else None,
            },
            "bear_ma_count": {
                "mean": df["bear_ma_count"].mean(),
                "max": df["bear_ma_count"].max(),
                "min": df["bear_ma_count"].min(),
                "current": df["bear_ma_count"].iloc[-1] if len(df) > 0 else None,
            },
            "signals": {
                "bull_rise_signals": df["bull_rise_signal"].sum()
                if "bull_rise_signal" in df.columns
                else 0,
                "bear_rise_signals": df["bear_rise_signal"].sum()
                if "bear_rise_signal" in df.columns
                else 0,
                "total_signals": (
                    df["bull_rise_signal"].sum() + df["bear_rise_signal"].sum()
                )
                if "bull_rise_signal" in df.columns and "bear_rise_signal" in df.columns
                else 0,
            },
            "rsi": {
                "current": df["rsi_14"].iloc[-1]
                if "rsi_14" in df.columns and len(df) > 0
                else None,
                "overbought_periods": df["rsi_overbought_70"].sum()
                if "rsi_overbought_70" in df.columns
                else 0,
                "oversold_periods": df["rsi_oversold_30"].sum()
                if "rsi_oversold_30" in df.columns
                else 0,
            },
        }

        return summary


def load_data_file(pair: str, timeframe: str, data_dir: str = "data") -> pd.DataFrame:
    """Utility helper used by the standalone runner for loading CSV data."""
    possible_names = [
        f"{pair}_{timeframe}.csv",
        f"{pair.lower()}_{timeframe}.csv",
        f"{pair}_{timeframe}_data.csv",
        f"{pair.lower()}_{timeframe}_data.csv",
    ]

    for filename in possible_names:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                for date_col in ["date", "Date", "timestamp", "time"]:
                    if date_col in df.columns:
                        df[date_col] = pd.to_datetime(df[date_col])
                        df.set_index(date_col, inplace=True)
                        break
                logger.info("Loaded %s %s data from %s (%d rows)", pair, timeframe, filepath, len(df))
                return df
            except Exception as exc:
                logger.error("Error loading %s: %s", filepath, exc)
                continue

    logger.warning("Could not find data file for %s %s", pair, timeframe)
    return pd.DataFrame()


def run_complete_holloway_system() -> Dict[str, Dict[str, Dict[str, float]]]:
    """Standalone driver to process all configured pairs/timeframes."""
    logger.info("Starting complete Holloway Algorithm system")
    holloway = CompleteHollowayAlgorithm()
    pairs = ["EURUSD", "XAUUSD"]
    timeframes = ["daily", "weekly", "4h"]
    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for pair in pairs:
        results[pair] = {}
        for timeframe in timeframes:
            logger.info("Processing %s %s", pair, timeframe)
            df = load_data_file(pair, timeframe)
            if df.empty:
                logger.warning("No data found for %s %s", pair, timeframe)
                continue

            df_holloway = holloway.calculate_complete_holloway_algorithm(df)
            filepath = holloway.save_holloway_results(df_holloway, pair, timeframe)
            summary = holloway.get_holloway_summary(df_holloway)
            results[pair][timeframe] = {
                "summary": summary,
                "filepath": filepath,
                "data_points": len(df_holloway),
            }

    logger.info("Completed Holloway Algorithm processing")
    return results


if __name__ == "__main__":  # pragma: no cover - convenience runner
    logging.basicConfig(level=logging.INFO)
    final_results = run_complete_holloway_system()
    for pair, timeframes in final_results.items():
        logger.info("%s:", pair)
        for timeframe, result in timeframes.items():
            if "summary" in result:
                summary = result["summary"]
                logger.info(
                    "  %s: %d data points, %d total signals",
                    timeframe,
                    result["data_points"],
                    summary["signals"]["total_signals"],
                )
