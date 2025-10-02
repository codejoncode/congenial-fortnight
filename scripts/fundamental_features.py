#!/usr/bin/env python3
"""Derived fundamental feature engineering utilities.

This module consolidates macroeconomic time series into daily, model-ready
features. It provides helpers for rate spreads, momentum metrics, and
positioning proxies used by both forecasting and signal generation pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class FundamentalFeatureConfig:
    resample_frequency: Optional[str] = "D"
    zscore_window: int = 60
    momentum_windows: tuple[int, ...] = (5, 10, 21, 63)
    yoy_window: int = 252  # Approximate trading days in a year


class FundamentalFeatureEngineer:
    """Engineer derived macro features for downstream models."""

    def __init__(self, data_dir: str | Path = "data", config: Optional[FundamentalFeatureConfig] = None):
        self.data_dir = Path(data_dir)
        self.config = config or FundamentalFeatureConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def enhance(self, fundamental_df: pd.DataFrame) -> pd.DataFrame:
        """Return the input dataframe augmented with derived macro features."""
        if fundamental_df is None or fundamental_df.empty:
            return fundamental_df

        df = fundamental_df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        df = df.sort_index()
        if self.config.resample_frequency:
            df = df.resample(self.config.resample_frequency).ffill()

        derived: Dict[str, pd.Series] = {}
        derived.update(self._rate_spreads(df))
        derived.update(self._inflation_features(df))
        derived.update(self._employment_growth_features(df))
        derived.update(self._volatility_oil_gold(df))
        derived.update(self._trade_balance_features(df))
        derived.update(self._currency_strength_features(df))
        derived.update(self._cftc_position_features(df))

        for name, series in derived.items():
            if series is None:
                continue
            series = series.ffill()
            series = series.fillna(0.0)
            df[name] = series

        return df.ffill()

    # ------------------------------------------------------------------
    # Feature blocks
    # ------------------------------------------------------------------
    def _rate_spreads(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        fed = df.get('FEDFUNDS')
        ecb = df.get('ECBDFR')
        dgs10 = df.get('DGS10')
        dgs2 = df.get('DGS2')
        dexuseu = df.get('DEXUSEU')
        vix = df.get('VIXCLS')

        features: Dict[str, pd.Series] = {}
        if fed is not None and ecb is not None:
            features['macro_us_eu_rate_spread'] = fed - ecb
            features['macro_rate_spread_delta_30'] = (fed - ecb) - (fed - ecb).shift(30)

        if dgs10 is not None and dgs2 is not None:
            features['macro_us_yield_curve'] = dgs10 - dgs2
            features['macro_yield_curve_slope_change'] = (dgs10 - dgs2).diff()

        if dexuseu is not None:
            features['macro_dollar_trend_21'] = dexuseu.pct_change(21)

        if vix is not None:
            features['macro_vix_zscore'] = self._zscore(vix, self.config.zscore_window)

        return features

    def _inflation_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        cpi_us = df.get('CPIAUCSL')
        cpi_eu = df.get('CP0000EZ19M086NEST')
        core_cpi = df.get('CPALTT01USM661S')

        features: Dict[str, pd.Series] = {}
        if cpi_us is not None and cpi_eu is not None:
            features['macro_inflation_spread'] = cpi_us.pct_change(12) - cpi_eu.pct_change(12)

        if cpi_us is not None:
            features['macro_inflation_yoy_us'] = cpi_us.pct_change(12)

        if cpi_eu is not None:
            features['macro_inflation_yoy_eu'] = cpi_eu.pct_change(12)

        if core_cpi is not None:
            features['macro_core_inflation_trend'] = core_cpi.pct_change(12) - core_cpi.pct_change(24)

        return features

    def _employment_growth_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        unemployment_us = df.get('UNRATE')
        unemployment_eu = df.get('LRHUTTTTDEM156S')
        payrolls = df.get('PAYEMS')
        indpro = df.get('INDPRO')

        features: Dict[str, pd.Series] = {}
        if unemployment_us is not None and unemployment_eu is not None:
            features['macro_unemployment_spread'] = unemployment_us - unemployment_eu

        if payrolls is not None:
            features['macro_payrolls_mom'] = payrolls.pct_change()
            features['macro_payrolls_trend_3m'] = payrolls.pct_change(3)

        if indpro is not None:
            features['macro_industrial_trend_6m'] = indpro.pct_change(6)

        return features

    def _volatility_oil_gold(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        oil_wti = df.get('DCOILWTICO')
        oil_brent = df.get('DCOILBRENTEU')
        gold = df.get('GOLDAMGBD228NLBM')
        vix = df.get('VIXCLS')

        features: Dict[str, pd.Series] = {}
        if oil_wti is not None:
            features['macro_oil_momentum_30'] = oil_wti.pct_change(30)

        if oil_brent is not None:
            features['macro_brent_wti_spread'] = oil_brent - oil_wti if oil_wti is not None else oil_brent

        if gold is not None:
            features['macro_gold_momentum_30'] = gold.pct_change(30)

        if vix is not None:
            features['macro_vix_trend_10'] = vix.pct_change(10)

        return features

    def _trade_balance_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        trade = df.get('BOPGSTB')
        features: Dict[str, pd.Series] = {}

        if trade is not None:
            features['macro_trade_balance_zscore'] = self._zscore(trade, 12)
            features['macro_trade_balance_momentum'] = trade.pct_change(3)

        return features

    def _currency_strength_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        usd_jpy = df.get('DEXJPUS')
        usd_cny = df.get('DEXCHUS')
        usd_eur = df.get('DEXUSEU')

        features: Dict[str, pd.Series] = {}
        if usd_eur is not None and usd_jpy is not None:
            features['macro_usd_cross_divergence'] = usd_eur.pct_change(21) - usd_jpy.pct_change(21)

        if usd_eur is not None and usd_cny is not None:
            features['macro_usd_asia_divergence'] = usd_eur.pct_change(21) - usd_cny.pct_change(21)

        return features

    def _cftc_position_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        cftc = self._load_cftc_positions()
        if cftc.empty:
            return {}

        merged = df.join(cftc, how='left')
        merged = merged.ffill()

        features: Dict[str, pd.Series] = {}
        if 'cftc_eur_net' in merged.columns:
            features['macro_cftc_eur_zscore'] = self._zscore(merged['cftc_eur_net'], 12)
            features['macro_cftc_eur_change_4w'] = merged['cftc_eur_net'].diff(28)

        if 'cftc_gold_net' in merged.columns:
            features['macro_cftc_gold_zscore'] = self._zscore(merged['cftc_gold_net'], 12)
            features['macro_cftc_gold_change_4w'] = merged['cftc_gold_net'].diff(28)

        return features

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _zscore(self, series: pd.Series, window: int) -> pd.Series:
        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std()
        return (series - rolling_mean) / (rolling_std + 1e-9)

    def _load_cftc_positions(self) -> pd.DataFrame:
        """Attempt to load cached CFTC net positioning data if available."""
        possible_files = list(self.data_dir.glob("CFTC_*_positions.csv"))
        if not possible_files:
            return pd.DataFrame()

        frames = []
        for file in possible_files:
            df = pd.read_csv(file)
            if 'date' not in df.columns:
                continue
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            frames.append(df)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, axis=1, join='outer')
        combined = combined.sort_index()
        return combined