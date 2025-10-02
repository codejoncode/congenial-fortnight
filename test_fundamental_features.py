import numpy as np
import pandas as pd

from scripts.fundamental_features import FundamentalFeatureEngineer


def _build_sample_dataframe(rows: int = 420) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="D")
    ramp = np.linspace(0, 1, rows)

    return pd.DataFrame(
        {
            "FEDFUNDS": 5 + 0.5 * ramp,
            "ECBDFR": 3 + 0.2 * ramp,
            "DGS10": 4 + 0.3 * ramp,
            "DGS2": 3.5 + 0.1 * ramp,
            "CPIAUCSL": 250 + 2 * ramp,
            "CP0000EZ19M086NEST": 220 + 1.5 * ramp,
            "CPALTT01USM661S": 240 + 1.7 * ramp,
            "UNRATE": 4 + 0.2 * ramp,
            "LRHUTTTTDEM156S": 5 - 0.1 * ramp,
            "PAYEMS": 150_000 + 500 * np.arange(rows),
            "INDPRO": 110 + 0.5 * ramp,
            "DCOILWTICO": 70 + np.sin(np.linspace(0, 6, rows)),
            "DCOILBRENTEU": 72 + np.sin(np.linspace(0, 6, rows) + 0.1),
            "GOLDAMGBD228NLBM": 1800 + 10 * ramp,
            "VIXCLS": 18 + np.cos(np.linspace(0, 3, rows)),
            "DEXUSEU": 0.92 + 0.01 * np.sin(np.linspace(0, 12, rows)),
            "DEXJPUS": 110 + 2 * np.cos(np.linspace(0, 12, rows)),
            "DEXCHUS": 6.8 + 0.05 * np.sin(np.linspace(0, 6, rows)),
            "BOPGSTB": -500 + 5 * np.sin(np.linspace(0, 2, rows)),
        },
        index=index,
    )


def test_fundamental_feature_engineer_adds_expected_columns():
    df = _build_sample_dataframe()
    engineer = FundamentalFeatureEngineer(data_dir="data")

    enhanced = engineer.enhance(df)

    expected_columns = {
        "macro_us_eu_rate_spread",
        "macro_us_yield_curve",
        "macro_inflation_spread",
        "macro_unemployment_spread",
        "macro_oil_momentum_30",
        "macro_gold_momentum_30",
        "macro_trade_balance_zscore",
        "macro_usd_cross_divergence",
    }

    missing = expected_columns - set(enhanced.columns)
    assert not missing, f"Missing engineered columns: {missing}"

    for column in expected_columns:
        series = enhanced[column]
        assert not series.isna().all(), f"Column {column} should contain non-NaN values"
        assert np.isfinite(series.iloc[-1]), f"Column {column} should be finite for latest observation"
