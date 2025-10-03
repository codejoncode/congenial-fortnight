import pandas as pd
import numpy as np
from forecasting_integration import add_holloway_features_safe


def make_fake_price_df(n=200):
    # create a synthetic time series of prices with realistic-ish noise
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n, freq='D')
    prices = 100 + np.cumsum(np.random.normal(scale=0.5, size=n))
    high = prices + np.abs(np.random.normal(scale=0.2, size=n))
    low = prices - np.abs(np.random.normal(scale=0.2, size=n))
    open_ = prices + np.random.normal(scale=0.1, size=n)
    close = prices
    vol = np.random.randint(100, 1000, size=n)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': vol,
    })
    return df


def test_add_holloway_features_basic():
    df = make_fake_price_df(250)
    out = add_holloway_features_safe(df, 'TEST')

    # Ensure returned dataframe is not None and has expected columns
    assert out is not None
    assert len(out) > 0
    for col in ['rv_20', 'rv_50', 'skew_20', 'kurt_20', 'mom_10']:
        assert col in out.columns, f"Missing engineered column: {col}"

    # rv_ratio should be finite and positive where present
    if 'rv_ratio' in out.columns:
        vals = out['rv_ratio'].dropna()
        assert (vals > 0).all()
