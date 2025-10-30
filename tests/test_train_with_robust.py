import pandas as pd
import numpy as np
from scripts.robust_lightgbm_config import train_with_robust_error_handling, create_robust_lgb_config_for_small_data


def make_dataset(n, constant=False):
    rng = np.random.RandomState(42)
    if constant:
        X = pd.DataFrame({f'f{i}': np.zeros(n) for i in range(5)})
    else:
        X = pd.DataFrame({f'f{i}': rng.randn(n) for i in range(5)})
    y = (X['f0'] + X['f1'] * 0.5 + rng.randn(n) * 0.1 > 0).astype(int)
    return X, y


def test_tiny_dataset_returns_none():
    X, y = make_dataset(30)
    model = train_with_robust_error_handling(X, y, create_robust_lgb_config_for_small_data(), timeframe_name='tiny')
    assert model is None


def test_constant_features_returns_none():
    X, y = make_dataset(200, constant=True)
    model = train_with_robust_error_handling(X, y, create_robust_lgb_config_for_small_data(), timeframe_name='constant')
    assert model is None


def test_healthy_dataset_trains():
    X, y = make_dataset(400)
    model = train_with_robust_error_handling(X, y, create_robust_lgb_config_for_small_data(), timeframe_name='healthy')
    assert model is not None
    assert hasattr(model, 'num_trees')
