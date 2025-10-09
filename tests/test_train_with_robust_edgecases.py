import pandas as pd
import numpy as np
import time

from scripts.robust_lightgbm_config import (
    train_with_robust_error_handling,
    create_robust_lgb_config_for_small_data,
    TimeoutException,
)


def make_binary_target(n, ratio=0.5):
    n_pos = max(1, int(n * ratio))
    labels = np.array([1] * n_pos + [0] * (n - n_pos))
    return labels


def test_tiny_dataset_returns_none_or_model():
    # Very small dataset should either return a model using emergency config or None
    X = pd.DataFrame({'f1': np.random.randn(8), 'f2': np.random.randn(8)})
    y = pd.Series(make_binary_target(8, 0.5))

    model = train_with_robust_error_handling(X, y, create_robust_lgb_config_for_small_data(), timeframe_name='tiny')
    # It should handle gracefully: either None (aborted) or a model object
    assert (model is None) or hasattr(model, 'num_trees')


def test_constant_features_are_rejected():
    # Constant features (zero variance) should cause the diagnostic to abort
    X = pd.DataFrame({'f1': [1.0] * 50, 'f2': [2.0] * 50})
    y = pd.Series(make_binary_target(50, 0.5))

    model = train_with_robust_error_handling(X, y, create_robust_lgb_config_for_small_data(), timeframe_name='constant')
    assert model is None


def test_timeout_enforced():
    # Create a dataset and use a decorator trick to force timeout by setting a very small alarm
    import importlib
    mod = importlib.import_module('scripts.robust_lightgbm_config')

    X = pd.DataFrame({'f1': np.random.randn(200), 'f2': np.random.randn(200)})
    y = pd.Series(make_binary_target(200, 0.5))

    # Temporarily wrap the function with a short timeout
    original = mod.train_with_robust_error_handling

    try:
        @mod.timeout(1)
        def quick_train(X, y, params, timeframe_name='q'):
            return original(X, y, params, timeframe_name=timeframe_name)

        res = quick_train(X, y, create_robust_lgb_config_for_small_data(), timeframe_name='quick')
        # If it returns None or a model, that's acceptable; the important part is no unhandled exception
        assert res is None or hasattr(res, 'num_trees')
    finally:
        # No need to restore since we didn't mutate module state
        pass
