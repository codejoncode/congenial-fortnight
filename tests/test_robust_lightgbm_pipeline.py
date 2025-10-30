import pandas as pd
import numpy as np
from scripts.robust_lightgbm_config import enhanced_lightgbm_training_pipeline


def make_synthetic_dataset(n=200):
    np.random.seed(42)
    X = pd.DataFrame({f'f{i}': np.random.randn(n) for i in range(5)})
    # simple binary target with some signal
    y = (X['f0'] + X['f1'] * 0.5 + np.random.randn(n) * 0.1 > 0).astype(int)
    df = X.copy()
    df['target'] = y
    return df


def test_enhanced_pipeline_runs():
    df = make_synthetic_dataset(300)
    datasets = {'Daily': df}
    models = enhanced_lightgbm_training_pipeline(datasets, target_column='target')
    assert isinstance(models, dict)
