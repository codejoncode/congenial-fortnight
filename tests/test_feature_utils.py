import pandas as pd
import numpy as np

from scripts.feature_utils import prune_features


def test_prune_na_and_zero_variance():
    df = pd.DataFrame({
        'good': [1, 2, 3, 4, 5],
        'mostly_nan': [None, None, None, 1, None],
        'constant': [1, 1, 1, 1, 1],
    })

    pruned, report = prune_features(df, na_pct_threshold=0.6, drop_zero_variance=True)
    # 'mostly_nan' has 80% NA -> should be dropped
    assert 'mostly_nan' in report['dropped_na_pct']
    # 'constant' has zero variance -> should be dropped
    assert 'constant' in report['dropped_zero_variance']
    # 'good' should remain
    assert 'good' in pruned.columns
    assert report['n_final_cols'] == 1
