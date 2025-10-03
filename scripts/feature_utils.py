"""Utility helpers for feature engineering and pruning"""
import os
import json
import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)


def prune_features(df, na_pct_threshold: float = 0.5, drop_zero_variance: bool = True) -> Tuple[object, Dict]:
    """Return (pruned_df, report) where report contains lists of dropped columns.

    - Drops columns with NA% > na_pct_threshold
    - Optionally drops numeric columns with variance == 0
    """
    try:
        import numpy as _np

        report = {
            'dropped_na_pct': [],
            'dropped_zero_variance': [],
            'initial_cols': list(df.columns) if hasattr(df, 'columns') else [],
            'final_cols': None,
        }

        if df is None or not hasattr(df, 'shape'):
            report['final_cols'] = report['initial_cols']
            return df, report

        n = max(1, int(df.shape[0]))
        na_frac = df.isnull().sum() / n
        drop_na_cols = na_frac[na_frac > na_pct_threshold].index.tolist()

        drop_var_cols = []
        if drop_zero_variance:
            numeric = df.select_dtypes(include=[_np.number])
            try:
                variances = numeric.var(skipna=True)
            except Exception:
                variances = numeric.var()
            drop_var_cols = variances[variances <= 0].index.tolist()

        to_drop = list(dict.fromkeys(drop_na_cols + drop_var_cols))

        pruned = df.drop(columns=to_drop, errors='ignore')

        report['dropped_na_pct'] = drop_na_cols
        report['dropped_zero_variance'] = drop_var_cols
        report['final_cols'] = list(pruned.columns)
        report['n_initial_cols'] = len(report['initial_cols'])
        report['n_final_cols'] = len(report['final_cols'])

        return pruned, report
    except Exception as e:
        logger.debug(f"prune_features failed: {e}")
        return df, {'error': str(e)}
