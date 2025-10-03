import os
import json
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def generate_schema_report(df, pair: str, pairs: List[str] = None, out_dir: str = None) -> Dict:
    """Generate an expanded schema report dict and optionally save it as JSON.

    Returns the report dict. If out_dir is provided, writes file to
    {out_dir}/schema_report_{pair}.json
    """
    try:
        import numpy as _np

        report = {}
        if df is None or not hasattr(df, 'shape'):
            report = {
                'pair': pair,
                'rows': 0,
                'cols': 0,
                'fund_cols': 0,
                'cross_pair_cols': 0,
                'sample_columns': [],
                'na_counts': {},
                'na_pct': {},
                'dtypes': {},
                'variance': {},
                'zero_counts': {},
                'basic_stats': {},
            }
        else:
            cols = list(df.columns)
            na_counts = df.isnull().sum().to_dict()
            na_pct = {k: (v / max(1, int(df.shape[0]))) for k, v in na_counts.items()}
            dtypes = {c: str(df[c].dtype) for c in cols}

            variance = {}
            zero_counts = {}
            basic_stats = {}
            for c in cols:
                try:
                    series = df[c]
                    if _np.issubdtype(series.dtype, _np.number):
                        variance[c] = float(series.var(skipna=True)) if series.size > 0 else None
                        zero_counts[c] = int((series == 0).sum())
                        basic_stats[c] = {
                            'min': None if series.size == 0 else float(series.min(skipna=True)),
                            'max': None if series.size == 0 else float(series.max(skipna=True)),
                            'mean': None if series.size == 0 else float(series.mean(skipna=True)),
                        }
                    else:
                        variance[c] = None
                        zero_counts[c] = None
                        basic_stats[c] = None
                except Exception:
                    variance[c] = None
                    zero_counts[c] = None
                    basic_stats[c] = None

            fund_cols = [c for c in cols if c.startswith('fund_')]
            cross_pair_keys = [p.lower() for p in (pairs or ['EURUSD', 'XAUUSD'])]
            cross_cols = [c for c in cols if any(k in c.lower() for k in cross_pair_keys) and not c.startswith('fund_')]

            report = {
                'pair': pair,
                'rows': int(df.shape[0]),
                'cols': len(cols),
                'fund_cols': len(fund_cols),
                'cross_pair_cols': len(cross_cols),
                'sample_columns': cols[:50],
                'na_counts': {k: int(v) for k, v in na_counts.items()},
                'na_pct': na_pct,
                'dtypes': dtypes,
                'variance': variance,
                'zero_counts': zero_counts,
                'basic_stats': basic_stats,
            }

        if out_dir:
            try:
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f'schema_report_{pair}.json')
                with open(out_path, 'w') as f:
                    json.dump(report, f, indent=2)
                logger.info(f"Saved schema report to {out_path}")
            except Exception as e:
                logger.debug(f"Failed to save schema report: {e}")

        return report
    except Exception as e:
        logger.debug(f"generate_schema_report failed: {e}")
        return {}


def save_prune_report(report: Dict, pair: str, out_dir: str = None) -> str:
    """Save a prune report dict to disk and return the path. Returns empty string on failure."""
    try:
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f'prune_report_{pair}.json')
            with open(out_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Saved prune report to {out_path}")
            return out_path
    except Exception as e:
        logger.debug(f"Failed to save prune report: {e}")
    return ''
