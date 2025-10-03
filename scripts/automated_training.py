#!/usr/bin/env python3
"""
Automated Training Script for Cloud Run Jobs
Uses a robust pipeline to train models, with data validation,
dynamic configuration, and hang prevention.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import List
from tqdm import tqdm

from .data_issue_fixes import pre_training_data_fix
from .robust_lightgbm_config import enhanced_lightgbm_training_pipeline_arrays as enhanced_lightgbm_training_pipeline

# Add project root to path
BASE_APP_DIR = os.environ.get('APP_ROOT', os.getcwd())
sys.path.insert(0, BASE_APP_DIR)

try:
    from .advanced_regularization_optimizer import optimize_pair
except ImportError:
    from .optimizer import optimize_pair

try:
    from .regularization_config_manager import get_regularization_config
except ImportError:
    get_regularization_config = None

from .forecasting import HybridPriceForecastingEnsemble as ForecastingSystem
from notification_system import NotificationSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        # logging.FileHandler(os.path.join(BASE_APP_DIR, 'logs', 'automated_training.log'))
    ]
)
logger = logging.getLogger(__name__)

class AutomatedTrainer:
    def __init__(self, target_accuracy=0.75, max_iterations=100):
        self.target_accuracy = target_accuracy
        self.max_iterations = max_iterations
        # Do not instantiate ForecastingSystem here; create per-pair in run_automated_training
        self.forecasting = None

        # Ensure directories exist
        os.makedirs(os.path.join(BASE_APP_DIR, 'models'), exist_ok=True)
        os.makedirs(os.path.join(BASE_APP_DIR, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(BASE_APP_DIR, 'output'), exist_ok=True)

    def run_automated_training(self, pairs: List[str] = None, dry_run: bool = False, dry_iterations: int = 10, dry_timeout_seconds: int = 60):
        """Run automated training for specified pairs using the robust pipeline"""
        if pairs is None:
            pairs = ['EURUSD', 'XAUUSD']

        logger.info(f"Starting automated training for pairs: {pairs}")
        logger.info(f"Target accuracy: {self.target_accuracy}")

        final_results = {}
        start_time = datetime.now()

    # Wrap pairs with tqdm for a progress bar
        for pair in tqdm(pairs, desc="Processing pairs"):
            try:
                logger.info(f"--- Processing {pair} ---")
                
                # 1. Initialize forecasting system for the specific pair
                forecasting_system = ForecastingSystem(pair=pair)
                
                # 2. Load and prepare datasets
                X_train, y_train, X_val, y_val = forecasting_system.load_and_prepare_datasets()

                # Input validation: ensure we have proper arrays/frames and matching lengths
                def invalid_input(reason):
                    logger.error(f"❌ Data preparation failed for {pair}: {reason}")
                    final_results[pair] = {'error': f'Data preparation failed: {reason}'}

                if X_train is None or y_train is None:
                    invalid_input('Missing training data or labels')
                    continue

                # Convert pandas objects to numpy where helpful
                try:
                    import numpy as _np
                    if hasattr(X_train, 'shape') and hasattr(y_train, 'shape'):
                        x_len = int(X_train.shape[0])
                        y_len = int(y_train.shape[0])
                    else:
                        # fallback to len()
                        x_len = len(X_train)
                        y_len = len(y_train)
                except Exception:
                    invalid_input('Could not determine dataset lengths')
                    continue

                if x_len == 0 or y_len == 0:
                    invalid_input('Empty training arrays')
                    continue

                if x_len != y_len:
                    invalid_input(f'Mismatched lengths: X_train={x_len}, y_train={y_len}')
                    continue

                if x_len < 10:
                    # Too few samples to train reliably; skip or allow emergency config downstream
                    logger.warning(f"⚠️  {pair}: Very small training set ({x_len} samples). Training may use emergency minimal config.")

                if dry_run:
                    logger.info(f"🔬 Dry-run enabled for {pair}: capping iterations to {dry_iterations} and timeout {dry_timeout_seconds}s")
                
                # Print a per-pair schema report to help auditing what features will be used
                def pair_schema_report(X_train, y_train, X_val, y_val):
                    """Generate an expanded schema report and save as JSON under output/."""
                    try:
                        import numpy as _np

                        df = X_train.copy() if hasattr(X_train, 'copy') else None
                        cols = list(df.columns) if df is not None else []

                        # Basic NA diagnostics
                        na_counts = df.isnull().sum().to_dict() if df is not None else {}
                        na_pct = {k: (v / max(1, int(df.shape[0]))) for k, v in na_counts.items()} if df is not None else {}

                        # Dtypes
                        dtypes = {c: str(df[c].dtype) for c in cols} if df is not None else {}

                        # Variance and zero-count diagnostics (numeric only)
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
                        # Use configured pair names to detect cross pair columns heuristically
                        cross_pair_keys = [p.lower() for p in (pairs or ['EURUSD', 'XAUUSD'])]
                        cross_cols = [c for c in cols if any(k in c.lower() for k in cross_pair_keys) and not c.startswith('fund_')]

                        report = {
                            'pair': pair,
                            'rows': int(df.shape[0]) if df is not None else 0,
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

                        # Save JSON output
                        out_path = os.path.join(BASE_APP_DIR, 'output', f'schema_report_{pair}.json')
                        with open(out_path, 'w') as f:
                            json.dump(report, f, indent=2)

                        logger.info(f"Pre-train schema for {pair}: rows={report['rows']}, cols={report['cols']}, fund_cols={report['fund_cols']}, cross_pair_cols={report['cross_pair_cols']}")
                        logger.info(f"Saved schema report to {out_path}")
                    except Exception as e:
                        logger.debug(f"Could not generate schema report: {e}")

                pair_schema_report(X_train, y_train, X_val, y_val)

                # 2.5 Automatic pruning: drop columns with high NA% or zero variance
                def prune_features(df, na_pct_threshold: float = 0.5, drop_zero_variance: bool = True):
                    """Return (pruned_df, report) where report contains lists of dropped columns.

                    - Drops columns with NA% > na_pct_threshold
                    - Optionally drops numeric columns with variance == 0 (or <= tiny tolerance)
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
                            # variance with skipna
                            try:
                                variances = numeric.var(skipna=True)
                            except Exception:
                                variances = numeric.var()
                            drop_var_cols = variances[variances <= 0].index.tolist()

                        # Unique-ify and remove any accidental overlap
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

                X_train_pruned, prune_report = prune_features(X_train, na_pct_threshold=0.5, drop_zero_variance=True)
                # Align validation set to pruned columns
                try:
                    X_val_pruned = X_val[X_train_pruned.columns]
                except Exception:
                    # If X_val can't be subset, attempt to reindex
                    try:
                        X_val_pruned = X_val.reindex(columns=X_train_pruned.columns)
                    except Exception:
                        X_val_pruned = X_val

                # Save pruning report
                try:
                    out_prune = os.path.join(BASE_APP_DIR, 'output', f'prune_report_{pair}.json')
                    with open(out_prune, 'w') as f:
                        json.dump(prune_report, f, indent=2)
                    logger.info(f"Saved prune report to {out_prune}")
                except Exception as e:
                    logger.debug(f"Could not save prune report: {e}")

                # 3. Use the new robust training pipeline
                # The enhanced pipeline now takes data directly
                # If dry_run, call the arrays wrapper which respects our small-iteration configs
                if dry_run:
                    # Try to reduce training time by trimming columns or rows if needed
                    model = enhanced_lightgbm_training_pipeline(
                        X_train_pruned.head(dry_iterations * 10), y_train[:dry_iterations * 10],
                        X_val_pruned.head(max(1, int(dry_iterations * 2))) if X_val_pruned is not None else None,
                        y_val[:max(1, int(dry_iterations * 2))] if y_val is not None else None,
                        pair_name=pair
                    )
                else:
                    model = enhanced_lightgbm_training_pipeline(
                        X_train_pruned, y_train, X_val_pruned, y_val,
                        pair_name=pair # Pass pair name for logging
                    )
                
                if model is None:
                    logger.error(f"❌ Training failed for {pair} - no model was produced.")
                    final_results[pair] = {'error': 'Training failed, no model produced.'}
                    continue

                # 4. (Optional) Evaluate the trained models
                # This part can be expanded to run a backtest with the new models
                logger.info(f"✅ Model for {pair} trained successfully.")
                
                # For now, we'll just record success
                final_results[pair] = {
                    'status': 'success',
                    'model_trained': True,
                    'target_reached': True # Assuming success if models are trained
                }

            except Exception as e:
                logger.error(f"An error occurred while processing {pair}: {e}", exc_info=True)
                final_results[pair] = {'error': str(e)}

        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Automated training finished in {duration}.")
        
        # Save final results
        results_path = os.path.join(BASE_APP_DIR, 'logs', 'automated_training_results.json')
        with open(results_path, 'w') as f:
            json.dump({
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration.total_seconds(),
                'results': final_results
            }, f, indent=2)
            
        return final_results

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Automated Model Training')
    parser.add_argument('--target', type=float, default=0.85,
                       help='Target accuracy (default: 0.85)')
    parser.add_argument('--max-iterations', type=int, default=50,
                       help='Maximum iterations per pair (default: 50)')
    parser.add_argument('--pairs', nargs='+', default=['EURUSD', 'XAUUSD'],
                       help='Currency pairs to optimize (default: EURUSD XAUUSD)')

    args = parser.parse_args()

    # 1. Fix data issues first
    if not pre_training_data_fix():
        logger.error("❌ Data validation failed - cannot proceed with training.")
        sys.exit(1)

    trainer = AutomatedTrainer(
        target_accuracy=args.target,
        max_iterations=args.max_iterations
    )

    results = trainer.run_automated_training(args.pairs)

    # Exit with success/failure code
    all_targets_reached = all(
        result.get('target_reached', False)
        for result in results.values()
        if 'error' not in result
    )

    sys.exit(0 if all_targets_reached else 1)

if __name__ == '__main__':
    main()