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
from pathlib import Path
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
from .feature_utils import prune_features

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

def ensure_environment_loaded():
    """Ensure .env is loaded and FRED key is available"""
    try:
        from dotenv import load_dotenv
    except ImportError:
        logger.warning("python-dotenv not available, skipping .env loading")
        return False

    env_paths = [
        Path(".env"),
        Path("../.env"),
        Path(os.getcwd()) / ".env",
        Path(os.environ.get('APP_ROOT', os.getcwd())) / ".env"
    ]

    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path, override=True)
            logger.info(f"Loaded environment from: {env_path}")
            break

    fred_key = os.getenv('FRED_API_KEY')
    if fred_key:
        logger.info(f"✅ FRED API key loaded: {fred_key[:8]}...")
        return True
    else:
        logger.warning("❌ FRED_API_KEY not found in environment")
        return False

class AutomatedTrainer:
    def train_until_target(self, pairs=None, max_attempts=10):
        """
        Repeatedly train models for each pair, deleting models before each run, until target accuracy is reached or max_attempts is hit.
        Logs best results for each pair.
        """
        import shutil
        if pairs is None:
            pairs = ['EURUSD', 'XAUUSD']
        best_results = {pair: {'validation_accuracy': 0, 'test_accuracy': 0} for pair in pairs}
        for attempt in range(1, max_attempts + 1):
            logger.info(f"\n=== Training Attempt {attempt}/{max_attempts} ===")
            # Delete models before each run
            models_dir = os.path.join(BASE_APP_DIR, 'models')
            if os.path.exists(models_dir):
                for f in os.listdir(models_dir):
                    file_path = os.path.join(models_dir, f)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        logger.warning(f"Could not delete model file {file_path}: {e}")
            # Run training
            results = self.run_automated_training(pairs=pairs)
            all_targets_reached = True
            for pair in pairs:
                result = results.get(pair, {})
                val_acc = result.get('validation_accuracy', 0)
                test_acc = result.get('test_accuracy', 0)
                if val_acc > best_results[pair]['validation_accuracy']:
                    best_results[pair]['validation_accuracy'] = val_acc
                if test_acc > best_results[pair]['test_accuracy']:
                    best_results[pair]['test_accuracy'] = test_acc
                if not result.get('target_reached', False):
                    all_targets_reached = False
            if all_targets_reached:
                logger.info(f"🎯 Target accuracy reached for all pairs on attempt {attempt}!")
                break
        logger.info("\n=== Best Results Across All Attempts ===")
        for pair in pairs:
            logger.info(f"{pair}: Best Validation Accuracy: {best_results[pair]['validation_accuracy']:.4f}, Best Test Accuracy: {best_results[pair]['test_accuracy']:.4f}")
        return best_results
    def __init__(self, target_accuracy=0.75, max_iterations=100):
        self.target_accuracy = target_accuracy
        self.max_iterations = max_iterations
        # Do not instantiate ForecastingSystem here; create per-pair in run_automated_training
        self.forecasting = None

        # Ensure directories exist
        os.makedirs(os.path.join(BASE_APP_DIR, 'models'), exist_ok=True)
        os.makedirs(os.path.join(BASE_APP_DIR, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(BASE_APP_DIR, 'output'), exist_ok=True)
    
    def walk_forward_cv(self, X, y, date_col, window_size=365, step_size=30, min_train=365):
        """
        Walk-forward cross-validation for time-series.
        Args:
            X: DataFrame of features (must include date_col)
            y: Series/array of targets
            date_col: str, name of date column in X
            window_size: int, number of days in test window
            step_size: int, number of days to move window forward each fold
            min_train: int, minimum number of days for training set
        Returns:
            List of (val_acc, test_acc) for each fold
        """
        import numpy as np
        from sklearn.metrics import accuracy_score
        results = []
        # Ensure data is sorted by date
        X = X.sort_values(date_col).reset_index(drop=True)
        y = y.loc[X.index]
        n = len(X)
        fold = 0
        for start in range(min_train, n - window_size, step_size):
            train_end = start
            valid_end = start + window_size // 2
            test_end = start + window_size
            if test_end > n:
                break
            X_train, X_val, X_test = X.iloc[:train_end], X.iloc[train_end:valid_end], X.iloc[valid_end:test_end]
            y_train, y_val, y_test = y.iloc[:train_end], y.iloc[train_end:valid_end], y.iloc[valid_end:test_end]
            if len(X_test) == 0 or len(X_val) == 0 or len(X_train) < min_train:
                continue
            # Train model (use your pipeline)
            model = enhanced_lightgbm_training_pipeline(X_train, y_train, X_val, y_val)
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)
            val_acc = accuracy_score(y_val, (val_pred > 0.5).astype(int))
            test_acc = accuracy_score(y_test, (test_pred > 0.5).astype(int))
            results.append((val_acc, test_acc))
            print(f"Fold {fold}: Val Acc={val_acc:.3f}, Test Acc={test_acc:.3f}")
            fold += 1
        if results:
            val_accs, test_accs = zip(*results)
            print(f"\nWalk-forward CV Results: {fold} folds")
            print(f"Mean Validation Accuracy: {np.mean(val_accs):.3f} ± {np.std(val_accs):.3f}")
            print(f"Mean Test Accuracy: {np.mean(test_accs):.3f} ± {np.std(test_accs):.3f}")
        else:
            print("No valid walk-forward folds found.")
        return results

    def run_automated_training(self, pairs: List[str] = None, dry_run: bool = False, dry_iterations: int = 10, dry_timeout_seconds: int = 60):
        """Run automated training for specified pairs using robust time-series splits and model saving."""
        import joblib
        if pairs is None:
            pairs = ['EURUSD', 'XAUUSD']

        logger.info(f"Starting automated training for pairs: {pairs}")
        logger.info(f"Target accuracy: {self.target_accuracy}")

        final_results = {}
        start_time = datetime.now()


        for pair in tqdm(pairs, desc="Processing pairs"):
            try:
                logger.info(f"--- Processing {pair} ---")
                forecasting_system = ForecastingSystem(pair=pair)
                # --- Feature Engineering ---
                feature_df = None
                if hasattr(forecasting_system, 'load_and_prepare_datasets'):
                    loaded = forecasting_system.load_and_prepare_datasets()
                    # Handle (X, y) or (X_train, y_train, X_val, y_val, X_test, y_test)
                    if isinstance(loaded, tuple) and len(loaded) == 6:
                        X_train, y_train, X_val, y_val, X_test, y_test = loaded
                    elif isinstance(loaded, tuple) and len(loaded) == 4:
                        X_train, y_train, X_val_full, y_val_full = loaded
                        n_val = len(X_val_full)
                        if n_val < 2:
                            raise ValueError("Not enough samples in validation set to split into val/test.")
                        split = n_val // 2
                        X_val, X_test = X_val_full.iloc[:split], X_val_full.iloc[split:]
                        y_val, y_test = y_val_full.iloc[:split], y_val_full.iloc[split:]
                    elif isinstance(loaded, tuple) and len(loaded) == 2:
                        X, y = loaded
                        n = len(X)
                        train_end = int(0.70 * n)
                        valid_end = int(0.85 * n)
                        X_train, X_val, X_test = X.iloc[:train_end], X.iloc[train_end:valid_end], X.iloc[valid_end:]
                        y_train, y_val, y_test = y.iloc[:train_end], y.iloc[train_end:valid_end], y.iloc[valid_end:]
                    else:
                        raise ValueError("load_and_prepare_datasets() returned unexpected format.")
                    # Try to get the feature_df for evaluation
                    if hasattr(forecasting_system, '_prepare_features'):
                        feature_df = forecasting_system._prepare_features()
                else:
                    feature_df = forecasting_system._prepare_features()
                    if feature_df is None or feature_df.empty:
                        error_msg = f"Feature engineering produced empty dataframe for {pair}. Cannot proceed with training."
                        logger.error(f"❌ {error_msg}")
                        raise ValueError(error_msg)
                    target_col = 'target_1d'
                    if target_col not in feature_df.columns:
                        target_col = next((c for c in feature_df.columns if c.startswith('target_')), None)
                    if not target_col:
                        error_msg = f"No target column found in features for {pair}. Available columns: {list(feature_df.columns)}"
                        logger.error(f"❌ {error_msg}")
                        raise ValueError(error_msg)
                    y = feature_df[target_col]
                    X = feature_df.drop(columns=[c for c in feature_df.columns if 'target' in c or 'next_close_change' in c], errors='ignore')
                    n = len(X)
                    train_end = int(0.70 * n)
                    valid_end = int(0.85 * n)
                    X_train, X_val, X_test = X.iloc[:train_end], X.iloc[train_end:valid_end], X.iloc[valid_end:]
                    y_train, y_val, y_test = y.iloc[:train_end], y.iloc[train_end:valid_end], y.iloc[valid_end:]

                # --- Model Training ---
                model = enhanced_lightgbm_training_pipeline(X_train, y_train, X_val, y_val, pair_name=pair)
                if model is None:
                    logger.error(f"❌ Training failed for {pair} - no model was produced.")
                    final_results[pair] = {'error': 'Training failed, no model produced.'}
                    continue

                # --- Save Model ---
                model_path = os.path.join(BASE_APP_DIR, 'models', f'{pair}_model.txt')
                try:
                    model.save_model(model_path)
                    logger.info(f"✅ Saved model for {pair} to {model_path}")
                except Exception as e:
                    logger.error(f"❌ Failed to save model for {pair}: {e}")

                # --- Evaluation ---
                from sklearn.metrics import accuracy_score
                val_pred = model.predict(X_val)
                val_pred = (val_pred > 0.5).astype(int)
                val_acc = accuracy_score(y_val, val_pred)
                logger.info(f"📊 {pair} Validation Accuracy: {val_acc:.4f} ({val_acc*100:.1f}%)")

                test_pred = model.predict(X_test)
                test_pred = (test_pred > 0.5).astype(int)
                test_acc = accuracy_score(y_test, test_pred)
                logger.info(f"📊 {pair} Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
                print(f"▶️ Validation Accuracy: {val_acc:.2%}")
                print(f"▶️ Test       Accuracy: {test_acc:.2%}")

                # --- Per-signal/feature evaluation ---
                try:
                    if feature_df is not None and hasattr(forecasting_system, 'evaluate_signal_features'):
                        eval_csv = f"{pair}_signal_evaluation.csv"
                        forecasting_system.evaluate_signal_features(feature_df, target_col='target_1d', output_csv=eval_csv)
                        logger.info(f"Per-signal evaluation printed and saved to {eval_csv}")
                except Exception as e:
                    logger.error(f"Signal evaluation failed for {pair}: {e}")

                final_results[pair] = {
                    'status': 'success',
                    'model_trained': True,
                    'validation_accuracy': val_acc,
                    'test_accuracy': test_acc,
                    'target_accuracy': self.target_accuracy,
                    'target_reached': val_acc >= self.target_accuracy
                }
            except Exception as e:
                logger.error(f"An error occurred while processing {pair}: {e}", exc_info=True)
                raise

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
    # Load .env file if present to ensure environment variables like FRED_API_KEY are available
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
            logger.info("✅ Loaded environment variables from .env")
        except ImportError:
            logger.warning("⚠️  python-dotenv not installed, environment variables may not be loaded from .env")

    import argparse

    parser = argparse.ArgumentParser(description='Automated Model Training')
    parser.add_argument('--target', type=float, default=0.85,
                       help='Target accuracy (default: 0.85)')
    parser.add_argument('--max-iterations', type=int, default=50,
                       help='Maximum iterations per pair (default: 50)')
    parser.add_argument('--pairs', nargs='+', default=['EURUSD', 'XAUUSD'],
                       help='Currency pairs to optimize (default: EURUSD XAUUSD)')
    parser.add_argument('--dry-run', action='store_true', help='Run a quick dry-run training with capped iterations')
    parser.add_argument('--dry-iterations', type=int, default=10, help='Number of iterations to approximate in dry-run (default: 10)')
    parser.add_argument('--na-threshold', type=float, default=0.5, help='NA% threshold for pruning (0-1)')
    parser.add_argument('--drop-zero-variance', action='store_true', help='Drop numeric zero-variance features during pruning')
    parser.add_argument('--loop-until-target', action='store_true', help='Loop training until target accuracy is reached or max attempts hit')
    parser.add_argument('--max-attempts', type=int, default=10, help='Maximum training attempts in loop mode (default: 10)')

    args = parser.parse_args()

    # Ensure environment is loaded and FRED key is available
    ensure_environment_loaded()

    # 1. Fix data issues first
    if not pre_training_data_fix():
        logger.error("❌ Data validation failed - cannot proceed with training.")
        sys.exit(1)

    trainer = AutomatedTrainer(
        target_accuracy=args.target,
        max_iterations=args.max_iterations
    )

    if args.loop_until_target:
        logger.info("Running training loop until target accuracy is reached or max attempts hit...")
        best_results = trainer.train_until_target(
            pairs=args.pairs,
            max_attempts=args.max_attempts
        )
    else:
        results = trainer.run_automated_training(
            pairs=args.pairs,
            dry_run=args.dry_run,
            dry_iterations=args.dry_iterations,
            dry_timeout_seconds=60
        )
        # Exit with success/failure code
        all_targets_reached = all(
            result.get('target_reached', False)
            for result in results.values()
            if 'error' not in result
        )
        sys.exit(0 if all_targets_reached else 1)

if __name__ == '__main__':
    main()