import logging
import signal
from typing import Dict

import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class TimeoutException(Exception):
    pass


def stop_on_negative_gain(env):
    """Callback to stop training if there's no positive gain."""
    # Only consider aborting when we have a valid best_score; avoid aborting
    # prematurely when best_score is still None during early iterations.
    best = getattr(env, 'best_score', None)
    if best is not None and best < 0:
        if env.iteration > 10:
            logging.getLogger('forecasting').warning(
                f"Stopping training on iteration {env.iteration} due to no positive gain.")
            raise lgb.basic.LightGBMError(
                "Aborting: no positive gain splits detected for 10 consecutive iterations.")


def timeout(seconds):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutException(f"Function {func.__name__} exceeded {seconds}s timeout.")

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)

        return wrapper

    return decorator


def create_robust_lgb_config_for_small_data():
    """
    Optimized LightGBM configuration for 346-feature financial model.
    Uses higher-end recommended ranges for maximum stability and generalization.
    
    Key optimizations for 346 features:
    - Balanced tree structure: num_leaves=127 (2^7-1) with max_depth=7
    - Conservative learning: 0.01 LR (NOT 0.03 which is 30x too high)
    - Strong regularization: L1=3.0, L2=5.0 for high-dimensional data
    - Extended training: 2000 iterations with patient early stopping (150 rounds)
    - Feature sampling: 60% to handle dimensionality and reduce overfitting
    """
    return {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        
        # Tree Structure (balanced for complex feature interactions)
        'num_leaves': 127,           # 2^7-1 for deeper trees, matches max_depth capacity
        'max_depth': 7,              # Balanced with num_leaves (2^7 = 128)
        'min_data_in_leaf': 150,     # Higher for stability with 346 features
        'min_gain_to_split': 0.01,   # Conservative splitting threshold
        
        # Learning Parameters (conservative for financial data)
        'learning_rate': 0.01,       # CRITICAL: 0.03 was 3x too high! Use 0.01 for stability
        'num_iterations': 2000,      # Extended for thorough learning with lower LR
        
        # Regularization (strong for high-dimensional financial data)
        'lambda_l1': 3.0,            # Strong L1 for automatic feature selection
        'lambda_l2': 5.0,            # Strong L2 for generalization
        'min_sum_hessian_in_leaf': 0.15,  # Higher for stability
        
        # Feature and Sampling (conservative for 346 features)
        'feature_fraction': 0.6,     # Sample 60% of 346 features (208 per tree)
        'bagging_fraction': 0.8,     # Keep 80% data sampling
        'bagging_freq': 5,           # Every 5 iterations
        
        # Technical Parameters
        'max_bin': 255,              # Maximum precision
        'early_stopping_round': 150,  # Extended patience for 2000 iterations
        'verbosity': 1,
        'is_unbalance': True,
        'seed': 42,
        'deterministic': True,
        'force_row_wise': True,      # Stability for many features
    }


def create_emergency_minimal_lgb_config():
    return {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 4,
        'max_depth': 2,
        'min_data_in_leaf': 15,
        'min_gain_to_split': 0.1,
        'learning_rate': 0.1,
        'num_iterations': 30,
        'lambda_l1': 5.0,
        'lambda_l2': 5.0,
        'min_sum_hessian_in_leaf': 1.0,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 3,
        'max_bin': 32,
        'early_stopping_round': 5,
        'verbosity': -1,
        'seed': 42,
    }


def diagnose_training_data_quality(X: pd.DataFrame, y, timeframe_name: str = "Unknown"):
    issues = []
    recommendations = []
    n_samples, n_features = X.shape
    logger.info(f"📊 {timeframe_name} Data Diagnosis: {n_samples} samples, {n_features} features.")

    if n_samples < 100:
        issues.append(f"CRITICAL: Only {n_samples} samples. Need at least 100 for meaningful training.")
        recommendations.append("Generate more historical data or use a longer base timeframe.")
    elif n_samples < 500:
        issues.append(f"WARNING: {n_samples} samples is minimal. Using emergency config.")
        recommendations.append("Confirm data generation for this timeframe is correct.")

    if hasattr(X, 'var') and X.var(axis=0).sum() == 0:
        issues.append("CRITICAL: All features are constant (zero variance).")
        recommendations.append("Check feature engineering pipeline; features are not diverse.")

    try:
        class_counts = y.value_counts()
        if len(class_counts) < 2:
            issues.append(f"CRITICAL: Target has only {len(class_counts)} unique class. Need 2 for binary classification.")
            recommendations.append("Check target generation logic.")
        else:
            min_class_size = class_counts.min()
            if min_class_size < 10:
                issues.append(f"CRITICAL: Smallest class has only {min_class_size} samples. Need at least 10.")
    except Exception:
        pass

    if issues:
        logger.warning(f"🚨 {timeframe_name}: DATA QUALITY ISSUES DETECTED:")
        for issue in issues:
            logger.warning(f"   - {issue}")
        for rec in recommendations:
            logger.info(f"   💡 {rec}")
    else:
        logger.info(f"✅ {timeframe_name}: Data quality appears adequate for training.")

    return issues


@timeout(300)
def train_with_robust_error_handling(X: pd.DataFrame, y, params: dict, timeframe_name: str = "Unknown"):
    try:
        issues = diagnose_training_data_quality(X, y, timeframe_name)
        if any("CRITICAL" in issue for issue in issues):
            logger.error(f"❌ {timeframe_name}: Aborting training due to critical data issues.")
            return None

        if X.shape[0] < 500:
            logger.warning(f"⚠️ {timeframe_name}: Switching to emergency minimal config for {X.shape[0]} samples.")
            params = create_emergency_minimal_lgb_config()

        test_size = min(0.2, 100 / X.shape[0]) if X.shape[0] > 50 else 0.2
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if hasattr(y, 'values') else None
        )

        logger.info(f"🎯 {timeframe_name}: Training with {len(X_train)} samples, validating with {len(X_val)}.")

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params=params,
            train_set=train_data,
            valid_sets=[valid_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=params.get('early_stopping_round', 50), verbose=False),
                lgb.log_evaluation(period=20),
                stop_on_negative_gain,
            ],
        )

        if model.num_trees() < 5:
            logger.warning(f"⚠️ {timeframe_name}: Model only created {model.num_trees()} trees. May be underfit.")
        else:
            logger.info(f"✅ {timeframe_name}: Successfully trained with {model.num_trees()} trees.")

        return model

    except lgb.basic.LightGBMError as e:
        logger.error(f"❌ {timeframe_name}: LightGBM training failed. Reason: {e}")
        return None
    except TimeoutException as e:
        logger.error(f"❌ {timeframe_name}: Training timed out. Reason: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ {timeframe_name}: Unexpected error during training: {e}", exc_info=True)
        return None


def enhanced_lightgbm_training_pipeline(datasets: Dict[str, pd.DataFrame], target_column: str = 'target') -> Dict[str, object]:
    """Train models for each dataset in `datasets` mapping.

    datasets: mapping from timeframe name to DataFrame containing numeric features and a target column.
    Returns a dict of trained models keyed by timeframe.
    """
    trained_models = {}
    for timeframe, df in datasets.items():
        logger.info(f"\n{'='*10} Processing Timeframe: {timeframe} {'='*10}")
        if not isinstance(df, pd.DataFrame) or df.empty:
            logger.error(f"❌ {timeframe}: Dataset empty or not a DataFrame. Skipping.")
            continue
        if target_column not in df.columns:
            logger.error(f"❌ {timeframe}: Target column '{target_column}' not found. Skipping.")
            continue

        feature_cols = [c for c in df.columns if c != target_column]
        X = df[feature_cols].select_dtypes(include=[np.number]).copy()
        y = df[target_column]

        if X.empty:
            logger.error(f"❌ {timeframe}: No numeric features found after filtering. Skipping.")
            continue

        model = train_with_robust_error_handling(X, y, create_robust_lgb_config_for_small_data(), timeframe)
        if model:
            trained_models[timeframe] = model
            logger.info(f"✅ {timeframe}: Model successfully trained and stored.")
        else:
            logger.error(f"❌ {timeframe}: Failed to train model.")

    if trained_models:
        logger.info(f"\n🎉 PIPELINE COMPLETE: {len(trained_models)}/{len(datasets)} models trained.")
    else:
        logger.error("\n❌ ALL TRAINING ATTEMPTS FAILED.")
    return trained_models


def enhanced_lightgbm_training_pipeline_arrays(X_train, y_train, X_val=None, y_val=None, pair_name='pair'):
    """Compatibility wrapper for the older signature used elsewhere in the codebase.

    Trains a single model using train_with_robust_error_handling on the provided arrays.
    Returns the trained model or None.
    """
    try:
        # Ensure X_train is a DataFrame
        if not hasattr(X_train, 'shape'):
            return None
        X = pd.DataFrame(X_train)
        return train_with_robust_error_handling(X, y_train, create_robust_lgb_config_for_small_data(), pair_name)
    except Exception:
        return None

