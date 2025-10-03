# LightGBM Configuration for Small/Insufficient Forex Trading Data
# Optimized parameters to handle data quantity and quality issues

import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import signal

# --- Custom Callbacks & Timeout ---

class TimeoutException(Exception): pass

def stop_on_negative_gain(env):
    """Callback to stop training if there's no positive gain."""
    # env.best_score is None when no valid splits are found
    if env.best_score is None or env.best_score < 0:
        # A small negative gain might be acceptable, but consistently no gain is a problem.
        # We check the iteration to avoid stopping too early.
        if env.iteration > 10:
            logging.getLogger('forecasting').warning(f"Stopping training on iteration {env.iteration} due to no positive gain.")
            raise lgb.basic.LightGBMError("Aborting: no positive gain splits detected for 10 consecutive iterations.")

def timeout(seconds):
    """Decorator to enforce a wall-clock timeout on a function."""
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutException(f"Training function {func.__name__} exceeded {seconds}s timeout.")
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)  # Disable the alarm
            return result
        return wrapper
    return decorator

# --- Configurations ---

def create_robust_lgb_config_for_small_data():
    """
    Create LightGBM configuration optimized for small/limited forex trading datasets.
    Addresses the "no further splits" and "best gain: -inf" warnings.
    """
    return {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 8,
        'max_depth': 4,
        'min_data_in_leaf': 10, # Increased from 5 for more robustness
        'min_gain_to_split': 0.01,
        'learning_rate': 0.05,
        'num_iterations': 100, # Increased cap, but early stopping is key
        'lambda_l1': 1.0,
        'lambda_l2': 1.0,
        'min_sum_hessian_in_leaf': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'max_bin': 64,
        'early_stopping_round': 10,
        'verbosity': 1,
        'is_unbalance': True,
        'seed': 42,
        'deterministic': True,
    }

def create_emergency_minimal_lgb_config():
    """
    Ultra-conservative LightGBM config for extremely limited data (< 500 samples).
    """
    return {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 4,
        'max_depth': 2,
        'min_data_in_leaf': 15, # Increased for very small data
        'min_gain_to_split': 0.1,
        'learning_rate': 0.1,
        'num_iterations': 30, # Cap iterations
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

# --- Core Logic ---

def diagnose_training_data_quality(X, y, pair_name="Unknown"):
    """
    Diagnose data quality issues that lead to LightGBM training warnings.
    """
    logger = logging.getLogger('forecasting')
    issues, recommendations = [], []
    
    n_samples, n_features = X.shape
    logger.info(f"📊 {timeframe_name} Data Diagnosis: {n_samples} samples, {n_features} features.")
    
    if n_samples < 100:
        issues.append(f"CRITICAL: Only {n_samples} samples. Need at least 100 for any meaningful training.")
        recommendations.append("Generate more historical data or use a longer base timeframe.")
    elif n_samples < 500:
        issues.append(f"WARNING: {n_samples} samples is minimal. Using emergency config.")
        recommendations.append("Confirm data generation for this timeframe is correct.")

    if hasattr(X, 'var') and X.var(axis=0).sum() == 0:
        issues.append("CRITICAL: All features are constant (zero variance).")
        recommendations.append("Check feature engineering pipeline; features are not diverse.")

    if hasattr(y, 'value_counts'):
        class_counts = y.value_counts()
        if len(class_counts) < 2:
            issues.append(f"CRITICAL: Target has only {len(class_counts)} unique class. Need 2 for binary classification.")
            recommendations.append("Check target generation logic.")
        else:
            min_class_size = class_counts.min()
            if min_class_size < 10:
                issues.append(f"CRITICAL: Smallest class has only {min_class_size} samples. Need at least 10.")
    
    if issues:
        logger.warning(f"🚨 {timeframe_name}: DATA QUALITY ISSUES DETECTED:")
        for issue in issues: logger.warning(f"   - {issue}")
        for rec in recommendations: logger.info(f"   💡 {rec}")
    else:
        logger.info(f"✅ {timeframe_name}: Data quality appears adequate for training.")
    
    return issues

@timeout(300) # 5-minute watchdog timer for the entire training function
def train_with_robust_error_handling(X, y, params, timeframe_name="Unknown"):
    """
    Train LightGBM with robust error handling, parameter adjustment, and timeouts.
    """
    logger = logging.getLogger('forecasting')
    
    try:
        issues = diagnose_training_data_quality(X, y, timeframe_name)
        if any("CRITICAL" in issue for issue in issues):
            logger.error(f"❌ {timeframe_name}: Aborting training due to critical data issues.")
            return None
        
        if X.shape[0] < 500:
            logger.warning(f"⚠️ {timeframe_name}: Switching to emergency minimal config for {X.shape[0]} samples.")
            params = create_emergency_minimal_lgb_config()
        
        # Stratified split is crucial for imbalanced datasets
        test_size = min(0.2, 100 / X.shape[0]) if X.shape[0] > 50 else 0.2
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logger.info(f"🎯 {timeframe_name}: Training with {len(X_train)} samples, validating with {len(X_val)}.")
        
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params=params,
            train_set=train_data,
            valid_sets=[valid_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=params.get('early_stopping_round', 10), verbose=False),
                lgb.log_evaluation(period=20),
                stop_on_negative_gain # Custom callback to prevent hangs
            ]
        )
        
        if model.num_trees() < 5:
            logger.warning(f"⚠️ {timeframe_name}: Model only created {model.num_trees()} trees. May be underfit or data is not predictive.")
        else:
            logger.info(f"✅ {timeframe_name}: Successfully trained with {model.num_trees()} trees.")
        
        return model
        
    except lgb.basic.LightGBMError as e:
        logger.error(f"❌ {timeframe_name}: LightGBM training failed gracefully. Reason: {e}")
        return None
    except TimeoutException as e:
        logger.error(f"❌ {timeframe_name}: Training timed out. Aborting. Reason: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ {timeframe_name}: An unexpected error occurred during training: {e}", exc_info=True)
        return None

def enhanced_lightgbm_training_pipeline(X_train, y_train, X_val, y_val, pair_name="Unknown", timeout_seconds=300):
    """
    Validates and trains LightGBM models for multiple timeframes with robust error handling.
    """
    logger = logging.getLogger('forecasting')
    logger.info("🚀 Starting Enhanced LightGBM Training Pipeline...")
    
    trained_models = {}
    
    for timeframe, df in datasets.items():
        logger.info(f"\n{'='*15} Processing Timeframe: {timeframe} {'='*15}")
        
        if not isinstance(df, pd.DataFrame) or df.empty:
            logger.error(f"❌ {timeframe}: Dataset is empty or not a DataFrame. Skipping.")
            continue
            
        if target_column not in df.columns:
            logger.error(f"❌ {timeframe}: Target column '{target_column}' not found. Skipping.")
            continue
        
        feature_cols = [col for col in df.columns if col != target_column]
        X = df[feature_cols].select_dtypes(include=np.number)
        y = df[target_column]
        
        if X.empty:
            logger.error(f"❌ {timeframe}: No numeric features found after filtering. Skipping.")
            continue
        
        # Use the robust training function
        model = train_with_robust_error_handling(X, y, create_robust_lgb_config_for_small_data(), timeframe)
        
        if model:
            trained_models[timeframe] = model
            logger.info(f"✅ {timeframe}: Model successfully trained and stored.")
        else:
            logger.error(f"❌ {timeframe}: Failed to train model. See logs above for details.")
            
    if trained_models:
        logger.info(f"\n🎉 PIPELINE COMPLETE: {len(trained_models)}/{len(datasets)} models were successfully trained.")
    else:
        logger.error("\n❌ ALL TRAINING ATTEMPTS FAILED. Please review data quality and feature engineering steps.")
        
    return trained_models
