#!/usr/bin/env python3
"""
Train Models with Integrated Fundamental Data
Retrains EURUSD and XAUUSD models using technical + fundamental features
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_integrated_data(pair: str) -> pd.DataFrame:
    """Load integrated data with fundamentals"""
    data_file = Path('data') / f"{pair}_with_fundamentals.csv"
    
    if not data_file.exists():
        raise FileNotFoundError(f"Integrated data file not found: {data_file}")
    
    logger.info(f"Loading {pair} data from {data_file}")
    df = pd.read_csv(data_file)
    
    # Handle timestamp column
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    
    logger.info(f"Loaded {len(df)} rows with {df.shape[1]} features")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
    
    return df


def create_target_variable(df: pd.DataFrame, forward_bars: int = 1) -> pd.DataFrame:
    """
    Create binary target variable for next-day price movement.
    
    Args:
        df: DataFrame with price data
        forward_bars: Number of bars to look ahead
        
    Returns:
        DataFrame with 'target' column added
    """
    # Assuming 'close' column exists
    if 'close' not in df.columns:
        # Try lowercase
        close_cols = [col for col in df.columns if 'close' in col.lower()]
        if close_cols:
            close_col = close_cols[0]
        else:
            raise ValueError("No 'close' price column found")
    else:
        close_col = 'close'
    
    # Create target: 1 if price goes up, 0 if it goes down
    df = df.copy()
    df['future_close'] = df[close_col].shift(-forward_bars)
    df['target'] = (df['future_close'] > df[close_col]).astype(int)
    
    # Drop rows with NaN target
    df = df.dropna(subset=['target', 'future_close'])
    
    # Drop the helper column
    df = df.drop(columns=['future_close'])
    
    logger.info(f"Target variable created: {df['target'].value_counts().to_dict()}")
    logger.info(f"Class balance: {df['target'].mean():.2%} positive class")
    
    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prepare features and target for training.
    
    Returns:
        tuple: (X, y, feature_names)
    """
    # Columns to exclude from features
    exclude_cols = ['target', 'id', 'time']
    
    # Also exclude any unnamed columns or index columns
    exclude_cols.extend([col for col in df.columns if 'unnamed' in col.lower() or col.startswith('_')])
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Handle any remaining NaN values
    X = df[feature_cols].copy()
    y = df['target'].copy()
    
    # Fill NaN with forward-fill, then backward-fill, then 0
    X = X.ffill().bfill().fillna(0)
    
    # Replace inf values
    X = X.replace([np.inf, -np.inf], 0)
    
    logger.info(f"Prepared features: {X.shape[1]} features, {len(X)} samples")
    logger.info(f"Feature names: {feature_cols[:10]}...")
    
    return X, y, feature_cols


def train_model(X_train, y_train, X_val, y_val, params=None):
    """
    Train LightGBM model.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        params: Optional model parameters
        
    Returns:
        Trained model
    """
    if params is None:
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'max_depth': 7,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1
        }
    
    logger.info("Training LightGBM model...")
    logger.info(f"Parameters: {params}")
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Train model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100)
        ]
    )
    
    return model


def evaluate_model(model, X_test, y_test, pair: str):
    """Evaluate model performance"""
    logger.info(f"\n{'='*80}")
    logger.info(f"EVALUATING {pair} MODEL")
    logger.info(f"{'='*80}")
    
    # Predictions
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except:
        auc = 0.0
    
    logger.info(f"Test Set Performance:")
    logger.info(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  F1 Score:  {f1:.4f}")
    logger.info(f"  ROC AUC:   {auc:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': model.feature_name(),
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    logger.info(f"\nTop 20 Most Important Features:")
    for idx, row in feature_importance.head(20).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.2f}")
    
    # Check for fundamental features in top 50
    top_50 = feature_importance.head(50)
    fundamental_indicators = ['cpi', 'gdp', 'fedfunds', 'unrate', 'dgs', 'vix', 'oil', 'dxy', 'exy']
    fundamental_features = [f for f in top_50['feature'].values if any(ind in f.lower() for ind in fundamental_indicators)]
    
    if fundamental_features:
        logger.info(f"\n🎯 Fundamental Features in Top 50:")
        for feat in fundamental_features:
            importance = top_50[top_50['feature'] == feat]['importance'].values[0]
            logger.info(f"  {feat}: {importance:.2f}")
    else:
        logger.warning("⚠️  No fundamental features in top 50 - fundamentals may not be useful")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'feature_importance': feature_importance
    }


def save_model(model, pair: str, metrics: dict):
    """Save trained model and metrics"""
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # Save model
    model_file = models_dir / f"{pair}_lightgbm_with_fundamentals.joblib"
    joblib.dump(model, model_file)
    logger.info(f"✅ Model saved to: {model_file}")
    
    # Save LightGBM native format
    model_txt = models_dir / f"{pair}_fundamentals_model.txt"
    model.save_model(str(model_txt))
    logger.info(f"✅ LightGBM model saved to: {model_txt}")
    
    # Save metrics
    metrics_file = models_dir / f"{pair}_fundamentals_metrics.json"
    import json
    with open(metrics_file, 'w') as f:
        json.dump({
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1': float(metrics['f1']),
            'auc': float(metrics['auc']),
            'training_date': datetime.now().isoformat(),
            'feature_count': len(model.feature_name())
        }, f, indent=2)
    logger.info(f"✅ Metrics saved to: {metrics_file}")


def train_pair(pair: str):
    """Train model for a specific pair"""
    logger.info(f"\n{'='*80}")
    logger.info(f"TRAINING {pair} MODEL WITH FUNDAMENTALS")
    logger.info(f"{'='*80}")
    
    try:
        # Load data
        df = load_integrated_data(pair)
        
        # Create target
        df = create_target_variable(df, forward_bars=1)
        
        # Prepare features
        X, y, feature_names = prepare_features(df)
        
        # Split data (80% train, 10% val, 10% test)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.1, shuffle=False  # Time series - don't shuffle
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.111, shuffle=False  # 0.111 * 0.9 = 0.1
        )
        
        logger.info(f"Data split:")
        logger.info(f"  Train: {len(X_train)} samples")
        logger.info(f"  Val:   {len(X_val)} samples")
        logger.info(f"  Test:  {len(X_test)} samples")
        
        # Train model
        model = train_model(X_train, y_train, X_val, y_val)
        
        # Evaluate
        metrics = evaluate_model(model, X_test, y_test, pair)
        
        # Save
        save_model(model, pair, metrics)
        
        return model, metrics
        
    except Exception as e:
        logger.error(f"❌ Error training {pair}: {e}", exc_info=True)
        return None, None


def main():
    """Main training function"""
    logger.info("="*80)
    logger.info("TRAINING MODELS WITH FUNDAMENTAL DATA")
    logger.info("="*80)
    
    pairs = ['EURUSD', 'XAUUSD']
    results = {}
    
    for pair in pairs:
        model, metrics = train_pair(pair)
        if model is not None:
            results[pair] = metrics
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("TRAINING SUMMARY")
    logger.info(f"{'='*80}")
    
    for pair, metrics in results.items():
        logger.info(f"{pair}:")
        logger.info(f"  ✅ Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        logger.info(f"  ✅ F1 Score: {metrics['f1']:.4f}")
        logger.info(f"  ✅ ROC AUC:  {metrics['auc']:.4f}")
    
    logger.info("\nModels saved with '_with_fundamentals' suffix")
    logger.info("Compare with old models to see impact of fundamental features!")


if __name__ == "__main__":
    main()
