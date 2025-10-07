#!/usr/bin/env python3
"""
Complete Model Reset and Retraining Pipeline
Implements the full cleanup, optimal configuration, and retraining process
for the 346-feature financial model.

This script:
1. Cleans up ALL existing models and artifacts
2. Validates data integrity
3. Trains with OPTIMAL configuration (LR=0.01, strong regularization)
4. Comprehensive validation and persistence
5. Generates detailed performance reports
"""
import sys
import os
import gc
import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'model_reset_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


def phase_1_cleanup_existing_models():
    """
    PHASE 1: Remove all existing model artifacts and clear memory
    Ensures a clean slate for retraining with optimal configuration
    """
    logger.info("="*80)
    logger.info("PHASE 1: CLEANUP EXISTING MODELS")
    logger.info("="*80)
    
    # Clear any existing model variables from memory
    if 'model' in globals():
        del globals()['model']
    if 'lgb_model' in globals():
        del globals()['lgb_model']
    
    # Force garbage collection
    gc.collect()
    logger.info("✅ Memory cleared")
    
    # Remove all model files
    model_patterns = [
        'models/*.txt',
        'models/*.pkl', 
        'models/*.joblib',
        'models/*.mdl',
        'models/*.json',
        '*.txt',  # Root level model files
        '*.pkl',
        '*.joblib'
    ]
    
    deleted_count = 0
    for pattern in model_patterns:
        for file_path in Path('.').glob(pattern):
            if 'model' in str(file_path).lower():
                try:
                    file_size = file_path.stat().st_size / 1024  # KB
                    file_path.unlink()
                    logger.info(f"   Deleted: {file_path} ({file_size:.1f} KB)")
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"   Could not delete {file_path}: {e}")
    
    logger.info(f"✅ Phase 1 Complete: {deleted_count} model files deleted\n")
    return deleted_count > 0


def phase_2_validate_data_integrity():
    """
    PHASE 2: Validate data integrity before training
    Ensures all required data files exist and have proper schema
    """
    logger.info("="*80)
    logger.info("PHASE 2: DATA INTEGRITY VALIDATION")
    logger.info("="*80)
    
    required_files = {
        'EURUSD': ['data/EURUSD_Daily.csv', 'data/EURUSD_H4.csv', 'data/EURUSD_Monthly.csv'],
        'XAUUSD': ['data/XAUUSD_Daily.csv', 'data/XAUUSD_H4.csv', 'data/XAUUSD_Monthly.csv']
    }
    
    all_valid = True
    for pair, files in required_files.items():
        logger.info(f"\nValidating {pair} data files:")
        for file_path in files:
            path = Path(file_path)
            if not path.exists():
                logger.error(f"   ❌ Missing: {file_path}")
                all_valid = False
            else:
                try:
                    df = pd.read_csv(path)
                    logger.info(f"   ✅ {path.name}: {len(df)} rows")
                except Exception as e:
                    logger.error(f"   ❌ Error reading {file_path}: {e}")
                    all_valid = False
    
    # Check fundamental data
    logger.info("\nValidating fundamental data files:")
    fundamental_files = list(Path('data').glob('*.csv'))
    fundamental_count = len([f for f in fundamental_files 
                            if not any(pair in f.name for pair in ['EURUSD', 'XAUUSD'])])
    logger.info(f"   Found {fundamental_count} fundamental data files")
    
    if all_valid:
        logger.info(f"\n✅ Phase 2 Complete: All data files validated\n")
    else:
        logger.error(f"\n❌ Phase 2 Failed: Some data files missing or corrupt\n")
    
    return all_valid


def phase_3_load_optimal_configuration():
    """
    PHASE 3: Load optimal configuration for 346-feature model
    Uses the research-backed optimal parameters
    """
    logger.info("="*80)
    logger.info("PHASE 3: OPTIMAL CONFIGURATION")
    logger.info("="*80)
    
    from scripts.robust_lightgbm_config import create_robust_lgb_config_for_small_data
    
    config = create_robust_lgb_config_for_small_data()
    
    logger.info("\n📊 Configuration Details:")
    logger.info(f"   Tree Structure:")
    logger.info(f"      - num_leaves: {config['num_leaves']} (2^7-1, balanced)")
    logger.info(f"      - max_depth: {config['max_depth']} (matches leaf capacity)")
    logger.info(f"      - min_data_in_leaf: {config['min_data_in_leaf']} (stability)")
    
    logger.info(f"\n   Learning Parameters:")
    logger.info(f"      - learning_rate: {config['learning_rate']} (optimal for 346 features)")
    logger.info(f"      - num_iterations: {config['num_iterations']} (extended training)")
    logger.info(f"      - early_stopping: {config['early_stopping_round']} (patient)")
    
    logger.info(f"\n   Regularization (STRONG for high dimensions):")
    logger.info(f"      - lambda_l1: {config['lambda_l1']} (feature selection)")
    logger.info(f"      - lambda_l2: {config['lambda_l2']} (generalization)")
    
    logger.info(f"\n   Feature Sampling:")
    logger.info(f"      - feature_fraction: {config['feature_fraction']} ({int(346*config['feature_fraction'])} of 346 features)")
    logger.info(f"      - bagging_fraction: {config['bagging_fraction']}")
    
    logger.info(f"\n✅ Phase 3 Complete: Optimal configuration loaded\n")
    return config


def phase_4_train_single_pair(pair: str, config: Dict) -> Dict:
    """
    PHASE 4: Train model for a single pair with comprehensive monitoring
    """
    logger.info("="*80)
    logger.info(f"PHASE 4: TRAINING {pair} WITH OPTIMAL CONFIG")
    logger.info("="*80)
    
    start_time = datetime.now()
    result = {
        'pair': pair,
        'status': 'failed',
        'error': None,
        'start_time': start_time.isoformat(),
        'validation_accuracy': None,
        'test_accuracy': None,
        'n_features': None,
        'n_samples': None,
        'best_iteration': None,
        'training_time_seconds': None
    }
    
    try:
        from scripts.forecasting import HybridPriceForecastingEnsemble
        import lightgbm as lgb
        from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
        
        # Initialize system
        logger.info(f"\n📊 Step 1/5: Initializing forecasting system for {pair}...")
        system = HybridPriceForecastingEnsemble(pair=pair)
        logger.info(f"✅ System initialized")
        
        # Feature engineering
        logger.info(f"\n📊 Step 2/5: Engineering features...")
        features = system._prepare_features()
        
        n_samples, n_features = features.shape
        result['n_samples'] = n_samples
        result['n_features'] = n_features
        
        logger.info(f"✅ Features prepared: {n_samples} samples × {n_features} features")
        
        # Feature breakdown
        fund_feats = len([c for c in features.columns if c.startswith('fund_')])
        h4_feats = len([c for c in features.columns if 'h4' in c.lower()])
        weekly_feats = len([c for c in features.columns if 'weekly' in c.lower()])
        
        logger.info(f"   Feature breakdown:")
        logger.info(f"      - Fundamental: {fund_feats}")
        logger.info(f"      - H4 timeframe: {h4_feats}")
        logger.info(f"      - Weekly timeframe: {weekly_feats}")
        logger.info(f"      - Other technical: {n_features - fund_feats - h4_feats - weekly_feats}")
        
        # Data splitting
        logger.info(f"\n📊 Step 3/5: Splitting data (time-series aware)...")
        target_col = 'target_1d'
        if target_col not in features.columns:
            target_cols = [c for c in features.columns if c.startswith('target_')]
            target_col = target_cols[0] if target_cols else None
            
        if not target_col:
            raise ValueError("No target column found")
        
        y = features[target_col]
        X = features.drop(columns=[c for c in features.columns 
                                  if 'target' in c or 'next_close' in c], 
                         errors='ignore')
        X = X.select_dtypes(include=[np.number])
        
        # Time-series split
        n = len(X)
        train_end = int(0.70 * n)
        valid_end = int(0.85 * n)
        
        X_train, X_val, X_test = X.iloc[:train_end], X.iloc[train_end:valid_end], X.iloc[valid_end:]
        y_train, y_val, y_test = y.iloc[:train_end], y.iloc[train_end:valid_end], y.iloc[valid_end:]
        
        logger.info(f"✅ Data split complete:")
        logger.info(f"      - Train: {len(X_train)} samples ({len(X_train)/n*100:.1f}%)")
        logger.info(f"      - Val:   {len(X_val)} samples ({len(X_val)/n*100:.1f}%)")
        logger.info(f"      - Test:  {len(X_test)} samples ({len(X_test)/n*100:.1f}%)")
        logger.info(f"      - Features: {len(X.columns)} numeric")
        
        # Class balance
        train_balance = y_train.value_counts()
        logger.info(f"      - Train balance: Class 0={train_balance.get(0, 0)}, Class 1={train_balance.get(1, 0)}")
        
        # Training
        logger.info(f"\n📊 Step 4/5: Training LightGBM with OPTIMAL config...")
        logger.info(f"   ⏳ Estimated time: 5-20 minutes (depends on early stopping)")
        logger.info(f"   Configuration: LR={config['learning_rate']}, Iterations={config['num_iterations']}")
        
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train with progress logging
        iteration_count = [0]
        def log_callback(env):
            iteration_count[0] = env.iteration
            if env.iteration % 200 == 0:
                logger.info(f"      Iteration {env.iteration}/{config['num_iterations']}")
        
        training_start = datetime.now()
        model = lgb.train(
            params=config,
            train_set=train_data,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'eval'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=config['early_stopping_round']),
                lgb.log_evaluation(period=200),
                log_callback
            ]
        )
        training_duration = (datetime.now() - training_start).total_seconds()
        
        result['best_iteration'] = int(model.best_iteration)
        result['training_time_seconds'] = training_duration
        
        logger.info(f"\n✅ Training completed:")
        logger.info(f"      - Best iteration: {model.best_iteration}")
        logger.info(f"      - Total trees: {model.num_trees()}")
        logger.info(f"      - Training time: {training_duration/60:.1f} minutes")
        
        # Evaluation
        logger.info(f"\n📊 Step 5/5: Comprehensive evaluation...")
        
        # Validation metrics
        val_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
        val_pred = (val_pred_proba > 0.5).astype(int)
        val_acc = accuracy_score(y_val, val_pred)
        val_auc = roc_auc_score(y_val, val_pred_proba)
        
        # Test metrics
        test_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
        test_pred = (test_pred_proba > 0.5).astype(int)
        test_acc = accuracy_score(y_test, test_pred)
        test_auc = roc_auc_score(y_test, test_pred_proba)
        
        result['validation_accuracy'] = float(val_acc)
        result['validation_auc'] = float(val_auc)
        result['test_accuracy'] = float(test_acc)
        result['test_auc'] = float(test_auc)
        
        logger.info(f"✅ Validation Performance:")
        logger.info(f"      - Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
        logger.info(f"      - AUC Score: {val_auc:.4f}")
        
        logger.info(f"✅ Test Performance:")
        logger.info(f"      - Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        logger.info(f"      - AUC Score: {test_auc:.4f}")
        
        # Classification report
        logger.info(f"\n📊 Detailed Classification Report:")
        report = classification_report(y_test, test_pred, output_dict=True)
        logger.info(f"      - Precision (Class 1): {report['1']['precision']:.3f}")
        logger.info(f"      - Recall (Class 1):    {report['1']['recall']:.3f}")
        logger.info(f"      - F1-Score (Class 1):  {report['1']['f1-score']:.3f}")
        
        # Feature importance
        feature_importance = model.feature_importance(importance_type='gain')
        top_features = sorted(enumerate(feature_importance), key=lambda x: x[1], reverse=True)[:20]
        
        logger.info(f"\n🎯 Top 20 Most Important Features:")
        for idx, (feat_idx, importance) in enumerate(top_features, 1):
            feat_name = X.columns[feat_idx]
            logger.info(f"      {idx}. {feat_name}: {importance:.0f}")
        
        # Save model
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_file = models_dir / f'{pair}_optimal_model_{timestamp}.txt'
        
        model.save_model(str(model_file), num_iteration=model.best_iteration)
        result['model_path'] = str(model_file)
        result['model_size_kb'] = model_file.stat().st_size / 1024
        
        logger.info(f"\n💾 Model saved:")
        logger.info(f"      - Path: {model_file}")
        logger.info(f"      - Size: {result['model_size_kb']:.1f} KB")
        
        # Signal evaluation
        if hasattr(system, 'evaluate_signal_features'):
            try:
                eval_csv = f"{pair}_signal_evaluation_{timestamp}.csv"
                system.evaluate_signal_features(features, target_col=target_col, output_csv=eval_csv)
                result['signal_eval_path'] = eval_csv
                logger.info(f"      - Signal evaluation: {eval_csv}")
            except Exception as e:
                logger.warning(f"      Signal evaluation failed: {e}")
        
        result['status'] = 'success'
        
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"\n❌ Training failed for {pair}: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    finally:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        result['end_time'] = end_time.isoformat()
        result['duration_seconds'] = duration
        
        logger.info(f"\n{'='*80}")
        logger.info(f"PHASE 4 SUMMARY FOR {pair}")
        logger.info(f"{'='*80}")
        logger.info(f"Status: {result['status'].upper()}")
        logger.info(f"Total Duration: {duration/60:.2f} minutes")
        
        if result['status'] == 'success':
            logger.info(f"✅ SUCCESS METRICS:")
            logger.info(f"   - Validation Acc: {result['validation_accuracy']:.4f}")
            logger.info(f"   - Test Acc:       {result['test_accuracy']:.4f}")
            logger.info(f"   - Best Iteration: {result['best_iteration']}")
            logger.info(f"   - Features Used:  {result['n_features']}")
        else:
            logger.info(f"❌ FAILED: {result['error']}")
        
        logger.info(f"{'='*80}\n")
    
    return result


def phase_5_final_summary(all_results: Dict):
    """
    PHASE 5: Generate final summary and save comprehensive results
    """
    logger.info("="*80)
    logger.info("PHASE 5: FINAL SUMMARY")
    logger.info("="*80)
    
    success_count = sum(1 for r in all_results.values() if r['status'] == 'success')
    total_count = len(all_results)
    
    logger.info(f"\n📊 Overall Results:")
    logger.info(f"   - Total pairs processed: {total_count}")
    logger.info(f"   - Successful: {success_count}")
    logger.info(f"   - Failed: {total_count - success_count}")
    
    for pair, result in all_results.items():
        status_icon = "✅" if result['status'] == 'success' else "❌"
        logger.info(f"\n{status_icon} {pair}:")
        logger.info(f"   Status: {result['status']}")
        logger.info(f"   Duration: {result['duration_seconds']/60:.2f} minutes")
        
        if result['status'] == 'success':
            logger.info(f"   Validation Acc: {result['validation_accuracy']:.4f} ({result['validation_accuracy']*100:.2f}%)")
            logger.info(f"   Test Acc:       {result['test_accuracy']:.4f} ({result['test_accuracy']*100:.2f}%)")
            logger.info(f"   Features: {result['n_features']}")
            logger.info(f"   Samples: {result['n_samples']}")
            logger.info(f"   Model: {result['model_path']}")
        else:
            logger.info(f"   Error: {result['error']}")
    
    # Save comprehensive results
    results_file = Path(f'model_reset_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(results_file, 'w') as f:
        json.dump({
            'reset_timestamp': datetime.now().isoformat(),
            'configuration_used': 'optimal_346_features',
            'total_pairs': total_count,
            'successful': success_count,
            'results': all_results
        }, f, indent=2)
    
    logger.info(f"\n💾 Results saved to: {results_file}")
    logger.info(f"\n{'='*80}")
    
    if success_count == total_count:
        logger.info("🎉 ALL TRAINING COMPLETED SUCCESSFULLY!")
    else:
        logger.info(f"⚠️  {total_count - success_count} PAIR(S) FAILED")
    
    logger.info(f"{'='*80}\n")
    
    return success_count == total_count


def main():
    """Main execution pipeline"""
    logger.info("\n" + "="*80)
    logger.info("COMPLETE MODEL RESET AND RETRAINING")
    logger.info("Using OPTIMAL Configuration for 346-Feature Financial Model")
    logger.info("="*80 + "\n")
    
    # Phase 1: Cleanup
    if not phase_1_cleanup_existing_models():
        logger.warning("No models found to delete (fresh start)")
    
    # Phase 2: Validate data
    if not phase_2_validate_data_integrity():
        logger.error("❌ Data validation failed - cannot proceed")
        return False
    
    # Phase 3: Load optimal config
    config = phase_3_load_optimal_configuration()
    
    # Phase 4: Train all pairs
    pairs = ['EURUSD', 'XAUUSD']
    all_results = {}
    
    for pair in pairs:
        result = phase_4_train_single_pair(pair, config)
        all_results[pair] = result
    
    # Phase 5: Final summary
    success = phase_5_final_summary(all_results)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
