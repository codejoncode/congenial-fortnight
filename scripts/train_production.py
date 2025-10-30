#!/usr/bin/env python3
"""
Robust Production Training Script
Trains models with comprehensive error handling and progress monitoring
"""
import sys
import os
import logging
import traceback
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Optional, Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'training_production_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

def load_environment():
    """Load environment variables"""
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
            logger.info("✅ Environment loaded from .env")
            
            # Verify FRED API key
            fred_key = os.getenv('FRED_API_KEY', '')
            if fred_key:
                logger.info(f"✅ FRED API key found: {fred_key[:8]}...")
            else:
                logger.warning("⚠️  FRED API key not found")
        except ImportError:
            logger.warning("⚠️  python-dotenv not installed")
    else:
        logger.warning("⚠️  .env file not found")


def train_single_pair(pair: str, timeout_minutes: int = 60) -> Dict:
    """Train model for a single pair with timeout and error handling"""
    logger.info(f"\n{'='*80}")
    logger.info(f"TRAINING {pair}")
    logger.info(f"{'='*80}\n")
    
    start_time = datetime.now()
    result = {
        'pair': pair,
        'status': 'failed',
        'error': None,
        'start_time': start_time.isoformat(),
        'end_time': None,
        'duration_seconds': None,
        'validation_accuracy': None,
        'test_accuracy': None,
        'n_features': None,
        'n_samples': None,
        'model_path': None,
        'signal_eval_path': None
    }
    
    try:
        # Import here to avoid issues if imports fail
        from scripts.forecasting import HybridPriceForecastingEnsemble
        from scripts.robust_lightgbm_config import create_robust_lgb_config_for_small_data
        import lightgbm as lgb
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        
        logger.info(f"Step 1/5: Initializing forecasting system for {pair}...")
        try:
            forecasting_system = HybridPriceForecastingEnsemble(pair=pair)
            logger.info(f"✅ Forecasting system initialized")
        except Exception as e:
            raise Exception(f"Failed to initialize forecasting system: {e}")
        
        logger.info(f"Step 2/5: Preparing features for {pair}...")
        try:
            feature_df = forecasting_system._prepare_features()
            
            if feature_df is None or feature_df.empty:
                raise Exception("Feature engineering returned empty dataframe")
            
            n_samples, n_features = feature_df.shape
            result['n_samples'] = n_samples
            result['n_features'] = n_features
            
            logger.info(f"✅ Features prepared: {n_samples} samples × {n_features} features")
            
            # Count feature types
            fund_feats = len([c for c in feature_df.columns if c.startswith('fund_')])
            h4_feats = len([c for c in feature_df.columns if 'h4' in c.lower()])
            weekly_feats = len([c for c in feature_df.columns if 'weekly' in c.lower()])
            logger.info(f"   - Fundamental: {fund_feats}")
            logger.info(f"   - H4: {h4_feats}")
            logger.info(f"   - Weekly: {weekly_feats}")
            logger.info(f"   - Other: {n_features - fund_feats - h4_feats - weekly_feats}")
            
        except Exception as e:
            raise Exception(f"Feature engineering failed: {e}")
        
        logger.info(f"Step 3/5: Splitting data for {pair}...")
        try:
            # Find target column
            target_col = 'target_1d'
            if target_col not in feature_df.columns:
                target_cols = [c for c in feature_df.columns if c.startswith('target_')]
                if not target_cols:
                    raise Exception(f"No target column found. Available: {list(feature_df.columns)[:10]}")
                target_col = target_cols[0]
                logger.info(f"Using target column: {target_col}")
            
            y = feature_df[target_col]
            X = feature_df.drop(columns=[c for c in feature_df.columns 
                                        if 'target' in c or 'next_close' in c], 
                               errors='ignore')
            
            # Remove non-numeric columns
            X = X.select_dtypes(include=[np.number])
            
            # Time-series split (no shuffle)
            n = len(X)
            train_end = int(0.70 * n)
            valid_end = int(0.85 * n)
            
            X_train = X.iloc[:train_end]
            X_val = X.iloc[train_end:valid_end]
            X_test = X.iloc[valid_end:]
            
            y_train = y.iloc[:train_end]
            y_val = y.iloc[train_end:valid_end]
            y_test = y.iloc[valid_end:]
            
            logger.info(f"✅ Data split:")
            logger.info(f"   - Train: {len(X_train)} samples")
            logger.info(f"   - Val:   {len(X_val)} samples")
            logger.info(f"   - Test:  {len(X_test)} samples")
            logger.info(f"   - Features: {len(X.columns)} numeric features")
            
            # Check class balance
            train_balance = y_train.value_counts()
            logger.info(f"   - Train class balance: {dict(train_balance)}")
            
        except Exception as e:
            raise Exception(f"Data splitting failed: {e}")
        
        logger.info(f"Step 4/5: Training LightGBM model for {pair}...")
        logger.info(f"   This may take 20-50 minutes...")
        try:
            # Get training config
            config = create_robust_lgb_config_for_small_data()
            logger.info(f"   - num_iterations: {config['num_iterations']}")
            logger.info(f"   - early_stopping: {config.get('early_stopping_round', 'N/A')}")
            logger.info(f"   - learning_rate: {config['learning_rate']}")
            logger.info(f"   - num_leaves: {config['num_leaves']}")
            
            # Create datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Train with callbacks for progress
            iteration_count = [0]
            
            def log_callback(env):
                iteration_count[0] = env.iteration
                if env.iteration % 100 == 0:
                    logger.info(f"   Iteration {env.iteration}/{config['num_iterations']}")
            
            logger.info(f"   Training started...")
            model = lgb.train(
                params=config,
                train_set=train_data,
                valid_sets=[valid_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=config.get('early_stopping_round', 50)),
                    lgb.log_evaluation(period=100),
                    log_callback
                ]
            )
            
            logger.info(f"✅ Training completed")
            logger.info(f"   - Total trees: {model.num_trees()}")
            logger.info(f"   - Best iteration: {model.best_iteration}")
            
        except Exception as e:
            raise Exception(f"Model training failed: {e}")
        
        logger.info(f"Step 5/5: Evaluating and saving model for {pair}...")
        try:
            # Evaluate
            val_pred = (model.predict(X_val) > 0.5).astype(int)
            test_pred = (model.predict(X_test) > 0.5).astype(int)
            
            val_acc = accuracy_score(y_val, val_pred)
            test_acc = accuracy_score(y_test, test_pred)
            
            result['validation_accuracy'] = float(val_acc)
            result['test_accuracy'] = float(test_acc)
            
            logger.info(f"✅ Model performance:")
            logger.info(f"   - Validation Accuracy: {val_acc:.4f} ({val_acc*100:.1f}%)")
            logger.info(f"   - Test Accuracy:       {test_acc:.4f} ({test_acc*100:.1f}%)")
            
            # Save model
            models_dir = Path(__file__).parent.parent / 'models'
            models_dir.mkdir(exist_ok=True)
            model_path = models_dir / f'{pair}_model.txt'
            
            model.save_model(str(model_path))
            result['model_path'] = str(model_path)
            logger.info(f"✅ Model saved to: {model_path}")
            
            # Signal evaluation
            if hasattr(forecasting_system, 'evaluate_signal_features'):
                try:
                    eval_csv = f"{pair}_signal_evaluation.csv"
                    forecasting_system.evaluate_signal_features(
                        feature_df, 
                        target_col=target_col, 
                        output_csv=eval_csv
                    )
                    result['signal_eval_path'] = eval_csv
                    logger.info(f"✅ Signal evaluation saved to: {eval_csv}")
                except Exception as e:
                    logger.warning(f"⚠️  Signal evaluation failed: {e}")
            
            result['status'] = 'success'
            
        except Exception as e:
            raise Exception(f"Model evaluation/saving failed: {e}")
        
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"❌ Training failed for {pair}: {e}")
        logger.error(traceback.format_exc())
    
    finally:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        result['end_time'] = end_time.isoformat()
        result['duration_seconds'] = duration
        
        logger.info(f"\n{'='*80}")
        logger.info(f"TRAINING SUMMARY FOR {pair}")
        logger.info(f"{'='*80}")
        logger.info(f"Status: {result['status'].upper()}")
        logger.info(f"Duration: {duration/60:.1f} minutes")
        if result['status'] == 'success':
            logger.info(f"Validation Accuracy: {result['validation_accuracy']:.4f}")
            logger.info(f"Test Accuracy: {result['test_accuracy']:.4f}")
            logger.info(f"Model: {result['model_path']}")
        else:
            logger.info(f"Error: {result['error']}")
        logger.info(f"{'='*80}\n")
    
    return result


def main():
    """Main training function"""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Production Model Training')
    parser.add_argument('--pairs', nargs='+', default=['EURUSD', 'XAUUSD'],
                       help='Pairs to train (default: EURUSD XAUUSD)')
    parser.add_argument('--timeout', type=int, default=60,
                       help='Timeout per pair in minutes (default: 60)')
    
    args = parser.parse_args()
    
    logger.info("\n" + "="*80)
    logger.info("PRODUCTION TRAINING START")
    logger.info("="*80)
    logger.info(f"Pairs: {args.pairs}")
    logger.info(f"Timeout: {args.timeout} minutes per pair")
    logger.info("="*80 + "\n")
    
    # Load environment
    load_environment()
    
    # Train each pair
    all_results = {}
    overall_start = datetime.now()
    
    for pair in args.pairs:
        result = train_single_pair(pair, args.timeout)
        all_results[pair] = result
    
    overall_end = datetime.now()
    overall_duration = (overall_end - overall_start).total_seconds()
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("FINAL TRAINING SUMMARY")
    logger.info("="*80)
    logger.info(f"Total Duration: {overall_duration/60:.1f} minutes")
    logger.info(f"Pairs Processed: {len(args.pairs)}")
    
    success_count = sum(1 for r in all_results.values() if r['status'] == 'success')
    logger.info(f"Successful: {success_count}/{len(args.pairs)}")
    
    for pair, result in all_results.items():
        status_icon = "✅" if result['status'] == 'success' else "❌"
        logger.info(f"\n{status_icon} {pair}:")
        logger.info(f"   Status: {result['status']}")
        logger.info(f"   Duration: {result['duration_seconds']/60:.1f} min")
        if result['status'] == 'success':
            logger.info(f"   Val Acc: {result['validation_accuracy']:.4f}")
            logger.info(f"   Test Acc: {result['test_accuracy']:.4f}")
            logger.info(f"   Features: {result['n_features']}")
            logger.info(f"   Samples: {result['n_samples']}")
        else:
            logger.info(f"   Error: {result['error']}")
    
    # Save results
    results_path = Path('training_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'start_time': overall_start.isoformat(),
            'end_time': overall_end.isoformat(),
            'duration_seconds': overall_duration,
            'pairs': all_results
        }, f, indent=2)
    
    logger.info(f"\n📄 Results saved to: {results_path}")
    logger.info("="*80 + "\n")
    
    # Exit code
    sys.exit(0 if success_count == len(args.pairs) else 1)


if __name__ == "__main__":
    main()
