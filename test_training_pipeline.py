#!/usr/bin/env python3
"""
Quick Training Test
Tests the training pipeline with minimal iterations to verify it works
"""
import sys
import os
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_training_pipeline():
    """Test the complete training pipeline quickly"""
    print("\n" + "="*80)
    print("QUICK TRAINING PIPELINE TEST")
    print("="*80 + "\n")
    
    try:
        # Load environment
        from dotenv import load_dotenv
        env_path = Path(__file__).parent.parent / '.env'
        if env_path.exists():
            load_dotenv(env_path)
            print("✅ Environment loaded")
        
        # Test 1: Import all required modules
        print("\nTest 1: Importing modules...")
        try:
            from scripts.forecasting import HybridPriceForecastingEnsemble
            from scripts.robust_lightgbm_config import create_robust_lgb_config_for_small_data
            import lightgbm as lgb
            import pandas as pd
            import numpy as np
            from sklearn.metrics import accuracy_score
            print("✅ All modules imported successfully")
        except ImportError as e:
            print(f"❌ Import failed: {e}")
            return False
        
        # Test 2: Initialize forecasting system
        print("\nTest 2: Initializing forecasting system...")
        try:
            system = HybridPriceForecastingEnsemble(pair='EURUSD')
            print("✅ Forecasting system initialized")
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False
        
        # Test 3: Feature engineering
        print("\nTest 3: Running feature engineering...")
        try:
            features = system._prepare_features()
            if features is None or features.empty:
                print("❌ Feature engineering returned empty dataframe")
                return False
            
            n_samples, n_features = features.shape
            print(f"✅ Features generated: {n_samples} samples × {n_features} features")
            
            # Check feature types
            fund_feats = len([c for c in features.columns if c.startswith('fund_')])
            print(f"   - Fundamental features: {fund_feats}")
            
            if fund_feats < 10:
                print(f"⚠️  Warning: Only {fund_feats} fundamental features (expected >10)")
            
        except Exception as e:
            print(f"❌ Feature engineering failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test 4: Data splitting
        print("\nTest 4: Splitting data...")
        try:
            target_col = 'target_1d'
            if target_col not in features.columns:
                target_cols = [c for c in features.columns if c.startswith('target_')]
                if not target_cols:
                    print(f"❌ No target column found")
                    return False
                target_col = target_cols[0]
            
            y = features[target_col]
            X = features.drop(columns=[c for c in features.columns 
                                      if 'target' in c or 'next_close' in c], 
                             errors='ignore')
            X = X.select_dtypes(include=[np.number])
            
            n = len(X)
            train_end = int(0.70 * n)
            valid_end = int(0.85 * n)
            
            X_train = X.iloc[:train_end]
            X_val = X.iloc[train_end:valid_end]
            y_train = y.iloc[:train_end]
            y_val = y.iloc[train_end:valid_end]
            
            print(f"✅ Data split:")
            print(f"   - Train: {len(X_train)} samples")
            print(f"   - Val:   {len(X_val)} samples")
            print(f"   - Features: {len(X.columns)} numeric features")
            
        except Exception as e:
            print(f"❌ Data splitting failed: {e}")
            return False
        
        # Test 5: Quick training (only 10 iterations)
        print("\nTest 5: Quick model training (10 iterations)...")
        try:
            # Override config for quick test
            config = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 8,
                'max_depth': 4,
                'learning_rate': 0.1,
                'num_iterations': 10,  # Very quick test
                'verbosity': -1,
                'seed': 42,
            }
            
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            model = lgb.train(
                params=config,
                train_set=train_data,
                valid_sets=[valid_data],
                callbacks=[lgb.log_evaluation(period=5)]
            )
            
            print(f"✅ Model trained: {model.num_trees()} trees")
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test 6: Evaluation
        print("\nTest 6: Model evaluation...")
        try:
            val_pred = (model.predict(X_val) > 0.5).astype(int)
            val_acc = accuracy_score(y_val, val_pred)
            
            print(f"✅ Validation accuracy: {val_acc:.4f} ({val_acc*100:.1f}%)")
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            return False
        
        # Test 7: Model saving
        print("\nTest 7: Model saving...")
        try:
            test_model_path = Path('models') / 'test_model.txt'
            test_model_path.parent.mkdir(exist_ok=True)
            model.save_model(str(test_model_path))
            
            # Verify file exists
            if test_model_path.exists():
                size = test_model_path.stat().st_size / 1024  # KB
                print(f"✅ Model saved: {test_model_path} ({size:.1f} KB)")
                
                # Clean up test model
                test_model_path.unlink()
                print("   (Test model deleted)")
            else:
                print("❌ Model file not created")
                return False
            
        except Exception as e:
            print(f"❌ Model saving failed: {e}")
            return False
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED")
        print("="*80)
        print("\nTraining pipeline is working correctly!")
        print("Ready for full production training.\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_training_pipeline()
    sys.exit(0 if success else 1)
