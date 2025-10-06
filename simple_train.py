#!/usr/bin/env python3
"""
Simple Direct Training - Works in Codespaces
"""
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_eurusd():
    """Train EURUSD pair"""
    print("\n" + "="*80)
    print(f"🚀 TRAINING EURUSD - Started at {datetime.now().strftime('%H:%M:%S')}")
    print("="*80)
    
    start_time = time.time()
    
    try:
        from scripts.forecasting import HybridPriceForecastingEnsemble
        
        # Initialize
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Initializing ensemble...")
        ensemble = HybridPriceForecastingEnsemble('EURUSD', data_dir='data', models_dir='models')
        print(f"✅ Initialized - Price: {ensemble.price_data.shape}, Fund: {ensemble.fundamental_data.shape}")
        
        # Prepare features
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Engineering features...")
        X_train, y_train, X_val, y_val = ensemble.load_and_prepare_datasets()
        
        if X_train is None:
            print("❌ Feature engineering failed")
            return False
            
        print(f"✅ Features ready - Train: {X_train.shape}, Val: {X_val.shape}")
        
        # Train with robust LightGBM config
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Training LightGBM...")
        
        from scripts.robust_lightgbm_config import enhanced_lightgbm_training_pipeline
        
        results = enhanced_lightgbm_training_pipeline(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            pair='EURUSD'
        )
        
        elapsed = time.time() - start_time
        print(f"\n✅ EURUSD TRAINING COMPLETE - {elapsed/60:.1f} minutes")
        print(f"Validation Accuracy: {results.get('validation_accuracy', 'N/A')}")
        print(f"Test Accuracy: {results.get('test_accuracy', 'N/A')}")
        
        return True
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ ERROR after {elapsed/60:.1f} minutes: {e}")
        import traceback
        traceback.print_exc()
        return False

def train_xauusd():
    """Train XAUUSD pair"""
    print("\n" + "="*80)
    print(f"🚀 TRAINING XAUUSD - Started at {datetime.now().strftime('%H:%M:%S')}")
    print("="*80)
    
    start_time = time.time()
    
    try:
        from scripts.forecasting import HybridPriceForecastingEnsemble
        
        # Initialize
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Initializing ensemble...")
        ensemble = HybridPriceForecastingEnsemble('XAUUSD', data_dir='data', models_dir='models')
        print(f"✅ Initialized - Price: {ensemble.price_data.shape}, Fund: {ensemble.fundamental_data.shape}")
        
        # Prepare features
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Engineering features...")
        X_train, y_train, X_val, y_val = ensemble.load_and_prepare_datasets()
        
        if X_train is None:
            print("❌ Feature engineering failed")
            return False
            
        print(f"✅ Features ready - Train: {X_train.shape}, Val: {X_val.shape}")
        
        # Train with robust LightGBM config
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Training LightGBM...")
        
        from scripts.robust_lightgbm_config import enhanced_lightgbm_training_pipeline
        
        results = enhanced_lightgbm_training_pipeline(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            pair='XAUUSD'
        )
        
        elapsed = time.time() - start_time
        print(f"\n✅ XAUUSD TRAINING COMPLETE - {elapsed/60:.1f} minutes")
        print(f"Validation Accuracy: {results.get('validation_accuracy', 'N/A')}")
        print(f"Test Accuracy: {results.get('test_accuracy', 'N/A')}")
        
        return True
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ ERROR after {elapsed/60:.1f} minutes: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    total_start = time.time()
    
    print("="*80)
    print("🎯 FOREX ML TRAINING PIPELINE")
    print("="*80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Ensure directories
    Path('logs').mkdir(exist_ok=True)
    Path('models').mkdir(exist_ok=True)
    
    # Train both pairs
    results = {}
    results['EURUSD'] = train_eurusd()
    results['XAUUSD'] = train_xauusd()
    
    # Summary
    total_elapsed = time.time() - total_start
    print("\n" + "="*80)
    print("📊 TRAINING SUMMARY")
    print("="*80)
    print(f"EURUSD: {'✅ SUCCESS' if results['EURUSD'] else '❌ FAILED'}")
    print(f"XAUUSD: {'✅ SUCCESS' if results['XAUUSD'] else '❌ FAILED'}")
    print(f"\nTotal Time: {total_elapsed/60:.1f} minutes")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    sys.exit(0 if all(results.values()) else 1)
