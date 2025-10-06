#!/usr/bin/env python3
"""
Ultra-Simple Training - Just LightGBM, No Complexity
"""
import sys
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_pair_simple(pair: str):
    """Train a single pair with minimal complexity"""
    print(f"\n{'='*80}")
    print(f"🚀 TRAINING {pair} - {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # 1. Initialize
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading data...")
        from scripts.forecasting import HybridPriceForecastingEnsemble
        ensemble = HybridPriceForecastingEnsemble(pair, data_dir='data', models_dir='models')
        print(f"✅ Price: {ensemble.price_data.shape}, Fund: {ensemble.fundamental_data.shape}")
        
        # 2. Prepare features
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Engineering features...")
        X_train, y_train, X_val, y_val = ensemble.load_and_prepare_datasets()
        
        if X_train is None or len(X_train) == 0:
            print("❌ No training data")
            return None
            
        print(f"✅ Train: {X_train.shape}, Val: {X_val.shape}")
        print(f"   Target balance: {y_train.mean():.1%} bull")
        
        # 3. Train LightGBM with simple config
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Training LightGBM...")
        
        model = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbosity=-1,
            force_col_wise=True,
            n_jobs=-1
        )
        
        # Fit with eval set
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)] if X_val is not None else None,
            eval_metric='binary_logloss'
        )
        
        # 4. Evaluate
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Evaluating...")
        
        y_train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        
        if X_val is not None and y_val is not None:
            y_val_pred = model.predict(X_val)
            val_acc = accuracy_score(y_val, y_val_pred)
        else:
            val_acc = None
        
        # 5. Save model
        model_path = Path('models') / f'{pair}_lightgbm_simple.joblib'
        joblib.dump(model, model_path)
        
        elapsed = time.time() - start_time
        
        # Results
        print(f"\n{'='*80}")
        print(f"✅ {pair} COMPLETE - {elapsed/60:.1f} minutes")
        print(f"{'='*80}")
        print(f"   Train Accuracy: {train_acc:.2%}")
        if val_acc:
            print(f"   Val Accuracy:   {val_acc:.2%}")
            print(f"   Improvement:    {val_acc - 0.517:.2%} (from 51.7% baseline)")
        print(f"   Model saved:    {model_path}")
        print(f"{'='*80}\n")
        
        return {
            'pair': pair,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'elapsed_minutes': elapsed/60,
            'model_path': str(model_path)
        }
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ ERROR after {elapsed/60:.1f} minutes: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    total_start = time.time()
    
    print("="*80)
    print("🎯 SIMPLE FOREX ML TRAINING")
    print("="*80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Ensure directories
    Path('logs').mkdir(exist_ok=True)
    Path('models').mkdir(exist_ok=True)
    
    # Train pairs
    results = []
    for pair in ['EURUSD', 'XAUUSD']:
        result = train_pair_simple(pair)
        if result:
            results.append(result)
    
    # Summary
    total_elapsed = time.time() - total_start
    print("\n" + "="*80)
    print("📊 FINAL SUMMARY")
    print("="*80)
    
    for r in results:
        status = f"✅ {r['pair']}: Val {r['val_accuracy']:.2%}" if r['val_accuracy'] else f"✅ {r['pair']}: Trained"
        print(status)
    
    print(f"\nTotal: {len(results)}/2 pairs trained successfully")
    print(f"Time: {total_elapsed/60:.1f} minutes")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    sys.exit(0 if len(results) == 2 else 1)

if __name__ == "__main__":
    main()
