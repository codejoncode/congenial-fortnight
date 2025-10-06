#!/usr/bin/env python3
"""
Full Training Script - Codespaces Safe
Prevents timeouts by running training with progress updates and keeping connection alive
"""
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# Configure logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/full_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def keepalive_print(message):
    """Print with timestamp to keep connection alive"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {message}", flush=True)

def train_pair_with_progress(pair: str):
    """Train a single pair with progress updates"""
    keepalive_print(f"\n{'='*80}")
    keepalive_print(f"🚀 Starting training for {pair}")
    keepalive_print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Import here to avoid loading before logging is set up
        from scripts.forecasting import HybridPriceForecastingEnsemble
        
        keepalive_print(f"📊 Initializing ensemble for {pair}...")
        ensemble = HybridPriceForecastingEnsemble(pair, data_dir='data', models_dir='models')
        
        keepalive_print(f"✅ Ensemble initialized")
        keepalive_print(f"   - Price data: {ensemble.price_data.shape}")
        keepalive_print(f"   - Fundamental data: {ensemble.fundamental_data.shape}")
        
        keepalive_print(f"\n🔧 Engineering features for {pair}...")
        features = ensemble._prepare_features()
        keepalive_print(f"✅ Features engineered: {features.shape[0]} observations, {features.shape[1]} features")
        
        if features.empty:
            keepalive_print(f"❌ ERROR: No features generated for {pair}")
            return False
        
        keepalive_print(f"\n🤖 Training ensemble models for {pair}...")
        keepalive_print(f"   Models: LightGBM, XGBoost, RandomForest, ExtraTrees")
        
        # Train with progress updates
        ensemble.train_full_ensemble()
        
        elapsed = time.time() - start_time
        keepalive_print(f"\n✅ {pair} training completed in {elapsed/60:.1f} minutes")
        
        # Signal evaluation
        keepalive_print(f"\n📊 Running signal evaluation for {pair}...")
        eval_csv = f"{pair}_signal_evaluation.csv"
        ensemble.evaluate_signal_features(features, target_col='target_1d', output_csv=eval_csv)
        keepalive_print(f"✅ Signal evaluation saved to {eval_csv}")
        
        return True
        
    except Exception as e:
        elapsed = time.time() - start_time
        keepalive_print(f"❌ ERROR training {pair} after {elapsed/60:.1f} minutes: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    keepalive_print("="*80)
    keepalive_print("🎯 FULL TRAINING PIPELINE - CODESPACES SAFE")
    keepalive_print("="*80)
    keepalive_print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Ensure directories exist
    Path('logs').mkdir(exist_ok=True)
    Path('models').mkdir(exist_ok=True)
    
    # Training configuration
    pairs = ['EURUSD', 'XAUUSD']
    total_start = time.time()
    
    results = {}
    
    for i, pair in enumerate(pairs, 1):
        keepalive_print(f"\n{'='*80}")
        keepalive_print(f"📍 PAIR {i}/{len(pairs)}: {pair}")
        keepalive_print(f"{'='*80}")
        
        # Heartbeat to keep connection alive
        keepalive_print(f"⏰ Training progress: {i-1}/{len(pairs)} pairs completed")
        
        success = train_pair_with_progress(pair)
        results[pair] = success
        
        if not success:
            keepalive_print(f"⚠️  Warning: {pair} training failed, continuing to next pair...")
    
    # Summary
    total_elapsed = time.time() - total_start
    keepalive_print("\n" + "="*80)
    keepalive_print("📊 TRAINING SUMMARY")
    keepalive_print("="*80)
    
    for pair, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        keepalive_print(f"   {pair}: {status}")
    
    successful = sum(results.values())
    keepalive_print(f"\n   Total: {successful}/{len(pairs)} pairs trained successfully")
    keepalive_print(f"   Total time: {total_elapsed/60:.1f} minutes")
    keepalive_print(f"   End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    keepalive_print("="*80)
    
    if successful == len(pairs):
        keepalive_print("\n🎉 ALL TRAINING COMPLETED SUCCESSFULLY! 🎉")
        sys.exit(0)
    else:
        keepalive_print(f"\n⚠️  Training completed with {len(pairs)-successful} failures")
        sys.exit(1)

if __name__ == "__main__":
    main()
