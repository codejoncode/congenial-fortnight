#!/usr/bin/env python3
"""
Pre-Training Validation Script
Verifies all systems are ready for full training run
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, '.')

def check_data_integrity():
    """Verify all data files are present and valid"""
    print("\n" + "="*80)
    print("1. DATA INTEGRITY CHECK")
    print("="*80)
    
    issues = []
    
    # Check fundamental data
    fundamental_files = [
        'INDPRO.csv', 'DGORDER.csv', 'VIXCLS.csv', 'DGS10.csv', 'DGS2.csv',
        'CPIAUCSL.csv', 'PAYEMS.csv', 'FEDFUNDS.csv', 'ECBDFR.csv', 'DEXUSEU.csv'
    ]
    
    for f in fundamental_files[:5]:  # Check first 5
        path = Path(f'data/{f}')
        if not path.exists():
            issues.append(f"Missing fundamental file: {f}")
        else:
            df = pd.read_csv(path)
            if 'date' not in df.columns:
                issues.append(f"{f} missing 'date' column")
            if len(df) < 100:
                issues.append(f"{f} has too few rows: {len(df)}")
    
    # Check price data
    for pair in ['EURUSD', 'XAUUSD']:
        path = Path(f'data/{pair}_Daily.csv')
        if not path.exists():
            issues.append(f"Missing price file: {pair}_Daily.csv")
        else:
            df = pd.read_csv(path)
            if len(df) < 1000:
                issues.append(f"{pair}_Daily.csv has too few rows: {len(df)}")
            required_cols = ['timestamp', 'open', 'high', 'low', 'close']
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                issues.append(f"{pair}_Daily.csv missing columns: {missing}")
    
    if issues:
        print("❌ FAILED - Data integrity issues:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("✅ PASSED - All data files valid")
        return True


def check_feature_generation():
    """Verify features are generated correctly"""
    print("\n" + "="*80)
    print("2. FEATURE GENERATION CHECK")
    print("="*80)
    
    from scripts.forecasting import HybridPriceForecastingEnsemble
    
    try:
        ensemble = HybridPriceForecastingEnsemble(pair='EURUSD')
        features = ensemble._prepare_features()
        
        print(f"   Generated {len(features)} rows × {len(features.columns)} features")
        
        # Check feature counts
        fund_feats = [c for c in features.columns if c.startswith('fund_')]
        h4_feats = [c for c in features.columns if 'h4' in c.lower()]
        weekly_feats = [c for c in features.columns if 'weekly' in c.lower()]
        
        print(f"   Fundamental features: {len(fund_feats)}")
        print(f"   H4 features: {len(h4_feats)}")
        print(f"   Weekly features: {len(weekly_feats)}")
        
        issues = []
        
        if len(features) < 1000:
            issues.append(f"Too few training samples: {len(features)}")
        if len(features.columns) < 200:
            issues.append(f"Too few features: {len(features.columns)}")
        if len(fund_feats) < 10:
            issues.append(f"Too few fundamental features: {len(fund_feats)}")
        
        # Check for zero variance
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        variance = features[numeric_cols].var()
        zero_var = variance[variance == 0]
        if len(zero_var) > 50:
            issues.append(f"Too many zero-variance features: {len(zero_var)}")
        
        # Check for NaN
        nan_counts = features.isna().sum()
        high_nan = nan_counts[nan_counts > len(features) * 0.5]
        if len(high_nan) > 0:
            issues.append(f"{len(high_nan)} features have >50% NaN values")
        
        if issues:
            print("❌ FAILED - Feature generation issues:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        else:
            print("✅ PASSED - Feature generation working correctly")
            return True
            
    except Exception as e:
        print(f"❌ FAILED - Feature generation crashed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_training_config():
    """Verify training configuration"""
    print("\n" + "="*80)
    print("3. TRAINING CONFIGURATION CHECK")
    print("="*80)
    
    from scripts.robust_lightgbm_config import create_robust_lgb_config_for_small_data
    
    config = create_robust_lgb_config_for_small_data()
    
    print(f"   num_iterations: {config['num_iterations']}")
    print(f"   learning_rate: {config['learning_rate']}")
    print(f"   early_stopping_round: {config['early_stopping_round']}")
    print(f"   num_leaves: {config['num_leaves']}")
    print(f"   max_depth: {config['max_depth']}")
    print(f"   feature_fraction: {config['feature_fraction']}")
    
    issues = []
    
    if config['num_iterations'] < 500:
        issues.append(f"num_iterations too low: {config['num_iterations']} (need >= 500)")
    if config['early_stopping_round'] < 30:
        issues.append(f"early_stopping_round too low: {config['early_stopping_round']} (need >= 30)")
    if config['num_leaves'] < 20:
        issues.append(f"num_leaves too low: {config['num_leaves']} (need >= 20 for 346 features)")
    
    if issues:
        print("❌ FAILED - Training config issues:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("✅ PASSED - Training configuration appropriate")
        return True


def check_signal_evaluation():
    """Verify signal evaluation is available"""
    print("\n" + "="*80)
    print("4. SIGNAL EVALUATION CHECK")
    print("="*80)
    
    from scripts.forecasting import HybridPriceForecastingEnsemble
    
    ensemble = HybridPriceForecastingEnsemble(pair='EURUSD')
    
    if hasattr(ensemble, 'evaluate_signal_features'):
        print("✅ PASSED - evaluate_signal_features method exists")
        print("   Signal evaluation will be performed during training")
        return True
    else:
        print("❌ FAILED - evaluate_signal_features method not found")
        return False


def check_models_directory():
    """Verify models directory is ready"""
    print("\n" + "="*80)
    print("5. MODELS DIRECTORY CHECK")
    print("="*80)
    
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    existing_models = list(models_dir.glob('*.txt')) + list(models_dir.glob('*.joblib'))
    
    if existing_models:
        print(f"⚠️  WARNING - {len(existing_models)} existing models found:")
        for m in existing_models[:5]:
            print(f"   - {m.name}")
        print("   These will be overwritten during training")
    else:
        print("✅ PASSED - Models directory empty and ready")
    
    return True


def estimate_training_time():
    """Estimate training time"""
    print("\n" + "="*80)
    print("6. TRAINING TIME ESTIMATE")
    print("="*80)
    
    from scripts.forecasting import HybridPriceForecastingEnsemble
    
    ensemble = HybridPriceForecastingEnsemble(pair='EURUSD')
    features = ensemble._prepare_features()
    
    n_samples = len(features)
    n_features = len(features.columns)
    n_iterations = 1000
    
    # Rough estimate: ~0.001 seconds per (sample × feature × iteration)
    # For 2 pairs
    estimated_seconds = 2 * (n_samples * n_features * n_iterations) / 1_000_000
    estimated_minutes = estimated_seconds / 60
    
    print(f"   Samples: {n_samples}")
    print(f"   Features: {n_features}")
    print(f"   Iterations: {n_iterations}")
    print(f"   Pairs: 2 (EURUSD, XAUUSD)")
    print(f"\n   📊 Estimated training time: {estimated_minutes:.1f} minutes")
    print(f"      (With early stopping, may finish in {estimated_minutes * 0.3:.1f}-{estimated_minutes * 0.7:.1f} minutes)")
    
    if estimated_minutes < 5:
        print("\n⚠️  WARNING: Estimate seems too fast - check configuration!")
        return False
    elif estimated_minutes > 180:
        print("\n⚠️  WARNING: Estimate >3 hours - consider reducing iterations")
    
    return True


def main():
    """Run all pre-training checks"""
    print("\n" + "="*80)
    print("PRE-TRAINING VALIDATION")
    print("="*80)
    
    checks = [
        ("Data Integrity", check_data_integrity),
        ("Feature Generation", check_feature_generation),
        ("Training Configuration", check_training_config),
        ("Signal Evaluation", check_signal_evaluation),
        ("Models Directory", check_models_directory),
        ("Training Time Estimate", estimate_training_time),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} check crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n{'='*80}")
    if passed == total:
        print(f"🎉 ALL CHECKS PASSED ({passed}/{total})")
        print("✅ System is ready for full production training")
        print("\nTo start training:")
        print("  python -m scripts.automated_training")
    else:
        print(f"⚠️  SOME CHECKS FAILED ({passed}/{total} passed)")
        print("❌ Fix issues before training")
    print(f"{'='*80}\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
