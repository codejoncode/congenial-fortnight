#!/usr/bin/env python3
"""
Training Diagnostic Script

Verifies that the forecasting system is ready for training:
- Forecasting class instantiation
- Feature engineering produces non-empty DataFrame
- Fundamentals attach when FRED key is available
- Data loading and consolidation work
"""

import os
import sys
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

# Load .env if present
env_path = BASE_DIR / '.env'
if env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
        print("✅ Loaded .env file")
    except ImportError:
        print("⚠️  python-dotenv not available, skipping .env load")

from scripts.forecasting import HybridPriceForecastingEnsemble

def run_diagnostic():
    """Run comprehensive diagnostic checks"""
    print("🔍 TRAINING DIAGNOSTIC")
    print("=" * 50)

    pair = 'EURUSD'


    try:
        print(f"1. Instantiating forecasting system for {pair}...")
        fs = HybridPriceForecastingEnsemble(pair)
        print("✅ Forecasting system instantiated successfully")

        print("\n2. Checking data loads (all timeframes)...")
        # Report all timeframes with user-specified labels
        tf_labels = [
            ("4H", getattr(fs, 'intraday_data', None)),
            ("Daily", getattr(fs, 'daily_data', None)),
            ("Weekly", getattr(fs, 'weekly_data', None)),
            ("Monthly", getattr(fs, 'monthly_data', None)),
        ]

        for label, df in tf_labels:
            shape = df.shape if df is not None and hasattr(df, 'shape') else 'None'
            status = "✅" if (df is not None and hasattr(df, 'empty') and not df.empty) else "⚠️  MISSING/EMPTY"
            print(f"   {label} data: {shape} {status}")
            # For Daily data, print the first few rows to verify content
            if label == "Daily" and df is not None and hasattr(df, 'head'):
                print("   Daily data sample:")
                print(df.head(5))

        # Also check price_data for completeness
        price_shape = fs.price_data.shape if hasattr(fs, 'price_data') and fs.price_data is not None else 'None'
        price_status = "✅" if (hasattr(fs, 'price_data') and fs.price_data is not None and not fs.price_data.empty) else "❌ EMPTY"
        print(f"   Consolidated price data: {price_shape} {price_status}")

        if fs.price_data is None or fs.price_data.empty:
            print("❌ Price data is empty - cannot proceed")
            return False

        print("\n3. Checking fundamentals...")
        # Check if FRED_API_KEY is loaded
        fred_key = os.getenv('FRED_API_KEY')
        if not fred_key:
            print("⚠️  FRED_API_KEY not loaded from environment/.env - fundamentals will not be available!")
        fundamentals = fs.fundamental_data if hasattr(fs, 'fundamental_data') else None
        if fundamentals is not None and not fundamentals.empty:
            print(f"✅ Fundamentals loaded: {fundamentals.shape}")
            fund_cols = [c for c in fundamentals.columns if c.startswith('fund_') or c.lower() not in ['date']]
            print(f"   Fundamental columns: {len(fund_cols)}")
            print(f"   Fundamental column names: {fund_cols}")
        else:
            print("⚠️  No fundamentals loaded (check FRED_API_KEY and .env setup)")

        print("\n4. Testing feature engineering...")
        feature_df = fs._prepare_features()
        if feature_df is not None and not feature_df.empty:
            print(f"✅ Feature engineering successful: {feature_df.shape}")
            print(f"   Total features: {len(feature_df.columns)}")
            target_cols = [c for c in feature_df.columns if 'target' in c]
            print(f"   Target columns: {target_cols}")
            fund_features = [c for c in feature_df.columns if c.startswith('fund_')]
            print(f"   Fundamental features: {len(fund_features)}")
            print(f"   Sample features: {list(feature_df.columns[:10])}")
        else:
            print("❌ Feature engineering failed - empty DataFrame")
            return False

        print("\n5. Checking for required methods...")
        required_methods = ['_get_cross_pair', '_load_daily_price_file', '_build_intraday_context', '_calculate_rsi']
        missing = []
        for method in required_methods:
            if not hasattr(fs, method):
                missing.append(method)
        if missing:
            print(f"⚠️  Missing methods: {missing}")
        else:
            print("✅ All required methods present")

        print("\n6. Data quality check...")
        if len(feature_df) < 100:
            print(f"⚠️  Small dataset: {len(feature_df)} rows")
        else:
            print(f"✅ Sufficient data: {len(feature_df)} rows")

        na_pct = feature_df.isnull().mean().mean() * 100
        if na_pct > 10:
            print(f"⚠️  High NA%: {na_pct:.1f}%")
        else:
            print(f"✅ Low NA%: {na_pct:.1f}%")

        print("\n🎉 DIAGNOSTIC COMPLETE - System appears ready for training!")
        return True

    except Exception as e:
        print(f"❌ Diagnostic failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_diagnostic()
    sys.exit(0 if success else 1)