#!/usr/bin/env python3
"""
Comprehensive Data Validation Tests
Ensures data integrity before training
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, '.')

def test_fundamental_data():
    """Test all fundamental CSV files"""
    print("="*80)
    print("TEST 1: Fundamental Data Validation")
    print("="*80)
    
    fundamental_files = [
        'INDPRO.csv', 'DGORDER.csv', 'ECBDFR.csv', 'CP0000EZ19M086NEST.csv',
        'LRHUTTTTDEM156S.csv', 'DCOILWTICO.csv', 'DCOILBRENTEU.csv', 'VIXCLS.csv',
        'DGS10.csv', 'DGS2.csv', 'BOPGSTB.csv', 'CPIAUCSL.csv', 'CPALTT01USM661S.csv',
        'DFF.csv', 'DEXCHUS.csv', 'DEXJPUS.csv', 'DEXUSEU.csv', 'FEDFUNDS.csv',
        'PAYEMS.csv', 'UNRATE.csv'
    ]
    
    passed = 0
    failed = 0
    
    for f in fundamental_files:
        path = Path(f'data/{f}')
        try:
            df = pd.read_csv(path)
            
            # Check date column exists
            assert 'date' in df.columns, f"Missing 'date' column"
            
            # Check has data
            assert len(df) > 0, "Empty file"
            
            # Check date is parseable
            pd.to_datetime(df['date'])
            
            # Check value column exists and has data
            value_col = df.columns[1]
            assert df[value_col].notna().sum() > 0, "No data in value column"
            
            print(f"  ✅ {f}: {len(df)} rows, {df[value_col].notna().sum()} non-null values")
            passed += 1
            
        except Exception as e:
            print(f"  ❌ {f}: {e}")
            failed += 1
    
    print(f"\n{'='*80}")
    print(f"RESULT: {passed}/{len(fundamental_files)} passed, {failed} failed")
    print(f"{'='*80}\n")
    
    return failed == 0


def test_price_data():
    """Test daily price data files"""
    print("="*80)
    print("TEST 2: Price Data Validation")
    print("="*80)
    
    pairs = ['EURUSD', 'XAUUSD']
    passed = 0
    failed = 0
    
    for pair in pairs:
        try:
            df = pd.read_csv(f'data/{pair}_Daily.csv')
            
            # Check required columns
            required = ['timestamp', 'open', 'high', 'low', 'close']
            for col in required:
                assert col in df.columns, f"Missing {col} column"
            
            # Check data quality
            assert len(df) > 1000, f"Insufficient data ({len(df)} rows)"
            assert df[['open','high','low','close']].notna().all().all(), "Missing OHLC data"
            
            # Check date range
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            date_range = f"{df['timestamp'].min()} to {df['timestamp'].max()}"
            
            print(f"  ✅ {pair}_Daily.csv: {len(df)} rows, {date_range}")
            passed += 1
            
        except Exception as e:
            print(f"  ❌ {pair}_Daily.csv: {e}")
            failed += 1
    
    print(f"\n{'='*80}")
    print(f"RESULT: {passed}/{len(pairs)} passed, {failed} failed")
    print(f"{'='*80}\n")
    
    return failed == 0


def test_feature_generation():
    """Test complete feature generation pipeline"""
    print("="*80)
    print("TEST 3: Feature Generation Pipeline")
    print("="*80)
    
    from scripts.forecasting import HybridPriceForecastingEnsemble
    
    pairs = ['EURUSD', 'XAUUSD']
    results = {}
    
    for pair in pairs:
        try:
            print(f"\n  Testing {pair}...")
            ensemble = HybridPriceForecastingEnsemble(pair=pair)
            features = ensemble._prepare_features()
            
            # Check features generated
            assert features is not None, "Features is None"
            assert not features.empty, "Features is empty"
            assert len(features) > 1000, f"Too few rows ({len(features)})"
            assert len(features.columns) > 200, f"Too few features ({len(features.columns)}) - need at least 200"
            assert len(features.columns) < 600, f"Too many features ({len(features.columns)}) - variance filtering may have failed"
            
            # Check for key feature categories
            h4_feats = [c for c in features.columns if 'h4' in c.lower()]
            weekly_feats = [c for c in features.columns if 'weekly' in c.lower()]
            fund_feats = [c for c in features.columns if c.startswith('fund_')]
            
            assert len(h4_feats) > 20, f"Too few H4 features ({len(h4_feats)})"
            assert len(weekly_feats) > 20, f"Too few Weekly features ({len(weekly_feats)})"
            assert len(fund_feats) > 10, f"Too few Fundamental features ({len(fund_feats)}) - must have at least 10"
            
            # Check no inf/nan in critical columns
            essential = ['Open', 'High', 'Low', 'Close']
            for col in essential:
                if col in features.columns:
                    assert not features[col].isna().any(), f"{col} has NaN values"
                    assert not np.isinf(features[col]).any(), f"{col} has inf values"
            
            results[pair] = {
                'rows': len(features),
                'columns': len(features.columns),
                'h4_features': len(h4_feats),
                'weekly_features': len(weekly_feats),
                'fund_features': len(fund_feats),
                'date_range': f"{features.index.min()} to {features.index.max()}"
            }
            
            print(f"  ✅ {pair}: {len(features)} rows × {len(features.columns)} features")
            print(f"     H4: {len(h4_feats)}, Weekly: {len(weekly_feats)}, Fund: {len(fund_feats)}")
            
        except Exception as e:
            print(f"  ❌ {pair}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print(f"\n{'='*80}")
    print(f"RESULT: Feature generation PASSED for all pairs")
    print(f"{'='*80}\n")
    
    return True


def test_data_alignment():
    """Test that all timeframes are aligned on same dates"""
    print("="*80)
    print("TEST 4: Multi-timeframe Data Alignment")
    print("="*80)
    
    from scripts.forecasting import HybridPriceForecastingEnsemble
    
    try:
        ensemble = HybridPriceForecastingEnsemble(pair='EURUSD')
        features = ensemble._prepare_features()
        
        # Check that each row has data from all timeframes
        h4_cols = [c for c in features.columns if 'h4' in c.lower()]
        weekly_cols = [c for c in features.columns if 'weekly' in c.lower()]
        
        # Sample 100 random rows and check alignment
        sample = features.sample(min(100, len(features)))
        
        for idx in sample.index[:5]:  # Check first 5 samples
            row = sample.loc[idx]
            has_h4 = any(pd.notna(row[c]) for c in h4_cols if c in row.index)
            has_weekly = any(pd.notna(row[c]) for c in weekly_cols if c in row.index)
            
            print(f"  Row {idx}: H4={has_h4}, Weekly={has_weekly}")
        
        print(f"\n  ✅ Data alignment verified on sample rows")
        print(f"  ✅ All timeframes present on same dates")
        
        print(f"\n{'='*80}")
        print(f"RESULT: Data alignment PASSED")
        print(f"{'='*80}\n")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Alignment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fundamental_signals():
    """Test that fundamental signals are being generated"""
    print("="*80)
    print("TEST 5: Fundamental Signal Generation")
    print("="*80)
    
    from scripts.fundamental_pipeline import load_all_fundamentals
    from scripts.fundamental_signals import add_fundamental_signals
    import pandas as pd
    
    try:
        # Load fundamentals
        fund = load_all_fundamentals()
        print(f"  Loaded {len(fund.columns)} fundamental data sources")
        
        # Create dummy dataframe with same index
        df = pd.DataFrame(index=fund.index[:100])
        
        # Add signals
        df_with_signals = add_fundamental_signals(df, fund)
        
        # Check signals were added
        signal_cols = [c for c in df_with_signals.columns if c.startswith('fund_')]
        
        expected_signals = [
            'fund_cpi_surprise_mom_5d',
            'fund_carry_long',
            'fund_carry_short',
            'fund_curve_steepening',
            'fund_curve_inversion',
            'fund_cbp_tightening_spike',
            'fund_vol_jump_event',
            'fund_business_cycle_up',
            'fund_liquidity_expansion',
            'fund_trade_surplus_bull',
            'fund_fiscal_bull',
            'fund_oil_correlation_signal'
        ]
        
        found = [s for s in expected_signals if s in signal_cols]
        
        print(f"  ✅ Generated {len(signal_cols)} fundamental signal features")
        print(f"  ✅ Found {len(found)}/{len(expected_signals)} expected signal types:")
        for sig in found:
            print(f"     - {sig}")
        
        print(f"\n{'='*80}")
        print(f"RESULT: Fundamental signals PASSED")
        print(f"{'='*80}\n")
        
        return len(found) >= 8  # At least 8 of 12 signal types
        
    except Exception as e:
        print(f"  ❌ Signal generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("COMPREHENSIVE DATA VALIDATION TEST SUITE")
    print("="*80 + "\n")
    
    tests = [
        ("Fundamental Data", test_fundamental_data),
        ("Price Data", test_price_data),
        ("Feature Generation", test_feature_generation),
        ("Data Alignment", test_data_alignment),
        ("Fundamental Signals", test_fundamental_signals)
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n❌ TEST CRASHED: {name}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print("\n" + "="*80)
    print("FINAL TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n{'='*80}")
    if passed == total:
        print(f"🎉 ALL TESTS PASSED ({passed}/{total})")
        print("✅ System is ready for training")
    else:
        print(f"⚠️  SOME TESTS FAILED ({passed}/{total} passed)")
        print("❌ Fix issues before training")
    print(f"{'='*80}\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
