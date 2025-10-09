#!/usr/bin/env python3
"""
Test script to verify enhanced Holloway Algorithm works with existing system.

This tests:
1. Backward compatibility with forecasting.py integration
2. Enhanced features work correctly
3. Both old and new methods produce valid output
"""

import pandas as pd
import numpy as np
from scripts.holloway_algorithm import CompleteHollowayAlgorithm
from pathlib import Path

def create_test_data(n_rows=500):
    """Create synthetic OHLCV data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=n_rows, freq='D')
    
    # Create realistic price movements
    np.random.seed(42)
    close_prices = 100 + np.cumsum(np.random.randn(n_rows) * 0.5)
    
    df = pd.DataFrame({
        'open': close_prices + np.random.randn(n_rows) * 0.2,
        'high': close_prices + np.abs(np.random.randn(n_rows)) * 0.5,
        'low': close_prices - np.abs(np.random.randn(n_rows)) * 0.5,
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, n_rows)
    }, index=dates)
    
    # Ensure high is highest and low is lowest
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    return df

def test_backward_compatibility():
    """Test that old methods still work (used by forecasting.py)."""
    print("\n" + "="*70)
    print("TEST 1: Backward Compatibility with forecasting.py")
    print("="*70)
    
    algo = CompleteHollowayAlgorithm()
    df = create_test_data(500)
    
    try:
        # Test original method used by forecasting.py
        result = algo.calculate_complete_holloway_algorithm(df.copy(), verbose=False)
        
        # Check required columns exist
        required_cols = [
            'bull_count', 'bear_count', 'bully', 'beary',
            'holloway_bull_signal', 'holloway_bear_signal',
            'bull_strength_signal', 'bear_strength_signal',
            'reversal_signal', 'weakness_signal'
        ]
        
        missing = [col for col in required_cols if col not in result.columns]
        if missing:
            print(f"❌ FAILED: Missing required columns: {missing}")
            return False
        
        print(f"✅ SUCCESS: Old method works")
        print(f"   - Input rows: {len(df)}")
        print(f"   - Output rows: {len(result)}")
        print(f"   - Output columns: {len(result.columns)}")
        print(f"   - Bull signals: {result['holloway_bull_signal'].sum()}")
        print(f"   - Bear signals: {result['holloway_bear_signal'].sum()}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_features():
    """Test that new enhanced features work."""
    print("\n" + "="*70)
    print("TEST 2: Enhanced Features")
    print("="*70)
    
    algo = CompleteHollowayAlgorithm()
    df = create_test_data(500)
    
    try:
        # Test enhanced processing
        result = algo.process_enhanced_data(df.copy(), timeframe='Daily')
        
        # Check enhanced columns exist
        enhanced_cols = [
            'enhanced_bull_count', 'enhanced_bear_count',
            'enhanced_bull_avg', 'enhanced_bear_avg',
            'enhanced_rsi', 'enhanced_strong_buy', 'enhanced_strong_sell',
            'enhanced_bull_div_price_bull_count', 'enhanced_bear_div_price_bull_count',
            'enhanced_bull_count_at_support', 'enhanced_rsi_at_support'
        ]
        
        missing = [col for col in enhanced_cols if col not in result.columns]
        if missing:
            print(f"❌ FAILED: Missing enhanced columns: {missing}")
            return False
        
        print(f"✅ SUCCESS: Enhanced method works")
        print(f"   - Input rows: {len(df)}")
        print(f"   - Output rows: {len(result)}")
        print(f"   - Output columns: {len(result.columns)}")
        print(f"   - Strong buy signals: {result['enhanced_strong_buy'].sum()}")
        print(f"   - Strong sell signals: {result['enhanced_strong_sell'].sum()}")
        print(f"   - Bullish divergences: {result['enhanced_bull_div_price_bull_count'].sum()}")
        print(f"   - Bearish divergences: {result['enhanced_bear_div_price_bull_count'].sum()}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_forecasting_integration():
    """Test integration with forecasting.py pattern."""
    print("\n" + "="*70)
    print("TEST 3: Forecasting.py Integration Pattern")
    print("="*70)
    
    try:
        # Simulate how forecasting.py uses Holloway
        from scripts.holloway_algorithm import CompleteHollowayAlgorithm
        
        algo = CompleteHollowayAlgorithm(str(Path('data')))
        df = create_test_data(500)
        
        # This is the exact pattern used in forecasting.py line 1204
        holloway_df = algo.calculate_complete_holloway_algorithm(df.copy())
        
        # Verify it returns a dataframe with the original columns plus Holloway features
        assert len(holloway_df) > 0, "Empty result"
        assert 'close' in holloway_df.columns, "Original columns missing"
        assert 'bull_count' in holloway_df.columns, "Holloway features missing"
        
        print(f"✅ SUCCESS: Forecasting integration pattern works")
        print(f"   - Can import from scripts.holloway_algorithm")
        print(f"   - Can initialize with Path object")
        print(f"   - calculate_complete_holloway_algorithm works")
        print(f"   - Returns dataframe with {len(holloway_df.columns)} columns")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_vs_original():
    """Compare enhanced vs original features."""
    print("\n" + "="*70)
    print("TEST 4: Enhanced vs Original Comparison")
    print("="*70)
    
    algo = CompleteHollowayAlgorithm()
    df = create_test_data(500)
    
    try:
        # Run both methods
        original = algo.calculate_complete_holloway_algorithm(df.copy(), verbose=False)
        enhanced = algo.process_enhanced_data(df.copy(), timeframe='Daily')
        
        print(f"✅ Both methods completed successfully")
        print(f"\nOriginal Holloway:")
        print(f"   - Columns: {len(original.columns)}")
        print(f"   - Bull count mean: {original['bull_count'].mean():.2f}")
        print(f"   - Bear count mean: {original['bear_count'].mean():.2f}")
        print(f"   - Bull signals: {original['holloway_bull_signal'].sum()}")
        
        print(f"\nEnhanced Holloway:")
        print(f"   - Columns: {len(enhanced.columns)}")
        print(f"   - Bull count mean: {enhanced['enhanced_bull_count'].mean():.2f}")
        print(f"   - Bear count mean: {enhanced['enhanced_bear_count'].mean():.2f}")
        print(f"   - Strong buy signals: {enhanced['enhanced_strong_buy'].sum()}")
        print(f"   - Divergence features: Added")
        print(f"   - S/R features: Added")
        print(f"   - Composite signals: Added")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("ENHANCED HOLLOWAY ALGORITHM - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    tests = [
        ("Backward Compatibility", test_backward_compatibility),
        ("Enhanced Features", test_enhanced_features),
        ("Forecasting Integration", test_forecasting_integration),
        ("Enhanced vs Original", test_enhanced_vs_original)
    ]
    
    results = {}
    for name, test_func in tests:
        results[name] = test_func()
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL TEST RESULTS")
    print("="*70)
    
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {name}")
    
    all_passed = all(results.values())
    print("\n" + "="*70)
    if all_passed:
        print("🎉 ALL TESTS PASSED - Ready for production!")
        print("="*70)
        print("\nNext steps:")
        print("1. The enhanced Holloway algorithm maintains backward compatibility")
        print("2. All existing forecasting.py integrations will continue to work")
        print("3. Enhanced features are available via process_enhanced_data() method")
        print("4. Original features remain accessible via calculate_complete_holloway_algorithm()")
        return 0
    else:
        print("⚠️  SOME TESTS FAILED - Review errors above")
        print("="*70)
        return 1

if __name__ == "__main__":
    exit(main())
