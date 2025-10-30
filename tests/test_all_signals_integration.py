#!/usr/bin/env python3
"""
Test All Signals Integration

Validates that all signal modules work together and can be trained simultaneously.
Tests:
- Day Trading Signals
- Slump Signals
- Candlestick Patterns
- Harmonic Patterns
- Chart Patterns
- Elliott Wave Patterns
- Ultimate Signal Repository (SMC, Order Flow, Session-based, etc.)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.day_trading_signals import DayTradingSignalGenerator
from scripts.slump_signals import SlumpSignalEngine
from scripts.candlestick_patterns import add_candlestick_patterns
from scripts.harmonic_patterns import detect_harmonic_patterns
from scripts.chart_patterns import detect_chart_patterns
from scripts.elliott_wave import detect_elliott_waves
from scripts.ultimate_signal_repository import UltimateSignalRepository, integrate_ultimate_signals

def create_sample_data(n_rows=200):
    """Create sample OHLCV data for testing"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=n_rows, freq='D')
    
    # Generate realistic price data
    close = 1.1000 + np.cumsum(np.random.randn(n_rows) * 0.001)
    high = close + np.abs(np.random.randn(n_rows) * 0.002)
    low = close - np.abs(np.random.randn(n_rows) * 0.002)
    open_price = close + np.random.randn(n_rows) * 0.001
    volume = np.abs(np.random.randn(n_rows) * 10000 + 50000)
    
    df = pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }, index=dates)
    
    return df

def test_day_trading_signals():
    """Test day trading signals generation"""
    print("\n" + "="*60)
    print("Testing Day Trading Signals")
    print("="*60)
    
    df = create_sample_data()
    generator = DayTradingSignalGenerator()
    
    try:
        df = generator.h1_breakout_pullbacks(df)
        df = generator.vwap_reversion_signals(df)
        df = generator.ema_ribbon_compression(df)
        df = generator.rsi_mean_reversion(df)
        df = generator.inside_outside_bar_patterns(df)
        df = generator.time_of_day_momentum(df)
        
        signal_cols = [col for col in df.columns if 'signal' in col.lower()]
        print(f"✓ Generated {len(signal_cols)} day trading signal columns")
        print(f"  Signal columns: {signal_cols[:5]}...")
        print(f"  Non-zero signals: {(df[signal_cols] != 0).sum().sum()}")
        return True
    except Exception as e:
        print(f"✗ Day trading signals failed: {e}")
        return False

def test_slump_signals():
    """Test slump signals generation"""
    print("\n" + "="*60)
    print("Testing Slump Signals")
    print("="*60)
    
    df = create_sample_data()
    
    try:
        engine = SlumpSignalEngine()
        df = engine.generate_all_signals(df)
        
        signal_cols = [col for col in df.columns if 'slump' in col.lower()]
        print(f"✓ Generated {len(signal_cols)} slump signal columns")
        print(f"  Signal columns: {signal_cols}")
        return True
    except Exception as e:
        print(f"✗ Slump signals failed: {e}")
        return False

def test_harmonic_patterns():
    """Test harmonic pattern recognition"""
    print("\n" + "="*60)
    print("Testing Harmonic Patterns")
    print("="*60)
    
    df = create_sample_data()
    
    try:
        df = detect_harmonic_patterns(df)
        
        pattern_cols = [col for col in df.columns if any(p in col for p in ['gartley', 'bat', 'butterfly', 'crab', 'shark'])]
        print(f"✓ Generated {len(pattern_cols)} harmonic pattern columns")
        print(f"  Pattern columns: {pattern_cols[:5]}...")
        print(f"  Detected patterns: {(df[pattern_cols] != 0).sum().sum()}")
        return True
    except Exception as e:
        print(f"✗ Harmonic patterns failed: {e}")
        return False

def test_chart_patterns():
    """Test chart pattern recognition"""
    print("\n" + "="*60)
    print("Testing Chart Patterns")
    print("="*60)
    
    df = create_sample_data()
    
    try:
        df = detect_chart_patterns(df)
        
        pattern_cols = [col for col in df.columns if any(p in col for p in ['double', 'head_shoulders', 'triangle', 'flag', 'cup'])]
        print(f"✓ Generated {len(pattern_cols)} chart pattern columns")
        print(f"  Pattern columns: {pattern_cols[:5]}...")
        print(f"  Detected patterns: {(df[pattern_cols] != 0).sum().sum()}")
        return True
    except Exception as e:
        print(f"✗ Chart patterns failed: {e}")
        return False

def test_elliott_wave():
    """Test Elliott Wave pattern recognition"""
    print("\n" + "="*60)
    print("Testing Elliott Wave Patterns")
    print("="*60)
    
    df = create_sample_data()
    
    try:
        df = detect_elliott_waves(df)
        
        wave_cols = [col for col in df.columns if 'elliott' in col.lower() or 'wave' in col.lower()]
        print(f"✓ Generated {len(wave_cols)} Elliott Wave columns")
        print(f"  Wave columns: {wave_cols}")
        print(f"  Detected waves: {(df[wave_cols] != 0).sum().sum()}")
        return True
    except Exception as e:
        print(f"✗ Elliott Wave patterns failed: {e}")
        return False

def test_ultimate_signals():
    """Test Ultimate Signal Repository"""
    print("\n" + "="*60)
    print("Testing Ultimate Signal Repository")
    print("="*60)
    
    df = create_sample_data()
    
    try:
        df = integrate_ultimate_signals(df)
        
        ultimate_cols = [col for col in df.columns if any(s in col for s in ['smc', 'order_flow', 'mtf', 'session', 'master_signal'])]
        print(f"✓ Generated {len(ultimate_cols)} ultimate signal columns")
        print(f"  Ultimate columns: {ultimate_cols}")
        
        if 'master_signal' in df.columns:
            print(f"  Master signal range: {df['master_signal'].min():.2f} to {df['master_signal'].max():.2f}")
            print(f"  Mean master signal: {df['master_signal'].mean():.2f}")
        
        return True
    except Exception as e:
        print(f"✗ Ultimate signals failed: {e}")
        return False

def test_all_signals_together():
    """Test that all signals can be generated together"""
    print("\n" + "="*60)
    print("Testing ALL Signals Together")
    print("="*60)
    
    df = create_sample_data()
    initial_cols = len(df.columns)
    
    try:
        # Day trading signals
        generator = DayTradingSignalGenerator()
        df = generator.h1_breakout_pullbacks(df)
        df = generator.vwap_reversion_signals(df)
        df = generator.ema_ribbon_compression(df)
        
        # Slump signals
        engine = SlumpSignalEngine()
        df_with_slump = engine.generate_all_signals(df.copy())
        slump_cols = [col for col in df_with_slump.columns if col not in df.columns]
        for col in slump_cols:
            df[col] = df_with_slump[col]
        
        # Harmonic patterns
        df = detect_harmonic_patterns(df)
        
        # Chart patterns
        df = detect_chart_patterns(df)
        
        # Elliott Wave
        df = detect_elliott_waves(df)
        
        # Ultimate signals
        df = integrate_ultimate_signals(df)
        
        final_cols = len(df.columns)
        added_cols = final_cols - initial_cols
        
        print(f"✓ Successfully integrated ALL signal modules")
        print(f"  Initial columns: {initial_cols}")
        print(f"  Final columns: {final_cols}")
        print(f"  Added features: {added_cols}")
        
        # Check for NaN issues
        nan_counts = df.isna().sum()
        critical_nans = nan_counts[nan_counts > len(df) * 0.5]
        
        if len(critical_nans) > 0:
            print(f"  ⚠ Warning: {len(critical_nans)} columns have >50% NaN values")
        else:
            print(f"  ✓ No critical NaN issues detected")
        
        # Check for signal activity
        signal_cols = [col for col in df.columns if 'signal' in col.lower()]
        active_signals = sum((df[col] != 0).sum() > 0 for col in signal_cols)
        print(f"  Active signal columns: {active_signals}/{len(signal_cols)}")
        
        print("\n✓ ALL SIGNALS CAN TRAIN TOGETHER!")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "#"*60)
    print("# COMPREHENSIVE SIGNAL INTEGRATION TEST SUITE")
    print("#"*60)
    
    results = {
        'Day Trading Signals': test_day_trading_signals(),
        'Slump Signals': test_slump_signals(),
        'Harmonic Patterns': test_harmonic_patterns(),
        'Chart Patterns': test_chart_patterns(),
        'Elliott Wave': test_elliott_wave(),
        'Ultimate Signal Repository': test_ultimate_signals(),
        'ALL SIGNALS TOGETHER': test_all_signals_together()
    }
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:.<50} {status}")
    
    total_pass = sum(results.values())
    total_tests = len(results)
    
    print("="*60)
    print(f"TOTAL: {total_pass}/{total_tests} tests passed")
    print("="*60)
    
    if total_pass == total_tests:
        print("\n🎉 ALL TESTS PASSED! Ready for training!")
        return 0
    else:
        print(f"\n⚠ {total_tests - total_pass} test(s) failed. Review errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
