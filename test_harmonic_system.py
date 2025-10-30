#!/usr/bin/env python3
"""
Test Harmonic Pattern System
Run comprehensive tests on pattern detection and trading
"""

import sys
sys.path.append('/workspaces/congenial-fortnight')

import pandas as pd
import numpy as np
from scripts.harmonic_pattern_trader import HarmonicPatternTrader
from backtest_harmonic_patterns import HarmonicPatternBacktest
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_pattern_detection():
    """Test pattern detection with different parameters"""
    print("\n" + "="*80)
    print("TEST 1: PATTERN DETECTION WITH DIFFERENT TOLERANCES")
    print("="*80)
    
    # Load data
    df = pd.read_csv('/workspaces/congenial-fortnight/data/EURUSD_H1.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Use recent data (last 5000 bars)
    df_recent = df.tail(5000).reset_index(drop=True)
    
    print(f"Testing on {len(df_recent)} recent bars")
    
    # Test different tolerance levels
    tolerances = [0.10, 0.08, 0.05]
    quality_thresholds = [0.60, 0.65, 0.70]
    
    for fib_tol in tolerances:
        for quality_threshold in quality_thresholds:
            trader = HarmonicPatternTrader(
                lookback=100,
                fib_tolerance=fib_tol,
                min_quality_score=quality_threshold
            )
            
            signals = trader.generate_signals(df_recent)
            
            print(f"\n   Fib Tolerance: {fib_tol:.1%}, Quality Threshold: {quality_threshold:.1%}")
            print(f"   → Patterns detected: {len(signals)}")
            
            if not signals.empty:
                print(f"   → Pattern types: {signals['pattern_type'].unique()}")
                print(f"   → Avg quality: {signals['quality_score'].mean():.2%}")
                print(f"   → Avg R:R (T2): {signals['risk_reward_t2'].mean():.2f}")


def test_synthetic_pattern():
    """Test with synthetic perfect Gartley pattern"""
    print("\n" + "="*80)
    print("TEST 2: SYNTHETIC PERFECT GARTLEY PATTERN")
    print("="*80)
    
    # Create synthetic Gartley pattern
    # X(low) - A(high) - B(low) - C(high) - D(low)
    
    X_price = 1.0500
    A_price = 1.0700  # Move up 200 pips
    B_price = 1.0576  # 0.618 retracement of XA = 1.0700 - (0.618 * 0.0200)
    C_price = 1.0650  # Move up from B
    D_price = 1.0543  # 0.786 retracement of XA = 1.0700 - (0.786 * 0.0200)
    
    # Create price bars
    bars = []
    base_time = pd.Timestamp('2025-01-01')
    
    # Generate smooth transitions between points
    def add_transition(start_price, end_price, num_bars, start_idx):
        """Add smooth transition between two price points"""
        for i in range(num_bars):
            pct = i / num_bars
            price = start_price + (end_price - start_price) * pct
            noise = np.random.normal(0, 0.0001)  # Small noise
            
            bars.append({
                'timestamp': base_time + pd.Timedelta(hours=start_idx + i),
                'open': price + noise,
                'high': price + abs(noise) + 0.0005,
                'low': price - abs(noise) - 0.0005,
                'close': price + noise,
                'volume': 1000
            })
    
    # X to A (uptrend)
    add_transition(X_price, A_price, 20, 0)
    
    # A to B (retracement)
    add_transition(A_price, B_price, 15, 20)
    
    # B to C (continuation)
    add_transition(B_price, C_price, 15, 35)
    
    # C to D (final retracement)
    add_transition(C_price, D_price, 15, 50)
    
    # Add some continuation after D
    add_transition(D_price, D_price + 0.0050, 20, 65)
    
    df_synthetic = pd.DataFrame(bars)
    
    print(f"Created synthetic Gartley pattern:")
    print(f"   X: {X_price:.4f}")
    print(f"   A: {A_price:.4f}")
    print(f"   B: {B_price:.4f} (0.618 retracement)")
    print(f"   C: {C_price:.4f}")
    print(f"   D: {D_price:.4f} (0.786 retracement)")
    
    # Test detection
    trader = HarmonicPatternTrader(
        lookback=80,
        fib_tolerance=0.10,  # More lenient for synthetic
        min_quality_score=0.60
    )
    
    signals = trader.generate_signals(df_synthetic)
    
    print(f"\n   Patterns detected: {len(signals)}")
    
    if not signals.empty:
        for idx, signal in signals.iterrows():
            print(f"\n   ✅ Detected: {signal['pattern_type']}")
            print(f"      Quality: {signal['quality_score']:.2%}")
            print(f"      Entry: {signal['entry']:.4f}")
            print(f"      Stop: {signal['stop_loss']:.4f}")
            print(f"      T1: {signal['target_1']:.4f} (R:R {signal['risk_reward_t1']:.2f})")
            print(f"      T2: {signal['target_2']:.4f} (R:R {signal['risk_reward_t2']:.2f})")
            print(f"      T3: {signal['target_3']:.4f} (R:R {signal['risk_reward_t3']:.2f})")
    else:
        print("   ❌ Pattern not detected - may need to adjust parameters")


def test_backtest_small():
    """Test backtest on small dataset"""
    print("\n" + "="*80)
    print("TEST 3: BACKTEST ON RECENT DATA")
    print("="*80)
    
    # Load recent data
    df = pd.read_csv('/workspaces/congenial-fortnight/data/EURUSD_H1.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Use last 10,000 bars (approx 1 year)
    df_test = df.tail(10000).reset_index(drop=True)
    
    print(f"Testing on {len(df_test)} bars")
    print(f"Period: {df_test.iloc[0]['timestamp']} to {df_test.iloc[-1]['timestamp']}")
    
    # Initialize with more lenient settings
    trader = HarmonicPatternTrader(
        lookback=100,
        fib_tolerance=0.08,  # 8% tolerance
        min_quality_score=0.65  # 65% minimum
    )
    
    backtest = HarmonicPatternBacktest(
        initial_balance=10000,
        risk_per_trade_pct=0.02,
        scale_out_percents=[0.50, 0.30, 0.20]
    )
    
    try:
        results = backtest.run_backtest(df_test, trader)
        backtest.print_summary(results)
        
        # Save results
        filepath = backtest.save_results(results, 'output/harmonic_backtests')
        print(f"\n✅ Results saved to: {filepath}")
        
    except Exception as e:
        logger.error(f"Backtest error: {e}", exc_info=True)
        print(f"❌ Backtest failed: {e}")


def test_pattern_statistics():
    """Show statistics about patterns in the data"""
    print("\n" + "="*80)
    print("TEST 4: PATTERN STATISTICS")
    print("="*80)
    
    df = pd.read_csv('/workspaces/congenial-fortnight/data/EURUSD_H1.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Check different data segments
    segments = {
        'Full history': df,
        'Last year': df.tail(8760),
        'Last 6 months': df.tail(4380),
        'Last 3 months': df.tail(2190)
    }
    
    for segment_name, segment_df in segments.items():
        # Very lenient detection
        trader = HarmonicPatternTrader(
            lookback=100,
            fib_tolerance=0.10,
            min_quality_score=0.50  # Very lenient
        )
        
        signals = trader.generate_signals(segment_df)
        
        print(f"\n   {segment_name}: {len(segment_df)} bars")
        print(f"   → Total patterns: {len(signals)}")
        
        if not signals.empty:
            print(f"   → Pattern types:")
            for pattern_type in signals['pattern_type'].unique():
                count = len(signals[signals['pattern_type'] == pattern_type])
                avg_quality = signals[signals['pattern_type'] == pattern_type]['quality_score'].mean()
                print(f"      • {pattern_type}: {count} (avg quality {avg_quality:.2%})")


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("HARMONIC PATTERN TRADING SYSTEM - COMPREHENSIVE TESTS")
    print("="*80)
    
    try:
        # Test 1: Different parameter combinations
        test_pattern_detection()
        
        # Test 2: Synthetic perfect pattern
        test_synthetic_pattern()
        
        # Test 3: Statistics across time periods
        test_pattern_statistics()
        
        # Test 4: Small backtest
        test_backtest_small()
        
        print("\n" + "="*80)
        print("ALL TESTS COMPLETE")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Test suite error: {e}", exc_info=True)
        print(f"\n❌ Test suite failed: {e}")


if __name__ == "__main__":
    main()
