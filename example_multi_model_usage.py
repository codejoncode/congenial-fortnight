#!/usr/bin/env python3
"""
Example Usage: Multi-Model Signal Aggregator
Demonstrates how to use the system with your existing infrastructure
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

from scripts.multi_model_signal_aggregator import MultiModelSignalAggregator

def example_1_basic_aggregation():
    """
    Example 1: Basic signal aggregation
    Show how to aggregate signals from multiple models
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Signal Aggregation")
    print("="*80)
    
    # Initialize aggregator
    aggregator = MultiModelSignalAggregator(pairs=['EURUSD', 'XAUUSD'])
    
    # Simulate signals from your models
    ml_signal = {
        'signal': 'long',
        'confidence': 0.78,
        'entry': 1.0850,
        'stop_loss': 1.0820,
        'setup_quality': 'GOOD',
        'reasoning': 'ML model predicts bullish movement with 78% confidence'
    }
    
    harmonic_signal = {
        'pattern_type': 'Butterfly',
        'direction': 'long',
        'quality_score': 0.85,
        'entry': 1.0848,
        'stop_loss': 1.0818,
        'target_1': 1.0890,
        'target_2': 1.0940,
        'target_3': 1.0990,
        'risk_reward_t1': 1.4,
        'risk_reward_t2': 3.1,
        'risk_reward_t3': 4.7,
        'X': 1.0800, 'A': 1.0900, 'B': 1.0850, 'C': 1.0880, 'D': 1.0848
    }
    
    quantum_signal = {
        'signal': 'bullish',
        'confidence': 0.72,
        'coherence': 0.65,
        'market_regime': 'trending'
    }
    
    # Aggregate signals
    signals = aggregator.aggregate_signals(
        ml_signal=ml_signal,
        harmonic_signal=harmonic_signal,
        quantum_signal=quantum_signal,
        pair='EURUSD',
        current_price=1.0850
    )
    
    print(f"\n✅ Generated {len(signals)} total signals\n")
    
    # Display signals
    for i, signal in enumerate(signals, 1):
        print(f"{i}. {signal['signal_type']} Signal")
        print(f"   Source: {signal['source']}")
        print(f"   Direction: {signal['direction'].upper()}")
        print(f"   Confidence: {signal['confidence']:.1%}")
        print(f"   Entry: {signal['entry']}")
        print(f"   Stop Loss: {signal['stop_loss']}")
        print(f"   Take Profits: TP1={signal['take_profit']['tp1']}, TP2={signal['take_profit']['tp2']}, TP3={signal['take_profit']['tp3']}")
        print(f"   R:R Ratios: {signal['risk_reward']['tp1']}:1, {signal['risk_reward']['tp2']}:1, {signal['risk_reward']['tp3']}:1")
        print(f"   Quality: {signal['setup_quality']}")
        print()


def example_2_confluence_detection():
    """
    Example 2: Confluence detection
    Show how the system detects when models agree
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Confluence Detection")
    print("="*80)
    
    aggregator = MultiModelSignalAggregator(pairs=['XAUUSD'])
    
    # All three models agree on direction (LONG)
    ml_signal = {
        'signal': 'long',
        'confidence': 0.85,
        'entry': 2050.00,
        'stop_loss': 2030.00,
        'setup_quality': 'EXCELLENT',
        'reasoning': 'Strong bullish setup'
    }
    
    harmonic_signal = {
        'pattern_type': 'Gartley',
        'direction': 'long',  # Same direction
        'quality_score': 0.88,
        'entry': 2048.00,
        'stop_loss': 2028.00,
        'target_1': 2088.00,
        'target_2': 2128.00,
        'target_3': 2168.00,
        'risk_reward_t1': 2.0,
        'risk_reward_t2': 4.0,
        'risk_reward_t3': 6.0,
        'X': 2000.0, 'A': 2100.0, 'B': 2050.0, 'C': 2080.0, 'D': 2048.0
    }
    
    quantum_signal = {
        'signal': 'bullish',  # Same direction
        'confidence': 0.80,
        'coherence': 0.75,
        'market_regime': 'strong_trend'
    }
    
    signals = aggregator.aggregate_signals(
        ml_signal=ml_signal,
        harmonic_signal=harmonic_signal,
        quantum_signal=quantum_signal,
        pair='XAUUSD',
        current_price=2050.00
    )
    
    # Check for confluence
    confluence_signals = [s for s in signals if 'CONFLUENCE' in s.get('signal_type', '')]
    triple_confluence = [s for s in signals if s.get('confluence_type') == 'TRIPLE']
    
    print(f"\n✅ Total signals: {len(signals)}")
    print(f"✅ Confluence signals: {len(confluence_signals)}")
    print(f"🚀 Triple confluence: {len(triple_confluence)}")
    
    if triple_confluence:
        sig = triple_confluence[0]
        print(f"\n🎯 TRIPLE CONFLUENCE DETECTED!")
        print(f"   All three models agree: {sig['direction'].upper()}")
        print(f"   Combined confidence: {sig['confidence']:.1%}")
        print(f"   Primary R:R: {sig['risk_reward']['primary']}:1")
        print(f"   TP1 R:R: {sig['risk_reward']['tp1']}:1")
        print(f"   TP2 R:R: {sig['risk_reward']['tp2']}:1")
        print(f"   TP3 R:R: {sig['risk_reward']['tp3']}:1")
        print(f"   Quality: {sig['setup_quality']}")


def example_3_risk_reward_tiers():
    """
    Example 3: Different R:R tiers
    Show how different signal types have different R:R ratios
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Risk:Reward Tiers")
    print("="*80)
    
    aggregator = MultiModelSignalAggregator(pairs=['EURUSD'])
    
    # Test different confidence levels
    test_cases = [
        {
            'name': 'Low Confidence (65%)',
            'ml_signal': {'signal': 'long', 'confidence': 0.65, 'entry': 1.0850, 'stop_loss': 1.0820,
                         'setup_quality': 'GOOD', 'reasoning': 'Minimum confidence signal'},
            'expected_rr': 2.0
        },
        {
            'name': 'Medium Confidence (75%)',
            'ml_signal': {'signal': 'long', 'confidence': 0.75, 'entry': 1.0850, 'stop_loss': 1.0820,
                         'setup_quality': 'GOOD', 'reasoning': 'Medium confidence signal'},
            'expected_rr': 2.5
        },
        {
            'name': 'High Confidence (85%)',
            'ml_signal': {'signal': 'long', 'confidence': 0.85, 'entry': 1.0850, 'stop_loss': 1.0820,
                         'setup_quality': 'EXCELLENT', 'reasoning': 'High confidence signal'},
            'expected_rr': 3.0
        }
    ]
    
    for test in test_cases:
        signals = aggregator.aggregate_signals(
            ml_signal=test['ml_signal'],
            harmonic_signal=None,
            quantum_signal=None,
            pair='EURUSD',
            current_price=1.0850
        )
        
        if signals:
            sig = signals[0]
            print(f"\n{test['name']}:")
            print(f"   Confidence: {sig['confidence']:.1%}")
            print(f"   R:R Tier: {sig['risk_reward']['tp1']}:1, {sig['risk_reward']['tp2']}:1, {sig['risk_reward']['tp3']}:1")
            print(f"   Risk: {sig['risk_pips']:.1f} pips")
            print(f"   Reward (TP2): {sig['reward_pips']['tp2']:.1f} pips")


def example_4_performance_tracking():
    """
    Example 4: Performance tracking
    Show how to track and export performance
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Performance Tracking")
    print("="*80)
    
    aggregator = MultiModelSignalAggregator(pairs=['EURUSD', 'XAUUSD'])
    
    # Generate multiple signals
    for i in range(3):
        ml_signal = {
            'signal': 'long' if i % 2 == 0 else 'short',
            'confidence': 0.70 + (i * 0.05),
            'entry': 1.0850,
            'stop_loss': 1.0820,
            'setup_quality': 'GOOD',
            'reasoning': f'Test signal {i+1}'
        }
        
        aggregator.aggregate_signals(
            ml_signal=ml_signal,
            harmonic_signal=None,
            quantum_signal=None,
            pair='EURUSD',
            current_price=1.0850
        )
    
    # Get performance summary
    summary = aggregator.get_signal_summary('EURUSD')
    
    print(f"\n📊 Performance Summary for EURUSD:")
    print(f"   Total Signals: {summary['total_signals']}")
    print(f"   Status: {summary.get('message', 'Active')}")
    
    # Export signals
    export_path = aggregator.export_signals('examples/signal_history.json')
    print(f"\n💾 Signals exported to: {export_path}")


def example_5_integration_pattern():
    """
    Example 5: Integration with existing system
    Show how to integrate with your current setup
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Integration Pattern")
    print("="*80)
    
    print("""
    # In your existing signal generation code:
    
    from scripts.multi_model_signal_aggregator import MultiModelSignalAggregator
    
    # Initialize once (e.g., in __init__ or at module level)
    aggregator = MultiModelSignalAggregator(pairs=['EURUSD', 'XAUUSD'])
    
    # When you have signals from your models:
    def generate_daily_signals(pair, df, ml_model):
        # 1. Get ML signal (your existing code)
        ml_predictions = ml_model.predict_proba(df)
        ml_signal = {
            'signal': 'long' if ml_predictions[1] > 0.5 else 'short',
            'confidence': max(ml_predictions),
            'entry': df['Close'].iloc[-1],
            'stop_loss': calculate_stop_loss(df),  # Your existing function
            'setup_quality': 'GOOD',
            'reasoning': 'ML model prediction'
        }
        
        # 2. Get harmonic signal (if available)
        harmonic_signal = detect_harmonic_patterns(df)  # Your existing function
        
        # 3. Get quantum signal (if available)
        quantum_signal = quantum_generator.get_quantum_signal()
        
        # 4. Aggregate all signals
        all_signals = aggregator.aggregate_signals(
            ml_signal=ml_signal,
            harmonic_signal=harmonic_signal,
            quantum_signal=quantum_signal,
            pair=pair,
            current_price=df['Close'].iloc[-1]
        )
        
        # 5. Use signals
        for signal in all_signals:
            # Send notification
            if signal['risk_reward']['primary'] >= 3.0:
                send_high_priority_notification(signal)
            
            # Log signal
            log_signal_to_database(signal)
            
            # Execute trade (if automated)
            if signal['signal_type'] == 'ULTRA':
                execute_trade(signal)
    
    """)
    
    print("See MULTI_MODEL_SIGNAL_SYSTEM.md for full integration guide")


def main():
    """Run all examples"""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║     MULTI-MODEL SIGNAL AGGREGATOR - USAGE EXAMPLES               ║
    ║     Ensuring 2:1 to 5:1+ Risk:Reward Ratios                      ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    example_1_basic_aggregation()
    example_2_confluence_detection()
    example_3_risk_reward_tiers()
    example_4_performance_tracking()
    example_5_integration_pattern()
    
    print("\n" + "="*80)
    print("✅ ALL EXAMPLES COMPLETED")
    print("="*80)
    print("\n📚 For more details, see:")
    print("   - MULTI_MODEL_SIGNAL_SYSTEM.md")
    print("   - scripts/multi_model_signal_aggregator.py")
    print("   - scripts/enhanced_signal_integration.py")
    print("\n🚀 Ready to generate optimal R:R signals!")


if __name__ == '__main__':
    main()
