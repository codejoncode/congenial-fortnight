#!/usr/bin/env python3
"""
Test Multi-Model Signal Aggregator
Validates signal generation, R:R ratios, and quality standards
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

from scripts.multi_model_signal_aggregator import MultiModelSignalAggregator
from scripts.enhanced_signal_integration import EnhancedSignalService


class TestMultiModelAggregator(unittest.TestCase):
    """Test suite for multi-model signal aggregator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.aggregator = MultiModelSignalAggregator(pairs=['EURUSD', 'XAUUSD'])
        
        # Create mock signals
        self.mock_ml_signal = {
            'signal': 'long',
            'confidence': 0.78,
            'entry': 1.0850,
            'stop_loss': 1.0820,
            'setup_quality': 'GOOD',
            'reasoning': 'High probability ML setup'
        }
        
        self.mock_harmonic_signal = {
            'pattern_type': 'Gartley',
            'direction': 'long',
            'quality_score': 0.82,
            'entry': 1.0848,
            'stop_loss': 1.0818,
            'target_1': 1.0890,
            'target_2': 1.0930,
            'target_3': 1.0980,
            'risk_reward_t1': 1.4,
            'risk_reward_t2': 2.8,
            'risk_reward_t3': 4.4,
            'X': 1.0800, 'A': 1.0900, 'B': 1.0850, 'C': 1.0880, 'D': 1.0848
        }
        
        self.mock_quantum_signal = {
            'signal': 'bullish',
            'confidence': 0.75,
            'coherence': 0.68,
            'market_regime': 'trending'
        }
    
    def test_aggregator_initialization(self):
        """Test aggregator initializes correctly"""
        self.assertEqual(len(self.aggregator.pairs), 2)
        self.assertIn('EURUSD', self.aggregator.pairs)
        self.assertIn('XAUUSD', self.aggregator.pairs)
        
        # Check R:R requirements
        self.assertEqual(self.aggregator.rr_requirements['HIGH_CONVICTION'], 2.0)
        self.assertEqual(self.aggregator.rr_requirements['HARMONIC'], 3.0)
        self.assertEqual(self.aggregator.rr_requirements['ULTRA'], 5.0)
    
    def test_ml_signal_validation(self):
        """Test ML signal validation"""
        # Valid signal
        self.assertTrue(self.aggregator._validate_ml_signal(self.mock_ml_signal))
        
        # Invalid - low confidence
        invalid_signal = self.mock_ml_signal.copy()
        invalid_signal['confidence'] = 0.50
        self.assertFalse(self.aggregator._validate_ml_signal(invalid_signal))
        
        # Invalid - missing fields
        invalid_signal = {'signal': 'long'}
        self.assertFalse(self.aggregator._validate_ml_signal(invalid_signal))
    
    def test_harmonic_signal_validation(self):
        """Test harmonic signal validation"""
        # Valid signal
        self.assertTrue(self.aggregator._validate_harmonic_signal(self.mock_harmonic_signal))
        
        # Invalid - low quality
        invalid_signal = self.mock_harmonic_signal.copy()
        invalid_signal['quality_score'] = 0.50
        self.assertFalse(self.aggregator._validate_harmonic_signal(invalid_signal))
        
        # Invalid - low R:R
        invalid_signal = self.mock_harmonic_signal.copy()
        invalid_signal['risk_reward_t2'] = 1.5
        self.assertFalse(self.aggregator._validate_harmonic_signal(invalid_signal))
    
    def test_quantum_signal_validation(self):
        """Test quantum signal validation"""
        # Valid signal
        self.assertTrue(self.aggregator._validate_quantum_signal(self.mock_quantum_signal))
        
        # Invalid - low confidence
        invalid_signal = self.mock_quantum_signal.copy()
        invalid_signal['confidence'] = 0.50
        self.assertFalse(self.aggregator._validate_quantum_signal(invalid_signal))
        
        # Invalid - low coherence
        invalid_signal = self.mock_quantum_signal.copy()
        invalid_signal['coherence'] = 0.20
        self.assertFalse(self.aggregator._validate_quantum_signal(invalid_signal))
    
    def test_ml_signal_formatting(self):
        """Test ML signal formatting with proper R:R"""
        formatted = self.aggregator._format_ml_signal(
            self.mock_ml_signal, 'EURUSD', datetime.now()
        )
        
        self.assertIsNotNone(formatted)
        self.assertEqual(formatted['pair'], 'EURUSD')
        self.assertEqual(formatted['direction'], 'long')
        self.assertEqual(formatted['source'], 'ml_ensemble')
        
        # Check R:R ratios
        self.assertGreaterEqual(formatted['risk_reward']['primary'], 2.0)
        self.assertIn('tp1', formatted['take_profit'])
        self.assertIn('tp2', formatted['take_profit'])
        self.assertIn('tp3', formatted['take_profit'])
        
        # Verify R:R increases
        self.assertLess(formatted['risk_reward']['tp1'], formatted['risk_reward']['tp2'])
        self.assertLess(formatted['risk_reward']['tp2'], formatted['risk_reward']['tp3'])
    
    def test_harmonic_signal_formatting(self):
        """Test harmonic signal formatting"""
        formatted = self.aggregator._format_harmonic_signal(
            self.mock_harmonic_signal, 'EURUSD', datetime.now()
        )
        
        self.assertIsNotNone(formatted)
        self.assertEqual(formatted['pair'], 'EURUSD')
        self.assertEqual(formatted['signal_type'], 'HARMONIC')
        self.assertEqual(formatted['pattern_name'], 'Gartley')
        
        # Check R:R ratios meet harmonic minimum (3:1)
        self.assertGreaterEqual(formatted['risk_reward']['primary'], 3.0)
        
        # Check pattern points
        self.assertIn('pattern_points', formatted)
        self.assertIn('X', formatted['pattern_points'])
    
    def test_signal_aggregation(self):
        """Test full signal aggregation"""
        signals = self.aggregator.aggregate_signals(
            ml_signal=self.mock_ml_signal,
            harmonic_signal=self.mock_harmonic_signal,
            quantum_signal=self.mock_quantum_signal,
            pair='EURUSD',
            current_price=1.0850
        )
        
        self.assertIsInstance(signals, list)
        self.assertGreater(len(signals), 0)
        
        # Check all signals have required fields
        for signal in signals:
            self.assertIn('signal_id', signal)
            self.assertIn('pair', signal)
            self.assertIn('direction', signal)
            self.assertIn('confidence', signal)
            self.assertIn('entry', signal)
            self.assertIn('stop_loss', signal)
            self.assertIn('take_profit', signal)
            self.assertIn('risk_reward', signal)
            
            # Verify minimum R:R
            self.assertGreaterEqual(signal['risk_reward']['primary'], 1.5)
    
    def test_confluence_detection(self):
        """Test confluence signal detection"""
        signals = self.aggregator.aggregate_signals(
            ml_signal=self.mock_ml_signal,
            harmonic_signal=self.mock_harmonic_signal,
            quantum_signal=self.mock_quantum_signal,
            pair='EURUSD',
            current_price=1.0850
        )
        
        # Should detect confluence (all signals are 'long'/'bullish')
        confluence_signals = [s for s in signals if 'CONFLUENCE' in s.get('signal_type', '')]
        self.assertGreater(len(confluence_signals), 0)
        
        # Confluence signals should have higher R:R
        for conf_sig in confluence_signals:
            self.assertGreaterEqual(conf_sig['risk_reward']['primary'], 3.0)
    
    def test_triple_confluence(self):
        """Test triple confluence signal generation"""
        # All three models agree on direction
        signals = self.aggregator.aggregate_signals(
            ml_signal=self.mock_ml_signal,
            harmonic_signal=self.mock_harmonic_signal,
            quantum_signal=self.mock_quantum_signal,
            pair='EURUSD',
            current_price=1.0850
        )
        
        # Should have a TRIPLE confluence signal
        ultra_signals = [s for s in signals if s.get('confluence_type') == 'TRIPLE']
        self.assertGreater(len(ultra_signals), 0)
        
        # Triple confluence should have highest R:R (4:1+)
        if ultra_signals:
            ultra_sig = ultra_signals[0]
            self.assertGreaterEqual(ultra_sig['risk_reward']['primary'], 4.0)
            self.assertEqual(ultra_sig['signal_type'], 'ULTRA')
    
    def test_signal_sorting(self):
        """Test signals are sorted by quality"""
        signals = self.aggregator.aggregate_signals(
            ml_signal=self.mock_ml_signal,
            harmonic_signal=self.mock_harmonic_signal,
            quantum_signal=self.mock_quantum_signal,
            pair='EURUSD',
            current_price=1.0850
        )
        
        # Signals should be sorted by confidence * R:R
        if len(signals) > 1:
            for i in range(len(signals) - 1):
                score_current = signals[i]['confidence'] * signals[i]['risk_reward']['primary']
                score_next = signals[i+1]['confidence'] * signals[i+1]['risk_reward']['primary']
                self.assertGreaterEqual(score_current, score_next)
    
    def test_disagreement_handling(self):
        """Test handling when models disagree"""
        # Create conflicting signals
        ml_long = self.mock_ml_signal.copy()
        ml_long['signal'] = 'long'
        
        harmonic_short = self.mock_harmonic_signal.copy()
        harmonic_short['direction'] = 'short'
        
        quantum_long = self.mock_quantum_signal.copy()
        quantum_long['signal'] = 'bullish'
        
        signals = self.aggregator.aggregate_signals(
            ml_signal=ml_long,
            harmonic_signal=harmonic_short,
            quantum_signal=quantum_long,
            pair='EURUSD',
            current_price=1.0850
        )
        
        # Should still generate individual signals
        self.assertGreater(len(signals), 0)
        
        # Should have mixed directions
        directions = [s['direction'] for s in signals]
        self.assertIn('long', directions)
        self.assertIn('short', directions)
    
    def test_performance_tracking(self):
        """Test performance statistics tracking"""
        initial_count = self.aggregator.performance_stats['EURUSD']['total_signals']
        
        signals = self.aggregator.aggregate_signals(
            ml_signal=self.mock_ml_signal,
            harmonic_signal=self.mock_harmonic_signal,
            quantum_signal=self.mock_quantum_signal,
            pair='EURUSD',
            current_price=1.0850
        )
        
        # Performance stats should be updated
        final_count = self.aggregator.performance_stats['EURUSD']['total_signals']
        self.assertEqual(final_count, initial_count + len(signals))
    
    def test_pip_calculations(self):
        """Test pip calculations are correct"""
        formatted = self.aggregator._format_ml_signal(
            self.mock_ml_signal, 'EURUSD', datetime.now()
        )
        
        # Check pip values
        self.assertIn('risk_pips', formatted)
        self.assertIn('reward_pips', formatted)
        
        # Risk pips should match stop loss distance
        entry = formatted['entry']
        sl = formatted['stop_loss']
        pip_value = self.aggregator.pip_values['EURUSD']
        
        expected_risk_pips = abs(entry - sl) / pip_value
        self.assertAlmostEqual(formatted['risk_pips'], expected_risk_pips, places=1)
        
        # Reward pips should match R:R ratios
        self.assertAlmostEqual(
            formatted['reward_pips']['tp1'],
            formatted['risk_pips'] * formatted['risk_reward']['tp1'],
            places=1
        )


class TestEnhancedSignalService(unittest.TestCase):
    """Test suite for enhanced signal service"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.service = EnhancedSignalService(pairs=['EURUSD', 'XAUUSD'])
    
    def test_service_initialization(self):
        """Test service initializes correctly"""
        self.assertEqual(len(self.service.pairs), 2)
        self.assertIsNotNone(self.service.aggregator)
        self.assertIsNotNone(self.service.pip_system)
        self.assertIsNotNone(self.service.harmonic_trader)
    
    def test_summary_generation(self):
        """Test summary generation"""
        # Create mock signals
        mock_signals = [
            {
                'signal_type': 'HIGH_CONVICTION',
                'direction': 'long',
                'confidence': 0.78,
                'risk_reward': {'primary': 2.5}
            },
            {
                'signal_type': 'HARMONIC',
                'direction': 'long',
                'confidence': 0.82,
                'risk_reward': {'primary': 3.5}
            }
        ]
        
        confluence_signals = []
        
        summary = self.service._generate_summary(mock_signals, confluence_signals, 'EURUSD')
        
        self.assertEqual(summary['status'], 'SIGNALS_AVAILABLE')
        self.assertEqual(summary['total_signals'], 2)
        self.assertIn('avg_confidence', summary)
        self.assertIn('avg_risk_reward', summary)
        self.assertIn('direction_agreement', summary)
    
    def test_no_signals_handling(self):
        """Test handling when no signals generated"""
        summary = self.service._generate_summary([], [], 'EURUSD')
        
        self.assertEqual(summary['status'], 'NO_SIGNALS')
        self.assertEqual(summary['recommendation'], 'WAIT')


def run_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("TESTING MULTI-MODEL SIGNAL AGGREGATOR")
    print("="*80 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add tests
    suite.addTests(loader.loadTestsFromTestCase(TestMultiModelAggregator))
    suite.addTests(loader.loadTestsFromTestCase(TestEnhancedSignalService))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
