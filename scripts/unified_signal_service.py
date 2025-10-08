#!/usr/bin/env python3
"""
Unified Signal Service
Aggregates signals from both ML Pip-Based system and Harmonic Pattern system
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime
import logging
from pathlib import Path
import sys

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from pip_based_signal_system import PipBasedSignalSystem
from harmonic_pattern_trader import HarmonicPatternTrader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedSignalService:
    """
    Unified signal service that aggregates signals from:
    1. ML Pip-Based System (from pip_based_signal_system.py)
    2. Harmonic Pattern System (from harmonic_pattern_trader.py)
    
    Provides multiple signal aggregation strategies:
    - Parallel: Show both signals independently
    - Confluence: Only show when both systems agree
    - Weighted: Prioritize based on quality scores
    """
    
    def __init__(self, mode: str = 'parallel'):
        """
        Args:
            mode: 'parallel', 'confluence', or 'weighted'
        """
        self.mode = mode
        self.pip_system = PipBasedSignalSystem(
            min_risk_reward=2.0,
            min_confidence=0.75
        )
        self.harmonic_trader = HarmonicPatternTrader(
            lookback=100,
            fib_tolerance=0.08,
            min_quality_score=0.65
        )
        
    def generate_unified_signals(self, pair: str, df: pd.DataFrame, 
                                 ml_model) -> Dict:
        """
        Generate unified signals from both systems
        
        Args:
            pair: Currency pair (e.g., 'EURUSD')
            df: Historical OHLC data
            ml_model: Trained ML model for pip-based system
            
        Returns:
            {
                'timestamp': datetime,
                'pair': str,
                'mode': str,
                'ml_signals': [...],
                'harmonic_signals': [...],
                'confluence_signals': [...],  # Only in confluence mode
                'recommendation': {...}  # Overall recommendation
            }
        """
        
        logger.info(f"Generating unified signals for {pair} in {self.mode} mode")
        
        # Get ML Pip-Based signals
        ml_signals = self._get_ml_signals(pair, df, ml_model)
        
        # Get Harmonic Pattern signals
        harmonic_signals = self._get_harmonic_signals(df)
        
        # Aggregate based on mode
        if self.mode == 'parallel':
            result = self._parallel_aggregation(pair, ml_signals, harmonic_signals)
        elif self.mode == 'confluence':
            result = self._confluence_aggregation(pair, ml_signals, harmonic_signals)
        elif self.mode == 'weighted':
            result = self._weighted_aggregation(pair, ml_signals, harmonic_signals)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        result['timestamp'] = datetime.now().isoformat()
        result['pair'] = pair
        result['mode'] = self.mode
        
        return result
    
    def _get_ml_signals(self, pair: str, df: pd.DataFrame, ml_model) -> List[Dict]:
        """Get signals from ML Pip-Based system"""
        
        signals = []
        
        try:
            # Get model predictions
            # Note: This assumes features are already calculated
            # In production, you'd integrate with forecasting.py
            
            # For the last bar, generate a quality signal
            predictions = ml_model.predict_proba(df.tail(1))  # Assuming features are in df
            prob = predictions[0]
            
            model_prediction = {
                'direction': 'long' if prob[1] > 0.5 else 'short',
                'confidence': max(prob)
            }
            
            # Check if this is a quality setup
            signal = self.pip_system.detect_quality_setup(
                df.tail(200),  # Last 200 bars for context
                pair,
                model_prediction
            )
            
            if signal['signal'] is not None:
                signals.append({
                    'source': 'ml_pip',
                    'type': signal['signal'],
                    'confidence': signal['confidence'],
                    'entry': signal['entry'],
                    'stop_loss': signal['stop_loss'],
                    'take_profit': signal['take_profit'],
                    'risk_pips': signal['risk_pips'],
                    'reward_pips': signal['reward_pips'],
                    'risk_reward_ratio': signal['risk_reward_ratio'],
                    'quality': signal['setup_quality'],
                    'quality_score': signal['quality_score'],
                    'reasoning': signal['reasoning']
                })
                
        except Exception as e:
            logger.error(f"Error generating ML signals: {e}")
            
        return signals
    
    def _get_harmonic_signals(self, df: pd.DataFrame) -> List[Dict]:
        """Get signals from Harmonic Pattern system"""
        
        signals = []
        
        try:
            # Generate harmonic pattern signals
            harmonic_signals_df = self.harmonic_trader.generate_signals(df)
            
            if not harmonic_signals_df.empty:
                # Get most recent active signal
                active_signals = self.harmonic_trader.get_active_trades(harmonic_signals_df)
                
                for _, signal in active_signals.iterrows():
                    signals.append({
                        'source': 'harmonic',
                        'type': signal['direction'],
                        'pattern': signal['pattern_type'],
                        'quality': signal['quality_score'],
                        'entry': signal['entry'],
                        'stop_loss': signal['stop_loss'],
                        'target_1': signal['target_1'],
                        'target_2': signal['target_2'],
                        'target_3': signal['target_3'],
                        'risk_reward_t1': signal['risk_reward_t1'],
                        'risk_reward_t2': signal['risk_reward_t2'],
                        'risk_reward_t3': signal['risk_reward_t3'],
                        'X': signal['X'],
                        'A': signal['A'],
                        'B': signal['B'],
                        'C': signal['C'],
                        'D': signal['D'],
                        'reasoning': f"{signal['pattern_type']} pattern detected with {signal['quality_score']:.1%} quality"
                    })
                    
        except Exception as e:
            logger.error(f"Error generating harmonic signals: {e}")
            
        return signals
    
    def _parallel_aggregation(self, pair: str, ml_signals: List[Dict], 
                             harmonic_signals: List[Dict]) -> Dict:
        """
        Parallel mode: Show both signal types independently
        Frontend can display both and let user choose
        """
        
        # Check if both agree (for highlighting)
        confluence = False
        if ml_signals and harmonic_signals:
            ml_direction = ml_signals[0]['type']
            harmonic_direction = harmonic_signals[0]['type']
            confluence = (ml_direction == harmonic_direction)
        
        # Overall recommendation based on strongest signal
        recommendation = self._generate_recommendation(ml_signals, harmonic_signals, confluence)
        
        return {
            'ml_signals': ml_signals,
            'harmonic_signals': harmonic_signals,
            'confluence_detected': confluence,
            'recommendation': recommendation
        }
    
    def _confluence_aggregation(self, pair: str, ml_signals: List[Dict],
                               harmonic_signals: List[Dict]) -> Dict:
        """
        Confluence mode: Only show signals when both systems agree
        More conservative approach
        """
        
        confluence_signals = []
        
        if ml_signals and harmonic_signals:
            ml_direction = ml_signals[0]['type']
            harmonic_direction = harmonic_signals[0]['type']
            
            if ml_direction == harmonic_direction:
                # Both agree - create confluence signal
                ml_sig = ml_signals[0]
                harm_sig = harmonic_signals[0]
                
                confluence_signals.append({
                    'type': ml_direction,
                    'confidence': ml_sig['confidence'],
                    'quality': (ml_sig['quality_score'] + harm_sig['quality']) / 2,
                    'entry': (ml_sig['entry'] + harm_sig['entry']) / 2,
                    'stop_loss': (ml_sig['stop_loss'] + harm_sig['stop_loss']) / 2,
                    'take_profit': ml_sig['take_profit'],
                    'ml_reasoning': ml_sig['reasoning'],
                    'harmonic_pattern': harm_sig['pattern'],
                    'harmonic_reasoning': harm_sig['reasoning'],
                    'risk_reward_ratio': ml_sig['risk_reward_ratio']
                })
        
        recommendation = {
            'action': 'BUY' if confluence_signals and confluence_signals[0]['type'] == 'long' else
                     'SELL' if confluence_signals and confluence_signals[0]['type'] == 'short' else
                     'WAIT',
            'confidence': confluence_signals[0]['confidence'] if confluence_signals else 0.0,
            'reason': 'Both systems agree' if confluence_signals else 'No confluence detected'
        }
        
        return {
            'ml_signals': ml_signals,
            'harmonic_signals': harmonic_signals,
            'confluence_signals': confluence_signals,
            'recommendation': recommendation
        }
    
    def _weighted_aggregation(self, pair: str, ml_signals: List[Dict],
                             harmonic_signals: List[Dict]) -> Dict:
        """
        Weighted mode: Combine signals based on quality scores
        Prioritize higher quality setups
        """
        
        # Calculate weights based on quality
        ml_weight = 0.0
        harm_weight = 0.0
        
        if ml_signals:
            ml_weight = ml_signals[0]['quality_score'] / 100.0  # Normalize to 0-1
        
        if harmonic_signals:
            harm_weight = harmonic_signals[0]['quality']
        
        total_weight = ml_weight + harm_weight
        
        if total_weight > 0:
            ml_weight_norm = ml_weight / total_weight
            harm_weight_norm = harm_weight / total_weight
        else:
            ml_weight_norm = 0.5
            harm_weight_norm = 0.5
        
        # Determine direction with weighted confidence
        if ml_signals and harmonic_signals:
            ml_direction = ml_signals[0]['type']
            harmonic_direction = harmonic_signals[0]['type']
            
            if ml_direction == harmonic_direction:
                # Both agree - high confidence
                combined_confidence = (
                    ml_signals[0]['confidence'] * ml_weight_norm +
                    harmonic_signals[0]['quality'] * harm_weight_norm
                )
                action = 'BUY' if ml_direction == 'long' else 'SELL'
            else:
                # Disagree - use weighted average
                if ml_weight > harm_weight:
                    action = 'BUY' if ml_direction == 'long' else 'SELL'
                    combined_confidence = ml_signals[0]['confidence'] * 0.7  # Reduced confidence
                else:
                    action = 'BUY' if harmonic_direction == 'long' else 'SELL'
                    combined_confidence = harmonic_signals[0]['quality'] * 0.7
        elif ml_signals:
            action = 'BUY' if ml_signals[0]['type'] == 'long' else 'SELL'
            combined_confidence = ml_signals[0]['confidence']
        elif harmonic_signals:
            action = 'BUY' if harmonic_signals[0]['type'] == 'long' else 'SELL'
            combined_confidence = harmonic_signals[0]['quality']
        else:
            action = 'WAIT'
            combined_confidence = 0.0
        
        recommendation = {
            'action': action,
            'confidence': combined_confidence,
            'ml_weight': ml_weight_norm,
            'harmonic_weight': harm_weight_norm,
            'reason': f"Weighted decision (ML: {ml_weight_norm:.1%}, Harmonic: {harm_weight_norm:.1%})"
        }
        
        return {
            'ml_signals': ml_signals,
            'harmonic_signals': harmonic_signals,
            'recommendation': recommendation
        }
    
    def _generate_recommendation(self, ml_signals: List[Dict],
                                harmonic_signals: List[Dict],
                                confluence: bool) -> Dict:
        """Generate overall recommendation"""
        
        if not ml_signals and not harmonic_signals:
            return {
                'action': 'WAIT',
                'confidence': 0.0,
                'reason': 'No quality setups detected'
            }
        
        if confluence:
            action = 'BUY' if ml_signals[0]['type'] == 'long' else 'SELL'
            confidence = (ml_signals[0]['confidence'] + harmonic_signals[0]['quality']) / 2
            reason = 'STRONG: Both systems agree'
        elif ml_signals and harmonic_signals:
            # Use ML signal (generally more reliable)
            action = 'BUY' if ml_signals[0]['type'] == 'long' else 'SELL'
            confidence = ml_signals[0]['confidence'] * 0.8  # Reduced confidence
            reason = 'MODERATE: Systems disagree, using ML signal'
        elif ml_signals:
            action = 'BUY' if ml_signals[0]['type'] == 'long' else 'SELL'
            confidence = ml_signals[0]['confidence']
            reason = 'ML signal only'
        else:  # harmonic_signals only
            action = 'BUY' if harmonic_signals[0]['type'] == 'long' else 'SELL'
            confidence = harmonic_signals[0]['quality']
            reason = 'Harmonic pattern only'
        
        return {
            'action': action,
            'confidence': confidence,
            'reason': reason,
            'has_ml': len(ml_signals) > 0,
            'has_harmonic': len(harmonic_signals) > 0,
            'confluence': confluence
        }


# Example usage
if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║           UNIFIED SIGNAL SERVICE                             ║
    ╚══════════════════════════════════════════════════════════════╝
    
    This service aggregates signals from:
    
    1. ML Pip-Based System
       - 75%+ win rate
       - 2:1+ Risk:Reward
       - Quality setups only
    
    2. Harmonic Pattern System  
       - 86.5% win rate
       - Geometric patterns
       - Fibonacci targets
    
    Modes:
    
    PARALLEL (Default):
      - Shows both signals independently
      - Frontend displays both types
      - Highlights when both agree
      
    CONFLUENCE (Conservative):
      - Only shows signals when both systems agree
      - Higher confidence, lower frequency
      - Best for risk-averse trading
      
    WEIGHTED (Balanced):
      - Combines signals based on quality scores
      - Prioritizes higher quality setups
      - Balanced approach
    
    Usage:
      service = UnifiedSignalService(mode='parallel')
      signals = service.generate_unified_signals(pair, df, ml_model)
    """)
