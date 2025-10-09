#!/usr/bin/env python3
"""
Multi-Model Signal Aggregator
Combines ML predictions, harmonic patterns, and technical confluence
Provides multiple signal types with varying R:R ratios (2:1 to 5:1+)
Ensures diversified, high-quality trading opportunities
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import logging
from pathlib import Path
import sys

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from pip_based_signal_system import PipBasedSignalSystem
    from harmonic_pattern_trader import HarmonicPatternTrader
    from signals import QuantumMultiTimeframeSignalGenerator
    from pattern_harmonic_detector import PatternHarmonicDetector
except ImportError:
    # Fallback if running from different location
    pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiModelSignalAggregator:
    """
    Aggregates signals from multiple models:
    1. ML Ensemble (existing pip-based system)
    2. Harmonic Pattern Detection (geometric patterns)
    3. Technical Confluence System (multi-indicator)
    4. Quantum Multi-Timeframe (cross-timeframe analysis)
    
    Provides diversified signals with performance tracking and varying R:R ratios
    """
    
    def __init__(self, pairs: List[str] = ['XAUUSD', 'EURUSD']):
        """
        Initialize multi-model aggregator
        
        Args:
            pairs: List of currency pairs to monitor
        """
        self.pairs = pairs
        self.signal_history = []
        
        # Initialize sub-systems
        self.pip_system = PipBasedSignalSystem(min_risk_reward=2.0, min_confidence=0.70)
        self.harmonic_trader = HarmonicPatternTrader(
            lookback=100,
            fib_tolerance=0.08,
            min_quality_score=0.65
        )
        
        # Model weights (adjust based on backtesting)
        self.model_weights = {
            'ml_ensemble': 0.40,        # Your calibrated ML models
            'harmonic_patterns': 0.35,  # 86.5% win rate system
            'quantum_mtf': 0.25         # Multi-timeframe quantum signals
        }
        
        # R:R requirements per signal type
        self.rr_requirements = {
            'HIGH_CONVICTION': 2.0,  # Minimum 2:1 - ML + Harmonic confluence
            'HARMONIC': 3.0,         # Minimum 3:1 - Pure harmonic patterns
            'SCALP': 1.5,            # Quick trades - 1.5:1 acceptable
            'SWING': 4.0,            # Longer holds - 4:1 minimum
            'ULTRA': 5.0             # Ultra high quality - 5:1+
        }
        
        # Performance tracking per pair
        self.performance_stats = {
            pair: {
                'total_signals': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
                'avg_rr': 0.0,
                'total_pips': 0.0,
                'by_signal_type': {}
            }
            for pair in pairs
        }
        
        # Pip values for calculations
        self.pip_values = {
            'EURUSD': 0.0001,
            'GBPUSD': 0.0001,
            'USDJPY': 0.01,
            'XAUUSD': 0.10,
            'USDCAD': 0.0001,
            'AUDUSD': 0.0001,
            'NZDUSD': 0.0001,
            'USDCHF': 0.0001
        }
    
    def aggregate_signals(self, 
                         ml_signal: Optional[Dict],
                         harmonic_signal: Optional[Dict],
                         quantum_signal: Optional[Dict],
                         pair: str,
                         current_price: float) -> List[Dict]:
        """
        Aggregate signals from all models
        Returns list of actionable trading signals with proper R:R ratios
        
        Args:
            ml_signal: Signal from ML pip-based system
            harmonic_signal: Signal from harmonic pattern detector
            quantum_signal: Signal from quantum multi-timeframe system
            pair: Currency pair
            current_price: Current market price
            
        Returns:
            List of formatted trading signals sorted by quality
        """
        aggregated_signals = []
        timestamp = datetime.now()
        
        logger.info(f"Aggregating signals for {pair} at {current_price}")
        
        # === MODEL 1: ML ENSEMBLE SIGNAL ===
        if ml_signal and self._validate_ml_signal(ml_signal):
            ml_trade = self._format_ml_signal(ml_signal, pair, timestamp)
            if ml_trade:
                aggregated_signals.append(ml_trade)
                logger.info(f"ML Signal: {ml_trade['signal_type']} - RR: {ml_trade['risk_reward']['primary']:.1f}:1")
        
        # === MODEL 2: HARMONIC PATTERN SIGNAL ===
        if harmonic_signal and self._validate_harmonic_signal(harmonic_signal):
            harmonic_trade = self._format_harmonic_signal(harmonic_signal, pair, timestamp)
            if harmonic_trade:
                aggregated_signals.append(harmonic_trade)
                logger.info(f"Harmonic Signal: {harmonic_trade['pattern_name']} - RR: {harmonic_trade['risk_reward']['primary']:.1f}:1")
        
        # === MODEL 3: QUANTUM MULTI-TIMEFRAME SIGNAL ===
        if quantum_signal and self._validate_quantum_signal(quantum_signal):
            quantum_trade = self._format_quantum_signal(quantum_signal, pair, timestamp, current_price)
            if quantum_trade:
                aggregated_signals.append(quantum_trade)
                logger.info(f"Quantum Signal: {quantum_trade['signal_type']} - RR: {quantum_trade['risk_reward']['primary']:.1f}:1")
        
        # === HYBRID SIGNALS (Multiple model agreement) ===
        hybrid_signals = self._create_hybrid_signals(
            ml_signal, harmonic_signal, quantum_signal, pair, timestamp, current_price
        )
        aggregated_signals.extend(hybrid_signals)
        
        # Sort by confidence and R:R (prioritize high-quality setups)
        aggregated_signals.sort(
            key=lambda x: (x['confidence'] * x['risk_reward']['primary']),
            reverse=True
        )
        
        # Store in history for performance tracking
        for signal in aggregated_signals:
            self.signal_history.append(signal)
            self.performance_stats[pair]['total_signals'] += 1
        
        logger.info(f"Generated {len(aggregated_signals)} total signals for {pair}")
        return aggregated_signals
    
    def _validate_ml_signal(self, signal: Dict) -> bool:
        """Validate ML ensemble signal meets quality standards"""
        if not signal:
            return False
        
        # Must have minimum confidence (from your calibrated models)
        if signal.get('confidence', 0) < 0.60:  # 60% minimum
            return False
        
        # Must have valid entry and risk parameters
        required_fields = ['entry', 'stop_loss', 'signal']
        if not all(field in signal for field in required_fields):
            return False
        
        # Ensure signal type is valid
        if signal.get('signal') not in ['long', 'short', 'bullish', 'bearish']:
            return False
        
        return True
    
    def _validate_harmonic_signal(self, signal: Dict) -> bool:
        """Validate harmonic pattern signal"""
        if not signal:
            return False
        
        # Must have minimum quality (harmonic system is 86.5% accurate)
        if signal.get('quality_score', 0) < 0.65:  # 65% minimum
            return False
        
        # Must have valid pattern and targets
        if not signal.get('pattern_type'):
            return False
        
        # Must meet minimum R:R for harmonic patterns
        if signal.get('risk_reward_t2', 0) < self.rr_requirements['HARMONIC']:
            return False
        
        return True
    
    def _validate_quantum_signal(self, signal: Dict) -> bool:
        """Validate quantum multi-timeframe signal"""
        if not signal:
            return False
        
        # Must have minimum confidence
        if signal.get('confidence', 0) < 0.60:
            return False
        
        # Must have valid direction
        if signal.get('signal') not in ['bullish', 'bearish', 'long', 'short']:
            return False
        
        # Check quantum coherence (timeframe agreement)
        if signal.get('coherence', 0) < 0.3:
            return False
        
        return True
    
    def _format_ml_signal(self, signal: Dict, pair: str, timestamp: datetime) -> Optional[Dict]:
        """
        Format ML ensemble signal into standardized format with proper R:R
        
        Returns signal with 2:1 to 4:1 R:R ratios based on confidence
        """
        try:
            entry = signal['entry']
            sl = signal['stop_loss']
            direction = signal['signal'].lower()
            
            # Normalize direction
            if direction in ['bullish', 'buy']:
                direction = 'long'
            elif direction in ['bearish', 'sell']:
                direction = 'short'
            
            # Calculate risk
            risk = abs(entry - sl)
            pip_value = self.pip_values.get(pair, 0.0001)
            risk_pips = risk / pip_value
            
            # Generate take profit levels based on confidence
            # Higher confidence = more aggressive targets
            confidence = signal.get('confidence', 0.65)
            
            # Base R:R ratios adjusted by confidence
            if confidence >= 0.85:
                # Ultra high confidence: 3:1, 4:1, 5:1
                rr_ratios = [3.0, 4.0, 5.0]
                signal_type = 'ULTRA'
            elif confidence >= 0.75:
                # High confidence: 2.5:1, 3.5:1, 4.5:1
                rr_ratios = [2.5, 3.5, 4.5]
                signal_type = 'HIGH_CONVICTION'
            else:
                # Standard: 2:1, 3:1, 4:1
                rr_ratios = [2.0, 3.0, 4.0]
                signal_type = 'HIGH_CONVICTION'
            
            if direction == 'long':
                tp1 = entry + (risk * rr_ratios[0])
                tp2 = entry + (risk * rr_ratios[1])
                tp3 = entry + (risk * rr_ratios[2])
            else:  # short
                tp1 = entry - (risk * rr_ratios[0])
                tp2 = entry - (risk * rr_ratios[1])
                tp3 = entry - (risk * rr_ratios[2])
            
            return {
                'signal_id': f"ML_{pair}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                'timestamp': timestamp.isoformat(),
                'pair': pair,
                'source': 'ml_ensemble',
                'signal_type': signal_type,
                'direction': direction,
                'confidence': confidence,
                'entry': round(entry, 5),
                'stop_loss': round(sl, 5),
                'take_profit': {
                    'tp1': round(tp1, 5),
                    'tp2': round(tp2, 5),
                    'tp3': round(tp3, 5)
                },
                'risk_reward': {
                    'primary': rr_ratios[0],
                    'tp1': rr_ratios[0],
                    'tp2': rr_ratios[1],
                    'tp3': rr_ratios[2]
                },
                'risk_pips': round(risk_pips, 1),
                'reward_pips': {
                    'tp1': round(risk_pips * rr_ratios[0], 1),
                    'tp2': round(risk_pips * rr_ratios[1], 1),
                    'tp3': round(risk_pips * rr_ratios[2], 1)
                },
                'setup_quality': signal.get('setup_quality', 'GOOD'),
                'reasoning': signal.get('reasoning', 'ML ensemble prediction with quality setup'),
                'model_weights': self.model_weights['ml_ensemble']
            }
        except Exception as e:
            logger.error(f"Error formatting ML signal: {e}")
            return None
    
    def _format_harmonic_signal(self, signal: Dict, pair: str, timestamp: datetime) -> Optional[Dict]:
        """
        Format harmonic pattern signal with 3:1 to 5:1 R:R ratios
        
        Harmonic patterns naturally provide excellent R:R due to Fibonacci targets
        """
        try:
            pattern_type = signal['pattern_type']
            direction = signal['direction'].lower()
            entry = signal['entry']
            sl = signal['stop_loss']
            
            # Harmonic patterns provide multiple Fibonacci targets
            tp1 = signal['target_1']
            tp2 = signal['target_2']
            tp3 = signal['target_3']
            
            # Calculate actual R:R ratios
            risk = abs(entry - sl)
            rr1 = abs(tp1 - entry) / risk
            rr2 = abs(tp2 - entry) / risk
            rr3 = abs(tp3 - entry) / risk
            
            pip_value = self.pip_values.get(pair, 0.0001)
            risk_pips = risk / pip_value
            
            return {
                'signal_id': f"HARM_{pair}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                'timestamp': timestamp.isoformat(),
                'pair': pair,
                'source': 'harmonic_patterns',
                'signal_type': 'HARMONIC',
                'pattern_name': pattern_type,
                'direction': direction,
                'confidence': signal['quality_score'],
                'entry': round(float(entry), 5),
                'stop_loss': round(float(sl), 5),
                'take_profit': {
                    'tp1': round(float(tp1), 5),
                    'tp2': round(float(tp2), 5),
                    'tp3': round(float(tp3), 5)
                },
                'risk_reward': {
                    'primary': round(float(rr2), 1),  # Use TP2 as primary target
                    'tp1': round(float(rr1), 1),
                    'tp2': round(float(rr2), 1),
                    'tp3': round(float(rr3), 1)
                },
                'risk_pips': round(float(risk_pips), 1),
                'reward_pips': {
                    'tp1': round(float(risk_pips * rr1), 1),
                    'tp2': round(float(risk_pips * rr2), 1),
                    'tp3': round(float(risk_pips * rr3), 1)
                },
                'pattern_points': {
                    'X': float(signal['X']),
                    'A': float(signal['A']),
                    'B': float(signal['B']),
                    'C': float(signal['C']),
                    'D': float(signal['D'])
                },
                'setup_quality': 'EXCELLENT' if signal['quality_score'] > 0.80 else 'GOOD',
                'reasoning': f"{pattern_type} harmonic pattern with {signal['quality_score']:.1%} quality score",
                'model_weights': self.model_weights['harmonic_patterns']
            }
        except Exception as e:
            logger.error(f"Error formatting harmonic signal: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _format_quantum_signal(self, signal: Dict, pair: str, timestamp: datetime, 
                               current_price: float) -> Optional[Dict]:
        """
        Format quantum multi-timeframe signal with 2:1 to 4:1 R:R
        
        Quantum signals leverage cross-timeframe agreement for high probability
        """
        try:
            direction = signal['signal'].lower()
            if direction in ['bullish', 'buy']:
                direction = 'long'
            elif direction in ['bearish', 'sell']:
                direction = 'short'
            
            confidence = signal['confidence']
            coherence = signal.get('coherence', 0.5)
            
            # Use current price as entry for quantum signals
            entry = current_price
            
            # Calculate stop loss based on quantum coherence
            # Higher coherence = tighter stop (more confident)
            pip_value = self.pip_values.get(pair, 0.0001)
            
            # Base risk: 30-50 pips depending on pair
            if pair == 'XAUUSD':
                base_risk_pips = 300  # Gold is more volatile
            else:
                base_risk_pips = 40
            
            # Adjust risk by coherence (higher coherence = tighter stop)
            risk_pips = base_risk_pips * (1.0 - (coherence * 0.3))
            risk = risk_pips * pip_value
            
            # Determine R:R based on confidence and coherence
            combined_score = (confidence + coherence) / 2
            
            if combined_score >= 0.80:
                rr_ratios = [3.0, 4.0, 5.0]
                signal_type = 'ULTRA'
            elif combined_score >= 0.70:
                rr_ratios = [2.5, 3.5, 4.5]
                signal_type = 'HIGH_CONVICTION'
            else:
                rr_ratios = [2.0, 3.0, 4.0]
                signal_type = 'HIGH_CONVICTION'
            
            if direction == 'long':
                sl = entry - risk
                tp1 = entry + (risk * rr_ratios[0])
                tp2 = entry + (risk * rr_ratios[1])
                tp3 = entry + (risk * rr_ratios[2])
            else:
                sl = entry + risk
                tp1 = entry - (risk * rr_ratios[0])
                tp2 = entry - (risk * rr_ratios[1])
                tp3 = entry - (risk * rr_ratios[2])
            
            return {
                'signal_id': f"QTM_{pair}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                'timestamp': timestamp.isoformat(),
                'pair': pair,
                'source': 'quantum_mtf',
                'signal_type': signal_type,
                'direction': direction,
                'confidence': confidence,
                'quantum_coherence': coherence,
                'entry': round(entry, 5),
                'stop_loss': round(sl, 5),
                'take_profit': {
                    'tp1': round(tp1, 5),
                    'tp2': round(tp2, 5),
                    'tp3': round(tp3, 5)
                },
                'risk_reward': {
                    'primary': rr_ratios[0],
                    'tp1': rr_ratios[0],
                    'tp2': rr_ratios[1],
                    'tp3': rr_ratios[2]
                },
                'risk_pips': round(risk_pips, 1),
                'reward_pips': {
                    'tp1': round(risk_pips * rr_ratios[0], 1),
                    'tp2': round(risk_pips * rr_ratios[1], 1),
                    'tp3': round(risk_pips * rr_ratios[2], 1)
                },
                'market_regime': signal.get('market_regime', 'unknown'),
                'setup_quality': 'EXCELLENT' if combined_score > 0.80 else 'GOOD',
                'reasoning': f"Multi-timeframe quantum signal with {coherence:.1%} coherence",
                'model_weights': self.model_weights['quantum_mtf']
            }
        except Exception as e:
            logger.error(f"Error formatting quantum signal: {e}")
            return None
    
    def _create_hybrid_signals(self, 
                              ml_signal: Optional[Dict],
                              harmonic_signal: Optional[Dict],
                              quantum_signal: Optional[Dict],
                              pair: str,
                              timestamp: datetime,
                              current_price: float) -> List[Dict]:
        """
        Create hybrid signals when multiple models agree
        
        These are the highest quality setups with best R:R potential (3:1 to 5:1+)
        """
        hybrid_signals = []
        
        # Check for ML + Harmonic confluence
        if ml_signal and harmonic_signal:
            ml_dir = ml_signal.get('signal', '').lower()
            harm_dir = harmonic_signal.get('direction', '').lower()
            
            # Normalize directions
            if ml_dir in ['bullish', 'buy']:
                ml_dir = 'long'
            elif ml_dir in ['bearish', 'sell']:
                ml_dir = 'short'
            
            if ml_dir == harm_dir:
                # Both models agree - create high-conviction hybrid signal
                hybrid = self._create_confluence_signal(
                    ml_signal, harmonic_signal, pair, timestamp,
                    'ML_HARMONIC', 'ml_ensemble + harmonic_patterns'
                )
                if hybrid:
                    hybrid_signals.append(hybrid)
                    logger.info(f"🔥 CONFLUENCE: ML + Harmonic both {ml_dir} - RR: {hybrid['risk_reward']['primary']:.1f}:1")
        
        # Check for ML + Quantum confluence
        if ml_signal and quantum_signal:
            ml_dir = ml_signal.get('signal', '').lower()
            qtm_dir = quantum_signal.get('signal', '').lower()
            
            # Normalize
            if ml_dir in ['bullish', 'buy']:
                ml_dir = 'long'
            elif ml_dir in ['bearish', 'sell']:
                ml_dir = 'short'
            if qtm_dir in ['bullish', 'buy']:
                qtm_dir = 'long'
            elif qtm_dir in ['bearish', 'sell']:
                qtm_dir = 'short'
            
            if ml_dir == qtm_dir:
                hybrid = self._create_confluence_signal(
                    ml_signal, quantum_signal, pair, timestamp,
                    'ML_QUANTUM', 'ml_ensemble + quantum_mtf'
                )
                if hybrid:
                    hybrid_signals.append(hybrid)
                    logger.info(f"🔥 CONFLUENCE: ML + Quantum both {ml_dir}")
        
        # Check for Harmonic + Quantum confluence
        if harmonic_signal and quantum_signal:
            harm_dir = harmonic_signal.get('direction', '').lower()
            qtm_dir = quantum_signal.get('signal', '').lower()
            
            # Normalize
            if qtm_dir in ['bullish', 'buy']:
                qtm_dir = 'long'
            elif qtm_dir in ['bearish', 'sell']:
                qtm_dir = 'short'
            
            if harm_dir == qtm_dir:
                hybrid = self._create_confluence_signal(
                    harmonic_signal, quantum_signal, pair, timestamp,
                    'HARMONIC_QUANTUM', 'harmonic_patterns + quantum_mtf'
                )
                if hybrid:
                    hybrid_signals.append(hybrid)
                    logger.info(f"🔥 CONFLUENCE: Harmonic + Quantum both {harm_dir}")
        
        # Triple confluence (all three models agree) - ULTRA signal
        if ml_signal and harmonic_signal and quantum_signal:
            ml_dir = ml_signal.get('signal', '').lower()
            harm_dir = harmonic_signal.get('direction', '').lower()
            qtm_dir = quantum_signal.get('signal', '').lower()
            
            # Normalize all
            if ml_dir in ['bullish', 'buy']:
                ml_dir = 'long'
            elif ml_dir in ['bearish', 'sell']:
                ml_dir = 'short'
            if qtm_dir in ['bullish', 'buy']:
                qtm_dir = 'long'
            elif qtm_dir in ['bearish', 'sell']:
                qtm_dir = 'short'
            
            if ml_dir == harm_dir == qtm_dir:
                hybrid = self._create_triple_confluence_signal(
                    ml_signal, harmonic_signal, quantum_signal, pair, timestamp
                )
                if hybrid:
                    hybrid_signals.append(hybrid)
                    logger.info(f"🚀 TRIPLE CONFLUENCE: All models {ml_dir} - RR: {hybrid['risk_reward']['primary']:.1f}:1")
        
        return hybrid_signals
    
    def _create_confluence_signal(self,
                                 signal1: Dict,
                                 signal2: Dict,
                                 pair: str,
                                 timestamp: datetime,
                                 confluence_type: str,
                                 sources: str) -> Optional[Dict]:
        """
        Create a hybrid signal from two agreeing models
        Use best parameters from both for optimal R:R
        """
        try:
            # Extract direction (already validated to match)
            dir1 = signal1.get('signal', signal1.get('direction', '')).lower()
            if dir1 in ['bullish', 'buy']:
                direction = 'long'
            elif dir1 in ['bearish', 'sell']:
                direction = 'short'
            else:
                direction = dir1
            
            # Average entry prices if both have them
            entry1 = signal1.get('entry', 0)
            entry2 = signal2.get('entry', 0)
            entry = (entry1 + entry2) / 2 if entry1 and entry2 else (entry1 or entry2)
            
            # Use tighter stop loss (more conservative)
            sl1 = signal1.get('stop_loss', 0)
            sl2 = signal2.get('stop_loss', 0)
            
            if direction == 'long':
                sl = max(sl1, sl2) if sl1 and sl2 else (sl1 or sl2)  # Tighter stop for long
            else:
                sl = min(sl1, sl2) if sl1 and sl2 else (sl1 or sl2)  # Tighter stop for short
            
            # Calculate risk
            risk = abs(entry - sl)
            pip_value = self.pip_values.get(pair, 0.0001)
            risk_pips = risk / pip_value
            
            # Confluence signals get more aggressive targets (3:1, 4:1, 5:1)
            rr_ratios = [3.0, 4.5, 6.0]
            
            if direction == 'long':
                tp1 = entry + (risk * rr_ratios[0])
                tp2 = entry + (risk * rr_ratios[1])
                tp3 = entry + (risk * rr_ratios[2])
            else:
                tp1 = entry - (risk * rr_ratios[0])
                tp2 = entry - (risk * rr_ratios[1])
                tp3 = entry - (risk * rr_ratios[2])
            
            # Combined confidence
            conf1 = signal1.get('confidence', signal1.get('quality_score', 0.70))
            conf2 = signal2.get('confidence', signal2.get('quality_score', 0.70))
            combined_confidence = (conf1 + conf2) / 2 * 1.15  # Boost for confluence
            combined_confidence = min(combined_confidence, 0.99)  # Cap at 99%
            
            return {
                'signal_id': f"{confluence_type}_{pair}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                'timestamp': timestamp.isoformat(),
                'pair': pair,
                'source': sources,
                'signal_type': 'CONFLUENCE',
                'confluence_type': confluence_type,
                'direction': direction,
                'confidence': round(combined_confidence, 3),
                'entry': round(entry, 5),
                'stop_loss': round(sl, 5),
                'take_profit': {
                    'tp1': round(tp1, 5),
                    'tp2': round(tp2, 5),
                    'tp3': round(tp3, 5)
                },
                'risk_reward': {
                    'primary': rr_ratios[0],
                    'tp1': rr_ratios[0],
                    'tp2': rr_ratios[1],
                    'tp3': rr_ratios[2]
                },
                'risk_pips': round(risk_pips, 1),
                'reward_pips': {
                    'tp1': round(risk_pips * rr_ratios[0], 1),
                    'tp2': round(risk_pips * rr_ratios[1], 1),
                    'tp3': round(risk_pips * rr_ratios[2], 1)
                },
                'setup_quality': 'ELITE',
                'reasoning': f"CONFLUENCE: {confluence_type.replace('_', ' + ')} agreement",
                'model_weights': 1.0  # Full weight for confluence
            }
        except Exception as e:
            logger.error(f"Error creating confluence signal: {e}")
            return None
    
    def _create_triple_confluence_signal(self,
                                        ml_signal: Dict,
                                        harmonic_signal: Dict,
                                        quantum_signal: Dict,
                                        pair: str,
                                        timestamp: datetime) -> Optional[Dict]:
        """
        Create ULTRA signal when all three models agree
        This is the highest quality setup - use most aggressive targets (5:1+)
        """
        try:
            # All directions already validated to match
            direction = ml_signal.get('signal', '').lower()
            if direction in ['bullish', 'buy']:
                direction = 'long'
            elif direction in ['bearish', 'sell']:
                direction = 'short'
            
            # Average all entry prices
            entry = (ml_signal.get('entry', 0) + 
                    harmonic_signal.get('entry', 0) + 
                    quantum_signal.get('entry', 0)) / 3
            
            # Use most conservative stop loss
            sl_ml = ml_signal.get('stop_loss', 0)
            sl_harm = harmonic_signal.get('stop_loss', 0)
            sl_qtm = quantum_signal.get('entry', 0)  # Quantum might not have explicit SL
            
            if direction == 'long':
                sl = max([s for s in [sl_ml, sl_harm, sl_qtm] if s > 0])
            else:
                sl = min([s for s in [sl_ml, sl_harm, sl_qtm] if s > 0])
            
            # Calculate risk
            risk = abs(entry - sl)
            pip_value = self.pip_values.get(pair, 0.0001)
            risk_pips = risk / pip_value
            
            # Triple confluence gets ultra-aggressive targets (4:1, 6:1, 8:1)
            rr_ratios = [4.0, 6.0, 8.0]
            
            if direction == 'long':
                tp1 = entry + (risk * rr_ratios[0])
                tp2 = entry + (risk * rr_ratios[1])
                tp3 = entry + (risk * rr_ratios[2])
            else:
                tp1 = entry - (risk * rr_ratios[0])
                tp2 = entry - (risk * rr_ratios[1])
                tp3 = entry - (risk * rr_ratios[2])
            
            # Combined confidence from all three models
            conf_ml = ml_signal.get('confidence', 0.70)
            conf_harm = harmonic_signal.get('quality_score', 0.70)
            conf_qtm = quantum_signal.get('confidence', 0.70)
            
            combined_confidence = (conf_ml + conf_harm + conf_qtm) / 3 * 1.25  # 25% boost
            combined_confidence = min(combined_confidence, 0.99)
            
            return {
                'signal_id': f"ULTRA_{pair}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                'timestamp': timestamp.isoformat(),
                'pair': pair,
                'source': 'ml_ensemble + harmonic_patterns + quantum_mtf',
                'signal_type': 'ULTRA',
                'confluence_type': 'TRIPLE',
                'direction': direction,
                'confidence': round(combined_confidence, 3),
                'entry': round(entry, 5),
                'stop_loss': round(sl, 5),
                'take_profit': {
                    'tp1': round(tp1, 5),
                    'tp2': round(tp2, 5),
                    'tp3': round(tp3, 5)
                },
                'risk_reward': {
                    'primary': rr_ratios[0],
                    'tp1': rr_ratios[0],
                    'tp2': rr_ratios[1],
                    'tp3': rr_ratios[2]
                },
                'risk_pips': round(risk_pips, 1),
                'reward_pips': {
                    'tp1': round(risk_pips * rr_ratios[0], 1),
                    'tp2': round(risk_pips * rr_ratios[1], 1),
                    'tp3': round(risk_pips * rr_ratios[2], 1)
                },
                'pattern_name': harmonic_signal.get('pattern_type', 'N/A'),
                'quantum_coherence': quantum_signal.get('coherence', 0),
                'setup_quality': 'LEGENDARY',
                'reasoning': '🚀 TRIPLE CONFLUENCE: All three models in perfect agreement',
                'model_weights': 1.0
            }
        except Exception as e:
            logger.error(f"Error creating triple confluence signal: {e}")
            return None
    
    def get_signal_summary(self, pair: str) -> Dict:
        """
        Get performance summary for a specific pair
        
        Returns:
            Dictionary with performance statistics
        """
        stats = self.performance_stats.get(pair, {})
        
        if stats.get('total_signals', 0) == 0:
            return {
                'pair': pair,
                'total_signals': 0,
                'message': 'No signals generated yet'
            }
        
        win_rate = stats['wins'] / (stats['wins'] + stats['losses']) if (stats['wins'] + stats['losses']) > 0 else 0
        
        return {
            'pair': pair,
            'total_signals': stats['total_signals'],
            'wins': stats['wins'],
            'losses': stats['losses'],
            'win_rate': round(win_rate * 100, 2),
            'avg_rr': round(stats['avg_rr'], 2),
            'total_pips': round(stats['total_pips'], 1),
            'by_signal_type': stats['by_signal_type']
        }
    
    def export_signals(self, filepath: str = None) -> str:
        """
        Export signal history to JSON file
        
        Args:
            filepath: Optional custom filepath
            
        Returns:
            Path to exported file
        """
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f"signals/multi_model_signals_{timestamp}.json"
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_signals': len(self.signal_history),
            'pairs': self.pairs,
            'model_weights': self.model_weights,
            'rr_requirements': self.rr_requirements,
            'performance_stats': self.performance_stats,
            'signals': self.signal_history
        }
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(self.signal_history)} signals to {filepath}")
        return filepath


def demo_usage():
    """Demonstrate multi-model signal aggregation"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║         MULTI-MODEL SIGNAL AGGREGATOR                            ║
    ║         Ensuring 2:1 to 5:1+ Risk:Reward Ratios                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    
    This system combines:
    1. ML Ensemble (pip-based) - 75%+ win rate, 2:1 to 4:1 R:R
    2. Harmonic Patterns - 86.5% win rate, 3:1 to 5:1 R:R
    3. Quantum Multi-Timeframe - Cross-timeframe analysis, 2:1 to 4:1 R:R
    
    Signal Types Generated:
    
    📊 INDIVIDUAL SIGNALS:
       • ML_ENSEMBLE: 2:1 to 4:1 R:R based on confidence
       • HARMONIC: 3:1 to 5:1 R:R (Fibonacci targets)
       • QUANTUM_MTF: 2:1 to 4:1 R:R (timeframe coherence)
    
    🔥 CONFLUENCE SIGNALS (2 models agree):
       • ML + HARMONIC: 3:1 to 6:1 R:R
       • ML + QUANTUM: 3:1 to 6:1 R:R
       • HARMONIC + QUANTUM: 3:1 to 6:1 R:R
    
    🚀 ULTRA SIGNALS (3 models agree):
       • TRIPLE CONFLUENCE: 4:1 to 8:1 R:R
       • Highest quality setups
       • Legendary risk:reward potential
    
    All signals include:
    • Multiple take-profit levels (TP1, TP2, TP3)
    • Clear entry, stop loss, and targets
    • Confidence scores
    • Quality ratings
    • Detailed reasoning
    """)
    
    # Example usage
    aggregator = MultiModelSignalAggregator(pairs=['EURUSD', 'XAUUSD'])
    
    # Mock signals for demonstration
    ml_signal = {
        'signal': 'long',
        'confidence': 0.78,
        'entry': 1.0850,
        'stop_loss': 1.0820,
        'setup_quality': 'GOOD',
        'reasoning': 'High probability ML setup'
    }
    
    harmonic_signal = {
        'pattern_type': 'Gartley',
        'direction': 'long',
        'quality_score': 0.82,
        'entry': 1.0848,
        'stop_loss': 1.0818,
        'target_1': 1.0890,
        'target_2': 1.0930,
        'target_3': 1.0980,
        'risk_reward_t2': 3.5,
        'X': 1.0800, 'A': 1.0900, 'B': 1.0850, 'C': 1.0880, 'D': 1.0848
    }
    
    quantum_signal = {
        'signal': 'bullish',
        'confidence': 0.75,
        'coherence': 0.68,
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
    
    print(f"\n✅ Generated {len(signals)} signals for EURUSD")
    print("\nSignal Breakdown:")
    for i, signal in enumerate(signals, 1):
        print(f"\n{i}. {signal['signal_type']} ({signal['source']})")
        print(f"   Direction: {signal['direction'].upper()}")
        print(f"   Confidence: {signal['confidence']:.1%}")
        print(f"   Entry: {signal['entry']}")
        print(f"   Stop Loss: {signal['stop_loss']}")
        print(f"   Take Profits: {signal['take_profit']}")
        print(f"   R:R Ratios: TP1={signal['risk_reward']['tp1']}:1, TP2={signal['risk_reward']['tp2']}:1, TP3={signal['risk_reward']['tp3']}:1")
        print(f"   Quality: {signal['setup_quality']}")
    
    # Export signals
    export_path = aggregator.export_signals()
    print(f"\n💾 Signals exported to: {export_path}")


if __name__ == '__main__':
    demo_usage()
