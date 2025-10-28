#!/usr/bin/env python3
"""
Enhanced Signal Integration
Connects Multi-Model Aggregator with existing signal generation and ML systems
Provides unified interface for generating diversified, high R:R signals
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging
from pathlib import Path
import sys
import joblib

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from multi_model_signal_aggregator import MultiModelSignalAggregator
from pip_based_signal_system import PipBasedSignalSystem
from harmonic_pattern_trader import HarmonicPatternTrader
from signals import QuantumMultiTimeframeSignalGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedSignalService:
    """
    Enhanced signal service that generates multiple high-quality signals
    with proper risk management (2:1 to 5:1+ R:R ratios)
    
    Integrates:
    - Your existing ML models (RF, XGB, etc.)
    - Harmonic pattern detection
    - Quantum multi-timeframe analysis
    - Multi-model signal aggregation
    """
    
    def __init__(self, 
                 pairs: List[str] = ['EURUSD', 'XAUUSD'],
                 models_dir: str = 'models',
                 data_dir: str = 'data'):
        """
        Initialize enhanced signal service
        
        Args:
            pairs: List of currency pairs
            models_dir: Directory containing trained ML models
            data_dir: Directory containing price data
        """
        self.pairs = pairs
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        
        # Initialize aggregator
        self.aggregator = MultiModelSignalAggregator(pairs=pairs)
        
        # Initialize sub-systems
        self.pip_system = PipBasedSignalSystem(min_risk_reward=2.0, min_confidence=0.70)
        self.harmonic_trader = HarmonicPatternTrader(
            lookback=100,
            fib_tolerance=0.08,
            min_quality_score=0.65
        )
        
        # Load ML models for each pair
        self.ml_models = {}
        self.load_ml_models()
        
        # Initialize quantum generators for each pair
        self.quantum_generators = {}
        for pair in pairs:
            try:
                self.quantum_generators[pair] = QuantumMultiTimeframeSignalGenerator(
                    pair=pair,
                    data_dir=str(data_dir),
                    models_dir=str(models_dir)
                )
                logger.info(f"Initialized quantum generator for {pair}")
            except Exception as e:
                logger.warning(f"Could not initialize quantum generator for {pair}: {e}")
    
    def load_ml_models(self):
        """Load trained ML models for each pair"""
        for pair in self.pairs:
            model_path = self.models_dir / f"{pair}_model.pkl"
            if model_path.exists():
                try:
                    self.ml_models[pair] = joblib.load(model_path)
                    logger.info(f"Loaded ML model for {pair}")
                except Exception as e:
                    logger.warning(f"Could not load model for {pair}: {e}")
            else:
                logger.warning(f"No model found for {pair} at {model_path}")
    
    def generate_all_signals(self, 
                            pair: str, 
                            df: pd.DataFrame,
                            current_price: Optional[float] = None) -> Dict:
        """
        Generate all signals for a pair using all available models
        
        Args:
            pair: Currency pair (e.g., 'EURUSD')
            df: Historical OHLC data with features
            current_price: Current market price (optional, will use latest close)
            
        Returns:
            Dictionary containing:
            - individual_signals: List of all signals
            - confluence_signals: List of confluence signals only
            - best_signal: Highest quality signal
            - summary: Performance and quality metrics
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Generating signals for {pair}")
        logger.info(f"{'='*60}")
        
        if current_price is None:
            current_price = df['Close'].iloc[-1]
        
        # === 1. Generate ML Signal ===
        ml_signal = self._generate_ml_signal(pair, df, current_price)
        
        # === 2. Generate Harmonic Signal ===
        harmonic_signal = self._generate_harmonic_signal(df)
        
        # === 3. Generate Quantum Signal ===
        quantum_signal = self._generate_quantum_signal(pair, current_price)
        
        # === 4. Aggregate signals ===
        all_signals = self.aggregator.aggregate_signals(
            ml_signal=ml_signal,
            harmonic_signal=harmonic_signal,
            quantum_signal=quantum_signal,
            pair=pair,
            current_price=current_price
        )
        
        # Separate confluence signals
        confluence_signals = [s for s in all_signals if 'confluence' in s.get('signal_type', '').lower()]
        individual_signals = [s for s in all_signals if 'confluence' not in s.get('signal_type', '').lower()]
        
        # Identify best signal
        best_signal = all_signals[0] if all_signals else None
        
        # Generate summary
        summary = self._generate_summary(all_signals, confluence_signals, pair)
        
        result = {
            'pair': pair,
            'timestamp': datetime.now().isoformat(),
            'current_price': current_price,
            'all_signals': all_signals,
            'individual_signals': individual_signals,
            'confluence_signals': confluence_signals,
            'best_signal': best_signal,
            'summary': summary,
            'total_count': len(all_signals),
            'confluence_count': len(confluence_signals)
        }
        
        logger.info(f"\n✅ Generated {len(all_signals)} total signals ({len(confluence_signals)} confluence)")
        if best_signal:
            logger.info(f"🏆 Best Signal: {best_signal['signal_type']} - {best_signal['direction'].upper()} - RR: {best_signal['risk_reward']['primary']:.1f}:1")
        
        return result
    
    def _generate_ml_signal(self, pair: str, df: pd.DataFrame, current_price: float) -> Optional[Dict]:
        """Generate signal from ML model"""
        try:
            if pair not in self.ml_models:
                logger.warning(f"No ML model available for {pair}")
                return None
            
            model = self.ml_models[pair]
            
            # Prepare features (assuming features are already in df)
            feature_cols = [col for col in df.columns 
                          if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Date', 'date']]
            
            if len(feature_cols) == 0:
                logger.warning("No features found in dataframe")
                return None
            
            # Get latest features
            X = df[feature_cols].iloc[-1:].values
            
            # Get prediction
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[0]
                confidence = max(proba)
                direction = 'long' if proba[1] > 0.5 else 'short'
            else:
                prediction = model.predict(X)[0]
                confidence = abs(prediction) if abs(prediction) <= 1 else 0.70
                direction = 'long' if prediction > 0 else 'short'
            
            # Create model prediction dict for pip system
            model_prediction = {
                'direction': direction,
                'confidence': confidence
            }
            
            # Get quality signal from pip system
            signal = self.pip_system.detect_quality_setup(
                df.tail(200),
                pair,
                model_prediction
            )
            
            if signal['signal'] is None:
                logger.info(f"ML: No quality setup detected (confidence: {confidence:.2%})")
                return None
            
            logger.info(f"ML Signal: {direction.upper()} - Confidence: {confidence:.2%} - Quality: {signal['setup_quality']}")
            return signal
            
        except Exception as e:
            logger.error(f"Error generating ML signal: {e}")
            return None
    
    def _generate_harmonic_signal(self, df: pd.DataFrame) -> Optional[Dict]:
        """Generate signal from harmonic pattern detector"""
        try:
            # Generate harmonic signals
            harmonic_df = self.harmonic_trader.generate_signals(df)
            
            if harmonic_df.empty:
                logger.info("Harmonic: No patterns detected")
                return None
            
            # Get active trades
            active_signals = self.harmonic_trader.get_active_trades(harmonic_df)
            
            if active_signals.empty:
                logger.info("Harmonic: No active patterns")
                return None
            
            # Get the best quality signal
            best_signal = active_signals.iloc[0]
            
            signal = {
                'pattern_type': best_signal['pattern_type'],
                'direction': best_signal['direction'],
                'quality_score': best_signal['quality_score'],
                'entry': best_signal['entry'],
                'stop_loss': best_signal['stop_loss'],
                'target_1': best_signal['target_1'],
                'target_2': best_signal['target_2'],
                'target_3': best_signal['target_3'],
                'risk_reward_t1': best_signal['risk_reward_t1'],
                'risk_reward_t2': best_signal['risk_reward_t2'],
                'risk_reward_t3': best_signal['risk_reward_t3'],
                'X': best_signal['X'],
                'A': best_signal['A'],
                'B': best_signal['B'],
                'C': best_signal['C'],
                'D': best_signal['D']
            }
            
            logger.info(f"Harmonic Signal: {signal['pattern_type']} - {signal['direction'].upper()} - Quality: {signal['quality_score']:.2%}")
            return signal
            
        except Exception as e:
            logger.error(f"Error generating harmonic signal: {e}")
            return None
    
    def _generate_quantum_signal(self, pair: str, current_price: float) -> Optional[Dict]:
        """Generate signal from quantum multi-timeframe system"""
        try:
            if pair not in self.quantum_generators:
                logger.warning(f"No quantum generator for {pair}")
                return None
            
            generator = self.quantum_generators[pair]
            
            # Get quantum signal
            signal = generator.get_quantum_signal()
            
            if signal.get('signal') in ['no_signal', 'error']:
                logger.info(f"Quantum: No signal - {signal.get('reason', 'unknown')}")
                return None
            
            logger.info(f"Quantum Signal: {signal['signal'].upper()} - Confidence: {signal['confidence']:.2%} - Coherence: {signal.get('coherence', 0):.2%}")
            return signal
            
        except Exception as e:
            logger.error(f"Error generating quantum signal: {e}")
            return None
    
    def _generate_summary(self, all_signals: List[Dict], confluence_signals: List[Dict], pair: str) -> Dict:
        """Generate summary statistics for signals"""
        
        if not all_signals:
            return {
                'status': 'NO_SIGNALS',
                'message': 'No quality setups detected',
                'recommendation': 'WAIT'
            }
        
        # Calculate metrics
        avg_confidence = np.mean([s['confidence'] for s in all_signals])
        avg_rr = np.mean([s['risk_reward']['primary'] for s in all_signals])
        
        # Count signal types
        signal_types = {}
        for signal in all_signals:
            sig_type = signal['signal_type']
            signal_types[sig_type] = signal_types.get(sig_type, 0) + 1
        
        # Check for agreement
        directions = [s['direction'] for s in all_signals]
        direction_counts = {d: directions.count(d) for d in set(directions)}
        dominant_direction = max(direction_counts, key=direction_counts.get)
        agreement_pct = direction_counts[dominant_direction] / len(directions)
        
        # Determine recommendation
        if confluence_signals:
            recommendation = 'STRONG_' + confluence_signals[0]['direction'].upper()
            quality = 'EXCELLENT'
        elif agreement_pct >= 0.67:  # 2/3 or more agree
            recommendation = dominant_direction.upper()
            quality = 'GOOD'
        else:
            recommendation = 'WAIT'
            quality = 'MIXED'
        
        return {
            'status': 'SIGNALS_AVAILABLE',
            'total_signals': len(all_signals),
            'confluence_signals': len(confluence_signals),
            'avg_confidence': round(avg_confidence, 3),
            'avg_risk_reward': round(avg_rr, 2),
            'signal_types': signal_types,
            'direction_agreement': {
                'dominant': dominant_direction,
                'agreement_pct': round(agreement_pct * 100, 1),
                'breakdown': direction_counts
            },
            'recommendation': recommendation,
            'quality': quality,
            'has_triple_confluence': any(s.get('confluence_type') == 'TRIPLE' for s in confluence_signals)
        }
    
    def generate_daily_signals(self) -> Dict:
        """
        Generate signals for all pairs (daily routine)
        
        Returns:
            Dictionary with signals for all pairs
        """
        logger.info("\n" + "="*80)
        logger.info("GENERATING DAILY SIGNALS FOR ALL PAIRS")
        logger.info("="*80)
        
        all_results = {}
        
        for pair in self.pairs:
            try:
                # Load data for pair
                df = self._load_pair_data(pair)
                
                if df is None or len(df) < 100:
                    logger.warning(f"Insufficient data for {pair}")
                    continue
                
                # Generate signals
                result = self.generate_all_signals(pair, df)
                all_results[pair] = result
                
            except Exception as e:
                logger.error(f"Error generating signals for {pair}: {e}")
                all_results[pair] = {
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        # Export results
        export_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_path = f"signals/daily_signals_{export_timestamp}.json"
        
        import json
        Path(export_path).parent.mkdir(parents=True, exist_ok=True)
        with open(export_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        logger.info(f"\n✅ Daily signals generated for {len(all_results)} pairs")
        logger.info(f"💾 Exported to: {export_path}")
        
        return all_results
    
    def _load_pair_data(self, pair: str) -> Optional[pd.DataFrame]:
        """Load historical data for a pair"""
        try:
            # Try multiple possible file locations
            possible_paths = [
                self.data_dir / f"{pair}.csv",
                self.data_dir / f"{pair}_H1.csv",
                self.data_dir / f"{pair}_D1.csv",
                Path('data') / f"{pair}.csv"
            ]
            
            for path in possible_paths:
                if path.exists():
                    df = pd.read_csv(path)
                    
                    # Ensure required columns
                    required = ['Open', 'High', 'Low', 'Close']
                    if not all(col in df.columns for col in required):
                        continue
                    
                    logger.info(f"Loaded {len(df)} bars for {pair} from {path}")
                    return df
            
            logger.warning(f"No data file found for {pair}")
            return None
            
        except Exception as e:
            logger.error(f"Error loading data for {pair}: {e}")
            return None
    
    def get_performance_report(self) -> Dict:
        """Get performance report for all pairs"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'pairs': {}
        }
        
        for pair in self.pairs:
            report['pairs'][pair] = self.aggregator.get_signal_summary(pair)
        
        return report


def demo():
    """Demonstrate enhanced signal service"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║         ENHANCED SIGNAL INTEGRATION SERVICE                      ║
    ║         Multiple Models, Optimal Risk Management                 ║
    ╚══════════════════════════════════════════════════════════════════╝
    
    This service provides:
    
    ✅ Multiple Signal Types:
       • Individual: ML, Harmonic, Quantum (2:1 to 4:1 R:R)
       • Confluence: 2 models agree (3:1 to 6:1 R:R)
       • Triple: All models agree (4:1 to 8:1 R:R)
    
    ✅ Proper Risk Management:
       • Every signal has 2:1+ R:R minimum
       • Multiple take-profit levels
       • Conservative stop losses
       • Pip-based calculations
    
    ✅ Quality Filtering:
       • Only high-probability setups
       • Confidence-based targets
       • Setup quality ratings
       • Performance tracking
    
    Usage:
    ------
    service = EnhancedSignalService(pairs=['EURUSD', 'XAUUSD'])
    
    # Generate signals for one pair
    result = service.generate_all_signals('EURUSD', df)
    
    # Generate daily signals for all pairs
    daily_results = service.generate_daily_signals()
    
    # Get performance report
    report = service.get_performance_report()
    """)
    
    # Initialize service
    service = EnhancedSignalService(pairs=['EURUSD', 'XAUUSD'])
    
    print("\n✅ Enhanced Signal Service initialized")
    print(f"   Pairs: {service.pairs}")
    print(f"   ML Models loaded: {list(service.ml_models.keys())}")
    print(f"   Quantum Generators: {list(service.quantum_generators.keys())}")
    
    print("\n💡 Ready to generate high-quality signals with 2:1 to 5:1+ R:R ratios!")


if __name__ == '__main__':
    demo()
