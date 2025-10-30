#!/usr/bin/env python3
"""
QuantumMultiTimeframeSignalGenerator - Advanced multi-timeframe signal generation

This module implements quantum-inspired multi-timeframe analysis for forex trading:
- Multi-timeframe feature fusion (H1, H4, Daily, Weekly)
- Cross-timeframe correlation analysis
- Quantum superposition-inspired signal combination
- Adaptive weighting based on market conditions
- Unified signal generation with confidence scoring

Features:
- Hierarchical timeframe analysis
- Quantum probability amplitude combination
- Market regime detection
- Adaptive signal filtering
- Cross-pair correlation integration

Usage:
    # Generate multi-timeframe signals
    generator = QuantumMultiTimeframeSignalGenerator('EURUSD')
    signals = generator.generate_unified_signals()

    # Get quantum-enhanced predictions
    quantum_signal = generator.get_quantum_signal()
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import joblib
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# Technical analysis
try:
    import ta
except ImportError:
    ta = None

# Pattern & harmonic detection
try:
    from .pattern_harmonic_detector import PatternHarmonicDetector
    from .fundamental_features import FundamentalFeatureEngineer
except ImportError:  # pragma: no cover - support running as script
    from pattern_harmonic_detector import PatternHarmonicDetector
    from fundamental_features import FundamentalFeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QuantumMultiTimeframeSignalGenerator:
    """
    Quantum-inspired multi-timeframe signal generator for forex trading.

    Combines signals from multiple timeframes using advanced fusion techniques
    inspired by quantum computing principles.
    """

    def __init__(self, pair: str, data_dir: str = "data", models_dir: str = "models"):
        """
        Initialize the quantum signal generator.

        Args:
            pair: Currency pair (e.g., 'EURUSD', 'XAUUSD')
            data_dir: Directory containing price data
            models_dir: Directory containing trained models
        """
        self.pair = pair
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)

        # Timeframe configurations
        self.timeframes = {
            'H1': {'period': '1H', 'weight': 0.2, 'lookback': 168},  # 1 week
            'H4': {'period': '4H', 'weight': 0.3, 'lookback': 168},  # 1 month
            'D1': {'period': '1D', 'weight': 0.4, 'lookback': 252},  # 1 year
            'W1': {'period': '1W', 'weight': 0.1, 'lookback': 104}   # 2 years
        }

        # Load data and models
        self.price_data = {}
        self.feature_data = {}
        self.timeframe_models = {}
        self.quantum_weights = {}

        self._load_data()
        self._load_models()

        # Signal generation parameters
        self.min_confidence = 0.6
        self.max_correlation_threshold = 0.8
        self.adaptive_weighting = True

    # Advanced feature utilities
    self.pattern_detector = PatternHarmonicDetector()
    self.fundamental_engineer = FundamentalFeatureEngineer(self.data_dir)
    self.feature_log_dir = Path("output") / "feature_logs"
    self.feature_log_dir.mkdir(parents=True, exist_ok=True)

    def _load_data(self):
        """Load price data for all timeframes."""
        try:
            # Load base daily data
            daily_file = self.data_dir / "raw" / f"{self.pair}_Daily.csv"
            if daily_file.exists():
                daily_data = pd.read_csv(daily_file)
                daily_data['Date'] = pd.to_datetime(daily_data['Date'])
                daily_data = daily_data.set_index('Date')

                # Resample to different timeframes
                self.price_data['D1'] = daily_data

                # Create H4 data (assuming daily data, we'll simulate)
                self.price_data['H4'] = self._resample_to_timeframe(daily_data, '4H')
                self.price_data['W1'] = self._resample_to_timeframe(daily_data, 'W')

                # For H1, we'll use the same as H4 for now (in practice, you'd have hourly data)
                self.price_data['H1'] = self.price_data['H4'].copy()

                logger.info(f"Loaded price data for {self.pair}: { {tf: len(df) for tf, df in self.price_data.items()} }")

            # Load fundamental data
            self.fundamental_data = self._load_fundamental_data()

        except Exception as e:
            logger.error(f"Error loading data: {e}")

    def _resample_to_timeframe(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample data to specified timeframe."""
        try:
            # Resample OHLC data
            resampled = df.resample(timeframe).agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum' if 'Volume' in df.columns else 'count'
            }).dropna()

            return resampled

        except Exception as e:
            logger.warning(f"Error resampling to {timeframe}: {e}")
            return df

    def _load_fundamental_data(self) -> pd.DataFrame:
        """Load fundamental economic data."""
        fundamental_df = pd.DataFrame()

        # Load key economic indicators
        key_series = [
            'FEDFUNDS',  # Fed Funds Rate
            'CPIAUCSL',  # US CPI
            'DEXUSEU',   # EUR/USD rate
            'GOLDAMGBD228NLBM',  # Gold price
            'DGS10'      # 10Y Treasury
        ]

        for series_id in key_series:
            csv_file = self.data_dir / f"{series_id}.csv"
            if csv_file.exists():
                try:
                    series_df = pd.read_csv(csv_file)
                    series_df['date'] = pd.to_datetime(series_df['date'])
                    series_df = series_df.set_index('date')

                    # Rename value column to series name
                    series_df = series_df.rename(columns={'value': series_id})

                    if fundamental_df.empty:
                        fundamental_df = series_df
                    else:
                        fundamental_df = fundamental_df.join(series_df, how='outer')

                except Exception as e:
                    logger.warning(f"Error loading {series_id}: {e}")

        if fundamental_df.empty:
            logger.info("Fundamental dataset empty or unavailable")
            return fundamental_df

        enhanced = self.fundamental_engineer.enhance(fundamental_df)
        logger.info(f"Loaded fundamental data with {len(enhanced)} observations and {len(enhanced.columns)} features")
        return enhanced

    def _load_models(self):
        """Load trained models for each timeframe."""
        try:
            # Try to load timeframe-specific models
            for tf in self.timeframes.keys():
                model_file = self.models_dir / f"{self.pair}_{tf}_model.joblib"
                if model_file.exists():
                    self.timeframe_models[tf] = joblib.load(model_file)
                    logger.info(f"Loaded {tf} model for {self.pair}")
                else:
                    logger.warning(f"{tf} model not found: {model_file}")

            # Load ensemble model as fallback
            ensemble_file = self.models_dir / f"{self.pair}_ensemble.joblib"
            if ensemble_file.exists():
                ensemble_data = joblib.load(ensemble_file)
                self.ensemble_model = ensemble_data
                logger.info(f"Loaded ensemble model as fallback")

        except Exception as e:
            logger.error(f"Error loading models: {e}")

    def _engineer_timeframe_features(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Engineer technical features for a specific timeframe.

        Args:
            df: Price data for the timeframe
            timeframe: Timeframe identifier

        Returns:
            DataFrame with engineered features
        """
        if df.empty:
            return df

        df = df.copy()

        # Basic price features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))

        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['Close'].rolling(period).mean()
            df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()

        # Volatility measures
        df['volatility_20'] = df['returns'].rolling(20).std()
        df['volatility_50'] = df['returns'].rolling(50).std()

        # Momentum indicators
        df['rsi_14'] = self._calculate_rsi(df['Close'], 14)
        df['macd'], df['macd_signal'], df['macd_hist'] = self._calculate_macd(df['Close'])

        # Trend indicators
        df['adx_14'] = self._calculate_adx(df, 14)

        # Support/Resistance levels
        df['pivot_point'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['r1'] = 2 * df['pivot_point'] - df['Low']
        df['s1'] = 2 * df['pivot_point'] - df['High']

        # Lagged features
        for lag in [1, 2, 3]:
            df[f'close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)

        # Target variables (future returns)
        for horizon in [1, 3, 5]:
            df[f'target_{horizon}d'] = df['Close'].shift(-horizon) / df['Close'] - 1

        # Timeframe-specific features
        tf_config = self.timeframes[timeframe]
        lookback = tf_config['lookback']

        # Rolling statistics
        df['rolling_mean_20'] = df['Close'].rolling(20).mean()
        df['rolling_std_20'] = df['Close'].rolling(20).std()
        df['rolling_skew_20'] = df['Close'].rolling(20).skew()
        df['rolling_kurt_20'] = df['Close'].rolling(20).kurt()

        # Volume-based indicators (if volume exists)
        if 'Volume' in df.columns:
            df['volume_sma_20'] = df['Volume'].rolling(20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_sma_20']

        # Add fundamental features if available
        if not self.fundamental_data.empty:
            # Merge fundamental data
            df = df.join(self.fundamental_data, how='left')

            # Forward fill missing fundamental data
            fundamental_cols = [col for col in df.columns if col in self.fundamental_data.columns]
            df[fundamental_cols] = df[fundamental_cols].fillna(method='ffill')

        # Pattern & harmonic detection
        df = self.pattern_detector.augment(df)

        # Drop rows with NaN values
        df = df.dropna()

        logger.info(f"Engineered {len(df.columns)} features for {timeframe} timeframe")
        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist

    def _calculate_adx(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average Directional Index."""
        high = df['High']
        low = df['Low']
        close = df['Close']

        # Calculate True Range
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)

        # Calculate Directional Movement
        dm_plus = np.where((high - high.shift(1)) > (low.shift(1) - low),
                          np.maximum(high - high.shift(1), 0), 0)
        dm_minus = np.where((low.shift(1) - low) > (high - high.shift(1)),
                           np.maximum(low.shift(1) - low, 0), 0)

        # Calculate Directional Indicators
        di_plus = 100 * (pd.Series(dm_plus).rolling(period).mean() / tr.rolling(period).mean())
        di_minus = 100 * (pd.Series(dm_minus).rolling(period).mean() / tr.rolling(period).mean())

        # Calculate ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(period).mean()

        return adx

    def _detect_market_regime(self, df: pd.DataFrame) -> str:
        """
        Detect current market regime based on volatility and trend.

        Args:
            df: Price data

        Returns:
            Market regime ('trending', 'ranging', 'volatile', 'calm')
        """
        try:
            # Calculate recent volatility
            recent_vol = df['returns'].tail(20).std()

            # Calculate trend strength
            sma_20 = df['Close'].rolling(20).mean()
            trend_strength = abs(df['Close'].iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1]

            # Determine regime
            if recent_vol > df['returns'].quantile(0.8):  # High volatility
                if trend_strength > 0.02:  # Strong trend
                    return 'volatile_trending'
                else:
                    return 'volatile'
            else:  # Low volatility
                if trend_strength > 0.01:  # Moderate trend
                    return 'trending'
                else:
                    return 'ranging'

        except Exception as e:
            logger.warning(f"Error detecting market regime: {e}")
            return 'unknown'

    def _calculate_timeframe_correlations(self) -> Dict[str, float]:
        """
        Calculate correlations between different timeframes.

        Returns:
            Dictionary of correlation coefficients
        """
        correlations = {}

        try:
            # Get common date range
            common_dates = None
            for tf, df in self.price_data.items():
                if common_dates is None:
                    common_dates = set(df.index)
                else:
                    common_dates = common_dates.intersection(set(df.index))

            if not common_dates:
                return correlations

            common_dates = sorted(list(common_dates))

            # Calculate correlations for returns
            for i, tf1 in enumerate(self.timeframes.keys()):
                for tf2 in list(self.timeframes.keys())[i+1:]:
                    if tf1 in self.price_data and tf2 in self.price_data:
                        df1 = self.price_data[tf1].loc[common_dates]
                        df2 = self.price_data[tf2].loc[common_dates]

                        if len(df1) > 10 and len(df2) > 10:
                            corr = df1['Close'].pct_change().corr(df2['Close'].pct_change())
                            if not np.isnan(corr):
                                correlations[f'{tf1}_{tf2}'] = float(corr)

        except Exception as e:
            logger.warning(f"Error calculating timeframe correlations: {e}")

        return correlations

    def _quantum_signal_fusion(self, timeframe_signals: Dict[str, Dict]) -> Dict:
        """
        Fuse signals from multiple timeframes using quantum-inspired methods.

        Args:
            timeframe_signals: Dictionary of signals from each timeframe

        Returns:
            Fused quantum signal
        """
        try:
            # Extract signal components
            directions = []
            strengths = []
            confidences = []

            for tf, signal in timeframe_signals.items():
                if signal and 'direction' in signal:
                    # Convert direction to numerical value
                    direction_val = 1 if signal['direction'] == 'bullish' else -1 if signal['direction'] == 'bearish' else 0
                    directions.append(direction_val)

                    strength = signal.get('strength', 0.5)
                    strengths.append(strength)

                    confidence = signal.get('confidence', 0.5)
                    confidences.append(confidence)

            if not directions:
                return {
                    'direction': 'neutral',
                    'strength': 0.0,
                    'confidence': 0.0,
                    'quantum_coherence': 0.0
                }

            # Quantum-inspired fusion
            directions = np.array(directions)
            strengths = np.array(strengths)
            confidences = np.array(confidences)

            # Calculate quantum coherence (alignment of signals)
            coherence = np.abs(np.mean(directions * strengths * confidences))

            # Adaptive weighting based on market conditions
            if self.adaptive_weighting:
                weights = self._calculate_adaptive_weights(timeframe_signals)
            else:
                weights = np.array([self.timeframes[tf]['weight'] for tf in timeframe_signals.keys()])

            # Normalize weights
            weights = weights / np.sum(weights)

            # Weighted fusion
            quantum_direction = np.sum(directions * strengths * confidences * weights)
            quantum_strength = np.sum(strengths * confidences * weights)
            quantum_confidence = coherence * np.mean(confidences)

            # Determine final direction
            if abs(quantum_direction) < 0.1:
                final_direction = 'neutral'
            elif quantum_direction > 0:
                final_direction = 'bullish'
            else:
                final_direction = 'bearish'

            return {
                'direction': final_direction,
                'strength': float(abs(quantum_direction)),
                'confidence': float(min(quantum_confidence, 1.0)),
                'quantum_coherence': float(coherence),
                'timeframe_contributions': {
                    tf: {
                        'direction': signal.get('direction', 'neutral'),
                        'weight': float(weights[i]),
                        'contribution': float(directions[i] * strengths[i] * confidences[i] * weights[i])
                    }
                    for i, (tf, signal) in enumerate(timeframe_signals.items())
                }
            }

        except Exception as e:
            logger.error(f"Error in quantum signal fusion: {e}")
            return {
                'direction': 'error',
                'strength': 0.0,
                'confidence': 0.0,
                'quantum_coherence': 0.0
            }

    def _calculate_adaptive_weights(self, timeframe_signals: Dict[str, Dict]) -> np.ndarray:
        """
        Calculate adaptive weights based on signal quality and market conditions.

        Args:
            timeframe_signals: Dictionary of signals from each timeframe

        Returns:
            Array of adaptive weights
        """
        try:
            base_weights = np.array([self.timeframes[tf]['weight'] for tf in timeframe_signals.keys()])

            # Adjust weights based on signal confidence
            confidence_adjustments = np.array([
                signal.get('confidence', 0.5) for signal in timeframe_signals.values()
            ])

            # Adjust weights based on market regime
            regime = self._detect_market_regime(self.price_data.get('D1', pd.DataFrame()))
            regime_weights = self._get_regime_weights(regime)

            # Combine adjustments
            adaptive_weights = base_weights * confidence_adjustments * regime_weights

            # Ensure minimum weight for each timeframe
            min_weight = 0.05
            adaptive_weights = np.maximum(adaptive_weights, min_weight)

            return adaptive_weights / np.sum(adaptive_weights)

        except Exception as e:
            logger.warning(f"Error calculating adaptive weights: {e}")
            return np.array([self.timeframes[tf]['weight'] for tf in timeframe_signals.keys()])

    def _get_regime_weights(self, regime: str) -> np.ndarray:
        """
        Get timeframe weights based on market regime.

        Args:
            regime: Current market regime

        Returns:
            Array of regime-based weights
        """
        # Different timeframes perform better in different regimes
        regime_weights = {
            'trending': np.array([0.1, 0.2, 0.5, 0.2]),  # Favor daily and weekly in trends
            'ranging': np.array([0.3, 0.3, 0.3, 0.1]),    # Balance across timeframes
            'volatile': np.array([0.4, 0.3, 0.2, 0.1]),   # Favor shorter timeframes
            'volatile_trending': np.array([0.2, 0.3, 0.4, 0.1]),  # Medium timeframes
            'calm': np.array([0.1, 0.2, 0.4, 0.3]),       # Favor longer timeframes
            'unknown': np.array([0.25, 0.25, 0.25, 0.25]) # Equal weights
        }

        return regime_weights.get(regime, regime_weights['unknown'])

    def _generate_timeframe_signal(self, timeframe: str) -> Optional[Dict]:
        """
        Generate signal for a specific timeframe.

        Args:
            timeframe: Timeframe identifier

        Returns:
            Signal dictionary or None if generation fails
        """
        try:
            if timeframe not in self.price_data:
                return None

            df = self.price_data[timeframe]
            if len(df) < 50:
                return None

            # Engineer features
            feature_df = self._engineer_timeframe_features(df, timeframe)

            if len(feature_df) < 10:
                return None

            # Get latest features
            latest_features = feature_df.iloc[-1:]
            feature_cols = [col for col in feature_df.columns
                          if not col.startswith('target_') and col != 'Close']

            X = latest_features[feature_cols].values

            # Feature logging for diagnostics
            self._log_feature_snapshot(timeframe, feature_cols, feature_df)

            # Generate prediction
            if timeframe in self.timeframe_models:
                # Use timeframe-specific model
                model = self.timeframe_models[timeframe]
                prediction = model.predict(X)[0]
            elif hasattr(self, 'ensemble_model') and 'models' in self.ensemble_model:
                # Use ensemble model
                prediction = 0
                model_count = 0

                for model_name, model in self.ensemble_model['models'].items():
                    if hasattr(model, 'predict'):
                        try:
                            pred = model.predict(X)[0] if X.ndim > 1 else model.predict(X.reshape(1, -1))[0]
                            prediction += pred
                            model_count += 1
                        except:
                            continue

                if model_count > 0:
                    prediction /= model_count
                else:
                    return None
            else:
                # Simple momentum-based signal
                recent_returns = df['Close'].pct_change().tail(5).mean()
                prediction = recent_returns

            # Determine signal direction and strength
            if abs(prediction) < 0.001:
                direction = 'neutral'
                strength = 0.0
            elif prediction > 0.002:
                direction = 'bullish'
                strength = min(abs(prediction) * 100, 1.0)
            elif prediction < -0.002:
                direction = 'bearish'
                strength = min(abs(prediction) * 100, 1.0)
            else:
                direction = 'neutral'
                strength = 0.0

            # Calculate confidence based on prediction magnitude and consistency
            confidence = min(abs(prediction) * 200, 1.0)

            # Additional factors for confidence
            recent_volatility = df['Close'].pct_change().tail(20).std()
            if recent_volatility > df['Close'].pct_change().quantile(0.8):
                confidence *= 0.8  # Reduce confidence in high volatility

            return {
                'timeframe': timeframe,
                'direction': direction,
                'strength': strength,
                'confidence': confidence,
                'prediction': float(prediction),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error generating {timeframe} signal: {e}")
            return None

    def _log_feature_snapshot(self, timeframe: str, feature_cols: List[str], feature_df: pd.DataFrame) -> None:
        """Persist a snapshot of the latest engineered features for diagnostics."""
        if feature_df.empty:
            return

        try:
            latest = feature_df.iloc[-1]
            recent_window = feature_df.tail(30)
            timestamp = datetime.utcnow()

            pattern_cols = [col for col in feature_cols if col.startswith('pattern_')]
            harmonic_cols = [col for col in feature_cols if col.startswith('harmonic_')]

            top_features = [
                {
                    'name': name,
                    'value': float(latest[name])
                }
                for name in latest[feature_cols].abs().sort_values(ascending=False).head(15).index
            ]

            pattern_summary = {
                name: {
                    'latest': float(latest[name]),
                    'mean_30': float(recent_window[name].mean()),
                    'max_30': float(recent_window[name].max())
                }
                for name in pattern_cols
            }

            harmonic_summary = {
                name: {
                    'latest': float(latest[name]),
                    'mean_30': float(recent_window[name].mean()),
                    'max_30': float(recent_window[name].max())
                }
                for name in harmonic_cols
            }

            log_payload = {
                'pair': self.pair,
                'timeframe': timeframe,
                'timestamp': timestamp.isoformat(),
                'pattern_summary': pattern_summary,
                'harmonic_summary': harmonic_summary,
                'top_features': top_features,
                'feature_count': len(feature_cols)
            }

            log_file = self.feature_log_dir / f"{self.pair}_{timeframe}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            with log_file.open('w') as fp:
                json.dump(log_payload, fp, indent=2)

        except Exception as exc:  # pragma: no cover - logging should not break signals
            logger.warning(f"Failed to log feature snapshot for {timeframe}: {exc}")

    def generate_unified_signals(self) -> Dict:
        """
        Generate unified signals across all timeframes using quantum fusion.

        Returns:
            Dictionary containing unified signal analysis
        """
        logger.info(f"Generating unified signals for {self.pair}")

        # Generate signals for each timeframe
        timeframe_signals = {}
        for tf in self.timeframes.keys():
            signal = self._generate_timeframe_signal(tf)
            if signal:
                timeframe_signals[tf] = signal

        if not timeframe_signals:
            return {
                'pair': self.pair,
                'error': 'No signals generated',
                'timestamp': datetime.now().isoformat()
            }

        # Apply quantum signal fusion
        quantum_signal = self._quantum_signal_fusion(timeframe_signals)

        # Calculate cross-timeframe correlations
        correlations = self._calculate_timeframe_correlations()

        # Detect market regime
        market_regime = self._detect_market_regime(self.price_data.get('D1', pd.DataFrame()))

        # Compile unified signal
        unified_signal = {
            'pair': self.pair,
            'timestamp': datetime.now().isoformat(),
            'market_regime': market_regime,
            'quantum_signal': quantum_signal,
            'timeframe_signals': timeframe_signals,
            'correlations': correlations,
            'signal_quality': self._assess_signal_quality(quantum_signal, timeframe_signals),
            'recommendations': self._generate_trading_recommendations(quantum_signal)
        }

        return unified_signal

    def get_quantum_signal(self) -> Dict:
        """
        Get the primary quantum-enhanced trading signal.

        Returns:
            Simplified quantum signal for trading
        """
        try:
            unified_signals = self.generate_unified_signals()

            if 'error' in unified_signals:
                return {
                    'pair': self.pair,
                    'signal': 'no_signal',
                    'confidence': 0.0,
                    'reason': unified_signals['error']
                }

            quantum_signal = unified_signals['quantum_signal']

            # Apply confidence threshold
            if quantum_signal['confidence'] < self.min_confidence:
                return {
                    'pair': self.pair,
                    'signal': 'no_signal',
                    'confidence': quantum_signal['confidence'],
                    'reason': 'Low confidence'
                }

            # Check for conflicting signals
            coherence = quantum_signal.get('quantum_coherence', 0)
            if coherence < 0.3:
                return {
                    'pair': self.pair,
                    'signal': 'no_signal',
                    'confidence': quantum_signal['confidence'],
                    'reason': 'Low coherence between timeframes'
                }

            return {
                'pair': self.pair,
                'signal': quantum_signal['direction'],
                'confidence': quantum_signal['confidence'],
                'strength': quantum_signal['strength'],
                'coherence': quantum_signal.get('quantum_coherence', 0),
                'market_regime': unified_signals.get('market_regime', 'unknown'),
                'timestamp': unified_signals['timestamp']
            }

        except Exception as e:
            logger.error(f"Error getting quantum signal: {e}")
            return {
                'pair': self.pair,
                'signal': 'error',
                'confidence': 0.0,
                'reason': str(e)
            }

    def _assess_signal_quality(self, quantum_signal: Dict, timeframe_signals: Dict) -> Dict:
        """Assess the overall quality of the generated signals."""
        try:
            quality = {
                'overall_quality': 'low',
                'confidence_level': 'low',
                'coherence_level': 'low',
                'consistency_score': 0.0
            }

            # Assess confidence
            confidence = quantum_signal.get('confidence', 0)
            if confidence > 0.8:
                quality['confidence_level'] = 'high'
            elif confidence > 0.6:
                quality['confidence_level'] = 'medium'
            else:
                quality['confidence_level'] = 'low'

            # Assess coherence
            coherence = quantum_signal.get('quantum_coherence', 0)
            if coherence > 0.7:
                quality['coherence_level'] = 'high'
            elif coherence > 0.5:
                quality['coherence_level'] = 'medium'
            else:
                quality['coherence_level'] = 'low'

            # Calculate consistency score
            directions = []
            for signal in timeframe_signals.values():
                if signal['direction'] == 'bullish':
                    directions.append(1)
                elif signal['direction'] == 'bearish':
                    directions.append(-1)
                else:
                    directions.append(0)

            if directions:
                consistency = np.abs(np.mean(directions))
                quality['consistency_score'] = float(consistency)

            # Overall quality assessment
            if (quality['confidence_level'] == 'high' and
                quality['coherence_level'] in ['high', 'medium'] and
                quality['consistency_score'] > 0.6):
                quality['overall_quality'] = 'high'
            elif (quality['confidence_level'] in ['high', 'medium'] and
                  quality['coherence_level'] in ['high', 'medium']):
                quality['overall_quality'] = 'medium'

            return quality

        except Exception as e:
            logger.error(f"Error assessing signal quality: {e}")
            return {'overall_quality': 'error'}

    def _generate_trading_recommendations(self, quantum_signal: Dict) -> List[str]:
        """Generate trading recommendations based on the quantum signal."""
        recommendations = []

        try:
            direction = quantum_signal.get('direction', 'neutral')
            confidence = quantum_signal.get('confidence', 0)
            coherence = quantum_signal.get('quantum_coherence', 0)

            if direction == 'bullish' and confidence > 0.7:
                recommendations.append("Strong bullish signal - consider long position")
                if coherence > 0.8:
                    recommendations.append("High coherence across timeframes - good entry signal")

            elif direction == 'bearish' and confidence > 0.7:
                recommendations.append("Strong bearish signal - consider short position")
                if coherence > 0.8:
                    recommendations.append("High coherence across timeframes - good entry signal")

            elif confidence < 0.5:
                recommendations.append("Low confidence signal - avoid trading or reduce position size")

            if coherence < 0.4:
                recommendations.append("Low coherence between timeframes - exercise caution")

            # Risk management recommendations
            recommendations.append("Always use stop-loss orders")
            recommendations.append("Consider position sizing based on confidence level")

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations = ["Error generating recommendations"]

        return recommendations

    def train_timeframe_models(self) -> Dict[str, bool]:
        """
        Train models for each timeframe.

        Returns:
            Dictionary indicating success for each timeframe
        """
        logger.info(f"Training timeframe models for {self.pair}")

        results = {}

        for tf, config in self.timeframes.items():
            try:
                if tf not in self.price_data:
                    results[tf] = False
                    continue

                # Engineer features
                feature_df = self._engineer_timeframe_features(self.price_data[tf], tf)

                if len(feature_df) < 100:
                    logger.warning(f"Insufficient data for {tf} training")
                    results[tf] = False
                    continue

                # Prepare training data
                feature_cols = [col for col in feature_df.columns
                              if not col.startswith('target_') and col != 'Close']
                target_col = 'target_1d'

                X = feature_df[feature_cols].values
                y = feature_df[target_col].values

                # Remove NaN values
                valid_idx = ~(np.isnan(X).any(axis=1) | np.isnan(y))
                X = X[valid_idx]
                y = y[valid_idx]

                if len(X) < 50:
                    results[tf] = False
                    continue

                # Train model
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )

                model.fit(X, y)

                # Save model
                model_file = self.models_dir / f"{self.pair}_{tf}_model.joblib"
                joblib.dump(model, model_file)

                self.timeframe_models[tf] = model
                results[tf] = True
                logger.info(f"Successfully trained {tf} model")

            except Exception as e:
                logger.error(f"Error training {tf} model: {e}")
                results[tf] = False

        return results


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description='Quantum Multi-Timeframe Signal Generator')
    parser.add_argument('--pair', required=True, help='Currency pair (EURUSD, XAUUSD)')
    parser.add_argument('--signal', action='store_true', help='Generate quantum signal')
    parser.add_argument('--unified', action='store_true', help='Generate unified signals')
    parser.add_argument('--train', action='store_true', help='Train timeframe models')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--models-dir', default='models', help='Models directory')

    args = parser.parse_args()

    # Initialize generator
    generator = QuantumMultiTimeframeSignalGenerator(
        pair=args.pair,
        data_dir=args.data_dir,
        models_dir=args.models_dir
    )

    if args.train:
        # Train models
        results = generator.train_timeframe_models()
        print(f"Training results: {results}")

    elif args.signal:
        # Generate quantum signal
        signal = generator.get_quantum_signal()
        print(f"Quantum Signal: {signal}")

    elif args.unified:
        # Generate unified signals
        signals = generator.generate_unified_signals()
        print(f"Unified Signals: {signals}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()