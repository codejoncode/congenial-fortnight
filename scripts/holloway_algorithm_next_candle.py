#!/usr/bin/env python3
"""
Enhanced Holloway Algorithm - Next Candle Prediction System

This system predicts not just direction, but full OHLC values for the next candle
with advanced pattern recognition and historical analysis.

Key Improvements:
1. Bull/Bear count crossing average detection (earliest signal)
2. Historical high/low support/resistance levels
3. W/M pattern detection from count lines
4. Explosion move detection (large point adjustments)
5. Mirroring behavior between bull and bear counts
6. Full OHLC prediction with confidence intervals
7. 75% accuracy tracking for candle accuracy

Target: 85%+ directional accuracy, 75%+ OHLC accuracy
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib
from typing import Dict, Tuple, List
import json
from datetime import datetime

warnings.filterwarnings('ignore')


class EnhancedHollowayPredictor:
    """
    Enhanced next-candle prediction system with full OHLC forecasting
    and advanced Holloway count analysis.
    """
    
    def __init__(self, data_folder='data', models_folder='models'):
        self.data_folder = Path(data_folder)
        self.models_folder = Path(models_folder)
        self.data_folder.mkdir(exist_ok=True)
        self.models_folder.mkdir(exist_ok=True)
        
        # Scalers
        self.direction_scaler = StandardScaler()
        self.ohlc_scaler = StandardScaler()
        
        # Models
        self.direction_model = None  # Predict bull/bear
        self.open_model = None       # Predict next open
        self.high_model = None       # Predict next high
        self.low_model = None        # Predict next low
        self.close_model = None      # Predict next close
        
        # Performance tracking
        self.prediction_history = []
        
    def calculate_moving_averages(self, df):
        """Calculate all required moving averages"""
        periods_exp = [5, 7, 10, 14, 20, 28, 50, 56, 100, 112, 200, 225]
        periods_sma = [5, 7, 10, 14, 20, 28, 50, 56, 100, 112, 200, 225]
        
        mas = {}
        for period in periods_exp:
            mas[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        for period in periods_sma:
            mas[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            
        return pd.DataFrame(mas, index=df.index)
    
    def calculate_holloway_signals(self, df, mas):
        """Calculate bull and bear counts with all Holloway conditions"""
        close = df['close']
        
        exp_periods = [5, 7, 10, 14, 20, 28, 50, 56, 100, 112, 200, 225]
        sma_periods = [5, 7, 10, 14, 20, 28, 50, 56, 100, 112, 200, 225]
        
        bull_signals = pd.DataFrame(index=df.index)
        bear_signals = pd.DataFrame(index=df.index)
        
        # Price vs MAs
        for period in exp_periods:
            col = f'ema_{period}'
            bull_signals[f'above_{col}'] = close > mas[col]
            bear_signals[f'below_{col}'] = close < mas[col]
            
        for period in sma_periods:
            col = f'sma_{period}'
            bull_signals[f'above_{col}'] = close > mas[col]
            bear_signals[f'below_{col}'] = close < mas[col]
        
        # EMA alignment
        for i, p1 in enumerate(exp_periods[:-1]):
            for p2 in exp_periods[i+1:]:
                bull_signals[f'ema{p1}_gt_ema{p2}'] = mas[f'ema_{p1}'] > mas[f'ema_{p2}']
                bear_signals[f'ema{p1}_lt_ema{p2}'] = mas[f'ema_{p1}'] < mas[f'ema_{p2}']
        
        # SMA alignment
        for i, p1 in enumerate(sma_periods[:-1]):
            for p2 in sma_periods[i+1:]:
                bull_signals[f'sma{p1}_gt_sma{p2}'] = mas[f'sma_{p1}'] > mas[f'sma_{p2}']
                bear_signals[f'sma{p1}_lt_sma{p2}'] = mas[f'sma_{p1}'] < mas[f'sma_{p2}']
        
        # EMA vs SMA
        for p1 in exp_periods:
            for p2 in sma_periods:
                bull_signals[f'ema{p1}_gt_sma{p2}'] = mas[f'ema_{p1}'] > mas[f'sma_{p2}']
                bear_signals[f'ema{p1}_lt_sma{p2}'] = mas[f'ema_{p1}'] < mas[f'sma_{p2}']
        
        # Fresh crossovers
        for period in exp_periods + sma_periods:
            if period in exp_periods:
                col = f'ema_{period}'
            else:
                col = f'sma_{period}'
            bull_signals[f'fresh_{col}'] = (close > mas[col]) & (close.shift(1) <= mas[col].shift(1))
            bear_signals[f'fresh_{col}'] = (close < mas[col]) & (close.shift(1) >= mas[col].shift(1))
        
        bull_count = bull_signals.astype(int).sum(axis=1)
        bear_count = bear_signals.astype(int).sum(axis=1)
        
        return bull_count, bear_count
    
    def calculate_holloway_averages(self, bull_count, bear_count, period=27):
        """Calculate smoothed averages (bully/beary)"""
        # Double EMA smoothing
        bully = bull_count.ewm(span=period, adjust=False).mean()
        bully = bully.ewm(span=period, adjust=False).mean()
        
        beary = bear_count.ewm(span=period, adjust=False).mean()
        beary = beary.ewm(span=period, adjust=False).mean()
        
        return bully, beary
    
    def detect_count_average_crossovers(self, bull_count, bear_count, bully, beary):
        """
        CRITICAL: Detect when counts cross their averages (earliest signal!)
        
        Key insight from your analysis:
        - Bull count dropping below bully = bearish (even if still above bear/beary)
        - Bear count dropping below beary = bullish
        - These are THE FASTEST signals
        """
        signals = pd.DataFrame(index=bull_count.index)
        
        # Fastest signals (count vs average)
        signals['bull_below_bully'] = (bull_count < bully) & (bull_count.shift(1) >= bully.shift(1))
        signals['bull_above_bully'] = (bull_count > bully) & (bull_count.shift(1) <= bully.shift(1))
        signals['bear_below_beary'] = (bear_count < beary) & (bear_count.shift(1) >= beary.shift(1))
        signals['bear_above_beary'] = (bear_count > beary) & (bear_count.shift(1) <= beary.shift(1))
        
        # Fast signals (count vs count)
        signals['bull_above_bear'] = (bull_count > bear_count) & (bull_count.shift(1) <= bear_count.shift(1))
        signals['bear_above_bull'] = (bear_count > bull_count) & (bear_count.shift(1) <= bull_count.shift(1))
        
        # Slower signals (average vs average)
        signals['bully_above_beary'] = (bully > beary) & (bully.shift(1) <= beary.shift(1))
        signals['beary_above_bully'] = (beary > bully) & (beary.shift(1) <= bully.shift(1))
        
        # Composite earliest signal
        signals['earliest_bullish'] = signals['bull_above_bully'] | signals['bear_below_beary']
        signals['earliest_bearish'] = signals['bull_below_bully'] | signals['bear_above_beary']
        
        return signals
    
    def detect_historical_levels(self, bull_count, bear_count, lookback=100):
        """
        Detect historical highs/lows that act as support/resistance.
        
        Key insight: When count reaches past high/low, price tends to reverse.
        """
        levels = pd.DataFrame(index=bull_count.index)
        
        # Rolling highs/lows
        levels['bull_high_100'] = bull_count.rolling(lookback).max()
        levels['bull_low_100'] = bull_count.rolling(lookback).min()
        levels['bear_high_100'] = bear_count.rolling(lookback).max()
        levels['bear_low_100'] = bear_count.rolling(lookback).min()
        
        # Distance to levels (normalized)
        levels['bull_to_high_pct'] = (levels['bull_high_100'] - bull_count) / bull_count
        levels['bull_to_low_pct'] = (bull_count - levels['bull_low_100']) / bull_count
        levels['bear_to_high_pct'] = (levels['bear_high_100'] - bear_count) / bear_count
        levels['bear_to_low_pct'] = (bear_count - levels['bear_low_100']) / bear_count
        
        # Near level flags (within 5%)
        levels['bull_near_high'] = levels['bull_to_high_pct'] < 0.05
        levels['bull_near_low'] = levels['bull_to_low_pct'] < 0.05
        levels['bear_near_high'] = levels['bear_to_high_pct'] < 0.05
        levels['bear_near_low'] = levels['bear_to_low_pct'] < 0.05
        
        return levels
    
    def detect_explosion_moves(self, bull_count, bear_count, threshold=10):
        """
        Detect "explosion" moves - large point adjustments from last increment.
        
        Key question: Does large sudden increase/decrease indicate false breakout?
        """
        explosions = pd.DataFrame(index=bull_count.index)
        
        # Calculate point changes
        bull_change = bull_count.diff()
        bear_change = bear_count.diff()
        
        # Detect explosions (change > threshold)
        explosions['bull_explosion_up'] = bull_change > threshold
        explosions['bull_explosion_down'] = bull_change < -threshold
        explosions['bear_explosion_up'] = bear_change > threshold
        explosions['bear_explosion_down'] = bear_change < -threshold
        
        # Track explosion magnitude
        explosions['bull_change_magnitude'] = bull_change.abs()
        explosions['bear_change_magnitude'] = bear_change.abs()
        
        # Rolling average of changes (to detect abnormal)
        avg_bull_change = bull_change.abs().rolling(20).mean()
        avg_bear_change = bear_change.abs().rolling(20).mean()
        
        explosions['bull_abnormal'] = explosions['bull_change_magnitude'] > (avg_bull_change * 2)
        explosions['bear_abnormal'] = explosions['bear_change_magnitude'] > (avg_bear_change * 2)
        
        return explosions
    
    def detect_mirroring_behavior(self, bull_count, bear_count, bully, beary):
        """
        Detect mirroring behavior when bull and bear counts trigger simultaneously.
        
        Key insight: Sometimes both trigger at same time, sometimes not.
        """
        mirrors = pd.DataFrame(index=bull_count.index)
        
        # Detect simultaneous triggers
        bull_below = bull_count < bully
        bear_above = bear_count > beary
        
        mirrors['mirror_bearish'] = bull_below & bear_above
        mirrors['mirror_bullish'] = (~bull_below) & (~bear_above)
        
        # Detect divergence (one triggers, other doesn't)
        mirrors['divergence_bull_only'] = bull_below & (~bear_above)
        mirrors['divergence_bear_only'] = (~bull_below) & bear_above
        
        return mirrors
    
    def detect_wm_patterns(self, bull_count, bear_count, window=20):
        """
        Detect W and M patterns in count lines for support/resistance.
        
        The peaks of M and bottoms of W act as key levels.
        """
        patterns = pd.DataFrame(index=bull_count.index)
        
        # Detect local peaks (M pattern peaks)
        bull_local_max = bull_count.rolling(window, center=True).max()
        bear_local_max = bear_count.rolling(window, center=True).max()
        
        patterns['bull_m_peak'] = bull_count == bull_local_max
        patterns['bear_m_peak'] = bear_count == bear_local_max
        
        # Detect local troughs (W pattern bottoms)
        bull_local_min = bull_count.rolling(window, center=True).min()
        bear_local_min = bear_count.rolling(window, center=True).min()
        
        patterns['bull_w_bottom'] = bull_count == bull_local_min
        patterns['bear_w_bottom'] = bear_count == bear_local_min
        
        # Track levels
        patterns['last_bull_m_peak'] = bull_count.where(patterns['bull_m_peak']).ffill()
        patterns['last_bull_w_bottom'] = bull_count.where(patterns['bull_w_bottom']).ffill()
        patterns['last_bear_m_peak'] = bear_count.where(patterns['bear_m_peak']).ffill()
        patterns['last_bear_w_bottom'] = bear_count.where(patterns['bear_w_bottom']).ffill()
        
        # Distance to pattern levels
        patterns['bull_to_last_peak'] = (patterns['last_bull_m_peak'] - bull_count) / bull_count
        patterns['bull_to_last_bottom'] = (bull_count - patterns['last_bull_w_bottom']) / bull_count
        
        return patterns
    
    def calculate_rsi(self, df, period=14):
        """Calculate RSI indicator"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def create_comprehensive_features(self, df):
        """
        Create comprehensive feature set including ALL Holloway insights.
        """
        print("Calculating comprehensive features...")
        
        # Core Holloway indicators
        mas = self.calculate_moving_averages(df)
        bull_count, bear_count = self.calculate_holloway_signals(df, mas)
        bully, beary = self.calculate_holloway_averages(bull_count, bear_count)
        
        # Advanced Holloway analysis
        crossovers = self.detect_count_average_crossovers(bull_count, bear_count, bully, beary)
        levels = self.detect_historical_levels(bull_count, bear_count)
        explosions = self.detect_explosion_moves(bull_count, bear_count)
        mirrors = self.detect_mirroring_behavior(bull_count, bear_count, bully, beary)
        wm_patterns = self.detect_wm_patterns(bull_count, bear_count)
        
        # RSI
        rsi = self.calculate_rsi(df)
        
        # Combine all features
        features = pd.DataFrame(index=df.index)
        
        # === CORE HOLLOWAY ===
        features['bull_count'] = bull_count
        features['bear_count'] = bear_count
        features['bully'] = bully
        features['beary'] = beary
        features['bull_minus_bear'] = bull_count - bear_count
        features['bully_minus_beary'] = bully - beary
        
        # === MOMENTUM ===
        features['bull_count_change'] = bull_count.diff()
        features['bear_count_change'] = bear_count.diff()
        features['bull_count_change_rate'] = features['bull_count_change'] / bull_count
        features['bear_count_change_rate'] = features['bear_count_change'] / bear_count
        
        # === CROSSOVER SIGNALS (FASTEST) ===
        for col in crossovers.columns:
            features[f'cross_{col}'] = crossovers[col].astype(int)
        
        # === HISTORICAL LEVELS ===
        for col in levels.columns:
            if col.endswith('_pct'):
                features[f'level_{col}'] = levels[col]
            else:
                features[f'level_{col}'] = levels[col].astype(int) if levels[col].dtype == bool else levels[col]
        
        # === EXPLOSIONS ===
        for col in explosions.columns:
            if 'magnitude' in col:
                features[f'exp_{col}'] = explosions[col]
            else:
                features[f'exp_{col}'] = explosions[col].astype(int)
        
        # === MIRRORING ===
        for col in mirrors.columns:
            features[f'mirror_{col}'] = mirrors[col].astype(int)
        
        # === W/M PATTERNS ===
        for col in wm_patterns.columns:
            if col.startswith('last_'):
                features[f'wm_{col}'] = wm_patterns[col]
            elif '_to_' in col:
                features[f'wm_{col}'] = wm_patterns[col]
            else:
                features[f'wm_{col}'] = wm_patterns[col].astype(int)
        
        # === RSI ===
        features['rsi'] = rsi
        features['rsi_above_50'] = (rsi > 50).astype(int)
        features['rsi_change'] = rsi.diff()
        
        # === PRICE ACTION ===
        features['close'] = df['close']
        features['returns'] = df['close'].pct_change()
        features['returns_5'] = df['close'].pct_change(5)
        features['atr'] = self.calculate_atr(df)
        features['high_low_range'] = (df['high'] - df['low']) / df['close']
        
        # === TARGETS ===
        features['next_direction'] = (df['close'].shift(-1) > df['close']).astype(int)  # 1=bull, 0=bear
        features['next_open'] = df['open'].shift(-1)
        features['next_high'] = df['high'].shift(-1)
        features['next_low'] = df['low'].shift(-1)
        features['next_close'] = df['close'].shift(-1)
        
        return features.dropna()
    
    def calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr
    
    def train_models(self, features, test_size=0.2):
        """
        Train both direction and OHLC prediction models.
        """
        print("\n" + "="*70)
        print("TRAINING ENHANCED HOLLOWAY PREDICTION SYSTEM")
        print("="*70)
        
        # Separate features and targets
        target_cols = ['next_direction', 'next_open', 'next_high', 'next_low', 'next_close']
        feature_cols = [col for col in features.columns if col not in target_cols and col != 'close']
        
        X = features[feature_cols]
        y_direction = features['next_direction']
        y_open = features['next_open']
        y_high = features['next_high']
        y_low = features['next_low']
        y_close = features['next_close']
        
        # Time series split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_dir_train, y_dir_test = y_direction.iloc[:split_idx], y_direction.iloc[split_idx:]
        
        print(f"\nTrain samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Features: {len(feature_cols)}")
        
        # === TRAIN DIRECTION MODEL ===
        print("\n1. Training Direction Model...")
        X_train_dir_scaled = self.direction_scaler.fit_transform(X_train)
        X_test_dir_scaled = self.direction_scaler.transform(X_test)
        
        self.direction_model = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            min_samples_split=20,
            subsample=0.8,
            random_state=42
        )
        self.direction_model.fit(X_train_dir_scaled, y_dir_train)
        
        dir_train_acc = self.direction_model.score(X_train_dir_scaled, y_dir_train)
        dir_test_acc = self.direction_model.score(X_test_dir_scaled, y_dir_test)
        
        print(f"   Training Accuracy: {dir_train_acc*100:.2f}%")
        print(f"   Testing Accuracy: {dir_test_acc*100:.2f}%")
        
        # === TRAIN OHLC MODELS ===
        print("\n2. Training OHLC Models...")
        X_train_ohlc_scaled = self.ohlc_scaler.fit_transform(X_train)
        X_test_ohlc_scaled = self.ohlc_scaler.transform(X_test)
        
        # Open model
        y_open_train, y_open_test = y_open.iloc[:split_idx], y_open.iloc[split_idx:]
        self.open_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        self.open_model.fit(X_train_ohlc_scaled, y_open_train)
        
        # High model
        y_high_train, y_high_test = y_high.iloc[:split_idx], y_high.iloc[split_idx:]
        self.high_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        self.high_model.fit(X_train_ohlc_scaled, y_high_train)
        
        # Low model
        y_low_train, y_low_test = y_low.iloc[:split_idx], y_low.iloc[split_idx:]
        self.low_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        self.low_model.fit(X_train_ohlc_scaled, y_low_train)
        
        # Close model
        y_close_train, y_close_test = y_close.iloc[:split_idx], y_close.iloc[split_idx:]
        self.close_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        self.close_model.fit(X_train_ohlc_scaled, y_close_train)
        
        # Evaluate OHLC
        ohlc_scores = {
            'open': self.open_model.score(X_test_ohlc_scaled, y_open_test),
            'high': self.high_model.score(X_test_ohlc_scaled, y_high_test),
            'low': self.low_model.score(X_test_ohlc_scaled, y_low_test),
            'close': self.close_model.score(X_test_ohlc_scaled, y_close_test)
        }
        
        print(f"   Open R²: {ohlc_scores['open']:.4f}")
        print(f"   High R²: {ohlc_scores['high']:.4f}")
        print(f"   Low R²: {ohlc_scores['low']:.4f}")
        print(f"   Close R²: {ohlc_scores['close']:.4f}")
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.direction_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 20 Most Important Features:")
        print(importance.head(20).to_string(index=False))
        
        # Save models
        print("\n3. Saving models...")
        joblib.dump(self.direction_model, self.models_folder / 'holloway_direction.pkl')
        joblib.dump(self.open_model, self.models_folder / 'holloway_open.pkl')
        joblib.dump(self.high_model, self.models_folder / 'holloway_high.pkl')
        joblib.dump(self.low_model, self.models_folder / 'holloway_low.pkl')
        joblib.dump(self.close_model, self.models_folder / 'holloway_close.pkl')
        joblib.dump(self.direction_scaler, self.models_folder / 'holloway_direction_scaler.pkl')
        joblib.dump(self.ohlc_scaler, self.models_folder / 'holloway_ohlc_scaler.pkl')
        
        print("✅ All models saved successfully!")
        
        print("\n" + "="*70)
        print(f"🎯 DIRECTION ACCURACY: {dir_test_acc*100:.2f}%")
        print(f"📊 OHLC R² (avg): {np.mean(list(ohlc_scores.values())):.4f}")
        print("="*70)
        
        return dir_test_acc, ohlc_scores, importance
    
    def predict_next_candle(self, df) -> Dict:
        """
        Predict full next candle: direction + OHLC values.
        """
        # Load models if needed
        self._load_models_if_needed()
        
        # Calculate features
        features = self.create_comprehensive_features(df)
        target_cols = ['next_direction', 'next_open', 'next_high', 'next_low', 'next_close']
        feature_cols = [col for col in features.columns if col not in target_cols and col != 'close']
        
        latest_features = features[feature_cols].iloc[-1:].values
        
        # Predict direction
        latest_dir_scaled = self.direction_scaler.transform(latest_features)
        direction_pred = self.direction_model.predict(latest_dir_scaled)[0]
        direction_proba = self.direction_model.predict_proba(latest_dir_scaled)[0]
        
        # Predict OHLC
        latest_ohlc_scaled = self.ohlc_scaler.transform(latest_features)
        open_pred = self.open_model.predict(latest_ohlc_scaled)[0]
        high_pred = self.high_model.predict(latest_ohlc_scaled)[0]
        low_pred = self.low_model.predict(latest_ohlc_scaled)[0]
        close_pred = self.close_model.predict(latest_ohlc_scaled)[0]
        
        # Current state
        current = df.iloc[-1]
        latest_feature_row = features.iloc[-1]
        
        # Build result
        result = {
            'timestamp': datetime.now().isoformat(),
            'current_price': float(current['close']),
            'prediction': {
                'direction': 'BULLISH' if direction_pred == 1 else 'BEARISH',
                'confidence': float(max(direction_proba) * 100),
                'probability_bullish': float(direction_proba[1] * 100),
                'probability_bearish': float(direction_proba[0] * 100),
                'ohlc': {
                    'open': float(open_pred),
                    'high': float(high_pred),
                    'low': float(low_pred),
                    'close': float(close_pred),
                    'range_pips': float(high_pred - low_pred)
                }
            },
            'holloway_state': {
                'bull_count': float(latest_feature_row['bull_count']),
                'bear_count': float(latest_feature_row['bear_count']),
                'bully': float(latest_feature_row['bully']),
                'beary': float(latest_feature_row['beary']),
                'bull_minus_bear': float(latest_feature_row['bull_minus_bear']),
                'rsi': float(latest_feature_row['rsi'])
            },
            'key_signals': self._extract_key_signals(latest_feature_row),
            'reasoning': self._generate_reasoning(latest_feature_row, direction_pred, direction_proba)
        }
        
        return result
    
    def _load_models_if_needed(self):
        """Load models if not already loaded"""
        if self.direction_model is None:
            self.direction_model = joblib.load(self.models_folder / 'holloway_direction.pkl')
            self.open_model = joblib.load(self.models_folder / 'holloway_open.pkl')
            self.high_model = joblib.load(self.models_folder / 'holloway_high.pkl')
            self.low_model = joblib.load(self.models_folder / 'holloway_low.pkl')
            self.close_model = joblib.load(self.models_folder / 'holloway_close.pkl')
            self.direction_scaler = joblib.load(self.models_folder / 'holloway_direction_scaler.pkl')
            self.ohlc_scaler = joblib.load(self.models_folder / 'holloway_ohlc_scaler.pkl')
    
    def _extract_key_signals(self, feature_row) -> Dict:
        """Extract key signals from feature row"""
        return {
            'earliest_bullish': bool(feature_row.get('cross_earliest_bullish', 0)),
            'earliest_bearish': bool(feature_row.get('cross_earliest_bearish', 0)),
            'bull_near_high': bool(feature_row.get('level_bull_near_high', 0)),
            'bull_near_low': bool(feature_row.get('level_bull_near_low', 0)),
            'explosion_detected': bool(feature_row.get('exp_bull_abnormal', 0) or feature_row.get('exp_bear_abnormal', 0)),
            'mirror_bearish': bool(feature_row.get('mirror_mirror_bearish', 0)),
            'wm_at_peak': bool(feature_row.get('wm_bull_m_peak', 0) or feature_row.get('wm_bear_m_peak', 0))
        }
    
    def _generate_reasoning(self, feature_row, direction_pred, proba) -> str:
        """Generate human-readable reasoning"""
        direction = 'BULLISH' if direction_pred == 1 else 'BEARISH'
        conf = max(proba) * 100
        
        reasons = []
        
        # Bull vs bear count
        bull_count = feature_row['bull_count']
        bear_count = feature_row['bear_count']
        if bull_count > bear_count:
            reasons.append(f"✓ Bull count ({bull_count:.0f}) > Bear count ({bear_count:.0f})")
        else:
            reasons.append(f"✗ Bear count ({bear_count:.0f}) > Bull count ({bull_count:.0f})")
        
        # Earliest signals
        if feature_row.get('cross_earliest_bullish', 0):
            reasons.append("✓ EARLIEST BULLISH signal (count crossed above average)")
        if feature_row.get('cross_earliest_bearish', 0):
            reasons.append("✗ EARLIEST BEARISH signal (count crossed below average)")
        
        # Historical levels
        if feature_row.get('level_bull_near_high', 0):
            reasons.append("⚠️ Bull count near historical high (potential reversal)")
        if feature_row.get('level_bull_near_low', 0):
            reasons.append("✓ Bull count near historical low (bounce likely)")
        
        # Explosions
        if feature_row.get('exp_bull_abnormal', 0):
            reasons.append("⚡ Bull explosion detected (large sudden move)")
        if feature_row.get('exp_bear_abnormal', 0):
            reasons.append("⚡ Bear explosion detected (large sudden move)")
        
        # Mirroring
        if feature_row.get('mirror_mirror_bearish', 0):
            reasons.append("✗ Mirror bearish (both counts trigger bearish)")
        
        # RSI
        rsi = feature_row['rsi']
        if rsi > 50:
            reasons.append(f"✓ RSI bullish at {rsi:.1f}")
        else:
            reasons.append(f"✗ RSI bearish at {rsi:.1f}")
        
        reasoning = f"\n{'='*60}\n"
        reasoning += f"PREDICTION: {direction} (Confidence: {conf:.1f}%)\n"
        reasoning += f"{'='*60}\n\n"
        reasoning += "Key Factors:\n"
        reasoning += "\n".join(f"  {r}" for r in reasons)
        
        return reasoning
    
    def generate_daily_report(self, df) -> str:
        """Generate comprehensive daily prediction report"""
        prediction = self.predict_next_candle(df)
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║      ENHANCED HOLLOWAY NEXT-CANDLE PREDICTION REPORT         ║
║                {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                     ║
╚══════════════════════════════════════════════════════════════╝

{prediction['reasoning']}

CURRENT MARKET STATE:
  Price: {prediction['current_price']:.2f}
  Bull Count: {prediction['holloway_state']['bull_count']:.0f}
  Bear Count: {prediction['holloway_state']['bear_count']:.0f}
  Bully (avg): {prediction['holloway_state']['bully']:.1f}
  Beary (avg): {prediction['holloway_state']['beary']:.1f}
  RSI: {prediction['holloway_state']['rsi']:.1f}

PROBABILITY BREAKDOWN:
  Bullish: {prediction['prediction']['probability_bullish']:.1f}%
  Bearish: {prediction['prediction']['probability_bearish']:.1f}%

PREDICTED NEXT CANDLE (OHLC):
  Open:  {prediction['prediction']['ohlc']['open']:.2f}
  High:  {prediction['prediction']['ohlc']['high']:.2f}
  Low:   {prediction['prediction']['ohlc']['low']:.2f}
  Close: {prediction['prediction']['ohlc']['close']:.2f}
  Range: {prediction['prediction']['ohlc']['range_pips']:.2f} pips

KEY SIGNALS:
  Earliest Signal: {'BULLISH ✓' if prediction['key_signals']['earliest_bullish'] else 'BEARISH ✗' if prediction['key_signals']['earliest_bearish'] else 'None'}
  Near Historical Level: {'YES ⚠️' if prediction['key_signals']['bull_near_high'] or prediction['key_signals']['bull_near_low'] else 'NO'}
  Explosion Move: {'YES ⚡' if prediction['key_signals']['explosion_detected'] else 'NO'}
  Mirroring: {'BEARISH ✗' if prediction['key_signals']['mirror_bearish'] else 'BULLISH ✓'}

RECOMMENDATION:
  Next candle predicted to be {prediction['prediction']['direction']}
  Confidence: {prediction['prediction']['confidence']:.1f}%
  
  {'⚠️  HIGH CONFIDENCE - Strong signal!' if prediction['prediction']['confidence'] > 75 else '⚠️  MODERATE - Use caution'}

{'='*60}
        """
        
        return report
    
    def track_accuracy(self, prediction: Dict, actual_candle: Dict) -> Dict:
        """
        Track prediction accuracy using 75% OHLC match criteria.
        
        Accuracy measured as:
        1. Direction correct (bull/bear)
        2. OHLC within 75% of actual values
        """
        # Direction accuracy
        predicted_dir = prediction['prediction']['direction']
        actual_dir = 'BULLISH' if actual_candle['close'] > actual_candle['open'] else 'BEARISH'
        direction_correct = predicted_dir == actual_dir
        
        # OHLC accuracy (within 75%)
        pred_ohlc = prediction['prediction']['ohlc']
        
        ohlc_accuracy = {}
        for key in ['open', 'high', 'low', 'close']:
            pred_val = pred_ohlc[key]
            actual_val = actual_candle[key]
            error_pct = abs(pred_val - actual_val) / actual_val
            within_75pct = error_pct <= 0.25
            ohlc_accuracy[key] = {
                'predicted': pred_val,
                'actual': actual_val,
                'error_pct': error_pct * 100,
                'within_75pct': within_75pct
            }
        
        # Overall candle accuracy (all OHLC within 75%)
        candle_accurate = all(v['within_75pct'] for v in ohlc_accuracy.values())
        
        # Combined accuracy
        fully_accurate = direction_correct and candle_accurate
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'direction_correct': direction_correct,
            'candle_accurate': candle_accurate,
            'fully_accurate': fully_accurate,
            'ohlc_breakdown': ohlc_accuracy,
            'prediction': prediction,
            'actual': actual_candle
        }
        
        # Track in history
        self.prediction_history.append(result)
        
        # Calculate rolling accuracy
        recent_100 = self.prediction_history[-100:] if len(self.prediction_history) >= 100 else self.prediction_history
        
        dir_acc = sum(r['direction_correct'] for r in recent_100) / len(recent_100) * 100
        candle_acc = sum(r['candle_accurate'] for r in recent_100) / len(recent_100) * 100
        full_acc = sum(r['fully_accurate'] for r in recent_100) / len(recent_100) * 100
        
        result['rolling_accuracy'] = {
            'direction_accuracy': dir_acc,
            'candle_accuracy_75pct': candle_acc,
            'fully_accurate': full_acc,
            'sample_size': len(recent_100)
        }
        
        return result


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║   ENHANCED HOLLOWAY NEXT-CANDLE PREDICTION SYSTEM v3.0       ║
    ╚══════════════════════════════════════════════════════════════╝
    
    NEW FEATURES:
    
    ✓ Count crossing average detection (FASTEST signal)
    ✓ Historical high/low support/resistance
    ✓ W/M pattern detection from count lines
    ✓ Explosion move detection
    ✓ Mirroring behavior analysis
    ✓ Full OHLC prediction (not just direction)
    ✓ 75% accuracy tracking
    
    USAGE:
    
    1. TRAIN MODELS:
       predictor = EnhancedHollowayPredictor()
       df = pd.read_csv('data/XAUUSD_4H.csv')
       features = predictor.create_comprehensive_features(df)
       predictor.train_models(features)
    
    2. PREDICT NEXT CANDLE:
       prediction = predictor.predict_next_candle(df)
       print(predictor.generate_daily_report(df))
    
    3. TRACK ACCURACY:
       # After actual candle closes
       actual = {
           'open': 2655.20,
           'high': 2658.40,
           'low': 2653.10,
           'close': 2657.30
       }
       accuracy = predictor.track_accuracy(prediction, actual)
       print(f"Direction: {accuracy['direction_correct']}")
       print(f"Candle: {accuracy['candle_accurate']}")
       print(f"Rolling accuracy: {accuracy['rolling_accuracy']}")
    
    TARGET: 85%+ direction, 75%+ OHLC accuracy
    """)
