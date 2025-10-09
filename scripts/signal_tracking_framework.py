#!/usr/bin/env python3
"""
Comprehensive Signal Tracking Framework
Creates a detailed CSV log of all predictions, features, and outcomes for analysis.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from scripts.forecasting import HybridPriceForecastingEnsemble
from scripts.data_issue_fixes import pre_training_data_fix

def load_trained_models() -> Dict[str, Any]:
    """Load the trained ensemble models for EURUSD and XAUUSD"""
    models = {}
    for pair in ['EURUSD', 'XAUUSD']:
        model_path = BASE_DIR / 'models' / f'{pair}_ensemble.joblib'
        if model_path.exists():
            try:
                import joblib
                models[pair] = joblib.load(model_path)
                print(f"✅ Loaded {pair} model")
            except Exception as e:
                print(f"⚠️  Could not load {pair} model: {e}")
                models[pair] = None
        else:
            print(f"⚠️  Model not found: {model_path}")
            models[pair] = None
    return models

def load_data_for_pair(pair: str) -> pd.DataFrame:
    """Load and prepare data for a currency pair"""
    data_file = BASE_DIR / 'data' / f'{pair}_Daily.csv'
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    df = pd.read_csv(data_file)

    # Handle timestamp column
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
    else:
        # Create date index from row number (fallback)
        df.index = pd.date_range(start='2000-01-01', periods=len(df), freq='D')

    # Keep only essential columns initially
    essential_cols = ['open', 'high', 'low', 'close']
    if 'volume' in df.columns:
        essential_cols.append('volume')

    df = df[essential_cols]

    return df

def engineer_comprehensive_features(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """Engineer comprehensive features across all timeframes and indicators"""

    # Basic price features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # Moving averages - multiple periods
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'sma_{period}'] = df['close'].rolling(period).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period).mean()

    # Volatility measures
    df['volatility_20'] = df['returns'].rolling(20).std()
    df['volatility_50'] = df['returns'].rolling(50).std()
    df['atr_14'] = calculate_atr(df, 14)

    # RSI - multiple periods
    for period in [7, 14, 21]:
        df[f'rsi_{period}'] = calculate_rsi(df['close'], period)

    # MACD
    df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['close'])

    # Bollinger Bands
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = calculate_bollinger_bands(df['close'])

    # Momentum indicators
    for period in [5, 10, 20]:
        df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
        df[f'roc_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period)

    # Volume-based features (if available, otherwise placeholder)
    if 'volume' in df.columns:
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    else:
        df['volume_sma_20'] = 1.0
        df['volume_ratio'] = 1.0

    # Price patterns
    df['doji'] = ((abs(df['open'] - df['close']) / (df['high'] - df['low'] + 1e-8)) < 0.1).astype(int)

    # Hammer pattern (simplified)
    body_size = abs(df['open'] - df['close'])
    upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
    lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
    df['hammer'] = ((lower_shadow > body_size * 2) & (upper_shadow < body_size)).astype(int)

    # Gap analysis
    df['gap_up'] = (df['open'] > df['close'].shift(1)).astype(int)
    df['gap_down'] = (df['open'] < df['close'].shift(1)).astype(int)

    # Support/Resistance levels (simplified)
    df['near_resistance'] = ((df['high'] - df['close']) / df['close'] < 0.01).astype(int)
    df['near_support'] = ((df['close'] - df['low']) / df['close'] < 0.01).astype(int)

    # Trend strength
    df['trend_strength'] = abs(df['close'] - df['close'].shift(20)) / df['volatility_20']

    # Load fundamental data if available
    fundamentals_file = BASE_DIR / 'data' / 'update_metadata.json'
    if fundamentals_file.exists():
        try:
            with open(fundamentals_file, 'r') as f:
                fundamentals_meta = json.load(f)

            # Add fundamental features (simplified - would need full integration)
            df['fed_funds_rate'] = 5.33  # Placeholder - would load from FRED
            df['unemployment_rate'] = 4.1  # Placeholder
            df['gdp_growth'] = 2.1  # Placeholder

        except Exception as e:
            print(f"Could not load fundamentals: {e}")

    # Target variable (next day direction)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

    return df

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Calculate MACD"""
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def calculate_bollinger_bands(series: pd.Series, period: int = 20, std_dev: int = 2):
    """Calculate Bollinger Bands"""
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

def generate_predictions(df: pd.DataFrame, model, feature_cols: List[str]) -> pd.DataFrame:
    """Generate predictions using the trained model"""
    # Prepare features for prediction
    pred_df = df.copy()

    # Get latest data point for prediction
    if len(df) < 50:  # Need enough history
        pred_df['prediction'] = 0.5
        pred_df['probability'] = 0.5
        return pred_df

    # Use last row for prediction (most recent complete data)
    latest_features = df[feature_cols].iloc[-1:].fillna(0)

    try:
        if model is not None:
            probabilities = model.predict_proba(latest_features.values)
            pred_df['prediction'] = (probabilities[:, 1] > 0.5).astype(int)
            pred_df['probability'] = probabilities[:, 1]
        else:
            # Fallback to simple rule
            pred_df['prediction'] = (df['close'] > df['sma_20']).astype(int)
            pred_df['probability'] = 0.5
    except Exception as e:
        print(f"Prediction error: {e}")
        pred_df['prediction'] = 0.5
        pred_df['probability'] = 0.5

    return pred_df

def create_signal_log(pair: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Create comprehensive signal log for a currency pair"""

    print(f"🔄 Processing {pair} signal log...")

    # Load data
    df = load_data_for_pair(pair)

    # Engineer comprehensive features
    df = engineer_comprehensive_features(df, pair)

    # Define feature columns (all numeric columns except target)
    feature_cols = [col for col in df.columns if col not in ['target'] and df[col].dtype in ['float64', 'int64']]

    # Load model
    models = load_trained_models()
    model = models.get(pair)

    # Generate predictions
    df = generate_predictions(df, model, feature_cols)

    # Filter date range if specified
    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]

    # Add metadata
    df['pair'] = pair
    df['date'] = df.index.date
    df['actual_outcome'] = df['target']  # Next day direction
    df['correct'] = (df['prediction'] == df['actual_outcome']).astype(int)

    # Select final columns for the log
    log_columns = [
        'pair', 'date', 'open', 'high', 'low', 'close',
        'prediction', 'probability', 'actual_outcome', 'correct'
    ] + feature_cols

    signal_log = df[log_columns].copy()

    return signal_log

def main():
    """Main function to generate comprehensive signal logs"""

    print("🎯 Starting Comprehensive Signal Tracking Framework")
    print("=" * 60)

    # Ensure data is ready
    if not pre_training_data_fix():
        print("❌ Data validation failed")
        return

    # Generate signal logs for both pairs
    all_logs = []

    for pair in ['EURUSD', 'XAUUSD']:
        try:
            signal_log = create_signal_log(pair)
            all_logs.append(signal_log)
            print(f"✅ Generated signal log for {pair}: {len(signal_log)} records")
        except Exception as e:
            print(f"❌ Failed to generate signal log for {pair}: {e}")

    if not all_logs:
        print("❌ No signal logs generated")
        return

    # Combine all logs
    if all_logs:
        print(f"Debug: Number of logs: {len(all_logs)}")
        for i, log in enumerate(all_logs):
            print(f"Log {i} shape: {log.shape}, columns: {len(log.columns)}")
            print(f"First few columns: {list(log.columns[:5])}")

        # Reset indexes and ensure clean concatenation
        cleaned_logs = []
        for log in all_logs:
            log_clean = log.reset_index(drop=True)
            cleaned_logs.append(log_clean)

        # Try to find common columns
        common_cols = set(cleaned_logs[0].columns)
        for log in cleaned_logs[1:]:
            common_cols = common_cols.intersection(set(log.columns))

        print(f"Common columns: {len(common_cols)}")

        # Keep only common columns
        final_logs = [log[list(common_cols)] for log in cleaned_logs]

        combined_log = pd.concat(final_logs, ignore_index=True, sort=False)
    else:
        print("❌ No signal logs to combine")
        return

    # Save to CSV
    output_dir = BASE_DIR / 'output'
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f'comprehensive_signal_log_{timestamp}.csv'

    combined_log.to_csv(output_file, index=False)

    print(f"💾 Saved comprehensive signal log to: {output_file}")
    print(f"📊 Total records: {len(combined_log)}")
    print(f"📈 Features included: {len(combined_log.columns) - 10}")  # Subtract metadata columns

    # Basic statistics
    for pair in ['EURUSD', 'XAUUSD']:
        pair_data = combined_log[combined_log['pair'] == pair]
        if len(pair_data) > 0:
            accuracy = pair_data['correct'].mean() * 100
            avg_probability = pair_data['probability'].mean()
            print(f"📊 {pair}: {len(pair_data)} signals, {accuracy:.1f}% accuracy, avg prob: {avg_probability:.3f}")

    print("\n🎉 Signal tracking framework complete!")
    print("Next: Analyze the CSV and create visualizations in the dashboard notebook.")

if __name__ == '__main__':
    main()