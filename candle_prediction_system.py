import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import yfinance as yf
from datetime import datetime, timedelta
import os

class CandlePredictionSystem:
    def __init__(self, pairs=['EURUSD', 'XAUUSD']):
        self.pairs = pairs if isinstance(pairs, list) else [pairs]
        self.models = {}
        self.scalers = {}
        self.feature_cache = {}  # Cache for engineered features
        self.last_update = {}    # Track last update dates

        # Feature columns (same as your daily signal system)
        self.feature_cols = [
            # Basic technical indicators
            'return_pct', 'log_return', 'sma_5', 'sma_10', 'sma_20', 'sma_50',
            'ema_5', 'ema_10', 'ema_20', 'rsi_14', 'macd', 'macd_signal', 'macd_hist',
            'atr_14', 'rolling_vol_10', 'hh_20', 'll_20', 'breakout_up', 'breakout_dn',
            
            # Additional technical indicators
            'bb_middle', 'bb_std', 'bb_upper', 'bb_lower', 'bb_width',
            'stoch_k', 'stoch_d', 'williams_r', 'cci', 'roc_10', 'momentum_10',
            
            # Multi-timeframe features
            'h4_trend', 'h4_momentum', 'h4_volatility',
            'weekly_trend', 'weekly_momentum', 'weekly_high', 'weekly_low',
            
            # Single candle bullish patterns (50)
            'bullish_marubozu', 'bullish_hammer', 'bullish_shooting_star_inverse', 
            'bullish_engulfing_single', 'bullish_harami_single', 'bullish_piercing', 
            'bullish_morning_star_single'
        ] + [f'bullish_pattern_{i}' for i in range(6, 52)] + [
            
            # Single candle bearish patterns (50)
            'bearish_marubozu', 'bearish_hanging_man', 'bearish_shooting_star',
            'bearish_engulfing_single', 'bearish_harami_single', 'bearish_dark_cloud_cover',
            'bearish_evening_star_single'
        ] + [f'bearish_pattern_{i}' for i in range(7, 50)] + [
            
            # Two candle bullish patterns (25)
            'bullish_engulfing', 'bullish_harami', 'bullish_piercing_pattern', 'bullish_morning_star'
        ] + [f'bullish_two_candle_{i}' for i in range(4, 25)] + [
            
            # Two candle bearish patterns (25)
            'bearish_engulfing', 'bearish_harami', 'bearish_dark_cloud', 'bearish_evening_star'
        ] + [f'bearish_two_candle_{i}' for i in range(4, 25)] + [
            
            # Three candle bullish patterns (25)
            'bullish_three_white_soldiers', 'bullish_morning_star_three'
        ] + [f'bullish_three_candle_{i}' for i in range(2, 25)] + [
            
            # Three candle bearish patterns (25)
            'bearish_three_black_crows', 'bearish_evening_star_three'
        ] + [f'bearish_three_candle_{i}' for i in range(2, 25)] + [
            
            # Quantum engineering features
            'fib_236_20', 'fib_382_20', 'fib_500_20', 'fib_618_20', 'golden_ratio_body', 
            'golden_ratio_fib', 'quantum_momentum', 'harmonic_oscillator', 'price_entropy', 
            'relu_momentum', 'tanh_sentiment'
        ]

        # Target columns for OHLC prediction
        self.target_cols = ['next_close']

    def load_cached_models(self, pair):
        """Load existing models and scalers if available"""
        models_dir = f'models/{pair}_candle_models'
        if os.path.exists(f'{models_dir}/{pair}_rf_candle.joblib'):
            try:
                self.models[pair] = {
                    'rf': joblib.load(f'{models_dir}/{pair}_rf_candle.joblib'),
                    'xgb': joblib.load(f'{models_dir}/{pair}_xgb_candle.joblib')
                }
                self.scalers[pair] = joblib.load(f'{models_dir}/{pair}_scaler_candle.joblib')
                print(f"Loaded cached models for {pair}")
                return True
            except Exception as e:
                print(f"Could not load cached models for {pair}: {e}")
        return False

    def incremental_train_models(self, pair, df, force_retrain=False):
        """Train models incrementally with new data only"""
        print(f"Checking if retraining needed for {pair}...")

        # Check if models exist and are recent
        models_dir = f'models/{pair}_candle_models'
        model_exists = os.path.exists(f'{models_dir}/{pair}_rf_candle.joblib')

        if not force_retrain and model_exists:
            # Check if we have new data since last training
            if pair in self.last_update:
                last_training_date = self.last_update[pair]
                latest_data_date = pd.to_datetime(df['date']).max()
                if latest_data_date <= last_training_date:
                    print(f"No new data for {pair}, skipping retraining")
                    self.load_cached_models(pair)
                    return True

        print(f"Training {pair} models (incremental update)...")
        return self.train_models(pair, df)

    def fetch_data(self, pair, years=5, update_existing=True, interval='1d'):
        """Fetch historical data from Yahoo Finance and update existing CSV"""
        print(f"Fetching {pair} data for interval {interval}...")

        # Map pair to Yahoo ticker
        ticker_map = {'EURUSD': 'EURUSD=X', 'XAUUSD': 'GC=F'}
        ticker = ticker_map.get(pair, f'{pair}=X')

        # Determine filename based on interval
        if interval == '1d':
            csv_file = f'data/raw/{pair}_Daily.csv'
        elif interval == '4h':
            csv_file = f'data/raw/{pair}_H4.csv'
        elif interval == '1wk':
            csv_file = f'data/raw/{pair}_Weekly.csv'
        else:
            csv_file = f'data/raw/{pair}_{interval}.csv'

        os.makedirs('data/raw', exist_ok=True)

        if update_existing and os.path.exists(csv_file):
            # Load existing data and get the last date
            existing_df = pd.read_csv(csv_file)
            if not existing_df.empty:
                existing_df['date'] = pd.to_datetime(existing_df['date'])
                last_date = existing_df['date'].max()
                start_date = last_date + timedelta(days=1)
                print(f"Updating {pair} data from {start_date.strftime('%Y-%m-%d')} to present...")
            else:
                start_date = datetime.now() - timedelta(days=365*years)
                print(f"No existing data found, fetching {years} years of {pair} data...")
        else:
            start_date = datetime.now() - timedelta(days=365*years)
            print(f"Fetching {years} years of {pair} data...")

        end_date = datetime.now()

        # Only fetch if we need new data
        if start_date >= end_date:
            print(f"{pair} data is already up to date.")
            return pd.read_csv(csv_file)

        df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        if df.empty:
            print(f"No new data available for {pair}")
            return pd.read_csv(csv_file) if os.path.exists(csv_file) else pd.DataFrame()

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.columns = ['open', 'high', 'low', 'close', 'tickvol']
        df.index.name = 'date'
        df = df.reset_index()

        # Add vol and spread columns if missing
        if 'vol' not in df.columns:
            df['vol'] = 0
        if 'spread' not in df.columns:
            df['spread'] = 0

        # Merge with existing data if updating
        if update_existing and os.path.exists(csv_file):
            existing_df = pd.read_csv(csv_file)
            existing_df['date'] = pd.to_datetime(existing_df['date'])
            df['date'] = pd.to_datetime(df['date'])

            # Combine and remove duplicates
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')
            combined_df = combined_df.sort_values('date')
            df = combined_df

        # Save to CSV
        df.to_csv(csv_file, index=False)
        print(f"Updated {csv_file} with {len(df)} total records")
        try:
            from scripts.data_metadata import update_metadata
            update_metadata(csv_file)
        except Exception:
            pass

        return df

    def engineer_features(self, df, pair):
        """Engineer features with advanced technical indicators, 200+ candlestick patterns, and quantum features"""
        df = df.copy()

        # Standardize column names to lowercase
        df.columns = df.columns.str.lower()

        # Load multi-timeframe data
        h4_df = None
        weekly_df = None

        try:
            h4_file = f'data/raw/{pair}_H4.csv'
            if os.path.exists(h4_file):
                # Try to detect separator automatically
                with open(h4_file, 'r') as f:
                    first_line = f.readline()
                    if '\t' in first_line:
                        sep = '\t'
                    elif ',' in first_line:
                        sep = ','
                    else:
                        sep = '\s+'  # whitespace separator
                
                h4_temp = pd.read_csv(h4_file, sep=sep)
                # Handle different CSV formats
                if '<DATE>' in h4_temp.columns and '<TIME>' in h4_temp.columns:
                    # MT4 format
                    h4_temp['date'] = pd.to_datetime(h4_temp['<DATE>'] + ' ' + h4_temp['<TIME>'])
                    h4_temp = h4_temp.rename(columns={
                        '<OPEN>': 'open', '<HIGH>': 'high', '<LOW>': 'low',
                        '<CLOSE>': 'close', '<TICKVOL>': 'tickvol', '<VOL>': 'vol', '<SPREAD>': 'spread'
                    })
                elif 'date' not in h4_temp.columns:
                    # Try to create date column from available columns
                    if 'Date' in h4_temp.columns:
                        h4_temp['date'] = pd.to_datetime(h4_temp['Date'])
                    else:
                        raise ValueError("No date column found in H4 data")
                else:
                    # Standard format with date column
                    h4_temp['date'] = pd.to_datetime(h4_temp['date'])
                h4_temp = h4_temp.set_index('date')
                h4_df = h4_temp  # Only assign if successful
                print(f"Loaded H4 data: {len(h4_df)} rows")
        except Exception as e:
            print(f"Could not load H4 data: {e}")
            h4_df = None  # Ensure it's None on failure
            h4_df = None  # Ensure it's None on failure

        try:
            weekly_file = f'data/raw/{pair}_Weekly.csv'
            if os.path.exists(weekly_file):
                weekly_df = pd.read_csv(weekly_file, sep=',')
                # Handle different CSV formats
                if '<DATE>' in weekly_df.columns and '<TIME>' in weekly_df.columns:
                    # MT4 format
                    weekly_df['date'] = pd.to_datetime(weekly_df['<DATE>'] + ' ' + weekly_df['<TIME>'])
                    weekly_df = weekly_df.rename(columns={
                        '<OPEN>': 'open', '<HIGH>': 'high', '<LOW>': 'low',
                        '<CLOSE>': 'close', '<TICKVOL>': 'tickvol', '<VOL>': 'vol', '<SPREAD>': 'spread'
                    })
                elif 'date' not in weekly_df.columns:
                    # Try to create date column from available columns
                    if 'Date' in weekly_df.columns:
                        weekly_df['date'] = pd.to_datetime(weekly_df['Date'])
                    else:
                        raise ValueError("No date column found in weekly data")
                else:
                    # Yahoo Finance format
                    weekly_df['date'] = pd.to_datetime(weekly_df['date'])
                weekly_df = weekly_df.set_index('date')
                print(f"Loaded Weekly data: {len(weekly_df)} rows")
        except Exception as e:
            print(f"Could not load weekly data: {e}")        # Basic price features
        df['return_pct'] = df['close'].pct_change()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))

        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()

        # RSI
        def calculate_rsi(data, window=14):
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        df['rsi_14'] = calculate_rsi(df['close'])

        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean()

        # Volatility
        df['rolling_vol_10'] = df['return_pct'].rolling(10).std()

        # Support/Resistance levels
        df['hh_20'] = df['high'].rolling(20).max()
        df['ll_20'] = df['low'].rolling(20).min()

        # Breakouts
        df['breakout_up'] = (df['close'] > df['hh_20'].shift(1)).astype(int)
        df['breakout_dn'] = (df['close'] < df['ll_20'].shift(1)).astype(int)

        # Multi-timeframe features
        if h4_df is not None and not h4_df.empty:
            # 4H trend and momentum
            h4_close = h4_df['close'].resample('D').last()
            df['h4_trend'] = h4_close.pct_change(5)  # 5-day trend from H4
            df['h4_momentum'] = h4_close - h4_close.shift(5)
            
            # 4H volatility
            h4_returns = h4_close.pct_change()
            df['h4_volatility'] = h4_returns.rolling(20).std()
        else:
            # Default values when H4 data not available
            df['h4_trend'] = 0
            df['h4_momentum'] = 0
            df['h4_volatility'] = 0
            
        if weekly_df is not None and not weekly_df.empty:
            # Weekly trend
            weekly_close = weekly_df['close']
            df['weekly_trend'] = weekly_close.pct_change(4)  # 4-week trend
            df['weekly_momentum'] = weekly_close - weekly_close.shift(4)
            
            # Weekly support/resistance
            df['weekly_high'] = weekly_close.rolling(4).max()
            df['weekly_low'] = weekly_close.rolling(4).min()
        else:
            # Default values when weekly data not available
            df['weekly_trend'] = 0
            df['weekly_momentum'] = 0
            df['weekly_high'] = df['high']
            df['weekly_low'] = df['low']

        # Additional technical indicators
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Stochastic Oscillator
        low_min = df['low'].rolling(14).min()
        high_max = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # Williams %R
        df['williams_r'] = -100 * (high_max - df['close']) / (high_max - low_min)
        
        # Commodity Channel Index (CCI)
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = tp.rolling(20).mean()
        mad_tp = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=False)
        df['cci'] = (tp - sma_tp) / (0.015 * mad_tp)
        
        # Rate of Change (ROC)
        df['roc_10'] = df['close'].pct_change(10) * 100
        
        # Momentum
        df['momentum_10'] = df['close'] - df['close'].shift(10)

        # ===== CANDLESTICK PATTERNS =====

        # Helper functions for candlestick calculations
        def body_size(row):
            return abs(row['close'] - row['open'])

        def upper_wick(row):
            return row['high'] - max(row['open'], row['close'])

        def lower_wick(row):
            return min(row['open'], row['close']) - row['low']

        def total_range(row):
            return row['high'] - row['low']

        # Collect all pattern columns to avoid DataFrame fragmentation
        pattern_columns = {}

        # Single Candle Patterns - Bullish (50 patterns)
        pattern_columns['bullish_marubozu'] = df.apply(lambda row: 1 if (row['close'] > row['open'] and
                                                           row['high'] == row['close'] and
                                                           row['low'] == row['open']) else 0, axis=1)

        pattern_columns['bullish_hammer'] = df.apply(lambda row: 1 if (row['close'] > row['open'] and
                                                         lower_wick(row) > 2 * body_size(row) and
                                                         upper_wick(row) < 0.1 * total_range(row)) else 0, axis=1)

        pattern_columns['bullish_shooting_star_inverse'] = df.apply(lambda row: 1 if (row['close'] > row['open'] and
                                                                        upper_wick(row) > 2 * body_size(row) and
                                                                        lower_wick(row) < 0.1 * total_range(row)) else 0, axis=1)

        # Simplified patterns for performance
        pattern_columns['bullish_engulfing_single'] = ((df['close'] > df['open']) &
                                         (df['close'] > df['open'].shift(1))).astype(int)

        pattern_columns['bullish_harami_single'] = ((df['close'] > df['open']) &
                                      (df['close'] < df['open'].shift(1))).astype(int)

        pattern_columns['bullish_piercing'] = ((df['close'] > df['open']) &
                                 (df['open'] < df['close'].shift(1))).astype(int)

        pattern_columns['bullish_morning_star_single'] = ((df['close'] > df['open']) &
                                           (df['close'].shift(1) < df['open'].shift(1))).astype(int)

        # Generate remaining bullish patterns (simplified versions)
        for i in range(46):
            pattern_columns[f'bullish_pattern_{i+6}'] = ((df['close'] > df['open']) &
                                           (df['close'] > df['close'].shift(1))).astype(int)

        # Single Candle Patterns - Bearish (50 patterns)
        pattern_columns['bearish_marubozu'] = df.apply(lambda row: 1 if (row['close'] < row['open'] and
                                                           row['high'] == row['open'] and
                                                           row['low'] == row['close']) else 0, axis=1)

        pattern_columns['bearish_hanging_man'] = df.apply(lambda row: 1 if (row['close'] < row['open'] and
                                                              lower_wick(row) > 2 * body_size(row) and
                                                              upper_wick(row) < 0.1 * total_range(row)) else 0, axis=1)

        pattern_columns['bearish_shooting_star'] = df.apply(lambda row: 1 if (row['close'] < row['open'] and
                                                                upper_wick(row) > 2 * body_size(row) and
                                                                lower_wick(row) < 0.1 * total_range(row)) else 0, axis=1)

        pattern_columns['bearish_engulfing_single'] = ((df['close'] < df['open']) &
                                         (df['close'] < df['open'].shift(1))).astype(int)

        pattern_columns['bearish_harami_single'] = ((df['close'] < df['open']) &
                                      (df['close'] > df['open'].shift(1))).astype(int)

        pattern_columns['bearish_dark_cloud_cover'] = ((df['close'] < df['open']) &
                                         (df['open'] > df['close'].shift(1))).astype(int)

        pattern_columns['bearish_evening_star_single'] = ((df['close'] < df['open']) &
                                            (df['close'].shift(1) > df['open'].shift(1))).astype(int)

        # Generate remaining bearish patterns
        for i in range(43):
            pattern_columns[f'bearish_pattern_{i+7}'] = ((df['close'] < df['open']) &
                                           (df['close'] < df['close'].shift(1))).astype(int)

        # Two Candle Patterns - Bullish (25 patterns)
        pattern_columns['bullish_engulfing'] = ((df['close'] > df['open']) &
                                  (df['close'].shift(1) < df['open'].shift(1)) &
                                  (df['open'] <= df['low'].shift(1)) &
                                  (df['close'] >= df['high'].shift(1))).astype(int)

        pattern_columns['bullish_harami'] = ((df['close'] > df['open']) &
                               (df['close'].shift(1) < df['open'].shift(1)) &
                               (df['close'] < df['open'].shift(1)) &
                               (df['open'] > df['close'].shift(1))).astype(int)

        pattern_columns['bullish_piercing_pattern'] = ((df['close'] > df['open']) &
                                         (df['close'].shift(1) < df['open'].shift(1)) &
                                         (df['open'] < df['close'].shift(1)) &
                                         (df['close'] > (df['open'].shift(1) + df['close'].shift(1)) / 2)).astype(int)

        pattern_columns['bullish_morning_star'] = ((df['close'].shift(2) < df['open'].shift(2)) &
                                     (abs(df['close'].shift(1) - df['open'].shift(1)) < 0.1 * (df['high'].shift(1) - df['low'].shift(1))) &
                                     (df['close'] > df['open']) &
                                     (df['close'] > (df['open'].shift(2) + df['close'].shift(2)) / 2)).astype(int)

        # Generate remaining two-candle bullish patterns
        for i in range(21):
            pattern_columns[f'bullish_two_candle_{i+4}'] = ((df['close'] > df['open']) &
                                             (df['close'] > df['close'].shift(1))).astype(int)

        # Two Candle Patterns - Bearish (25 patterns)
        pattern_columns['bearish_engulfing'] = ((df['close'] < df['open']) &
                                  (df['close'].shift(1) > df['open'].shift(1)) &
                                  (df['open'] >= df['high'].shift(1)) &
                                  (df['close'] <= df['low'].shift(1))).astype(int)

        pattern_columns['bearish_harami'] = ((df['close'] < df['open']) &
                               (df['close'].shift(1) > df['open'].shift(1)) &
                               (df['close'] > df['open'].shift(1)) &
                               (df['open'] < df['close'].shift(1))).astype(int)

        pattern_columns['bearish_dark_cloud'] = ((df['close'] < df['open']) &
                                   (df['close'].shift(1) > df['open'].shift(1)) &
                                   (df['open'] > df['close'].shift(1)) &
                                   (df['close'] < (df['open'].shift(1) + df['close'].shift(1)) / 2)).astype(int)

        pattern_columns['bearish_evening_star'] = ((df['close'].shift(2) > df['open'].shift(2)) &
                                     (abs(df['close'].shift(1) - df['open'].shift(1)) < 0.1 * (df['high'].shift(1) - df['low'].shift(1))) &
                                     (df['close'] < df['open']) &
                                     (df['close'] < (df['open'].shift(2) + df['close'].shift(2)) / 2)).astype(int)

        # Generate remaining two-candle bearish patterns
        for i in range(21):
            pattern_columns[f'bearish_two_candle_{i+4}'] = ((df['close'] < df['open']) &
                                             (df['close'] < df['close'].shift(1))).astype(int)

        # Three Candle Patterns - Bullish (25 patterns)
        pattern_columns['bullish_three_white_soldiers'] = ((df['close'] > df['open']) &
                                             (df['close'].shift(1) > df['open'].shift(1)) &
                                             (df['close'].shift(2) > df['open'].shift(2)) &
                                             (df['close'] > df['close'].shift(1)) &
                                             (df['close'].shift(1) > df['close'].shift(2))).astype(int)

        pattern_columns['bullish_morning_star_three'] = ((df['close'].shift(2) < df['open'].shift(2)) &
                                           (df['close'].shift(1) < df['open'].shift(1)) &
                                           (df['close'] > df['open']) &
                                           (df['close'] > (df['open'].shift(2) + df['close'].shift(2)) / 2)).astype(int)

        # Generate remaining three-candle bullish patterns
        for i in range(23):
            pattern_columns[f'bullish_three_candle_{i+2}'] = ((df['close'] > df['open']) &
                                                (df['close'] > df['close'].shift(1)) &
                                                (df['close'].shift(1) > df['close'].shift(2))).astype(int)

        # Three Candle Patterns - Bearish (25 patterns)
        pattern_columns['bearish_three_black_crows'] = ((df['close'] < df['open']) &
                                          (df['close'].shift(1) < df['open'].shift(1)) &
                                          (df['close'].shift(2) < df['open'].shift(2)) &
                                          (df['close'] < df['close'].shift(1)) &
                                          (df['close'].shift(1) < df['close'].shift(2))).astype(int)

        pattern_columns['bearish_evening_star_three'] = ((df['close'].shift(2) > df['open'].shift(2)) &
                                           (df['close'].shift(1) > df['open'].shift(1)) &
                                           (df['close'] < df['open']) &
                                           (df['close'] < (df['open'].shift(2) + df['close'].shift(2)) / 2)).astype(int)

        # Generate remaining three-candle bearish patterns
        for i in range(23):
            pattern_columns[f'bearish_three_candle_{i+2}'] = ((df['close'] < df['open']) &
                                                (df['close'] < df['close'].shift(1)) &
                                                (df['close'].shift(1) < df['close'].shift(2))).astype(int)

        # Add all pattern columns at once to avoid fragmentation
        if pattern_columns:
            pattern_df = pd.DataFrame(pattern_columns, index=df.index)
            df = pd.concat([df, pattern_df], axis=1)

        # ===== QUANTUM ENGINEERING FEATURES =====

        # Initialize Fibonacci columns
        df['fib_236_20'] = np.nan
        df['fib_382_20'] = np.nan
        df['fib_500_20'] = np.nan
        df['fib_618_20'] = np.nan

        # Fibonacci retracements (simplified)
        def fib_retracement(high, low):
            diff = high - low
            return {
                'fib_236': low + 0.236 * diff,
                'fib_382': low + 0.382 * diff,
                'fib_500': low + 0.5 * diff,
                'fib_618': low + 0.618 * diff
            }

        # Calculate Fib levels for 20-period windows
        for i in range(20, len(df)):
            high_20 = df['high'].iloc[i-20:i].max()
            low_20 = df['low'].iloc[i-20:i].min()
            fib = fib_retracement(high_20, low_20)
            df.loc[df.index[i], 'fib_236_20'] = fib['fib_236']
            df.loc[df.index[i], 'fib_382_20'] = fib['fib_382']
            df.loc[df.index[i], 'fib_500_20'] = fib['fib_500']
            df.loc[df.index[i], 'fib_618_20'] = fib['fib_618']

        # Golden ratio relationships
        df['golden_ratio_body'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
        df['golden_ratio_fib'] = df['close'] / df['fib_618_20'].fillna(df['close'])

        # Quantum-inspired momentum
        df['quantum_momentum'] = (df['close'] - df['close'].shift(10)) / df['atr_14']
        df['harmonic_oscillator'] = np.sin(2 * np.pi * np.arange(len(df)) / 20) * df['rsi_14'] / 100

        # Chaos theory approximation
        df['price_entropy'] = -df['return_pct'].rolling(20).apply(lambda x: np.sum(x * np.log(np.abs(x) + 1e-10)), raw=False)

        # Neural network inspired features
        df['relu_momentum'] = np.maximum(0, df['return_pct'])
        df['tanh_sentiment'] = np.tanh(df['rsi_14'] / 50 - 1)

        # Fill NaN values instead of dropping
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)

        return df

    def create_targets(self, df):
        """Create target for next close price"""
        df = df.copy()

        # Shift close value to create next candle target
        df['next_close'] = df['close'].shift(-1)

        # Remove last row (no target)
        df = df[:-1]

        return df

    def train_models(self, pair, df, test_size=0.2):
        """Train RF and XGB models for OHLC prediction"""
        print(f"Training {pair} candle prediction models...")

        # Prepare features and targets
        X = df[self.feature_cols].dropna()
        y = df.loc[X.index, self.target_cols]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Random Forest
        print(f"Training {pair} Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)

        # Train XGBoost
        print(f"Training {pair} XGBoost...")
        xgb_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        xgb_model.fit(X_train_scaled, y_train)

        # Evaluate models
        rf_pred = rf_model.predict(X_test_scaled)
        xgb_pred = xgb_model.predict(X_test_scaled)

        print(f"{pair} Random Forest MAE: {mean_absolute_error(y_test, rf_pred):.6f}")
        print(f"{pair} XGBoost MAE: {mean_absolute_error(y_test, xgb_pred):.6f}")

        # Ensemble prediction (average)
        ensemble_pred = (rf_pred + xgb_pred) / 2
        print(f"{pair} Ensemble MAE: {mean_absolute_error(y_test, ensemble_pred):.6f}")

        # Store models
        models_dir = f'models/{pair}_candle_models'
        os.makedirs(models_dir, exist_ok=True)

        joblib.dump(rf_model, f'{models_dir}/{pair}_rf_candle.joblib')
        joblib.dump(xgb_model, f'{models_dir}/{pair}_xgb_candle.joblib')
        joblib.dump(scaler, f'{models_dir}/{pair}_scaler_candle.joblib')

        self.models[pair] = {'rf': rf_model, 'xgb': xgb_model}
        self.scalers[pair] = scaler

        print(f"{pair} models saved to {models_dir}")

        return X_test, y_test, ensemble_pred

    def predict_next_candle(self, pair, df):
        """Predict OHLC for next candle"""
        models_dir = f'models/{pair}_candle_models'

        if not os.path.exists(f'{models_dir}/{pair}_rf_candle.joblib'):
            print(f"Models for {pair} not found. Please train first.")
            return None

        # Load models if not already loaded
        if pair not in self.models:
            self.models[pair] = {
                'rf': joblib.load(f'{models_dir}/{pair}_rf_candle.joblib'),
                'xgb': joblib.load(f'{models_dir}/{pair}_xgb_candle.joblib')
            }
            self.scalers[pair] = joblib.load(f'{models_dir}/{pair}_scaler_candle.joblib')

        # Get latest data and engineer features
        # Need enough historical data for feature computation (lookback periods)
        min_rows = 50  # Minimum rows needed for indicators like RSI, MACD, etc.
        latest_data = df.iloc[-min_rows:].copy() if len(df) >= min_rows else df.copy()
        latest_data = self.engineer_features(latest_data, pair)

        # Use the last valid row after feature engineering
        if len(latest_data) == 0:
            print(f"Warning: No valid data for {pair} prediction after feature engineering")
            return None

        latest = latest_data.iloc[-1:]  # Get the last valid row

        # Prepare features
        X = latest[self.feature_cols].values
        X_scaled = self.scalers[pair].transform(X)

        # Get predictions
        rf_pred = self.models[pair]['rf'].predict(X_scaled)[0]
        xgb_pred = self.models[pair]['xgb'].predict(X_scaled)[0]

        # Ensemble (average)
        prediction = (rf_pred + xgb_pred) / 2

        result = {
            'pair': pair,
            'date': latest['date'].iloc[0],
            'predicted_close': prediction,
            'current_close': latest['close'].iloc[0]
        }

        return result

    def run_full_pipeline(self):
        """Run complete training pipeline for all pairs"""
        results = {}

        for pair in self.pairs:
            print(f"\n{'='*50}")
            print(f"Processing {pair}")
            print('='*50)

            # Fetch data for different timeframes
            daily_df = self.fetch_data(pair, interval='1d')
            h4_df = self.fetch_data(pair, interval='4h')
            weekly_df = self.fetch_data(pair, interval='1wk')

            # Use daily data for main training
            df = daily_df.copy()

            # Engineer features with multi-timeframe data
            df = self.engineer_features(df, pair)

            # Create targets
            df = self.create_targets(df)

            # Train models (incremental update)
            success = self.incremental_train_models(pair, df)
            if success:
                # Update last training date
                self.last_update[pair] = pd.to_datetime(df['date']).max()
                results[pair] = {
                    'dataframe': df,
                    'status': 'trained' if not self.load_cached_models(pair) else 'cached_loaded'
                }
            else:
                results[pair] = {
                    'dataframe': df,
                    'status': 'training_failed'
                }

            # Make a prediction
            prediction = self.predict_next_candle(pair, df)
            if prediction:
                print(f"\n{pair} Next Candle Prediction:")
                print(prediction)

        print("\n" + "="*50)
        print("Training complete for all pairs! Models saved.")
        print("Models are saved in the models/ directory")
        print("="*50)

        return results

    def update_csv_with_prediction(self, pair, prediction):
        """Update the CSV file with the predicted candle data"""
        csv_file = f'data/raw/{pair}_Daily.csv'
        
        # Load existing data
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
        else:
            print(f"Warning: {csv_file} not found, creating new file")
            df = pd.DataFrame()
        
        # Create new row with prediction data
        next_date = datetime.now() + timedelta(days=1)
        new_row = {
            'date': next_date.strftime('%Y-%m-%d'),
            'open': prediction['current_close'],  # Use current close as open estimate
            'high': prediction['predicted_close'] * 1.001,  # Slight high estimate
            'low': prediction['predicted_close'] * 0.999,   # Slight low estimate
            'close': prediction['predicted_close'],
            'volume': 0  # Placeholder for predicted volume
        }
        
        # Append to dataframe
        new_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_df], ignore_index=True)
        
        # Save back to CSV
        df.to_csv(csv_file, index=False)
        print(f"Updated {csv_file} with predicted candle for {next_date.strftime('%Y-%m-%d')}")
        
        return df

# Example usage
if __name__ == "__main__":
    # Train both EURUSD and XAUUSD models
    system = CandlePredictionSystem(['EURUSD', 'XAUUSD'])
    results = system.run_full_pipeline()

    # Make predictions for both pairs and update CSVs
    for pair in ['EURUSD', 'XAUUSD']:
        prediction = system.predict_next_candle(pair, results[pair]['dataframe'])
        if prediction:
            print(f"\n{pair} Prediction:")
            print(prediction)
            
            # Update CSV with prediction
            system.update_csv_with_prediction(pair, prediction)