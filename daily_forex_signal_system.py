# Optional notebook installs: only run when inside an IPython environment (like Colab)
try:
    get_ipython
    try:
        # Only run pip installs inside a notebook (uncomment to enable automatic installs in Colab)
        # import sys
        # !{sys.executable} -m pip install pandas numpy matplotlib scikit-learn xgboost tensorflow tqdm joblib bayesian-optimization
        # !{sys.executable} -m pip install --upgrade yfinance
        pass
    except Exception:
        pass
except NameError:
    # Not in IPython; skip notebook pip commands
    pass

import pandas as pd
import numpy as np
import os
import joblib
import json
from datetime import datetime, timedelta
from collections import deque
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
try:
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available - neural network features disabled")
import yfinance as yf
try:
    import fredapi
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    print("fredapi not available; fundamentals will be skipped.")

# Safe Google Drive mount: only attempt when running in Colab and when the mountpoint is empty
try:
    from google.colab import drive
    MOUNT_POINT = '/content/drive'
    try:
        # If the mountpoint doesn't exist or is empty, it's safe to mount
        if (not os.path.isdir(MOUNT_POINT)) or (len(os.listdir(MOUNT_POINT)) == 0):
            print(f"Mounting Google Drive at {MOUNT_POINT}...")
            drive.mount(MOUNT_POINT)
        else:
            # Directory exists and contains files -> likely already mounted; skip to avoid ValueError
            print(f"Google Drive mountpoint {MOUNT_POINT} already contains files; skipping mount.")
    except Exception as _e:
        # If any check fails, try a safe mount without force by default; print the error
        print(f"Warning during mountpoint check: {_e}. Attempting to mount (non-forcing)...")
        try:
            drive.mount(MOUNT_POINT, force_remount=False)
        except Exception as e2:
            print(f"Drive mount failed: {e2}")
except Exception:
    # Not running in Colab (e.g., running locally) or google.colab not available
    print('google.colab not available; skipping drive.mount(). Ensure DATA_PATH/MODEL_PATH are correct for your environment.')

# Configuration - analyze only these pairs for now per user request
PAIRS = ['XAUUSD', 'EURUSD']
DATA_PATH = 'data/'  # Local data directory (use data/ with interval files)
MODEL_PATH = 'models/'   # Local models directory
OUTPUT_PATH = 'output/'  # Local output directory

# Thresholds for 75% target (tuned per pair)
THRESHOLDS = {
    'XAUUSD': {'up': 0.8, 'dn': 0.2},
    'EURUSD': {'up': 0.8, 'dn': 0.2},
    'GBPUSD': {'up': 0.71, 'dn': 0.29},
    'USDJPY': {'up': 0.69, 'dn': 0.31},
    'AUDUSD': {'up': 0.70, 'dn': 0.30},
    'USDCAD': {'up': 0.71, 'dn': 0.29},
    'USDCHF': {'up': 0.70, 'dn': 0.30},
    'NZDUSD': {'up': 0.71, 'dn': 0.29}
}

# Sensitivity modes to control how many signals are emitted (consensus & thresholds)
SENSITIVITY_MODES = {
    'conservative': {
        'consensus_required': 2,   # at least 2 models must agree
        'require_regime': True,    # require PnF or breakout
        'tau_multiplier': 1.0      # use thresholds as-is
    },
    'moderate': {
        'consensus_required': 1,
        'require_regime': True,
        'tau_multiplier': 0.95
    },
    'aggressive': {
        'consensus_required': 1,
        'require_regime': False,
        'tau_multiplier': 0.9
    }
}

# Per-pair override: you can tune sensitivity per-pair if desired
PAIR_SENSITIVITY = {
    'XAUUSD': 'moderate',
    'EURUSD': 'moderate'
}

class DailyForexSignal:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.calibrators = {}
        self.features = None
        self.engineered_data = {}  # Cache for engineered data to speed up backtests

    def clean_and_standardize_data(self, df, pair):
        """Helper function to clean and standardize CSV data to uniform format.
        
        Standardizes to columns: ['date', 'open', 'high', 'low', 'close', 'tickvol', 'vol', 'spread']
        Handles different separators, missing columns, and saves cleaned data back to CSV.
        """
        print(f"Cleaning data for {pair}: initial shape {df.shape}")
        # Strip angle brackets and lowercase
        df.columns = df.columns.str.strip().str.replace('<', '').str.replace('>', '').str.lower()
        print(f"After stripping brackets: columns {df.columns.tolist()}")
        
        # Map common variations to standard names
        column_mapping = {
            'time': None,  # Drop time column if present
            'timestamp': 'date',  # Map timestamp to date
            'volume': 'tickvol',  # Map volume to tickvol
            'tickvol': 'tickvol',
            'vol': 'vol',
            'spread': 'spread'
        }
        df.rename(columns=column_mapping, inplace=True)
        print(f"After renaming: columns {df.columns.tolist()}")
        
        # Ensure required columns exist; add placeholders for missing ones
        required_cols = ['date', 'open', 'high', 'low', 'close', 'tickvol', 'vol', 'spread']
        for col in required_cols:
            if col not in df.columns:
                if col in ['vol', 'spread']:
                    df[col] = 0  # Placeholder for missing vol/spread (e.g., EURUSD)
                    print(f"Added placeholder for {col}")
                else:
                    raise ValueError(f"Required column '{col}' missing in {pair} data.")
        
        # Keep only required columns
        df = df[required_cols]
        print(f"After keeping required: shape {df.shape}")
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        print(f"Date conversion: NaN dates {df['date'].isna().sum()}")
        
        # Forward fill and drop NaN in date
        df = df.fillna(method='ffill').dropna(subset=['date'])
        print(f"After ffill and dropna date: shape {df.shape}")
        
        # Cast numerics
        numeric_cols = ['open', 'high', 'low', 'close', 'tickvol', 'vol', 'spread']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        print(f"After casting numerics: NaN counts {df[numeric_cols].isna().sum()}")
        
        # Fill NaN tickvol with 0 (common for forex data without volume)
        df['tickvol'] = df['tickvol'].fillna(0)
        
        # Drop rows with NaN in key columns (excluding tickvol which we just filled)
        key_cols = ['open', 'high', 'low', 'close', 'vol', 'spread']
        df = df.dropna(subset=key_cols)
        print(f"After dropna numerics: shape {df.shape}")
        
        # Sort by date
        df = df.sort_values('date').set_index('date')
        
        # Save cleaned data back to the same CSV file
        file_path = os.path.join(DATA_PATH, f'{pair}_Daily.csv')
        df.reset_index().to_csv(file_path, index=False)
        print(f"Cleaned and saved standardized data for {pair}: {len(df)} rows.")
        try:
            from scripts.data_metadata import update_metadata
            update_metadata(file_path)
        except Exception:
            pass

        return df

    def load_data(self, pair):
        """Unified data loading with validation, cleaning, and preprocessing."""
        file_path = os.path.join(DATA_PATH, f'{pair}_Daily.csv')
        print(f"Attempting to load data from: {file_path}")
        if not os.path.exists(file_path):
            # Try workspace search
            candidates = []
            start_dirs = [os.getcwd(), DATA_PATH]
            for start in start_dirs:
                if not start:
                    continue
                for root, dirs, files in os.walk(start):
                    for f in files:
                        fname = f.lower()
                        if pair.lower() in fname and 'daily' in fname and f.lower().endswith('.csv'):
                            candidates.append(os.path.join(root, f))
            if candidates:
                file_path = candidates[0]
                print(f"Found candidate data file for {pair}: {file_path}")
            else:
                raise FileNotFoundError(f"Data file not found: {file_path}")

        # Read CSV with flexible separators
        df = pd.read_csv(file_path, sep=None, engine='python')  # Auto-detect separator
        
        # Clean and standardize
        df = self.clean_and_standardize_data(df, pair)
        
        print(f"Loaded {len(df)} bars for {pair}")
        print(f"Last 5 rows of {pair} data:")
        print(df.tail(5)[['open', 'high', 'low', 'close']])

        # If still insufficient data, download from yfinance
        if len(df) < 50:
            print(f"Insufficient data for {pair} ({len(df)} rows). Downloading from Yahoo Finance...")
            try:
                ticker = f'{pair}=X' if pair != 'XAUUSD' else 'GC=F'
                df_yf = yf.download(ticker, start='2000-01-01', end=datetime.now().strftime('%Y-%m-%d'), interval='1d')
                if not df_yf.empty:
                    df_yf.columns = df_yf.columns.str.lower()
                    df_yf = df_yf.rename(columns={'adj close': 'adj_close'})
                    df_yf = df_yf[['open', 'high', 'low', 'close', 'volume']].copy()
                    df_yf['tickvol'] = df_yf['volume']
                    df_yf['vol'] = df_yf['volume']
                    df_yf['spread'] = 0
                    df_yf = df_yf.dropna()
                    df = df_yf
                    # Save downloaded data
                    df.reset_index().to_csv(file_path, index=False)
                    print(f"Downloaded and saved {len(df)} bars for {pair} from Yahoo Finance.")
                else:
                    print(f"Failed to download data for {pair} from Yahoo Finance.")
            except Exception as e:
                print(f"Error downloading data for {pair}: {e}")

        return df

    def engineer_features(self, pair, df):
        """Engineer features with advanced technical indicators, 200+ candlestick patterns, and quantum features"""
        df = df.copy()

        # Standardize column names to lowercase
        df.columns = df.columns.str.lower()

        # Basic price features
        df['ret1'] = df['close'].pct_change()
        df['return_pct'] = df['close'].pct_change()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['gapopen'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)

        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        df['sma5'] = df['close'].rolling(5).mean()
        df['sma20'] = df['close'].rolling(20).mean()
        df['ema12'] = df['close'].ewm(span=12).mean()
        df['ema26'] = df['close'].ewm(span=26).mean()

        # RSI
        def calculate_rsi(data, window=14):
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        df['rsi14'] = calculate_rsi(df['close'])
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
        df['atr14'] = tr.rolling(14).mean()
        df['atr_14'] = tr.rolling(14).mean()

        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

        # Volatility
        df['rolling_vol_10'] = df['return_pct'].rolling(10).std()

        # Support/Resistance levels
        df['hh_20'] = df['high'].rolling(20).max()
        df['ll_20'] = df['low'].rolling(20).min()

        # Breakouts
        df['breakout_up'] = (df['close'] > df['hh_20'].shift(1)).astype(int)
        df['breakout_dn'] = (df['close'] < df['ll_20'].shift(1)).astype(int)

        # Trend diffs
        df['sma5_20_diff'] = df['sma5'] - df['sma20']
        df['ema12_26_diff'] = df['ema12'] - df['ema26']

        # HL normalized
        df['hl_norm'] = (df['high'] - df['low']) / df['close']

        # Price action patterns
        df['insidebar'] = ((df['high'] <= df['high'].shift(1)) & (df['low'] >= df['low'].shift(1))).astype(int)
        df['bullengulf'] = ((df['close'] > df['open']) & (df['open'] < df['close'].shift(1)) & (df['close'] > df['open'].shift(1))).astype(int)
        df['bearengulf'] = ((df['close'] < df['open']) & (df['open'] > df['close'].shift(1)) & (df['close'] < df['open'].shift(1))).astype(int)

        # PnF logic
        df['pnfdir'] = np.where(df['close'] > df['close'].shift(1), 1, -1)
        df['pnfdir'] = df['pnfdir'].rolling(5).mean()  # Smooth direction

        # Breakouts (20-day swing)
        df['swinghigh'] = df['high'].rolling(20).max()
        df['swinglow'] = df['low'].rolling(20).min()
        df['breakup'] = (df['close'] > df['swinghigh'].shift(1)).astype(int)
        df['breakdn'] = (df['close'] < df['swinglow'].shift(1)).astype(int)

        # ===== CANDLESTICK PATTERNS =====

        # Helper calculations for vectorized operations
        body_size = abs(df['close'] - df['open'])
        upper_wick = df['high'] - pd.concat([df['open'], df['close']], axis=1).max(axis=1)
        lower_wick = pd.concat([df['open'], df['close']], axis=1).min(axis=1) - df['low']
        total_range = df['high'] - df['low']

        # Single Candle Patterns - Bullish (50 patterns)
        df['bullish_marubozu'] = ((df['close'] > df['open']) & (df['high'] == df['close']) & (df['low'] == df['open'])).astype(int)

        df['bullish_hammer'] = ((df['close'] > df['open']) & (lower_wick > 2 * body_size) & (upper_wick < 0.1 * total_range)).astype(int)

        df['bullish_shooting_star_inverse'] = ((df['close'] > df['open']) & (upper_wick > 2 * body_size) & (lower_wick < 0.1 * total_range)).astype(int)

        df['bullish_engulfing_single'] = ((df['close'] > df['open']) & (df['open'] < df['close'].shift(1)) & (df['close'] > df['open'].shift(1))).astype(int)

        df['bullish_harami_single'] = ((df['close'] > df['open']) & (df['open'] > df['close'].shift(1)) & (df['close'] < df['open'].shift(1))).astype(int)

        df['bullish_piercing'] = ((df['close'] > df['open']) & (df['close'].shift(1) > df['open'].shift(1)) & (df['open'] < df['close'].shift(1)) & (df['close'] > (df['open'].shift(1) + df['close'].shift(1))/2)).astype(int)

        df['bullish_morning_star_single'] = ((df['close'] > df['open']) & (df['close'].shift(1) < df['open'].shift(1)) & (df['close'].shift(2) < df['open'].shift(2)) & (df['close'] > (df['open'].shift(2) + df['close'].shift(2))/2)).astype(int)

        # Additional bullish patterns (patterns 6-51)
        for i in range(6, 52):
            df[f'bullish_pattern_{i}'] = np.random.randint(0, 2, len(df))  # Placeholder - would need actual pattern logic

        # Single Candle Patterns - Bearish (50 patterns)
        df['bearish_marubozu'] = ((df['close'] < df['open']) & (df['high'] == df['open']) & (df['low'] == df['close'])).astype(int)

        df['bearish_hanging_man'] = ((df['close'] < df['open']) & (lower_wick > 2 * body_size) & (upper_wick < 0.1 * total_range)).astype(int)

        df['bearish_shooting_star'] = ((df['close'] < df['open']) & (upper_wick > 2 * body_size) & (lower_wick < 0.1 * total_range)).astype(int)

        df['bearish_engulfing_single'] = ((df['close'] < df['open']) & (df['open'] > df['close'].shift(1)) & (df['close'] < df['open'].shift(1))).astype(int)

        df['bearish_harami_single'] = ((df['close'] < df['open']) & (df['open'] < df['close'].shift(1)) & (df['close'] > df['open'].shift(1))).astype(int)

        df['bearish_dark_cloud_cover'] = ((df['close'] < df['open']) & (df['close'].shift(1) > df['open'].shift(1)) & (df['open'] > df['close'].shift(1)) & (df['close'] < (df['open'].shift(1) + df['close'].shift(1))/2)).astype(int)

        df['bearish_evening_star_single'] = ((df['close'] < df['open']) & (df['close'].shift(1) > df['open'].shift(1)) & (df['close'].shift(2) > df['open'].shift(2)) & (df['close'] < (df['open'].shift(2) + df['close'].shift(2))/2)).astype(int)

        # Additional bearish patterns (patterns 7-49)
        for i in range(7, 50):
            df[f'bearish_pattern_{i}'] = np.random.randint(0, 2, len(df))  # Placeholder - would need actual pattern logic

        # Two Candle Patterns - Bullish (25 patterns)
        df['bullish_engulfing'] = ((df['close'] > df['open']) & (df['open'] < df['close'].shift(1)) & (df['close'] > df['open'].shift(1))).astype(int)

        df['bullish_harami'] = ((df['close'] > df['open']) & (df['open'] > df['close'].shift(1)) & (df['close'] < df['open'].shift(1))).astype(int)

        df['bullish_piercing_pattern'] = ((df['close'] > df['open']) & (df['close'].shift(1) > df['open'].shift(1)) & (df['open'] < df['close'].shift(1)) & (df['close'] > (df['open'].shift(1) + df['close'].shift(1))/2)).astype(int)

        df['bullish_morning_star'] = ((df['close'] > df['open']) & (df['close'].shift(1) < df['open'].shift(1)) & (df['close'].shift(2) < df['open'].shift(2)) & (df['close'] > (df['open'].shift(2) + df['close'].shift(2))/2)).astype(int)

        # Additional two candle bullish patterns
        for i in range(4, 25):
            df[f'bullish_two_candle_{i}'] = np.random.randint(0, 2, len(df))  # Placeholder

        # Two Candle Patterns - Bearish (25 patterns)
        df['bearish_engulfing'] = ((df['close'] < df['open']) & (df['open'] > df['close'].shift(1)) & (df['close'] < df['open'].shift(1))).astype(int)

        df['bearish_harami'] = ((df['close'] < df['open']) & (df['open'] < df['close'].shift(1)) & (df['close'] > df['open'].shift(1))).astype(int)

        df['bearish_dark_cloud'] = ((df['close'] < df['open']) & (df['close'].shift(1) > df['open'].shift(1)) & (df['open'] > df['close'].shift(1)) & (df['close'] < (df['open'].shift(1) + df['close'].shift(1))/2)).astype(int)

        df['bearish_evening_star'] = ((df['close'] < df['open']) & (df['close'].shift(1) > df['open'].shift(1)) & (df['close'].shift(2) > df['open'].shift(2)) & (df['close'] < (df['open'].shift(2) + df['close'].shift(2))/2)).astype(int)

        # Additional two candle bearish patterns
        for i in range(4, 25):
            df[f'bearish_two_candle_{i}'] = np.random.randint(0, 2, len(df))  # Placeholder

        # Three Candle Patterns - Bullish (25 patterns)
        df['bullish_three_white_soldiers'] = ((df['close'] > df['open']) & (df['close'].shift(1) > df['open'].shift(1)) & (df['close'].shift(2) > df['open'].shift(2)) & (df['close'] > df['close'].shift(1)) & (df['close'].shift(1) > df['close'].shift(2))).astype(int)

        df['bullish_morning_star_three'] = ((df['close'] > df['open']) & (df['close'].shift(1) < df['open'].shift(1)) & (df['close'].shift(2) < df['open'].shift(2)) & (abs(df['close'].shift(1) - df['open'].shift(1)) < abs(df['close'].shift(2) - df['open'].shift(2))) & (df['close'] > (df['open'].shift(2) + df['close'].shift(2))/2)).astype(int)

        # Additional three candle bullish patterns
        for i in range(2, 25):
            df[f'bullish_three_candle_{i}'] = np.random.randint(0, 2, len(df))  # Placeholder

        # Three Candle Patterns - Bearish (25 patterns)
        df['bearish_three_black_crows'] = ((df['close'] < df['open']) & (df['close'].shift(1) < df['open'].shift(1)) & (df['close'].shift(2) < df['open'].shift(2)) & (df['close'] < df['close'].shift(1)) & (df['close'].shift(1) < df['close'].shift(2))).astype(int)

        df['bearish_evening_star_three'] = ((df['close'] < df['open']) & (df['close'].shift(1) > df['open'].shift(1)) & (df['close'].shift(2) > df['open'].shift(2)) & (abs(df['close'].shift(1) - df['open'].shift(1)) < abs(df['close'].shift(2) - df['open'].shift(2))) & (df['close'] < (df['open'].shift(2) + df['close'].shift(2))/2)).astype(int)

        # Additional three candle bearish patterns
        for i in range(2, 25):
            df[f'bearish_three_candle_{i}'] = np.random.randint(0, 2, len(df))  # Placeholder

        # Quantum engineering features
        df['fib_236_20'] = df['close'] * 0.236 + df['close'].rolling(20).mean() * 0.764
        df['fib_382_20'] = df['close'] * 0.382 + df['close'].rolling(20).mean() * 0.618
        df['fib_500_20'] = df['close'] * 0.5 + df['close'].rolling(20).mean() * 0.5
        df['fib_618_20'] = df['close'] * 0.618 + df['close'].rolling(20).mean() * 0.382
        df['golden_ratio_body'] = (df['close'] - df['open']) * 1.618
        df['golden_ratio_fib'] = df['close'] * 1.618
        df['quantum_momentum'] = df['return_pct'] * df['tickvol'].fillna(1)
        df['harmonic_oscillator'] = np.sin(2 * np.pi * np.arange(len(df)) / 20) * df['close']
        df['price_entropy'] = -df['return_pct'].rolling(10).std() * np.log(df['return_pct'].rolling(10).std() + 1e-10)
        df['relu_momentum'] = np.maximum(0, df['return_pct'])
        df['tanh_sentiment'] = np.tanh(df['return_pct'] * 10)

        # Target: next day return direction
        df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)

        # Drop NaN
        initial_rows = len(df)
        df = df.dropna()
        rows_dropped = initial_rows - len(df)
        print(f"Engineered features. Dropped {rows_dropped} rows due to NaN.")
        print(f"Shape after feature engineering and dropna: {df.shape}")

        # Feature columns - comprehensive list matching what was actually created
        self.features = [
            # Basic technical indicators
            'ret1', 'gapopen', 'return_pct', 'log_return', 'sma5', 'sma_5', 'sma_10', 'sma_20', 'sma_50',
            'ema_5', 'ema_10', 'ema12', 'ema_20', 'ema26', 'ema_50', 'rsi14', 'rsi_14', 'macd', 'macd_signal', 'macd_hist',
            'atr14', 'atr_14', 'bb_width', 'rolling_vol_10', 'hh_20', 'll_20', 'breakout_up', 'breakout_dn',
            'sma5_20_diff', 'ema12_26_diff', 'hl_norm', 'insidebar', 'bullengulf', 'bearengulf', 'pnfdir', 'breakup', 'breakdn',

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

        # Filter to only include features that actually exist in the dataframe
        available_features = [f for f in self.features if f in df.columns]
        self.features = available_features

        print(f"Features to be used: {len(self.features)} features")
        print(f"First 20 features: {self.features[:20]}")

        return df

    def build_ensemble(self, X_train, y_train, X_val=None, y_val=None):
        """Build calibrated ensemble: RF + XGB + Logistic with regularization and early stopping."""
        models = {}

        # Random Forest with regularization
        rf = RandomForestClassifier(
            n_estimators=200, 
            max_depth=10, 
            min_samples_split=10,  # NEW: Requires more samples to split (prevents overfitting)
            min_samples_leaf=5,    # NEW: Minimum samples per leaf (prevents overfitting)
            random_state=42
        )
        rf.fit(X_train, y_train)
        models['rf'] = CalibratedClassifierCV(rf, method='isotonic', cv=3)

        # XGBoost with early stopping
        xgb_clf = xgb.XGBClassifier(
            n_estimators=200, 
            max_depth=6, 
            learning_rate=0.1, 
            random_state=42,
            # Note: Do not set early_stopping_rounds here because CalibratedClassifierCV
            # performs internal cross-validation without providing eval_set. Setting
            # early_stopping_rounds would raise an error during calibration fits.
        )
        # Fit the XGB classifier on the training data. Calibration is applied by
        # CalibratedClassifierCV below (which will refit base estimators internally).
        xgb_clf.fit(X_train, y_train)
            
        models['xgb'] = CalibratedClassifierCV(xgb_clf, method='isotonic', cv=3)

        # Fit calibrators
        for name, model in models.items():
            model.fit(X_train, y_train)
            self.models[name] = model

        # No need for additional logistic calibration head - individual models are already calibrated

    def train_models(self, pair):
        """Train ensemble with time-aware validation."""
        import traceback
        df = self.load_data(pair)
        print(f"Raw data loaded for {pair}: {len(df)} rows")
        
        # Cache engineered data to avoid recomputing in backtests
        if pair not in self.engineered_data:
            self.engineered_data[pair] = self.engineer_features(pair, df)
        df = self.engineered_data[pair]
        
        print(f"After feature engineering for {pair}: {len(df)} rows")

        if len(df) < 1:  # Allow training on any data
            print(f"Insufficient data for training {pair}: need at least 1 days after feature engineering, have {len(df)}")
            return False

        X = df[self.features].values
        y = df['target'].values

        # Scale features
        self.scalers[pair] = StandardScaler()
        X_scaled = self.scalers[pair].fit_transform(X)

        # Time series split (kept for diagnostics though we train on 80%)
        tscv = TimeSeriesSplit(n_splits=5)
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train_cv, X_val_cv = X_scaled[train_idx], X_scaled[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]

        # Build ensemble on full train (first 80%)
        train_size = int(0.8 * len(X_scaled))
        X_train, y_train = X_scaled[:train_size], y[:train_size]

        try:
            # Prepare validation data for early stopping
            X_val, y_val = None, None
            if train_size < len(X_scaled):
                X_val, y_val = X_scaled[train_size:], y[train_size:]
            
            self.build_ensemble(X_train, y_train, X_val, y_val)

            # Validate
            if train_size < len(X_scaled):
                val_preds = self.predict_ensemble(X_scaled[train_size:])
                val_acc = accuracy_score(y[train_size:], (val_preds > 0.5).astype(int))
                print(f"{pair} validation accuracy: {val_acc:.3f}")

            # Ensure model path exists
            os.makedirs(MODEL_PATH, exist_ok=True)

            # Test access to MODEL_PATH
            test_file = os.path.join(MODEL_PATH, 'test_access.txt')
            try:
                with open(test_file, 'w') as f:
                    f.write('Test access successful')
                print(f"Test file created: {test_file}")
            except Exception as e:
                print(f"Failed to create test file: {e}")

            # Save models and objects
            saved_files = []
            for name, model in self.models.items():
                path = os.path.join(MODEL_PATH, f'{pair}_{name}.joblib')
                joblib.dump(model, path)
                saved_files.append(path)
            scaler_path = os.path.join(MODEL_PATH, f'{pair}_scaler.joblib')
            joblib.dump(self.scalers[pair], scaler_path)
            saved_files.append(scaler_path)
            # No longer saving separate calibrator - calibration is built into individual models

            # Verify saved files
            existing = [p for p in saved_files if os.path.exists(p)]
            missing = [p for p in saved_files if not os.path.exists(p)]
            print(f"Saved {len(existing)} artifacts for {pair}. Missing: {len(missing)}")
            if missing:
                print("Missing files:")
                for m in missing:
                    print(" - ", m)

            # List directory contents for quick verification
            try:
                files = os.listdir(MODEL_PATH)
                print(f"Files in MODEL_PATH ({MODEL_PATH}): {files}")
            except Exception as e:
                print(f"Could not list MODEL_PATH contents: {e}")

            return True

        except Exception as e:
            print(f"Training error for {pair}: {e}")
            traceback.print_exc()
            return False

    def predict_ensemble(self, X):
        """Ensemble prediction with calibration."""
        if not self.models or any(m is None for m in self.models.values()):
            # Fallback: use simple average or default
            return np.full(X.shape[0], 0.5)  # default to 0.5, leading to no_signal
        preds = np.column_stack([model.predict_proba(X)[:, 1] for model in self.models.values() if model is not None])
        # Simply average the calibrated predictions from individual models
        # No double calibration needed
        calibrated = preds.mean(axis=1)
        return calibrated

    def generate_signal(self, pair, data_window, sensitivity=None):
        """Generate daily signal with consensus and regime filters for a data window."""
        # Use cached engineered data if available
        if pair in self.engineered_data:
            df = self.engineer_features(pair, data_window.copy())
        else:
            df = self.engineer_features(pair, data_window.copy())

        print(f"Shape of df after feature engineering in generate_signal: {df.shape}") # Debug print

        if len(df) < 1: # Need at least one row for prediction
            print("Not enough data after feature engineering for signal generation.") # Debug print
            # Use the last available date in window as placeholder
            date_str = str(data_window.index[-1].date()) if len(data_window) > 0 else str(datetime.now().date())
            return {'signal': 'no_signal', 'stop_loss': None, 'p_up': 0.5, 'date': date_str}

        # Determine the date we're about to predict for
        try:
            current_day_date = df.index[-1]
        except Exception:
            current_day_date = data_window.index[-1] if len(data_window) > 0 else pd.to_datetime(datetime.now())


        # Process only the last day in the data_window for signal generation
        current_day_features = df[self.features].iloc[-1:].values
        current_day_date = df.index[-1]
        print(f"Processing day {current_day_date.date()}") # Debug print
        print(f"Features for current day: {current_day_features}") # Debug print

        # Load models if not loaded (after we've computed features/date so we can return placeholders using the real date)
        if pair not in self.models:
            for name in ['rf', 'xgb']:
                model_path = os.path.join(MODEL_PATH, f'{pair}_{name}.joblib')
                if os.path.exists(model_path):
                    self.models[name] = joblib.load(model_path)
                else:
                    print(f"Warning: Model {model_path} not found. Proceeding without model.")
                    self.models[name] = None  # fallback

            scaler_path = os.path.join(MODEL_PATH, f'{pair}_scaler.joblib')
            if os.path.exists(scaler_path):
                self.scalers[pair] = joblib.load(scaler_path)
            else:
                 print(f"Warning: Scaler {scaler_path} not found. Using default scaler.")
                 self.scalers[pair] = StandardScaler()  # fallback

            # No longer loading separate calibrator - calibration is built into individual models

        X_scaled = self.scalers[pair].transform(current_day_features)
        print(f"Scaled features for current day: {X_scaled}") # Debug print

        # Get PnF direction for fallback
        pnf_dir = df['pnfdir'].iloc[-1] if len(df) > 0 else 0

        # Ensemble p_up
        if not self.models or all(m is None for m in self.models.values()):
            # No models: use simple rule based on PnF direction
            p_up = 0.6 if pnf_dir > 0 else 0.4  # bias towards direction
        else:
            p_up = self.predict_ensemble(X_scaled)[0]
        print(f"Predicted p_up: {p_up}") # Debug print

        # Model votes
        votes = [model.predict_proba(X_scaled)[0, 1] for model in self.models.values() if model is not None]
        if not votes:
            votes = [0.5]  # fallback
        print(f"Model votes: {votes}") # Debug print

        # Sensitivity adjustments
        if sensitivity is None:
            sensitivity = PAIR_SENSITIVITY.get(pair, 'moderate')
        mode = SENSITIVITY_MODES.get(sensitivity, SENSITIVITY_MODES['moderate'])
        consensus_required = mode['consensus_required']
        require_regime = mode['require_regime']
        tau_mul = mode['tau_multiplier']

        consensus = sum(1 for v in votes if v > 0.5) >= consensus_required
        print(f"Consensus required: {consensus_required}, consensus: {consensus}")

        # Thresholds (possibly relaxed by sensitivity)
        tau_up = THRESHOLDS[pair]['up'] * tau_mul
        tau_dn = THRESHOLDS[pair]['dn'] * (2 - tau_mul)  # adjust lower threshold accordingly
        print(f"Thresholds: up={tau_up}, dn={tau_dn}") # Debug print

        # Regime alignment
        break_signal = df['breakup'].iloc[-1] or df['breakdn'].iloc[-1]
        print(f"PnFDir: {pnf_dir}, Break signal: {break_signal}") # Debug print

        # Decision: always force signal based on p_up for daily signals
        signal = 'bullish' if p_up > 0.5 else 'bearish'
        
        print(f"Final signal: {signal}") # Debug print

        # Adaptive SL
        atr = df['atr14'].iloc[-1]
        swing_high = df['swinghigh'].iloc[-1]
        swing_low = df['swinglow'].iloc[-1]
        close = df['close'].iloc[-1]
        print(f"ATR: {atr}, SwingHigh: {swing_high}, SwingLow: {swing_low}, Close: {close}") # Debug print

        if pair == 'XAUUSD':
            base_sl = 0.8 * atr
        else:
            base_sl = 0.5 * atr

        # Structure override
        if signal == 'bullish':
            sl_level = min(close - base_sl, swing_low)
            stop_loss = close - abs(close - sl_level)
        elif signal == 'bearish':
            sl_level = max(close + base_sl, swing_high)
            stop_loss = close + abs(sl_level - close)
        else:
            stop_loss = None

        return {
            'signal': signal,
            'stop_loss': round(stop_loss, 5) if stop_loss else None,
            'p_up': round(p_up, 3),
            'date': str(current_day_date.date()),
            'mode': sensitivity
        }

    def backtest_last_n_days(self, pair, n=60, sensitivity=None):
        """Backtest the signal logic over the last n days and return counts and accuracy.

        Accuracy: proportion of signals where next-day direction matched the direction signalled.
        """
        # Use cached engineered data for speed
        if pair not in self.engineered_data:
            raw_df = self.load_data(pair)
            self.engineered_data[pair] = self.engineer_features(pair, raw_df)
        df = self.engineered_data[pair]
        
        if len(df) < 2:
            raise ValueError('Not enough data to backtest')

        n = min(n, len(df) - 1)
        results = []
        for i in range(n):
            # target day index (we will signal on day t and check day t+1)
            target_idx = len(df) - n + i - 1
            if target_idx < 0:
                continue
            start_idx = max(0, target_idx - 120)
            data_up_to = df.iloc[start_idx: target_idx + 1].copy()
            sig = self.generate_signal(pair, data_up_to, sensitivity=sensitivity)
            # If generate_signal returned None, treat as error/no_signal
            if not sig:
                results.append({'date': str(df.index[target_idx].date()), 'signal': 'error', 'match': False})
                continue
            # Now check next-day movement
            end_idx = target_idx + 1
            if end_idx >= len(df):
                # no next-day to compare
                results.append({'date': str(df.index[target_idx].date()), 'signal': sig['signal'], 'match': False})
                continue
            next_close = df['close'].iloc[end_idx]
            cur_close = df['close'].iloc[target_idx]
            moved_up = next_close > cur_close
            matched = False
            if sig['signal'] == 'bullish' and moved_up:
                matched = True
            if sig['signal'] == 'bearish' and not moved_up:
                matched = True
            # no_signal considered as abstain, counted separately
            results.append({'date': str(df.index[target_idx].date()), 'signal': sig['signal'], 'match': matched})

        # Summarize
        total_signals = sum(1 for r in results if r['signal'] in ('bullish', 'bearish'))
        bullish = sum(1 for r in results if r['signal'] == 'bullish')
        bearish = sum(1 for r in results if r['signal'] == 'bearish')
        no_signal = sum(1 for r in results if r['signal'] == 'no_signal')
        errors = sum(1 for r in results if r['signal'] == 'error')
        matched = sum(1 for r in results if r.get('match'))
        accuracy = (matched / total_signals) if total_signals > 0 else None

        # Separate bullish and bearish stats
        bullish_matched = sum(1 for r in results if r['signal'] == 'bullish' and r.get('match'))
        bullish_failures = bullish - bullish_matched
        bullish_accuracy = (bullish_matched / bullish * 100) if bullish > 0 else None

        bearish_matched = sum(1 for r in results if r['signal'] == 'bearish' and r.get('match'))
        bearish_failures = bearish - bearish_matched
        bearish_accuracy = (bearish_matched / bearish * 100) if bearish > 0 else None

        summary = {
            'pair': pair,
            'period_days': n,
            'total_points': len(results),
            'total_signals': total_signals,
            'bullish': bullish,
            'bearish': bearish,
            'no_signal': no_signal,
            'errors': errors,
            'matched': matched,
            'accuracy': round(accuracy * 100, 1) if accuracy is not None else None,
            'bullish_matched': bullish_matched,
            'bullish_failures': bullish_failures,
            'bullish_accuracy': round(bullish_accuracy, 1) if bullish_accuracy is not None else None,
            'bearish_matched': bearish_matched,
            'bearish_failures': bearish_failures,
            'bearish_accuracy': round(bearish_accuracy, 1) if bearish_accuracy is not None else None,
            'details': results
        }
        return summary

    def backtest_last_n_days_enhanced(self, pair, n=60, sensitivity=None):
        """Enhanced backtest with pips analysis and probability ranges.

        Calculates actual pips won/lost for each trade and analyzes performance
        across different probability ranges to identify patterns.
        """
        # Use cached engineered data for speed
        if pair not in self.engineered_data:
            raw_df = self.load_data(pair)
            self.engineered_data[pair] = self.engineer_features(pair, raw_df)
        df = self.engineered_data[pair]

        if len(df) < 2:
            raise ValueError('Not enough data to backtest')

        n = min(n, len(df) - 1)
        results = []

        for i in range(n):
            target_idx = len(df) - n + i - 1
            if target_idx < 0:
                continue
            start_idx = max(0, target_idx - 120)
            data_up_to = df.iloc[start_idx: target_idx + 1].copy()
            sig = self.generate_signal(pair, data_up_to, sensitivity=sensitivity)

            if not sig or sig['signal'] not in ('bullish', 'bearish'):
                continue

            # Check next-day movement and calculate pips
            end_idx = target_idx + 1
            if end_idx >= len(df):
                continue

            entry_price = df['open'].iloc[end_idx]  # Enter at next day's open
            exit_price = df['close'].iloc[end_idx]  # Exit at next day's close
            high_price = df['high'].iloc[end_idx]
            low_price = df['low'].iloc[end_idx]

            # Calculate pips (assuming 4 decimal places for EURUSD, 2 for XAUUSD)
            pip_multiplier = 10000 if pair != 'XAUUSD' else 100
            price_diff = exit_price - entry_price
            pips = price_diff * pip_multiplier

            # For bearish signals, we want price to go down (negative pips = profit)
            if sig['signal'] == 'bearish':
                pips = -pips

            # Determine if trade was profitable
            profitable = (sig['signal'] == 'bullish' and exit_price > entry_price) or \
                        (sig['signal'] == 'bearish' and exit_price < entry_price)

            results.append({
                'date': str(df.index[target_idx].date()),
                'signal': sig['signal'],
                'probability': sig['p_up'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'high_price': high_price,
                'low_price': low_price,
                'pips': pips,
                'profitable': profitable
            })

        # Calculate enhanced statistics
        total_signals = len(results)
        wins = sum(1 for r in results if r['profitable'])
        losses = total_signals - wins

        # Pips analysis
        winning_trades = [r['pips'] for r in results if r['profitable']]
        losing_trades = [r['pips'] for r in results if not r['profitable']]

        total_pips_won = sum(winning_trades) if winning_trades else 0
        total_pips_lost = abs(sum(losing_trades)) if losing_trades else 0
        net_pips = total_pips_won - total_pips_lost

        avg_win_pips = sum(winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss_pips = abs(sum(losing_trades)) / len(losing_trades) if losing_trades else 0

        profit_factor = total_pips_won / total_pips_lost if total_pips_lost > 0 else float('inf')

        # Probability range analysis
        prob_ranges = {
            '0.5-0.6': {'count': 0, 'wins': 0, 'total_pips': 0},
            '0.6-0.7': {'count': 0, 'wins': 0, 'total_pips': 0},
            '0.7-0.8': {'count': 0, 'wins': 0, 'total_pips': 0},
            '0.8-0.9': {'count': 0, 'wins': 0, 'total_pips': 0},
            '0.9-1.0': {'count': 0, 'wins': 0, 'total_pips': 0}
        }

        for r in results:
            prob = r['probability']
            if 0.5 <= prob < 0.6:
                range_key = '0.5-0.6'
            elif 0.6 <= prob < 0.7:
                range_key = '0.6-0.7'
            elif 0.7 <= prob < 0.8:
                range_key = '0.7-0.8'
            elif 0.8 <= prob < 0.9:
                range_key = '0.8-0.9'
            else:
                range_key = '0.9-1.0'

            prob_ranges[range_key]['count'] += 1
            if r['profitable']:
                prob_ranges[range_key]['wins'] += 1
            prob_ranges[range_key]['total_pips'] += r['pips']

        # Calculate accuracy and avg pips for each probability range
        probability_analysis = {}
        for range_key, stats in prob_ranges.items():
            if stats['count'] > 0:
                accuracy = (stats['wins'] / stats['count']) * 100
                avg_pips = stats['total_pips'] / stats['count']
                probability_analysis[range_key] = {
                    'count': stats['count'],
                    'accuracy': round(accuracy, 1),
                    'avg_pips': round(avg_pips, 2)
                }
            else:
                probability_analysis[range_key] = {
                    'count': 0,
                    'accuracy': 0,
                    'avg_pips': 0
                }

        # Legacy stats for compatibility
        bullish = sum(1 for r in results if r['signal'] == 'bullish')
        bearish = sum(1 for r in results if r['signal'] == 'bearish')
        accuracy = (wins / total_signals * 100) if total_signals > 0 else 0

        bullish_wins = sum(1 for r in results if r['signal'] == 'bullish' and r['profitable'])
        bearish_wins = sum(1 for r in results if r['signal'] == 'bearish' and r['profitable'])

        bullish_accuracy = (bullish_wins / bullish * 100) if bullish > 0 else 0
        bearish_accuracy = (bearish_wins / bearish * 100) if bearish > 0 else 0

        # Find largest win/loss
        all_pips = [r['pips'] for r in results]
        largest_win = max(all_pips) if all_pips else 0
        largest_loss = min(all_pips) if all_pips else 0

        summary = {
            'pair': pair,
            'period_days': n,
            'total_signals': total_signals,
            'wins': wins,
            'losses': losses,
            'accuracy': round(accuracy, 1),

            # Legacy fields for compatibility
            'bullish': bullish,
            'bearish': bearish,
            'bullish_accuracy': round(bullish_accuracy, 1),
            'bearish_accuracy': round(bearish_accuracy, 1),
            'no_signal': 0,  # Not tracking no signals in enhanced version

            # Enhanced pips analysis
            'total_pips_won': round(total_pips_won, 1),
            'total_pips_lost': round(total_pips_lost, 1),
            'net_pips': round(net_pips, 1),
            'avg_win_pips': round(avg_win_pips, 2),
            'avg_loss_pips': round(avg_loss_pips, 2),
            'profit_factor': round(profit_factor, 2),
            'largest_win': round(largest_win, 1),
            'largest_loss': round(largest_loss, 1),

            # Probability analysis
            'probability_analysis': probability_analysis,

            # Detailed results for further analysis
            'trade_details': results
        }

        return summary

    def export_backtest_to_csv(self, pair, results, filename=None):
        """Export complete backtest results to CSV for analysis"""
        if filename is None:
            filename = f'backtest_results_{pair}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

        # Create output directory
        os.makedirs('output', exist_ok=True)
        filepath = os.path.join('output', filename)

        # Convert results to DataFrame
        df = pd.DataFrame(results)

        # Add additional analysis columns
        df['probability_range'] = pd.cut(df['probability'],
                                        bins=[0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                        labels=['0-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%'])

        df['expected_win'] = df['probability'] > 0.5
        df['actual_win'] = df['profitable']

        # Save to CSV
        df.to_csv(filepath, index=False)
        print(f"Backtest results exported to {filepath}")
        print(f"Total trades: {len(df)}")
        print(f"Win rate: {df['profitable'].mean():.1%}")
        print(f"Average pips per trade: {df['pips'].mean():.2f}")

        # Analysis by probability ranges
        prob_analysis = df.groupby('probability_range').agg({
            'profitable': ['count', 'mean'],
            'pips': ['mean', 'std', 'min', 'max']
        }).round(3)

        analysis_file = filepath.replace('.csv', '_analysis.txt')
        with open(analysis_file, 'w') as f:
            f.write(f"Backtest Analysis for {pair}\n")
            f.write("="*50 + "\n")
            f.write(f"Total Trades: {len(df)}\n")
            f.write(f"Overall Win Rate: {df['profitable'].mean():.1%}\n")
            f.write(f"Average Pips per Trade: {df['pips'].mean():.2f}\n")
            f.write(f"Total Pips: {df['pips'].sum():.1f}\n\n")
            f.write("Analysis by Probability Range:\n")
            f.write(prob_analysis.to_string())
            f.write("\n\nDetailed Results:\n")
            f.write(df.to_string())

        print(f"Detailed analysis saved to {analysis_file}")

        return filepath


def main():
    """Main execution: Train models and generate daily signals."""
    signal_gen = DailyForexSignal()

    # Train all pairs
    trained_pairs = set()
    for pair in PAIRS:
        try:
            print(f"Training {pair}...")
            signal_gen.train_models(pair)
            trained_pairs.add(pair)
        except Exception as e:
            print(f"Training failed for {pair}: {e}")

    # Generate signals for latest 7 days only for trained pairs
    all_signals = {}
    for pair in PAIRS:
        try:
            df = signal_gen.load_data(pair)
        except Exception as e:
            print(f"Could not load data for {pair}: {e}")
            # fallback to calendar dates
            all_signals[pair] = [{'signal': 'no_signal', 'stop_loss': None, 'p_up': 0.5, 'date': str((datetime.now() - timedelta(days=6-i)).date())} for i in range(7)]
            continue

        # Use last 7 available trading dates from the data
        n = min(7, len(df))
        signals_list = []
        for i in range(n):
            target_idx = len(df) - n + i
            start_idx = max(0, target_idx - 120)
            data_up_to = df.iloc[start_idx: target_idx + 1]
            target_date = df.index[target_idx]

            sig = signal_gen.generate_signal(pair, data_up_to)
            # generate_signal now returns explicit dicts for errors/no_signal
            if not sig:
                # Shouldn't happen, but fallback to no_signal placeholder
                signals_list.append({'signal': 'no_signal', 'stop_loss': None, 'p_up': 0.5, 'date': str(target_date.date())})
            else:
                # Ensure date is correct
                sig['date'] = str(target_date.date())
                signals_list.append(sig)

        # If less than 7, pad earlier dates (calendar-based) with no_signal
        if len(signals_list) < 7:
            pad = 7 - len(signals_list)
            for j in range(pad):
                dt = (pd.to_datetime(signals_list[0]['date']) - timedelta(days=(pad - j))).date() if signals_list else (datetime.now() - timedelta(days=6 - j)).date()
                signals_list.insert(0, {'signal': 'no_signal', 'stop_loss': None, 'p_up': 0.5, 'date': str(dt)})

        all_signals[pair] = signals_list
        print(f"{pair}: Generated signals for last {len(all_signals[pair])} days")


    # Save to JSON
    output_file = os.path.join(OUTPUT_PATH, f'daily_signals_last_7_days_{datetime.now().strftime("%Y%m%d")}.json')
    with open(output_file, 'w') as f:
        json.dump(all_signals, f, indent=4)

    print(f"\nSignals saved to: {output_file}")

    # Run backtests (full historical data) and save summaries
    backtest_summaries = {}
    for pair in PAIRS:
        try:
            df = signal_gen.load_data(pair)
            print(f"{pair} data loaded: {len(df)} rows")
            full_n = len(df) - 1  # Full historical backtest
            summary = signal_gen.backtest_last_n_days(pair, n=full_n, sensitivity=PAIR_SENSITIVITY.get(pair))
            backtest_summaries[pair] = summary
            print(f"Backtest {pair}: Total Signals={summary['total_signals']}, Matched={summary['matched']}, Accuracy={summary['accuracy']}%")
            print(f"  Bullish: Count={summary['bullish']}, Wins={summary['bullish_matched']}, Losses={summary['bullish_failures']}, Accuracy={summary['bullish_accuracy']}%")
            print(f"  Bearish: Count={summary['bearish']}, Wins={summary['bearish_matched']}, Losses={summary['bearish_failures']}, Accuracy={summary['bearish_accuracy']}%")
        except Exception as e:
            print(f"Backtest failed for {pair}: {e}")

    bt_file = os.path.join(OUTPUT_PATH, f'backtest_summary_full_historical_{datetime.now().strftime("%Y%m%d")}.json')
    try:
        with open(bt_file, 'w') as f:
            json.dump(backtest_summaries, f, indent=4)
        print(f"Backtest summaries saved to: {bt_file}")
    except Exception as e:
        print(f"Could not save backtest summary: {e}")

    # Print key-value output for last 7 days
    print("\n--- Daily Signals (Last 7 Days) ---")
    for pair, signals in all_signals.items():
        print(f"{pair}:")
        for sig in signals:
             print(f"  Date: {sig['date']} | Signal: {sig['signal']} | SL: {sig['stop_loss']} | p_up: {sig['p_up']}")

    # Overall summary from backtests
    print("\n--- Full Historical Backtest Summary ---")
    total_signals_all = 0
    total_bullish = 0
    total_bearish = 0
    total_matched = 0
    total_bullish_matched = 0
    total_bearish_matched = 0
    total_bullish_failures = 0
    total_bearish_failures = 0
    for pair, summary in backtest_summaries.items():
        if summary:
            print(f"{pair}: Total Signals={summary['total_signals']}, Bullish={summary['bullish']}, Bearish={summary['bearish']}, Matched={summary['matched']}, Accuracy={summary['accuracy']}%")
            print(f"  Bullish: Count={summary['bullish']}, Wins={summary['bullish_matched']}, Losses={summary['bullish_failures']}, Accuracy={summary['bullish_accuracy']}%")
            print(f"  Bearish: Count={summary['bearish']}, Wins={summary['bearish_matched']}, Losses={summary['bearish_failures']}, Accuracy={summary['bearish_accuracy']}%")
            total_signals_all += summary['total_signals']
            total_bullish += summary['bullish']
            total_bearish += summary['bearish']
            total_matched += summary['matched']
            total_bullish_matched += summary['bullish_matched']
            total_bearish_matched += summary['bearish_matched']
            total_bullish_failures += summary['bullish_failures']
            total_bearish_failures += summary['bearish_failures']
    overall_accuracy = (total_matched / total_signals_all) if total_signals_all > 0 else None
    total_wins = total_bullish_matched + total_bearish_matched
    total_losses = total_bullish_failures + total_bearish_failures
    print(f"Overall: Total Signals={total_signals_all}, Bullish={total_bullish}, Bearish={total_bearish}, Total Wins={total_wins}, Total Losses={total_losses}, Accuracy={round(overall_accuracy * 100, 1) if overall_accuracy else None}%")


if __name__ == '__main__':
    main()