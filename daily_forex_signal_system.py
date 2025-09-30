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
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
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
DATA_PATH = '/content/drive/MyDrive/forex_data/'  # Update if different
MODEL_PATH = '/content/drive/MyDrive/forex_models/'
OUTPUT_PATH = '/content/drive/MyDrive/forex_signals/'

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
        
        # Drop rows with NaN in key columns
        df = df.dropna(subset=numeric_cols)
        print(f"After dropna numerics: shape {df.shape}")
        
        # Sort by date
        df = df.sort_values('date').set_index('date')
        
        # Save cleaned data back to the same CSV file
        file_path = os.path.join(DATA_PATH, f'{pair}_Daily.csv')
        df.reset_index().to_csv(file_path, index=False)
        print(f"Cleaned and saved standardized data for {pair}: {len(df)} rows.")
        
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
        """Robust daily features: price action, technicals, PnF."""
        # Basic price features
        df['ret1'] = df['close'].pct_change()
        df['gapopen'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)

        # Moving averages
        df['sma5'] = df['close'].rolling(5).mean()
        df['sma20'] = df['close'].rolling(20).mean()
        df['ema12'] = df['close'].ewm(span=12).mean()
        df['ema26'] = df['close'].ewm(span=26).mean()

        # Oscillators
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi14'] = 100 - (100 / (1 + rs))

        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr14'] = tr.rolling(14).mean()

        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

        # MACD
        df['macd'] = df['ema12'] - df['ema26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()

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

        # Target: next day return direction
        df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)

        # Drop NaN
        initial_rows = len(df)
        df = df.dropna()
        rows_dropped = initial_rows - len(df)
        print(f"Engineered features. Dropped {rows_dropped} rows due to NaN.") # Debug print
        print(f"Shape after feature engineering and dropna: {df.shape}") # Debug print


        # Feature columns
        self.features = [
            'ret1', 'gapopen', 'sma5', 'sma20', 'ema12', 'ema26', 'rsi14', 'atr14',
            'bb_width', 'macd', 'macd_signal', 'sma5_20_diff', 'ema12_26_diff',
            'hl_norm', 'insidebar', 'bullengulf', 'bearengulf', 'pnfdir', 'breakup', 'breakdn'
        ]
        print(f"Features to be used: {self.features}") # Debug print


        return df

    def build_ensemble(self, X_train, y_train):
        """Build calibrated ensemble: RF + XGB + Logistic."""
        models = {}

        # Random Forest
        rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        rf.fit(X_train, y_train)
        models['rf'] = CalibratedClassifierCV(rf, method='isotonic', cv=3)

        # XGBoost
        xgb_clf = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
        xgb_clf.fit(X_train, y_train)
        models['xgb'] = CalibratedClassifierCV(xgb_clf, method='isotonic', cv=3)

        # Fit calibrators
        for name, model in models.items():
            model.fit(X_train, y_train)
            self.models[name] = model

        # Logistic calibration head
        ensemble_preds = np.column_stack([model.predict_proba(X_train)[:, 1] for model in self.models.values()])
        self.calibrators['logistic'] = CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=50, random_state=42),
            method='isotonic', cv=3
        )
        self.calibrators['logistic'].fit(ensemble_preds, y_train)

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
            self.build_ensemble(X_train, y_train)

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
            calibrator_path = os.path.join(MODEL_PATH, f'{pair}_calibrator.joblib')
            joblib.dump(self.calibrators['logistic'], calibrator_path)
            saved_files.append(calibrator_path)

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
        if self.calibrators['logistic'] is not None:
            calibrated = self.calibrators['logistic'].predict_proba(preds)[:, 1]
        else:
            calibrated = preds.mean(axis=1)  # simple average if no calibrator
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

            calibrator_path = os.path.join(MODEL_PATH, f'{pair}_calibrator.joblib')
            if os.path.exists(calibrator_path):
                self.calibrators['logistic'] = joblib.load(calibrator_path)
            else:
                 print(f"Warning: Calibrator {calibrator_path} not found. Using default calibrator.")
                 # fallback to a simple calibrator or skip calibration
                 self.calibrators['logistic'] = None

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