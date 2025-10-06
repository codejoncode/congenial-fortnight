#!/usr/bin/env python3
"""
HybridPriceForecastingEnsemble - Advanced ensemble forecasting for forex trading

This module implements a sophisticated hybrid ensemble forecasting system that combines:
- Classical statistical models (Prophet, StatsForecast)
- Machine learning models (LightGBM, Random Forest)
- Deep learning models (LSTM, RNN)
- Meta-model stacking with Ridge regression

Features:
- Multi-timeframe feature engineering (H4, Daily, Weekly)
- Cross-pair correlation analysis
- Fundamental data integration
- Automated model selection and weighting
- Confidence scoring and uncertainty estimation
- Backtesting with realistic trading costs

Usage:
    # Train ensemble for EURUSD
    ensemble = HybridPriceForecastingEnsemble('EURUSD')
    ensemble.train_full_ensemble()

    # Generate forecast
    forecast = ensemble.generate_forecast(days_ahead=5)

    # Get trading signal with confidence
    signal = ensemble.get_trading_signal()
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

try:
    from .fundamental_pipeline import FundamentalDataPipeline as FundamentalFeatureEngineer
except ImportError:  # pragma: no cover - support standalone execution
    from fundamental_pipeline import FundamentalDataPipeline as FundamentalFeatureEngineer

try:
    from .holloway_algorithm import CompleteHollowayAlgorithm
except ImportError:  # pragma: no cover - support standalone execution
    from holloway_algorithm import CompleteHollowayAlgorithm

try:
    from .day_trading_signals import DayTradingSignalGenerator
except ImportError:  # pragma: no cover - support standalone execution
    from day_trading_signals import DayTradingSignalGenerator

try:
    from .slump_signals import SlumpSignalEngine
except ImportError:  # pragma: no cover - support standalone execution
    from slump_signals import SlumpSignalEngine

try:
    from .harmonic_patterns import detect_harmonic_patterns
except ImportError:  # pragma: no cover - support standalone execution
    from harmonic_patterns import detect_harmonic_patterns

try:
    from .chart_patterns import detect_chart_patterns
except ImportError:  # pragma: no cover - support standalone execution
    from chart_patterns import detect_chart_patterns

try:
    from .elliott_wave import detect_elliott_waves
except ImportError:  # pragma: no cover - support standalone execution
    from elliott_wave import detect_elliott_waves

try:
    from .ultimate_signal_repository import UltimateSignalRepository, integrate_ultimate_signals
except ImportError:  # pragma: no cover - support standalone execution
    from ultimate_signal_repository import UltimateSignalRepository, integrate_ultimate_signals

# ML and statistical libraries
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from lightgbm import LGBMClassifier
import lightgbm as lgb
from xgboost import XGBClassifier

# Time series and forecasting
try:
    from prophet import Prophet
except ImportError:
    Prophet = None

try:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, ETS, Theta
except ImportError:
    StatsForecast = None

# Deep learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
except ImportError:
    tf = None

        # Technical analysis
try:
    import ta
    from ta.pattern import *
except ImportError:
    ta = None

try:
    import pywt
except ImportError:
    pywt = None

# API clients
try:
    from fredapi import Fred
except ImportError:
    Fred = None

try:
    from alpha_vantage.foreignexchange import ForeignExchange
except ImportError:
    ForeignExchange = None

try:
    import finnhub
except ImportError:
    finnhub = None
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _normalize_price_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize common price CSV variants to canonical columns and a DatetimeIndex.
    Handles:
      - MetaTrader-style headers: '<DATE>' and '<TIME>'
      - Single combined 'timestamp' column
      - lowercase headers like 'open', 'high', etc.
    Returns a DataFrame indexed by datetime with columns Open/High/Low/Close/Volume when possible.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # Trim whitespace of column names and map to lowercase for matching
    orig_cols = list(df.columns)
    col_map_low = {c: c.strip() for c in orig_cols}
    df = df.rename(columns=col_map_low)

    lower_cols = {c.lower(): c for c in df.columns}

    # Build datetime index
    if '<date>' in lower_cols and '<time>' in lower_cols:
        date_col = lower_cols['<date>']
        time_col = lower_cols['<time>']
        combined = df[date_col].astype(str).str.strip() + ' ' + df[time_col].astype(str).str.strip()
        df['_datetime'] = pd.to_datetime(combined, errors='coerce')
        df = df.dropna(subset=['_datetime']).set_index('_datetime')
    elif 'timestamp' in lower_cols:
        ts_col = lower_cols['timestamp']
        df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
        df = df.dropna(subset=[ts_col]).set_index(ts_col)
    elif 'date' in lower_cols:
        dcol = lower_cols['date']
        df[dcol] = pd.to_datetime(df[dcol], errors='coerce')
        df = df.dropna(subset=[dcol]).set_index(dcol)
    else:
        # Try to find any parsable datetime column
        parsed = False
        for c in df.columns:
            try:
                ser = pd.to_datetime(df[c], errors='coerce')
                if ser.notna().sum() > 0:
                    df[c] = ser
                    df = df.dropna(subset=[c]).set_index(c)
                    parsed = True
                    break
            except Exception:
                continue
        if not parsed:
            return pd.DataFrame()

    # Normalize OHLCV names
    rename_map = {}
    for c in df.columns:
        key = c.strip().lower()
        if key in ('<open>', 'open'):
            rename_map[c] = 'Open'
        if key in ('<high>', 'high'):
            rename_map[c] = 'High'
        if key in ('<low>', 'low'):
            rename_map[c] = 'Low'
        if key in ('<close>', 'close'):
            rename_map[c] = 'Close'
        if key in ('<tickvol>', 'tickvol'):
            rename_map[c] = 'TickVolume'
        if key in ('<vol>', 'vol', 'volume'):
            rename_map[c] = 'Volume'
        if key in ('<spread>', 'spread'):
            rename_map[c] = 'Spread'

    if rename_map:
        df = df.rename(columns=rename_map)

    # Coerce numeric types for numeric-looking columns
    for c in df.columns:
        if c not in ([],):
            try:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            except Exception:
                pass

    return df

try:
    from .robust_lightgbm_config import enhanced_lightgbm_training_pipeline
except ImportError:
    from robust_lightgbm_config import enhanced_lightgbm_training_pipeline

class HybridPriceForecastingEnsemble:
    def evaluate_signal_features(self, feature_df: pd.DataFrame, target_col: str = 'target_1d', output_csv: str = None):
        """
        Evaluate each signal/feature for predictive power against the target.
        Outputs a summary table (printed and optionally saved as CSV) with:
        - Feature name
        - Accuracy (thresholded at 0.5 if numeric, or direct if binary)
        - Hit rate (fraction of nonzero signals that are correct)
        - Correlation with target
        """
        import pandas as pd
        import numpy as np
        results = []
        if feature_df.empty or target_col not in feature_df.columns:
            print("No features or target column for signal evaluation.")
            return
        y = feature_df[target_col]
        for col in feature_df.columns:
            if col == target_col or col.startswith('next_close_change') or col in ['Open','High','Low','Close','Volume','returns','log_returns']:
                continue
            x = feature_df[col]
            # Only evaluate numeric or boolean features
            if not np.issubdtype(x.dtype, np.number):
                continue
            # Accuracy: threshold at 0.5 if not binary
            if set(x.dropna().unique()).issubset({0,1}):
                pred = x
            else:
                pred = (x > 0.5).astype(int)
            acc = (pred == y).mean()
            # Hit rate: fraction of nonzero signals that are correct
            nonzero = pred != 0
            hit_rate = ((pred[nonzero] == y[nonzero]).mean() if nonzero.sum() > 0 else np.nan)
            # Correlation
            try:
                corr = np.corrcoef(x.fillna(0), y.fillna(0))[0,1]
            except Exception:
                corr = np.nan
            results.append({
                'feature': col,
                'accuracy': acc,
                'hit_rate': hit_rate,
                'correlation': corr
            })
        df_results = pd.DataFrame(results).sort_values('accuracy', ascending=False)
        print("\n=== Per-Signal/Feature Evaluation ===")
        print(df_results.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        if output_csv:
            df_results.to_csv(output_csv, index=False)
            print(f"Signal evaluation results saved to {output_csv}")
    """
    Advanced hybrid ensemble forecasting system for forex price prediction.

    Combines multiple forecasting approaches:
    - Statistical: Prophet, ARIMA, ETS, Theta
    - ML: LightGBM, XGBoost, Random Forest
    - DL: LSTM, BiLSTM
    - Meta-learning: Ridge regression for model stacking
    """

    def __init__(self, pair: str, data_dir: str = "data", models_dir: str = "models"):
        """
        Initialize the hybrid forecasting ensemble.

        Args:
            pair: Currency pair (e.g., 'EURUSD', 'XAUUSD')
            data_dir: Directory containing price data
            models_dir: Directory to save trained models
        """
        self.pair = pair
        # Ensure data_dir is an absolute path relative to the project root
        self.base_dir = Path(__file__).resolve().parent.parent
        self.data_dir = self.base_dir / data_dir
        self.models_dir = self.base_dir / models_dir
        self.models_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)

        # Model storage
        self.models = {}
        self.scalers = {}
        self.meta_model = None
        self.feature_importance = {}

        # Configuration - default to daily forecasting targeting next-day direction
        if self.pair in ['EURUSD', 'XAUUSD']:
            # These flagship pairs leverage intraday data but still predict next day moves
            self.forecast_horizon = 1  # Next trading day
            self.lookback_window = 250  # ~1 trading year of daily observations
            self.timeframe = 'Daily'
        else:
            # Other pairs fall back to daily setup as well
            self.forecast_horizon = 1
            self.lookback_window = 200
            self.timeframe = 'Daily'

        self.validation_splits = 5

        # Model configurations
        self.model_configs = self._get_model_configs()

        # Load data across timeframes
        self.intraday_data = self._load_intraday_data()
        self.monthly_data = self._load_monthly_data()




        # Load daily data as the consolidated price data (true daily OHLC)
        # This will be set after price_data is loaded below
        self.daily_data = None

        # Load weekly data from the correct file (e.g., EURUSD_Weekly.csv)
        weekly_path = self.data_dir / f'{self.pair}_Weekly.csv'
        if weekly_path.exists():
            try:
                weekly_df = pd.read_csv(weekly_path)
                # Try to parse a datetime index from timestamp/date columns
                if 'timestamp' in weekly_df.columns:
                    weekly_df['datetime'] = pd.to_datetime(weekly_df['timestamp'], errors='coerce')
                    weekly_df = weekly_df.set_index('datetime')
                elif 'date' in weekly_df.columns:
                    weekly_df['datetime'] = pd.to_datetime(weekly_df['date'], errors='coerce')
                    weekly_df = weekly_df.set_index('datetime')
                else:
                    # fallback: try to parse first column as date
                    try:
                        weekly_df.index = pd.to_datetime(weekly_df.iloc[:, 0], errors='coerce')
                    except Exception:
                        pass
                # Normalize OHLCV columns
                rename_map = {}
                for col in weekly_df.columns:
                    key = col.strip().lower()
                    if key in ('open',):
                        rename_map[col] = 'Open'
                    if key in ('high',):
                        rename_map[col] = 'High'
                    if key in ('low',):
                        rename_map[col] = 'Low'
                    if key in ('close',):
                        rename_map[col] = 'Close'
                    if key in ('volume',):
                        rename_map[col] = 'Volume'
                if rename_map:
                    weekly_df = weekly_df.rename(columns=rename_map)
                # Only keep standard columns if present
                keep_cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in weekly_df.columns]
                self.weekly_data = weekly_df[keep_cols]
            except Exception as e:
                self.logger.error(f"Error loading weekly data for {self.pair}: {e}")
                self.weekly_data = pd.DataFrame()
        else:
            self.logger.warning(f"Weekly data file not found: {weekly_path}")
            self.weekly_data = pd.DataFrame()

        # _load_price_data may call methods that aren't bound (left in module scope).
        # Attempt to use it, otherwise construct a sensible daily price DataFrame

        try:
            self.price_data = self._load_price_data()
        except Exception:
            # Fallback: prefer intraday-derived daily OHLC, otherwise resample monthly
            try:
                daily = pd.DataFrame()
                if hasattr(self, 'intraday_data') and not self.intraday_data.empty:
                    daily_ohlc, intraday_features = self._build_intraday_context(self.intraday_data)
                    daily = daily_ohlc
                elif hasattr(self, 'monthly_data') and not self.monthly_data.empty:
                    monthly_features = self.monthly_data.rename(columns=lambda c: f"monthly_{c}")
                    monthly_features = monthly_features.resample('1D').ffill()
                    daily = monthly_features
                else:
                    daily = pd.DataFrame()

                self.price_data = daily
            except Exception:
                self.price_data = pd.DataFrame()

        # Set daily_data to the true daily/consolidated price data
        self.daily_data = self.price_data

        self.fundamental_engineer = FundamentalFeatureEngineer(self.data_dir)
        self.fundamental_data = self._load_fundamental_data()
        # _get_cross_pair may not exist in some refactored versions; be resilient
        try:
            self.cross_pair = self._get_cross_pair()
        except Exception:
            # Fallback mapping for core pairs
            cross_map = {
                'EURUSD': 'XAUUSD',
                'XAUUSD': 'EURUSD'
            }
            self.cross_pair = cross_map.get(self.pair)

        # Feature engineering
        self.feature_columns = []
        self._holloway_algo = CompleteHollowayAlgorithm(str(self.data_dir))
        self._day_trading_signals = DayTradingSignalGenerator()
        self._slump_signals = SlumpSignalEngine()

    def _get_model_configs(self) -> Dict:
        """Get configuration for all base models."""
        return {
            # Statistical Models
            'prophet': {
                'enabled': Prophet is not None,
                'type': 'statistical',
                'params': {
                    'seasonality_mode': 'multiplicative',
                    'yearly_seasonality': True,
                    'weekly_seasonality': True,
                    'daily_seasonality': False,
                    'changepoint_prior_scale': 0.05
                }
            },
            'auto_arima': {
                'enabled': StatsForecast is not None,
                'type': 'statistical',
                'params': {'season_length': 5}  # Trading days
            },
            'ets': {
                'enabled': StatsForecast is not None,
                'type': 'statistical',
                'params': {'season_length': 5}
            },
            'theta': {
                'enabled': StatsForecast is not None,
                'type': 'statistical',
                'params': {}
            },

            # Machine Learning Models with Enhanced Regularization
            'lightgbm': {
                'enabled': True,
                'type': 'ml',
                'params': {
                    'n_estimators': 2000,
                    'learning_rate': 0.03,  # Lower for better generalization
                    'max_depth': 6,
                    'num_leaves': 31,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'min_child_samples': 20,  # Regularization
                    'min_child_weight': 0.001,
                    'reg_alpha': 0.1,  # L1 regularization
                    'reg_lambda': 0.1,  # L2 regularization
                    'random_state': 42,
                    'verbosity': -1,
                    'force_col_wise': True,
                    'boosting_type': 'gbdt'
                },
                'early_stopping': {
                    'enabled': True,
                    'rounds': 100,
                    'metric': 'binary_logloss'
                }
            },
            'xgboost': {
                'enabled': True,
                'type': 'ml',
                'params': {
                    'n_estimators': 2000,
                    'learning_rate': 0.03,  # Lower for stability
                    'max_depth': 6,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'min_child_weight': 3,  # Regularization
                    'gamma': 0.2,  # Minimum loss reduction
                    'alpha': 0.1,  # L1 regularization
                    'lambda': 1.0,  # L2 regularization
                    'scale_pos_weight': 1,
                    'random_state': 42,
                    'verbosity': 0,
                    'tree_method': 'auto'
                },
                'early_stopping': {
                    'enabled': True,
                    'rounds': 100,
                    'metric': 'logloss'
                }
            },
            'random_forest': {
                'enabled': True,
                'type': 'ml',
                'params': {
                    'n_estimators': 800,  # Increased for better ensemble
                    'max_depth': 12,
                    'min_samples_split': 10,  # Increased regularization
                    'min_samples_leaf': 4,   # Increased regularization
                    'max_features': 'sqrt',  # Feature regularization
                    'min_impurity_decrease': 0.0001,  # Regularization
                    'max_samples': 0.8,      # Bootstrap regularization
                    'random_state': 42,
                    'n_jobs': -1,
                    'oob_score': True        # Out-of-bag scoring
                },
                'validation': {
                    'use_oob': True,
                    'target_score': 0.85
                }
            },
            'extra_trees': {
                'enabled': True,
                'type': 'ml',
                'params': {
                    'n_estimators': 800,
                    'max_depth': 12,
                    'min_samples_split': 10,  # Regularization
                    'min_samples_leaf': 4,   # Regularization
                    'max_features': 'sqrt',  # Feature regularization
                    'min_impurity_decrease': 0.0001,
                    'max_samples': 0.8,      # Bootstrap regularization
                    'random_state': 42,
                    'n_jobs': -1,
                    'bootstrap': True        # Enable bootstrapping
                },
                'validation': {
                    'target_score': 0.85
                }
            },

            # Deep Learning Models with Advanced Regularization
            'lstm': {
                'enabled': tf is not None,
                'type': 'dl',
                'params': {
                    'units': 128,           # Increased capacity
                    'dropout': 0.3,         # Increased dropout
                    'recurrent_dropout': 0.2, # RNN-specific dropout
                    'epochs': 200,          # More epochs with early stopping
                    'batch_size': 64,       # Larger batch for stability
                    'learning_rate': 0.001,
                    'l1_reg': 0.01,        # L1 regularization
                    'l2_reg': 0.01,        # L2 regularization
                    'loss': 'binary_crossentropy',  # Binary classification
                    'activation': 'sigmoid',  # Sigmoid for binary
                    'optimizer': 'adam',
                    'metrics': ['accuracy', 'binary_accuracy']
                },
                'early_stopping': {
                    'enabled': True,
                    'monitor': 'val_loss',
                    'patience': 25,        # Increased patience
                    'min_delta': 0.0001,
                    'restore_best_weights': True
                },
                'callbacks': {
                    'reduce_lr': {
                        'enabled': True,
                        'monitor': 'val_loss',
                        'factor': 0.5,
                        'patience': 15,
                        'min_lr': 1e-7
                    }
                }
            },
            'bilstm': {
                'enabled': tf is not None,
                'type': 'dl',
                'params': {
                    'units': 128,
                    'dropout': 0.3,
                    'recurrent_dropout': 0.2,
                    'epochs': 200,
                    'batch_size': 64,
                    'learning_rate': 0.001,
                    'l1_reg': 0.01,
                    'l2_reg': 0.01,
                    'loss': 'binary_crossentropy',  # Binary classification
                    'activation': 'sigmoid',  # Sigmoid for binary
                    'optimizer': 'adam',
                    'metrics': ['accuracy', 'binary_accuracy']
                },
                'early_stopping': {
                    'enabled': True,
                    'monitor': 'val_loss',
                    'patience': 25,
                    'min_delta': 0.0001,
                    'restore_best_weights': True
                },
                'callbacks': {
                    'reduce_lr': {
                        'enabled': True,
                        'monitor': 'val_loss',
                        'factor': 0.5,
                        'patience': 15,
                        'min_lr': 1e-7
                    }
                }
            }
        }

    def _load_fundamental_data(self):
        """Loads and merges fundamental economic data."""
        try:
            # The engineer is already initialized in __init__
            fundamental_data = self.fundamental_engineer.load_all_series_as_df()
            if fundamental_data.empty:
                self.logger.critical("FATAL: Fundamental dataset is empty after processing. Halting pipeline.")
                raise RuntimeError("FATAL: Fundamental dataset is empty after processing. Pipeline halted.")
            self.logger.info(f"Successfully loaded fundamental data. Shape: {fundamental_data.shape}")
            return fundamental_data
        except Exception as e:
            self.logger.critical(f"FATAL: Could not load fundamental data: {e}. Halting pipeline.")
            raise RuntimeError(f"FATAL: Could not load fundamental data: {e}. Pipeline halted.")

    def _load_price_data(self):
        """
        Consolidates price data from different timeframes.
        """
        try:
            daily_from_file = self._load_daily_price_file()

            intraday_daily = pd.DataFrame()
            intraday_features = pd.DataFrame()
            if not self.intraday_data.empty:
                intraday_daily, intraday_features = self._build_intraday_context(self.intraday_data)

            if daily_from_file.empty and intraday_daily.empty:
                daily = self._load_fallback_ohlc()
            elif daily_from_file.empty:
                daily = intraday_daily.copy()
            elif intraday_daily.empty:
                daily = daily_from_file.copy()
            else:
                daily = self._combine_daily_sources(daily_from_file, intraday_daily)

            if daily.empty:
                logger.warning(f"No price data available for {self.pair}")
                return daily

            if not intraday_features.empty:
                daily = daily.join(intraday_features, how='left')

            if not self.monthly_data.empty:
                monthly_features = self.monthly_data.rename(columns=lambda c: f"monthly_{c}")
                monthly_features = monthly_features.resample('1D').ffill()
                daily = daily.join(monthly_features, how='left')

            daily = daily.sort_index()
            daily = daily[~daily.index.duplicated(keep='last')]

            logger.info(
                "%s consolidated dataset summary -> observations: %d (daily), start: %s, end: %s",
                self.pair,
                len(daily),
                daily.index.min().date(),
                daily.index.max().date()
            )

            return daily

        except Exception as e:
            logger.error(f"Error loading price data: {e}")
            return pd.DataFrame()

    def _load_intraday_data(self, pair: Optional[str] = None):
        """Loads H4 data for the specified pair."""
        pair = pair or self.pair
        intraday_path = self.data_dir / f'{pair}_H4.csv'
        self.logger.info(f"Loading H4 data from: {intraday_path}")
        try:
            # Read permissively and then attempt to detect date/time columns
            # Let pandas infer delimiter (commonly comma) so CSVs with commas parse correctly
            data = pd.read_csv(intraday_path, engine='python', header=0)
            # Detect and construct datetime index
            if '<DATE>' in data.columns and '<TIME>' in data.columns:
                data['datetime'] = pd.to_datetime(data['<DATE>'].astype(str) + ' ' + data['<TIME>'].astype(str), errors='coerce')
                data = data.set_index('datetime')
            elif 'Date' in data.columns and 'Time' in data.columns:
                data['datetime'] = pd.to_datetime(data['Date'].astype(str) + ' ' + data['Time'].astype(str), errors='coerce')
                data = data.set_index('datetime')
            elif 'Date' in data.columns:
                data['datetime'] = pd.to_datetime(data['Date'], errors='coerce')
                data = data.set_index('datetime')
            elif 'date' in data.columns:
                data['datetime'] = pd.to_datetime(data['date'], errors='coerce')
                data = data.set_index('datetime')
            else:
                # fallback: try to parse first column as date
                try:
                    data.index = pd.to_datetime(data.iloc[:, 0], errors='coerce')
                except Exception:
                    # Leave as-is; other logic will handle missing index
                    pass
            # Normalize common column names (both uppercase <OPEN> and lowercase 'open')
            rename_map = {}
            if '<OPEN>' in data.columns:
                rename_map['<OPEN>'] = 'Open'
            if '<HIGH>' in data.columns:
                rename_map['<HIGH>'] = 'High'
            if '<LOW>' in data.columns:
                rename_map['<LOW>'] = 'Low'
            if '<CLOSE>' in data.columns:
                rename_map['<CLOSE>'] = 'Close'
            if '<TICKVOL>' in data.columns:
                rename_map['<TICKVOL>'] = 'Volume'

            # lowercase variants
            if 'open' in data.columns and 'Open' not in data.columns:
                rename_map['open'] = 'Open'
            if 'high' in data.columns and 'High' not in data.columns:
                rename_map['high'] = 'High'
            if 'low' in data.columns and 'Low' not in data.columns:
                rename_map['low'] = 'Low'
            if 'close' in data.columns and 'Close' not in data.columns:
                rename_map['close'] = 'Close'
            if 'volume' in data.columns and 'Volume' not in data.columns:
                rename_map['volume'] = 'Volume'

            if rename_map:
                data = data.rename(columns=rename_map)

            # Ensure we have the expected columns; if not, log and return empty
            expected_cols = ['Open', 'High', 'Low', 'Close']
            if not all(col in data.columns for col in expected_cols):
                self.logger.error(f"Error loading H4 data for {pair}: missing OHLC columns after normalization")
                return pd.DataFrame()

            # Volume is optional
            cols = ['Open', 'High', 'Low', 'Close'] + (['Volume'] if 'Volume' in data.columns else [])
            self.intraday_data = data[cols]
            self.logger.info(f"Successfully loaded H4 data for {pair}. Shape: {self.intraday_data.shape}")
            return self.intraday_data
        except FileNotFoundError:
            self.logger.error(f"H4 data file not found at {intraday_path}")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error loading H4 data for {pair}: {e}")
            return pd.DataFrame()

    def _load_monthly_data(self):
        """Loads monthly data for the specified pair."""
        monthly_path = self.data_dir / f'{self.pair}_Monthly.csv'
        self.logger.info(f"Loading Monthly data from: {monthly_path}")
        try:
            # Let pandas infer delimiter for monthly CSVs as well
            data = pd.read_csv(monthly_path, engine='python', header=0)
            # Normalize date column
            if '<DATE>' in data.columns:
                data['Date'] = pd.to_datetime(data['<DATE>'], errors='coerce')
                data = data.set_index('Date')
            elif 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
                data = data.set_index('Date')
            elif 'date' in data.columns:
                data['Date'] = pd.to_datetime(data['date'], errors='coerce')
                data = data.set_index('Date')
            else:
                try:
                    data.index = pd.to_datetime(data.iloc[:, 0], errors='coerce')
                except Exception:
                    pass
            # Normalize monthly column names similar to intraday loader
            rename_map = {}
            if '<OPEN>' in data.columns:
                rename_map['<OPEN>'] = 'Open'
            if '<HIGH>' in data.columns:
                rename_map['<HIGH>'] = 'High'
            if '<LOW>' in data.columns:
                rename_map['<LOW>'] = 'Low'
            if '<CLOSE>' in data.columns:
                rename_map['<CLOSE>'] = 'Close'
            if '<TICKVOL>' in data.columns:
                rename_map['<TICKVOL>'] = 'Volume'
            if 'open' in data.columns and 'Open' not in data.columns:
                rename_map['open'] = 'Open'
            if 'high' in data.columns and 'High' not in data.columns:
                rename_map['high'] = 'High'
            if 'low' in data.columns and 'Low' not in data.columns:
                rename_map['low'] = 'Low'
            if 'close' in data.columns and 'Close' not in data.columns:
                rename_map['close'] = 'Close'
            if 'volume' in data.columns and 'Volume' not in data.columns:
                rename_map['volume'] = 'Volume'

            if rename_map:
                data = data.rename(columns=rename_map)

            expected_cols = ['Open', 'High', 'Low', 'Close']
            if not all(col in data.columns for col in expected_cols):
                self.logger.error(f"Error loading Monthly data for {self.pair}: missing OHLC columns after normalization")
                return pd.DataFrame()

            cols = ['Open', 'High', 'Low', 'Close'] + (['Volume'] if 'Volume' in data.columns else [])
            self.monthly_data = data[cols]
            self.logger.info(f"Successfully loaded Monthly data for {self.pair}. Shape: {self.monthly_data.shape}")
            return self.monthly_data
        except FileNotFoundError:
            self.logger.error(f"Monthly data file not found at {monthly_path}")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error loading Monthly data for {self.pair}: {e}")
            return pd.DataFrame()

    def _load_fundamental_data(self):
        """Loads and merges fundamental economic data."""
        try:
            # The engineer is already initialized in __init__
            fundamental_data = self.fundamental_engineer.load_all_series_as_df()
            if fundamental_data.empty:
                self.logger.warning("Fundamental dataset is empty after processing.")
                return pd.DataFrame()
            self.logger.info(f"Successfully loaded fundamental data. Shape: {fundamental_data.shape}")
            return fundamental_data
        except Exception as e:
            self.logger.error(f"Could not load fundamental data: {e}")
            return pd.DataFrame()

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer comprehensive technical and fundamental features.

        Args:
            df: DataFrame with OHLC data

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
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = df['Close'].rolling(period).mean()
            df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()

        # Volatility measures (remove ATR as it's candle size dependent)
        df['volatility_20'] = df['returns'].rolling(20).std()
        df['volatility_50'] = df['returns'].rolling(50).std()
        # df['atr_14'] = self._calculate_atr(df, 14)  # Removed: candle size dependent

        # Momentum indicators
        df['rsi_14'] = self._calculate_rsi(df['Close'], 14)
        df['macd'], df['macd_signal'], df['macd_hist'] = self._calculate_macd(df['Close'])

        # Add Holloway Algorithm features
        df = self._calculate_holloway_features(df)

        # Add day trading signals
        df = self._add_day_trading_signals(df)

        # Add slump signals
        df = self._add_slump_signals(df)

        # Add candlestick pattern features
        df = self._add_candlestick_patterns(df)

        # Add harmonic pattern features
        df = self._add_harmonic_patterns(df)

        # Add chart pattern features
        df = self._add_chart_patterns(df)

        # Add Elliott Wave signals
        df = self._add_elliott_wave_signals(df)

        # Add Ultimate Signal Repository (SMC, Order Flow, Session-based, etc.)
        df = self._add_ultimate_signals(df)

        # Trend indicators (remove ADX as it uses TR which is candle size dependent)
        # df['adx_14'] = self._calculate_adx(df, 14)  # Removed: depends on candle ranges

        # Support/Resistance levels (remove pivot points as they use OHLC ranges)
        # df['pivot_point'] = (df['High'] + df['Low'] + df['Close']) / 3  # Removed: candle size dependent
        # df['r1'] = 2 * df['pivot_point'] - df['Low']  # Removed: candle size dependent
        # df['s1'] = 2 * df['pivot_point'] - df['High']  # Removed: candle size dependent

        # Volume-based indicators (if volume exists)
        if 'Volume' in df.columns:
            df['volume_sma_20'] = df['Volume'].rolling(20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_sma_20']

        # Time-based features
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['week_of_year'] = df.index.isocalendar().week.astype(int)
        df['month_of_year'] = df.index.month

        # Interaction features
        df['high_x_day'] = df['High'] * (df['day_of_week'] + 1)
        df['low_x_day'] = df['Low'] * (df['day_of_week'] + 1)
        df['close_x_month'] = df['Close'] * df['month_of_year']

        # Rolling statistics
        for window in [5, 10, 20, 50]:
            df[f'returns_roll_mean_{window}'] = df['returns'].rolling(window).mean()
            df[f'returns_roll_std_{window}'] = df['returns'].rolling(window).std()
            df[f'volatility_roll_mean_{window}'] = df['volatility_20'].rolling(window).mean()

        # Fourier features
        close_fft = np.fft.fft(df['Close'].fillna(0))
        for i in [1, 2, 3, 5, 10]:
            df[f'fft_real_{i}'] = close_fft.real[i]
            df[f'fft_imag_{i}'] = close_fft.imag[i]

        # Wavelet features
        if pywt:
            for level in range(3):
                try:
                    coeffs = pywt.wavedec(df['Close'].fillna(0), 'db1', level=level+1)
                    for i, c in enumerate(coeffs):
                        df[f'wavelet_level{level}_{i}'] = c[0] if len(c) > 0 else 0
                except Exception as e:
                    self.logger.warning(f"Could not calculate wavelet features for level {level}: {e}")

        # Statistical features
        df['skewness_20'] = df['returns'].rolling(20).skew()
        df['kurtosis_20'] = df['returns'].rolling(20).kurt()

        # Lagged features
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)

        # Target variables (binary classification: 1 for bull, 0 for bear)
        for horizon in [1, 3, 5]:
            df[f'target_{horizon}d'] = (df['Close'].shift(-horizon) > df['Open'].shift(-horizon)).astype(int)

        # Multi-timeframe technical indicators
        df = self._add_multi_timeframe_indicators(df)

        # Introduce cross-pair correlation features before cleaning
        df = self._add_cross_pair_features(df)

        # Add multi-timeframe Holloway features
        df = self._add_multi_timeframe_holloway_features(df)

        # ------------------------------------------------------------------
        # Attach fundamental features (macro & company-level) if available
        # ------------------------------------------------------------------
        try:
            if hasattr(self, 'fundamental_data') and self.fundamental_data is not None and not self.fundamental_data.empty:
                fund = self.fundamental_data.copy()

                # Ensure datetime index
                if not isinstance(fund.index, pd.DatetimeIndex):
                    try:
                        fund.index = pd.to_datetime(fund.index)
                    except Exception:
                        # some fundamental loaders use a 'date' column
                        if 'date' in fund.columns:
                            fund.index = pd.to_datetime(fund['date'])
                            fund = fund.drop(columns=['date'], errors='ignore')

                # Resample to daily and forward-fill to align with price dates
                try:
                    fund_daily = fund.resample('D').ffill()
                except Exception:
                    fund_daily = fund.copy()

                # Reindex to price dataframe index and forward-fill
                aligned = fund_daily.reindex(df.index).fillna(method='ffill').fillna(method='bfill')

                # Prefix fundamental columns to avoid name collisions
                prefixed = {c: f'fund_{c}' for c in aligned.columns}
                aligned = aligned.rename(columns=prefixed)

                # Convert numeric columns where possible
                for col in aligned.columns:
                    aligned[col] = pd.to_numeric(aligned[col], errors='coerce')

                # Join into main feature df
                df = df.join(aligned, how='left')

                logger.info(f"Attached {len(aligned.columns)} fundamental features (prefixed) to {self.pair}")
        except Exception as e:
            logger.warning(f"Failed to attach fundamental features: {e}")

        # Drop rows with NaN values, but be more lenient with technical indicators
        # Only drop rows where essential columns (OHLC) have NaN
        essential_cols = ['Open', 'High', 'Low', 'Close', 'returns']
        df_clean = df.dropna(subset=essential_cols)

        # Replace infinite values with NaN, then fill
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)

        # Fill NaN values in technical indicators with forward/backward fill or zeros
        df_clean = df_clean.fillna(method='ffill').fillna(method='bfill').fillna(0)

        logger.info(f"Engineered {len(df_clean.columns)} features from {len(df_clean)} observations")
        # Ensure common technical indicator columns exist to avoid KeyErrors later
        expected_tech_cols = [
            'Volume', 'tick_volume_sum',
            'ema_7', 'sma_7', 'ema_14', 'sma_14', 'ema_28', 'sma_28',
            'ema_56', 'sma_56', 'ema_112', 'sma_112', 'ema_225', 'sma_225',
            'volume_sma_20', 'volume_ratio'
        ]

        for col in expected_tech_cols:
            if col not in df_clean.columns:
                df_clean[col] = 0.0

        return df_clean

    def _prepare_features(self) -> pd.DataFrame:
        """
        Compatibility shim for external optimizers.

        Returns a DataFrame with engineered features and a numeric next-day change
        column named 'next_close_change' (used by older optimizer code).
        """
        # Engineer features from the loaded price data
        feature_df = self._engineer_features(self.price_data)

        # Enforce: pipeline must halt if fundamental data is missing or empty
        if not hasattr(self, 'fundamental_data') or self.fundamental_data is None or self.fundamental_data.empty:
            self.logger.critical("FATAL: Fundamental data is missing or empty during feature preparation. Halting pipeline.")
            raise RuntimeError("FATAL: Fundamental data is missing or empty during feature preparation. Pipeline halted.")

        if feature_df.empty:
            return feature_df

        # Ensure a consistent numeric next-day close change column exists
        try:
            feature_df['next_close_change'] = (feature_df['Close'].shift(-1) - feature_df['Close']) / feature_df['Close']
        except Exception:
            # Fallback: if Close is missing or malformed, create zeros
            feature_df['next_close_change'] = 0.0

        # Provide a binary target for compatibility as well
        if 'target_1d' not in feature_df.columns:
            try:
                feature_df['target_1d'] = (feature_df['Close'].shift(-1) > feature_df['Open'].shift(-1)).astype(int)
            except Exception:
                feature_df['target_1d'] = 0

        return feature_df


    def _add_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds candlestick pattern features to the dataframe.
        """
        if df.empty or ta is None:
            return df

        # List of all available candlestick pattern functions from the 'ta' library
        pattern_functions = [
            CDL2CROWS, CDL3BLACKCROWS, CDL3INSIDE, CDL3LINESTRIKE, CDL3OUTSIDE,
            CDL3STARSINSOUTH, CDL3WHITESOLDIERS, CDLABANDONEDBABY, CDLADVANCEBLOCK,
            CDLBELTHOLD, CDLBREAKAWAY, CDLCLOSINGMARUBOZU, CDLCONCEALBABYSWALL,
            CDLCOUNTERATTACK, CDLDARKCLOUDCOVER, CDLDOJI, CDLDOJISTAR,
            CDLDRAGONFLYDOJI, CDLENGULFING, CDLEVENINGDOJISTAR, CDLEVENINGSTAR,
            CDLGAPSIDESIDEWHITE, CDLGRAVESTONEDOJI, CDLHAMMER, CDLHANGINGMAN,
            CDLHARAMI, CDLHARAMICROSS, CDLHIGHWAVE, CDLHIKKAKE, CDLHIKKAKEMOD,
            CDLHOMINGPIGEON, CDLIDENTICAL3CROWS, CDLINNECK, CDLINVERTEDHAMMER,
            CDLKICKING, CDLKICKINGBYLENGTH, CDLLADDERBOTTOM, CDLLONGLEGGEDDOJI,

            CDLLONGLINE, CDLMARUBOZU, CDLMATCHINGLOW, CDLMATHOLD, CDLMORNINGDOJISTAR,
            CDLMORNINGSTAR, CDLONNECK, CDLPIERCING, CDLRICKSHAWMAN,
            CDLRISEFALL3METHODS, CDLSEPARATINGLINES, CDLSHOOTINGSTAR,
            CDLSHORTLINE, CDLSPINNINGTOP, CDLSTALLEDPATTERN, CDLSTICKSANDWICH,
            CDLTAKURI, CDLTASUKIGAP, CDLTHRUSTING, CDLTRISTAR, CDLUNIQUE3RIVER,
            CDLUPSIDEGAP2CROWS, CDLXSIDEGAP3METHODS
        ]

        df_patterns = df.copy()

        for pattern_func in pattern_functions:
            try:
                # The pattern functions are classes that need to be instantiated
                indicator = pattern_func(
                    open=df_patterns['Open'],
                    high=df_patterns['High'],
                    low=df_patterns['Low'],
                    close=df_patterns['Close']
                )
                # The result is in a method, usually with the same name as the function
                pattern_name = pattern_func.__name__.lower()
                result_series = getattr(indicator, indicator.name.lower())()
                if result_series is not None:
                    df_patterns[pattern_name] = result_series
            except Exception as e:
                self.logger.warning(f"Could not calculate candlestick pattern {pattern_func.__name__}: {e}")

        return df_patterns

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range."""
        high = df['High']
        low = df['Low']
        close = df['Close'].shift(1)

        tr = pd.concat([
            high - low,
            (high - close).abs(),
            (low - close).abs()
        ], axis=1).max(axis=1)

        return tr.rolling(period).mean()

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

    def _calculate_holloway_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate and integrate features from the Complete Holloway Algorithm.
        """
        try:
            # Run the complete Holloway algorithm
            holloway_df = self._holloway_algo.calculate_complete_holloway_algorithm(df.copy())

            # Merge all Holloway features back into the main dataframe
            # Prefix Holloway features to avoid column name collisions
            holloway_features = holloway_df.add_prefix('holloway_')
            
            # Align indices before merging
            df = df.join(holloway_features, how='left')

        except Exception as e:
            logger.error(f"Error calculating Holloway features: {e}")

        return df

    def _add_day_trading_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add day trading signals to the dataframe.
        """
        try:
            # Generate all day trading signals
            signals_df = self._day_trading_signals.generate_all_signals(df.copy())
            
            # Only select signal columns (those ending with '_signal')
            signal_columns = [col for col in signals_df.columns if col.endswith('_signal')]
            
            # Merge only the signal columns into main dataframe
            if signal_columns:
                signals_only_df = signals_df[signal_columns]
                df = df.join(signals_only_df, how='left')
                logger.info(f"Successfully added {len(signal_columns)} day trading signals")
            else:
                logger.warning("No signal columns found in day trading signals")
            
        except Exception as e:
            logger.error(f"Error calculating day trading signals: {e}")

        return df

    def _add_slump_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add slump signals to the dataframe.
        """
        try:
            # Generate all slump signals
            signals_df = self._slump_signals.generate_all_signals(df.copy())
            
            # Only select signal columns (those ending with '_signal')
            signal_columns = [col for col in signals_df.columns if col.endswith('_signal')]
            
            # Merge only the signal columns into main dataframe with suffix to avoid conflicts
            if signal_columns:
                signals_only_df = signals_df[signal_columns]
                df = df.join(signals_only_df, how='left', rsuffix='_slump')
                logger.info(f"Successfully added {len(signal_columns)} slump signals")
            else:
                logger.warning("No signal columns found in slump signals")
            
        except Exception as e:
            logger.error(f"Error calculating slump signals: {e}")

        return df

    def _add_harmonic_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add harmonic pattern signals to the dataframe.
        """
        try:
            df = detect_harmonic_patterns(df)
            logger.info("Successfully added harmonic pattern signals")
        except Exception as e:
            logger.error(f"Error calculating harmonic patterns: {e}")
        
        return df

    def _add_chart_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add chart pattern signals to the dataframe.
        """
        try:
            df = detect_chart_patterns(df)
            logger.info("Successfully added chart pattern signals")
        except Exception as e:
            logger.error(f"Error calculating chart patterns: {e}")
        
        return df

    def _add_elliott_wave_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Elliott Wave pattern signals to the dataframe.
        """
        try:
            df = detect_elliott_waves(df)
            logger.info("Successfully added Elliott Wave signals")
        except Exception as e:
            logger.error(f"Error calculating Elliott Wave signals: {e}")
        
        return df

    def _add_ultimate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Ultimate Signal Repository signals to the dataframe.
        Includes: SMC, Order Flow, Multi-TF Confluence, Session-based trading, etc.
        """
        try:
            df = integrate_ultimate_signals(df)
            logger.info("Successfully added Ultimate Signal Repository features")
        except Exception as e:
            logger.error(f"Error calculating Ultimate Signal Repository: {e}")
        
        return df

    def _calculate_holloway_for_timeframe(self, tf: str) -> Optional[pd.DataFrame]:
        """
        Helper function to calculate Holloway features for a single timeframe
        This function is designed to be called in parallel.
        """
        try:
            # Load raw data for the specific timeframe
            tf_loader_map = {
                'H4': lambda p: self._load_daily_price_file(p, timeframe_hint='H4'),
                'Daily': self._load_daily_price_file,
                'Weekly': lambda p: self._load_daily_price_file(p, timeframe_hint='Weekly'),
                'Monthly': self._load_monthly_data
            }
            
            loader_func = tf_loader_map.get(tf)
            if not loader_func:
                return None

            # Pass pair argument correctly
            if tf in ['H4', 'Weekly', 'Daily']:
                    tf_data = loader_func(self.pair)
            else:
                    tf_data = loader_func()

            if tf_data.empty or len(tf_data) < 225:
                logger.warning(f"Skipping Holloway for {tf} due to insufficient data ({len(tf_data)} rows)")
                return None

            # Calculate Holloway features for this timeframe
            holloway_tf_df = self._holloway_algo.calculate_complete_holloway_algorithm(tf_data.copy())
            
            if holloway_tf_df.empty:
                return None

            # Prefix columns with timeframe
            holloway_tf_df = holloway_tf_df.add_prefix(f'holloway_{tf.lower()}_')
            
            return holloway_tf_df

        except Exception as e:
            logger.error(f"Error processing Holloway features for {tf} timeframe: {e}")
            return None

    def _add_multi_timeframe_holloway_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate and integrate Holloway features from all available timeframes in parallel.
        """
        if df.empty:
            return df

        logger.info("Calculating and merging multi-timeframe Holloway features in parallel...")
        timeframes = ['H4', 'Daily', 'Weekly', 'Monthly']
        
        results = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(self._calculate_holloway_for_timeframe)(tf) for tf in timeframes
        )

        for i, tf in enumerate(timeframes):
            holloway_tf_df = results[i]
            if holloway_tf_df is not None and not holloway_tf_df.empty:
                try:
                    # Align with the main dataframe's index and forward-fill
                    aligned_features = holloway_tf_df.reindex(df.index).fillna(method='ffill')
                    
                    # Join the aligned features, adding a suffix to avoid conflicts
                    df = df.join(aligned_features, how='left', rsuffix=f'_{tf.lower()}_holloway')
                    logger.info(f"Successfully merged Holloway features for {tf} timeframe.")
                except Exception as e:
                    logger.error(f"Error merging Holloway features for {tf} timeframe: {e}")

        return df

    def _add_multi_timeframe_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich feature set with multi-timeframe technical indicators."""
        if df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return df

        base_minutes = self._infer_base_frequency_minutes(df.index)
        if base_minutes is None:
            return df

        timeframe_minutes = {
            'H1': 60,
            'H4': 240,
            'D1': 1440,
            'W1': 10080,
            'M1': 43200,
        }
        timeframe_freq = {
            'H1': '1H',
            'H4': '4H',
            'D1': '1D',
            'W1': '1W',
            'M1': '1M',
        }

        # Ensure the index is sorted for resampling
        df = df.sort_index()

        tolerance_factor = 0.8  # Allow small variance for calendar-based periods

        for tf_name, minutes in timeframe_minutes.items():
            if minutes < base_minutes * tolerance_factor:
                continue  # Skip faster timeframes than the source data

            freq = timeframe_freq[tf_name]

            ohlc = df[['Open', 'High', 'Low', 'Close']]
            try:
                resampled = ohlc.resample(freq).agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last'
                }).dropna(how='all')
            except (ValueError, TypeError):
                continue

            if resampled.empty:
                continue

            features = pd.DataFrame(index=resampled.index)
            features[f'{tf_name.lower()}_ema_20'] = resampled['Close'].ewm(span=20, min_periods=5).mean()
            features[f'{tf_name.lower()}_ema_50'] = resampled['Close'].ewm(span=50, min_periods=10).mean()
            features[f'{tf_name.lower()}_ema_200'] = resampled['Close'].ewm(span=200, min_periods=20).mean()
            features[f'{tf_name.lower()}_sma_20'] = resampled['Close'].rolling(20, min_periods=5).mean()
            features[f'{tf_name.lower()}_sma_50'] = resampled['Close'].rolling(50, min_periods=10).mean()
            features[f'{tf_name.lower()}_slope_5'] = resampled['Close'].pct_change(5)
            features[f'{tf_name.lower()}_slope_10'] = resampled['Close'].pct_change(10)
            features[f'{tf_name.lower()}_rsi_14'] = self._calculate_rsi(resampled['Close'], 14)

            # Align multi-timeframe features back to base timeframe using forward-fill
            aligned = features.reindex(df.index).fillna(method='ffill')
            df = df.join(aligned, how='left')

        return df

    def _infer_base_frequency_minutes(self, index: pd.DatetimeIndex) -> Optional[float]:
        """Infer approximate base frequency in minutes for the given datetime index."""
        if len(index) < 2:
            return None

        try:
            freq = pd.infer_freq(index[:10])
        except ValueError:
            freq = None

        if freq:
            try:
                offset = pd.tseries.frequencies.to_offset(freq)
                if offset.delta is not None:
                    return offset.delta.total_seconds() / 60
                return offset.nanos / 1e9 / 60
            except (AttributeError, ValueError):
                pass

        deltas = index.to_series().diff().dropna()
        if deltas.empty:
            return None

        median_delta = deltas.median()
        return median_delta.total_seconds() / 60

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

    def train_full_ensemble(self):
        """Train the complete ensemble of models for the specified currency pair."""
        logger.info(f"Starting full ensemble training for {self.pair}...")

        # Load and prepare the data
        price_data = self._load_price_data()
        if price_data.empty:
            logger.warning(f"No price data available for {self.pair}. Skipping training.")
            return

        feature_df = self._prepare_features()
        if feature_df.empty:
            logger.warning(f"No features available for {self.pair}. Skipping training.")
            return

        # Split features and target
        X = feature_df.drop(columns=['target_1d', 'next_close_change'], errors='ignore')
        y = feature_df['target_1d'] if 'target_1d' in feature_df else None

        # Time-based train/validation split
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:] if y is not None else (None, None)

        # Train each model in the ensemble
        for model_name, config in self.model_configs.items():
            if not config.get('enabled', False):
                logger.info(f"Skipping disabled model: {model_name}")
                continue

            logger.info(f"Training {model_name} model...")
            model = None

            try:
                if model_name == 'prophet':
                    # Prophet requires a specific DataFrame format
                    prophet_df = X_train.reset_index()[['Date', 'Close']]
                    prophet_df.columns = ['ds', 'y']

                    model = Prophet(**config['params'])
                    model.fit(prophet_df)

                elif model_name == 'auto_arima':
                    model = AutoARIMA(**config['params'])
                    model.fit(X_train, y_train)

                elif model_name == 'ets':
                    model = ETS(**config['params'])
                    model.fit(X_train, y_train)

                elif model_name == 'theta':
                    model = Theta(**config['params'])
                    model.fit(X_train, y_train)

                elif model_name == 'lightgbm':
                    model_params = config.get('params', {})
                    fit_params = {}
                    early_stopping_config = config.get('early_stopping', {})

                    if early_stopping_config.get('enabled'):
                        fit_params = {
                            'eval_set': [(X_val, y_val)],
                            'eval_metric': early_stopping_config.get('metric', 'binary_logloss'),
                            'callbacks': [
                                lgb.early_stopping(
                                    stopping_rounds=early_stopping_config.get('rounds', 100),
                                    verbose=False
                                )
                            ]
                        }
                    
                    # Use the enhanced pipeline for LightGBM
                    model = enhanced_lightgbm_training_pipeline(
                        X_train, y_train, X_val, y_val,
                        model_params=model_params,
                        fit_params=fit_params
                    )
                    if model is None:
                        logger.warning(f"Enhanced LightGBM training failed. Skipping.")
                        continue

                elif model_name == 'xgboost':
                    model = XGBClassifier(**config['params'])
                    model.fit(X_train, y_train)

                elif model_name == 'random_forest':
                    model = RandomForestClassifier(**config['params'])
                    model.fit(X_train, y_train)

                elif model_name == 'extra_trees':
                    model = ExtraTreesClassifier(**config['params'])
                    model.fit(X_train, y_train)

                # Save the trained model
                model_path = self.models_dir / f"{self.pair}_{model_name}.joblib"
                joblib.dump(model, model_path)
                self.models[model_name] = model

                logger.info(f"Trained and saved {model_name} model for {self.pair}")

            except Exception as e:
                logger.error(f"Error training {model_name} model: {e}")

        logger.info(f"Ensemble training completed for {self.pair}")

        # === Per-signal/feature evaluation ===
        # Always print and save per-signal evaluation after training
        try:
            eval_csv = f"{self.pair}_signal_evaluation.csv"
            self.evaluate_signal_features(feature_df, target_col='target_1d', output_csv=eval_csv)
            logger.info(f"Per-signal evaluation printed and saved to {eval_csv}")
        except Exception as e:
            logger.error(f"Signal evaluation failed: {e}")

    def generate_forecast(self, days_ahead: int = 1) -> pd.DataFrame:
        """Generate forecast for the specified number of days ahead."""
        logger.info(f"Generating {days_ahead}-day forecast for {self.pair}...")

        # Ensure models are loaded
        if not self.models:
            logger.warning(f"No trained models found for {self.pair}. Please train the ensemble first.")
            return pd.DataFrame()

        # Prepare the input data for forecasting
        feature_df = self._prepare_features()
        if feature_df.empty:
            logger.warning(f"No features available for {self.pair}. Cannot generate forecast.")
            return pd.DataFrame()

        # Use the latest available data for prediction
        latest_data = feature_df.iloc[-1:]

        # Collect predictions from each model
        predictions = {}
        for model_name, model in self.models.items():
            try:
                if model_name in ['prophet', 'auto_arima', 'ets', 'theta']:
                    # These models require the entire history for forecasting
                    model.fit(feature_df.drop(columns=['target_1d', 'next_close_change'], errors='ignore'), feature_df['target_1d'])
                    forecast = model.predict(future=pd.date_range(start=latest_data.index[-1] + timedelta(days=1), periods=days_ahead, freq='B'))
                    predictions[model_name] = forecast

                elif model_name in ['lightgbm', 'xgboost', 'random_forest', 'extra_trees']:
                    # Tree-based models can predict directly on the latest data
                    pred = model.predict(latest_data.drop(columns=['target_1d', 'next_close_change'], errors='ignore'))
                    predictions[model_name] = pred

            except Exception as e:
                logger.error(f"Error generating forecast with {model_name}: {e}")

        # Combine predictions - for tree-based models, take the mean prediction
        if predictions:
            try:
                combined_forecast = pd.DataFrame(predictions).mean(axis=1)
                combined_forecast.name = 'forecast'
                logger.info(f"Forecast generated for {self.pair}: {combined_forecast.values}")
                return combined_forecast
            except Exception as e:
                logger.error(f"Error combining forecasts: {e}")

        logger.warning(f"No valid forecasts generated for {self.pair}.")
        return pd.DataFrame()

    def get_trading_signal(self) -> str:
        """Get the trading signal based on the ensemble forecast."""
        logger.info(f"Generating trading signal for {self.pair}...")

        # Generate a short-term forecast (next day)
        forecast = self.generate_forecast(days_ahead=1)

        if forecast.empty:
            logger.warning(f"No forecast data available for signal generation.")
            return "Hold"

        # Simple signal logic: buy if forecasted return is positive, sell if negative
        signal = "Hold"
        try:
            if forecast.iloc[-1] > 0:
                signal = "Buy"
            elif forecast.iloc[-1] < 0:
                signal = "Sell"
        except Exception as e:
            logger.error(f"Error determining trading signal: {e}")

        logger.info(f"Trading signal for {self.pair}: {signal}")
        return signal

    def load_and_prepare_datasets(self, train_size_split=0.8):
        """
        Loads data, engineers features, and splits it into training and validation sets.
        This is a high-level function for the automated training pipeline.
        """
        logger.info(f"Loading and preparing datasets for {self.pair}...")

        feature_df = self._prepare_features()
        if feature_df.empty:
            logger.error(f"Feature engineering failed for {self.pair}. Cannot proceed.")
            return None, None, None, None

        # Define features (X) and target (y)
        target_col = 'target_1d'
        if target_col not in feature_df.columns:
            logger.error(f"Target column '{target_col}' not found in the dataframe.")
            return None, None, None, None
            
        y = feature_df[target_col]
        
        # Drop target and other leakage-prone columns
        cols_to_drop = [c for c in feature_df.columns if 'target' in c or 'next_close_change' in c]
        X = feature_df.drop(columns=cols_to_drop)

        # Perform time-based split
        train_size = int(len(X) * train_size_split)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:] if y is not None else (None, None)

        logger.info(f"Data prepared: X_train: {X_train.shape}, X_val: {X_val.shape}")

        return X_train, y_train, X_val, y_val


# Backwards-compatibility shims: ensure critical helpers exist on the class
# Some CI/tests or older orchestrators expect these methods to be present
# even when parts of the file were refactored. Provide small, safe
# implementations that call the normalized loaders where possible.
def _compat_get_cross_pair(self) -> Optional[str]:
    cross_map = {
        'EURUSD': 'XAUUSD',
        'XAUUSD': 'EURUSD'
    }
    return cross_map.get(getattr(self, 'pair', None))


def _compat_load_daily_price_file(self, pair: Optional[str] = None, timeframe_hint: str = 'Daily') -> pd.DataFrame:
    """Safe fallback loader that tries common candidate files and uses
    the module normalizer when available.
    """
    try:
        pair = pair or getattr(self, 'pair', None)
        root = Path(os.getcwd()) / 'data'
        candidates = [
            root / f"{pair}_Daily.csv",
            root / f"{pair}_daily.csv",
            root / f"{pair}.csv",
            root / f"{pair}_D1.csv",
            root / f"{pair}_H4.csv",
            root / f"{pair}_Monthly.csv",
        ]

        for c in candidates:
            if not c.exists():
                continue
            try:
                raw = pd.read_csv(c, engine='python')
                df = _normalize_price_dataframe(raw)
                if df.empty:
                    continue
                if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
                    return df[['Open', 'High', 'Low', 'Close']]
            except Exception:
                continue

        # Last resort: aggregate intraday data if available
        intr = getattr(self, 'intraday_data', None)
        if intr is not None and not getattr(intr, 'empty', True):
            try:
                daily, _ = self._build_intraday_context(intr)
                if not daily.empty:
                    return daily
            except Exception:
                pass

    except Exception:
        pass
    return pd.DataFrame()


def _compat_build_intraday_context(self, hourly: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Thin wrapper that delegates to the class implementation if present,
    otherwise performs a conservative aggregation.
    """
    try:
        if hasattr(self, '_build_intraday_context') and callable(getattr(self, '_build_intraday_context')):
            # If the method exists (maybe via normal class def), call it
            return getattr(self, '_build_intraday_context')(hourly)
    except Exception:
        pass

    # Conservative fallback aggregation
    if hourly is None or getattr(hourly, 'empty', True) or not isinstance(hourly.index, pd.DatetimeIndex):
        return pd.DataFrame(), pd.DataFrame()

    try:
        hourly = hourly.sort_index()
        daily_ohlc = hourly[['Open', 'High', 'Low', 'Close']].resample('1D').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
        }).dropna(how='all')

        intraday_features = pd.DataFrame(index=daily_ohlc.index)
        if 'Volume' in hourly.columns:
            intraday_features['intraday_volume_sum'] = hourly['Volume'].resample('1D').sum()
        return daily_ohlc.dropna(subset=['Open', 'High', 'Low', 'Close'], how='any'), intraday_features
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


# Attach shims to the class only if the methods are missing
if not hasattr(HybridPriceForecastingEnsemble, '_get_cross_pair'):
    HybridPriceForecastingEnsemble._get_cross_pair = _compat_get_cross_pair

if not hasattr(HybridPriceForecastingEnsemble, '_load_daily_price_file'):
    HybridPriceForecastingEnsemble._load_daily_price_file = _compat_load_daily_price_file

if not hasattr(HybridPriceForecastingEnsemble, '_build_intraday_context'):
    HybridPriceForecastingEnsemble._build_intraday_context = _compat_build_intraday_context


# Provide minimal module-level helper implementations for core indicators
# These will be attached to the class if the full implementations are
# missing (for older/refactored code paths). They are conservative and
# avoid raising errors when data is partially available.
def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
    try:
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta).clip(lower=0).rolling(period).mean()
        rs = gain / (loss + 1e-12)
        return 100 - (100 / (1 + rs))
    except Exception:
        return pd.Series(index=prices.index, dtype=float)


def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    try:
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    except Exception:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)


def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
    try:
        high = df['High']
        low = df['Low']
        close = df['Close'].shift(1)
        tr = pd.concat([
            (high - low).abs(),
            (high - close).abs(),
            (low - close).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    except Exception:
        return pd.Series(dtype=float)


def _calculate_holloway_features(self, df: pd.DataFrame) -> pd.DataFrame:
    # Conservative stub: try to call real holloway algo if present, else noop
    try:
        if hasattr(self, '_holloway_algo') and getattr(self, '_holloway_algo') is not None:
            hdf = self._holloway_algo.calculate_complete_holloway_algorithm(df.copy())
            if hdf is None or hdf.empty:
                return df
            return df.join(hdf.add_prefix('holloway_'), how='left')
    except Exception:
        pass
    return df


def _calculate_holloway_for_timeframe(self, tf: str) -> Optional[pd.DataFrame]:
    # Minimal implementation: attempt to load timeframe data and run holloway
    try:
        if tf == 'H1':
            tf_data = self._load_intraday_data(self.pair)
        elif tf == 'Monthly':
            tf_data = self._load_monthly_data()
        else:
            tf_data = self._load_daily_price_file(self.pair)

        if tf_data is None or getattr(tf_data, 'empty', True):
            return None
        if hasattr(self, '_holloway_algo') and self._holloway_algo is not None:
            res = self._holloway_algo.calculate_complete_holloway_algorithm(tf_data.copy())
            if res is not None and not res.empty:
                return res.add_prefix(f'holloway_{tf.lower()}_')
    except Exception:
        pass
    return None


def _add_multi_timeframe_holloway_features(self, df: pd.DataFrame) -> pd.DataFrame:
    # Conservative: do nothing if holloway features unavailable
    return df


def _add_multi_timeframe_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
    # Minimal multi-timeframe indicator integration: noop
    return df


def _infer_base_frequency_minutes(self, index: pd.DatetimeIndex) -> Optional[float]:
    try:
        if len(index) < 2:
            return None
        deltas = index.to_series().diff().dropna()
        if deltas.empty:
            return None
        median_delta = deltas.median()
        return median_delta.total_seconds() / 60
    except Exception:
        return None


def _calculate_adx(self, df: pd.DataFrame, period: int) -> pd.Series:
    try:
        high = df['High']
        low = df['Low']
        close = df['Close']
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        dm_plus = np.where((high - high.shift(1)) > (low.shift(1) - low),
                          np.maximum(high - high.shift(1), 0), 0)
        dm_minus = np.where((low.shift(1) - low) > (high - high.shift(1)),
                           np.maximum(low.shift(1) - low, 0), 0)
        di_plus = 100 * (pd.Series(dm_plus).rolling(period).mean() / tr.rolling(period).mean())
        di_minus = 100 * (pd.Series(dm_minus).rolling(period).mean() / tr.rolling(period).mean())
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus + 1e-12)
        return dx.rolling(period).mean()
    except Exception:
        return pd.Series(dtype=float)


def _add_cross_pair_features(self, df: pd.DataFrame) -> pd.DataFrame:
    # Attempt to call the existing implementation if present; else noop
    try:
        if hasattr(self, '_get_cross_pair'):
            # reuse the class method if available
            return HybridPriceForecastingEnsemble._add_cross_pair_features(self, df) if hasattr(HybridPriceForecastingEnsemble, '_add_cross_pair_features') else df
    except Exception:
        pass
    return df


# Attach any of these implementations to the class if still missing
for _name in ['_calculate_rsi','_calculate_macd','_calculate_atr','_calculate_holloway_features','_calculate_holloway_for_timeframe','_add_multi_timeframe_holloway_features','_add_multi_timeframe_indicators','_infer_base_frequency_minutes','_calculate_adx','_add_cross_pair_features']:
    if not hasattr(HybridPriceForecastingEnsemble, _name):
        setattr(HybridPriceForecastingEnsemble, _name, globals().get(_name))
