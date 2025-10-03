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
    from .fundamental_features import FundamentalFeatureEngineer
except ImportError:  # pragma: no cover - support standalone execution
    from fundamental_features import FundamentalFeatureEngineer

try:
    from .holloway_algorithm import CompleteHollowayAlgorithm
except ImportError:  # pragma: no cover - support standalone execution
    from holloway_algorithm import CompleteHollowayAlgorithm

# ML and statistical libraries
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from lightgbm import LGBMClassifier
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
except ImportError:
    ta = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HybridPriceForecastingEnsemble:
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
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)

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
        self.price_data = self._load_price_data()
        self.fundamental_engineer = FundamentalFeatureEngineer(self.data_dir)
        self.fundamental_data = self._load_fundamental_data()
        self.cross_pair = self._get_cross_pair()

        # Feature engineering
        self.feature_columns = []
        self._holloway_algo = CompleteHollowayAlgorithm(str(self.data_dir))

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

    def _load_price_data(self) -> pd.DataFrame:
        """Load historical price data using provided daily files and intraday context."""
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

    def _load_intraday_data(self, pair: Optional[str] = None) -> pd.DataFrame:
        """Load intraday (H1) price data if available."""
        pair = pair or self.pair
        csv_file = self.data_dir / f"{pair}_H1.csv"
        if not csv_file.exists():
            return pd.DataFrame()

        try:
            df = pd.read_csv(csv_file, sep='\t')
            if '<DATE>' not in df.columns or '<TIME>' not in df.columns:
                logger.warning(f"Unexpected intraday format for {pair}: {csv_file}")
                return pd.DataFrame()

            df['Date'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])
            df = df.sort_values('Date').set_index('Date')

            rename_map = {
                '<OPEN>': 'Open',
                '<HIGH>': 'High',
                '<LOW>': 'Low',
                '<CLOSE>': 'Close',
                '<TICKVOL>': 'TickVolume',
                '<VOL>': 'RealVolume',
                '<SPREAD>': 'Spread'
            }
            df = df.rename(columns=rename_map)
            df['Volume'] = df['TickVolume']

            logger.info(
                "%s intraday coverage -> bars: %d, span: %s to %s",
                pair,
                len(df),
                df.index.min().date(),
                df.index.max().date()
            )

            return df[['Open', 'High', 'Low', 'Close', 'Volume', 'TickVolume', 'RealVolume', 'Spread']]
        except Exception as e:
            logger.error(f"Error loading intraday data for {pair}: {e}")
            return pd.DataFrame()

    def _load_monthly_data(self, pair: Optional[str] = None) -> pd.DataFrame:
        """Load monthly price data used for slower timeframe context."""
        pair = pair or self.pair
        csv_file = self.data_dir / f"{pair}_Monthly.csv"
        if not csv_file.exists():
            return pd.DataFrame()

        try:
            df = pd.read_csv(csv_file, sep='\t')
            if '<DATE>' not in df.columns:
                logger.warning(f"Unexpected monthly format for {pair}: {csv_file}")
                return pd.DataFrame()

            df['Date'] = pd.to_datetime(df['<DATE>'])
            df = df.sort_values('Date').set_index('Date')

            rename_map = {
                '<OPEN>': 'Open',
                '<HIGH>': 'High',
                '<LOW>': 'Low',
                '<CLOSE>': 'Close',
                '<TICKVOL>': 'TickVolume',
                '<VOL>': 'RealVolume',
                '<SPREAD>': 'Spread'
            }
            df = df.rename(columns=rename_map)

            logger.info(
                "%s monthly coverage -> rows: %d, span: %s to %s",
                pair,
                len(df),
                df.index.min().date(),
                df.index.max().date()
            )

            return df[['Open', 'High', 'Low', 'Close', 'TickVolume', 'RealVolume', 'Spread']]
        except Exception as e:
            logger.error(f"Error loading monthly data for {pair}: {e}")
            return pd.DataFrame()

    def _load_fallback_ohlc(self) -> pd.DataFrame:
        """Fallback loader when no intraday data is available."""
        if not self.monthly_data.empty:
            monthly = self.monthly_data.copy()
            daily = monthly[['Open', 'High', 'Low', 'Close']].resample('1D').ffill()
            tick_volume = monthly.get('TickVolume', pd.Series(index=monthly.index, dtype=float)).resample('1D').ffill()
            daily['Volume'] = tick_volume
            daily['tick_volume_sum'] = tick_volume
            return daily.dropna(subset=['Open', 'High', 'Low', 'Close'])

        logger.warning(f"No intraday or monthly OHLC data found for {self.pair}")
        return pd.DataFrame()

    def _load_fundamental_data(self) -> pd.DataFrame:
        """Load fundamental economic data."""
        fundamental_df = pd.DataFrame()

        # Load comprehensive economic indicators from FRED
        key_series = [
            'FEDFUNDS',      # Federal Funds Rate
            'DFF',           # Federal Funds Target Rate
            'CPIAUCSL',      # Consumer Price Index
            'CPALTT01USM661S', # Core CPI
            'UNRATE',        # Unemployment Rate
            'PAYEMS',        # Nonfarm Payrolls
            'INDPRO',        # Industrial Production Index
            'DGORDER',       # Durable Goods Orders
            'DEXUSEU',       # USD/EUR Exchange Rate
            'DEXJPUS',       # USD/JPY Exchange Rate
            'DEXCHUS',       # USD/China Exchange Rate
            'ECBDFR',        # ECB Deposit Facility Rate
            'CP0000EZ19M086NEST', # Eurozone CPI
            'LRHUTTTTDEM156S',   # Eurozone Unemployment
            'DCOILWTICO',    # WTI Crude Oil Price
            'DCOILBRENTEU',  # Brent Crude Oil Price
            'VIXCLS',        # CBOE Volatility Index
            'DGS10',         # 10-Year Treasury Rate
            'DGS2',          # 2-Year Treasury Rate
            'BOPGSTB'        # US Trade Balance
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

    def _load_daily_price_file(self, pair: Optional[str] = None, timeframe_hint: str = 'Daily') -> pd.DataFrame:
        """Load the provided daily CSV for the specified pair if available."""
        pair = pair or self.pair
        
        # Dynamically create candidate filenames based on hint
        timeframe_variants = [timeframe_hint.upper(), timeframe_hint.lower()]
        if 'daily' in timeframe_hint.lower():
            timeframe_variants.extend(['D1', 'd1'])

        candidate_files = []
        for variant in set(timeframe_variants):
            candidate_files.append(self.data_dir / f"{pair}_{variant}.csv")
        
        # Add generic fallback
        if 'daily' in timeframe_hint.lower():
            candidate_files.append(self.data_dir / f"{pair}.csv")
            candidate_files.append(self.data_dir / "raw" / f"{pair}_Daily.csv")


        for csv_file in candidate_files:
            if not csv_file.exists():
                continue

            try:
                df = pd.read_csv(csv_file)

                if '<DATE>' in df.columns and '<TIME>' in df.columns:
                    df['Date'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])
                elif '<DATE>' in df.columns:
                    df['Date'] = pd.to_datetime(df['<DATE>'])
                elif 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                elif 'date' in df.columns:
                    df['Date'] = pd.to_datetime(df['date'])
                else:
                    logger.warning(f"Unable to identify date column in {csv_file}")
                    continue

                df = df.sort_values('Date').set_index('Date')

                rename_map = {
                    '<OPEN>': 'Open',
                    '<HIGH>': 'High',
                    '<LOW>': 'Low',
                    '<CLOSE>': 'Close',
                    '<VOL>': 'Volume',
                    '<TICKVOL>': 'TickVolume',
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume',
                    'tick_volume': 'TickVolume',
                    'spread': 'Spread'
                }
                available_map = {k: v for k, v in rename_map.items() if k in df.columns}
                if available_map:
                    df = df.rename(columns=available_map)

                if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
                    logger.warning(f"Daily file {csv_file} missing OHLC columns after normalization")
                    continue

                numeric_cols = [col for col in df.columns if col not in ['symbol', 'currency']]
                df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

                df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
                df = df[~df.index.duplicated(keep='last')]

                logger.info(
                    "%s daily coverage -> rows: %d, span: %s to %s from %s",
                    pair,
                    len(df),
                    df.index.min().date(),
                    df.index.max().date(),
                    csv_file.name
                )

                return df
            except Exception as e:
                logger.warning(f"Error loading daily data from {csv_file}: {e}")

        return pd.DataFrame()

    def _build_intraday_context(self, hourly: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Aggregate hourly candles into daily OHLC and derive intraday statistics."""
        if hourly.empty or not isinstance(hourly.index, pd.DatetimeIndex):
            return pd.DataFrame(), pd.DataFrame()

        hourly = hourly.sort_index()

        daily_ohlc = hourly[['Open', 'High', 'Low', 'Close']].resample('1D').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        }).dropna(how='all')

        intraday_features = pd.DataFrame(index=daily_ohlc.index)

        if 'Volume' in hourly.columns:
            intraday_features['intraday_volume_sum'] = hourly['Volume'].resample('1D').sum()
            intraday_features['intraday_volume_std'] = hourly['Volume'].resample('1D').std()

        if 'TickVolume' in hourly.columns:
            intraday_features['intraday_tick_volume_sum'] = hourly['TickVolume'].resample('1D').sum()

        if 'RealVolume' in hourly.columns:
            intraday_features['intraday_real_volume_sum'] = hourly['RealVolume'].resample('1D').sum()

        if 'Spread' in hourly.columns:
            intraday_features['intraday_spread_mean'] = hourly['Spread'].resample('1D').mean()

        intraday_returns = hourly['Close'].pct_change()
        intraday_range = (hourly['High'] - hourly['Low'])

        intraday_features['intraday_close_std'] = hourly['Close'].resample('1D').std()
        intraday_features['intraday_return_volatility'] = intraday_returns.resample('1D').std()
        intraday_features['intraday_return_sum'] = intraday_returns.resample('1D').sum()
        intraday_features['intraday_range_mean'] = intraday_range.resample('1D').mean()
        intraday_features['intraday_range_max'] = intraday_range.resample('1D').max()
        intraday_features['intraday_range_std'] = intraday_range.resample('1D').std()
        intraday_features['intraday_bar_count'] = hourly['Close'].resample('1D').count()

        intraday_features = intraday_features.replace([np.inf, -np.inf], np.nan)

        return daily_ohlc.dropna(subset=['Open', 'High', 'Low', 'Close'], how='any'), intraday_features

    def _combine_daily_sources(self, primary: pd.DataFrame, supplemental: pd.DataFrame) -> pd.DataFrame:
        """Merge provided daily candles with intraday-derived aggregates."""
        combined = primary.copy()
        for col in ['Open', 'High', 'Low', 'Close']:
            if col not in combined.columns:
                combined[col] = supplemental[col]
            else:
                combined[col] = combined[col].combine_first(supplemental[col])

        combined = combined.dropna(subset=['Open', 'High', 'Low', 'Close'])
        combined = combined[~combined.index.duplicated(keep='last')]

        return combined

    def _get_cross_pair(self) -> Optional[str]:
        """Identify the complementary pair used for cross-asset signals."""
        cross_map = {
            'EURUSD': 'XAUUSD',
            'XAUUSD': 'EURUSD'
        }
        return cross_map.get(self.pair)

    def _add_cross_pair_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Integrate cross-pair correlation features to enhance signal quality."""
        if df.empty:
            return df

        cross_pair = self.cross_pair
        if not cross_pair:
            return df

        cross_daily = self._load_daily_price_file(cross_pair)
        if cross_daily.empty:
            cross_intraday = self._load_intraday_data(cross_pair)
            if not cross_intraday.empty:
                cross_daily, _ = self._build_intraday_context(cross_intraday)

        if cross_daily.empty:
            logger.warning(f"Cross-pair data unavailable for {cross_pair}")
            return df

        cross_daily = cross_daily[['Close']].rename(columns={'Close': f'{cross_pair}_close'})
        cross_daily[f'{cross_pair}_returns'] = cross_daily[f'{cross_pair}_close'].pct_change()

        aligned_cross = cross_daily.reindex(df.index).fillna(method='ffill')

        enriched = df.join(aligned_cross, how='left')

        base_returns = enriched.get('returns')
        cross_returns = enriched.get(f'{cross_pair}_returns')

        if base_returns is not None and cross_returns is not None:
            enriched[f'corr_5_{cross_pair.lower()}'] = base_returns.rolling(5).corr(cross_returns)
            enriched[f'corr_20_{cross_pair.lower()}'] = base_returns.rolling(20).corr(cross_returns)
            enriched[f'return_spread_{cross_pair.lower()}'] = base_returns - cross_returns

        cross_prices = enriched.get(f'{cross_pair}_close')
        if cross_prices is not None:
            safe_cross = cross_prices.replace(0, np.nan)
            enriched[f'price_ratio_{cross_pair.lower()}'] = enriched['Close'] / safe_cross
            enriched[f'price_spread_{cross_pair.lower()}'] = enriched['Close'] - cross_prices

        return enriched

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

    def _calculate_holloway_for_timeframe(self, tf: str) -> Optional[pd.DataFrame]:
        """
        Helper function to calculate Holloway features for a single timeframe.
        This function is designed to be called in parallel.
        """
        try:
            # Load raw data for the specific timeframe
            tf_loader_map = {
                'H1': self._load_intraday_data,
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
        timeframes = ['H1', 'H4', 'Daily', 'Weekly', 'Monthly']
        
        results = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(self._calculate_holloway_for_timeframe)(tf) for tf in timeframes
        )

        for i, tf in enumerate(timeframes):
            holloway_tf_df = results[i]
            if holloway_tf_df is not None and not holloway_tf_df.empty:
                try:
                    # Align with the main dataframe's index and forward-fill
                    aligned_features = holloway_tf_df.reindex(df.index).fillna(method='ffill')
                    
                    # Join the aligned features
                    df = df.join(aligned_features, how='left')
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

    def _load_daily_price_file(self, pair: Optional[str] = None, timeframe_hint: str = 'Daily') -> pd.DataFrame:
        """Load the provided daily CSV for the specified pair if available."""
        pair = pair or self.pair
        
        # Dynamically create candidate filenames based on hint
        timeframe_variants = [timeframe_hint.upper(), timeframe_hint.lower()]
        if 'daily' in timeframe_hint.lower():
            timeframe_variants.extend(['D1', 'd1'])

        candidate_files = []
        for variant in set(timeframe_variants):
            candidate_files.append(self.data_dir / f"{pair}_{variant}.csv")
        
        # Add generic fallback
        if 'daily' in timeframe_hint.lower():
            candidate_files.append(self.data_dir / f"{pair}.csv")
            candidate_files.append(self.data_dir / "raw" / f"{pair}_Daily.csv")


        for csv_file in candidate_files:
            if not csv_file.exists():
                continue

            try:
                df = pd.read_csv(csv_file)

                if '<DATE>' in df.columns and '<TIME>' in df.columns:
                    df['Date'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])
                elif '<DATE>' in df.columns:
                    df['Date'] = pd.to_datetime(df['<DATE>'])
                elif 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                elif 'date' in df.columns:
                    df['Date'] = pd.to_datetime(df['date'])
                else:
                    logger.warning(f"Unable to identify date column in {csv_file}")
                    continue

                df = df.sort_values('Date').set_index('Date')

                rename_map = {
                    '<OPEN>': 'Open',
                    '<HIGH>': 'High',
                    '<LOW>': 'Low',
                    '<CLOSE>': 'Close',
                    '<VOL>': 'Volume',
                    '<TICKVOL>': 'TickVolume',
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume',
                    'tick_volume': 'TickVolume',
                    'spread': 'Spread'
                }
                available_map = {k: v for k, v in rename_map.items() if k in df.columns}
                if available_map:
                    df = df.rename(columns=available_map)

                if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
                    logger.warning(f"Daily file {csv_file} missing OHLC columns after normalization")
                    continue

                numeric_cols = [col for col in df.columns if col not in ['symbol', 'currency']]
                df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

                df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
                df = df[~df.index.duplicated(keep='last')]

                logger.info(
                    "%s daily coverage -> rows: %d, span: %s to %s from %s",
                    pair,
                    len(df),
                    df.index.min().date(),
                    df.index.max().date(),
                    csv_file.name
                )

                return df
            except Exception as e:
                logger.warning(f"Error loading daily data from {csv_file}: {e}")

        return pd.DataFrame()

    def _build_intraday_context(self, hourly: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Aggregate hourly candles into daily OHLC and derive intraday statistics."""
        if hourly.empty or not isinstance(hourly.index, pd.DatetimeIndex):
            return pd.DataFrame(), pd.DataFrame()

        hourly = hourly.sort_index()

        daily_ohlc = hourly[['Open', 'High', 'Low', 'Close']].resample('1D').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        }).dropna(how='all')

        intraday_features = pd.DataFrame(index=daily_ohlc.index)

        if 'Volume' in hourly.columns:
            intraday_features['intraday_volume_sum'] = hourly['Volume'].resample('1D').sum()
            intraday_features['intraday_volume_std'] = hourly['Volume'].resample('1D').std()

        if 'TickVolume' in hourly.columns:
            intraday_features['intraday_tick_volume_sum'] = hourly['TickVolume'].resample('1D').sum()

        if 'RealVolume' in hourly.columns:
            intraday_features['intraday_real_volume_sum'] = hourly['RealVolume'].resample('1D').sum()

        if 'Spread' in hourly.columns:
            intraday_features['intraday_spread_mean'] = hourly['Spread'].resample('1D').mean()

        intraday_returns = hourly['Close'].pct_change()
        intraday_range = (hourly['High'] - hourly['Low'])

        intraday_features['intraday_close_std'] = hourly['Close'].resample('1D').std()
        intraday_features['intraday_return_volatility'] = intraday_returns.resample('1D').std()
        intraday_features['intraday_return_sum'] = intraday_returns.resample('1D').sum()
        intraday_features['intraday_range_mean'] = intraday_range.resample('1D').mean()
        intraday_features['intraday_range_max'] = intraday_range.resample('1D').max()
        intraday_features['intraday_range_std'] = intraday_range.resample('1D').std()
        intraday_features['intraday_bar_count'] = hourly['Close'].resample('1D').count()

        intraday_features = intraday_features.replace([np.inf, -np.inf], np.nan)

        return daily_ohlc.dropna(subset=['Open', 'High', 'Low', 'Close'], how='any'), intraday_features

    def _combine_daily_sources(self, primary: pd.DataFrame, supplemental: pd.DataFrame) -> pd.DataFrame:
        """Merge provided daily candles with intraday-derived aggregates."""
        combined = primary.copy()
        for col in ['Open', 'High', 'Low', 'Close']:
            if col not in combined.columns:
                combined[col] = supplemental[col]
            else:
                combined[col] = combined[col].combine_first(supplemental[col])

        combined = combined.dropna(subset=['Open', 'High', 'Low', 'Close'])
        combined = combined[~combined.index.duplicated(keep='last')]

        return combined

    def _get_cross_pair(self) -> Optional[str]:
        """Identify the complementary pair used for cross-asset signals."""
        cross_map = {
            'EURUSD': 'XAUUSD',
            'XAUUSD': 'EURUSD'
        }
        return cross_map.get(self.pair)

    def _add_cross_pair_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Integrate cross-pair correlation features to enhance signal quality."""
        if df.empty:
            return df

        cross_pair = self.cross_pair
        if not cross_pair:
            return df

        cross_daily = self._load_daily_price_file(cross_pair)
        if cross_daily.empty:
            cross_intraday = self._load_intraday_data(cross_pair)
            if not cross_intraday.empty:
                cross_daily, _ = self._build_intraday_context(cross_intraday)

        if cross_daily.empty:
            logger.warning(f"Cross-pair data unavailable for {cross_pair}")
            return df

        cross_daily = cross_daily[['Close']].rename(columns={'Close': f'{cross_pair}_close'})
        cross_daily[f'{cross_pair}_returns'] = cross_daily[f'{cross_pair}_close'].pct_change()

        aligned_cross = cross_daily.reindex(df.index).fillna(method='ffill')

        enriched = df.join(aligned_cross, how='left')

        base_returns = enriched.get('returns')
        cross_returns = enriched.get(f'{cross_pair}_returns')

        if base_returns is not None and cross_returns is not None:
            enriched[f'corr_5_{cross_pair.lower()}'] = base_returns.rolling(5).corr(cross_returns)
            enriched[f'corr_20_{cross_pair.lower()}'] = base_returns.rolling(20).corr(cross_returns)
            enriched[f'return_spread_{cross_pair.lower()}'] = base_returns - cross_returns

        cross_prices = enriched.get(f'{cross_pair}_close')
        if cross_prices is not None:
            safe_cross = cross_prices.replace(0, np.nan)
            enriched[f'price_ratio_{cross_pair.lower()}'] = enriched['Close'] / safe_cross
            enriched[f'price_spread_{cross_pair.lower()}'] = enriched['Close'] - cross_prices

        return enriched

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

    def _calculate_holloway_for_timeframe(self, tf: str) -> Optional[pd.DataFrame]:
        """
        Helper function to calculate Holloway features for a single timeframe.
        This function is designed to be called in parallel.
        """
        try:
            # Load raw data for the specific timeframe
            tf_loader_map = {
                'H1': self._load_intraday_data,
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
        timeframes = ['H1', 'H4', 'Daily', 'Weekly', 'Monthly']
        
        results = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(self._calculate_holloway_for_timeframe)(tf) for tf in timeframes
        )

        for i, tf in enumerate(timeframes):
            holloway_tf_df = results[i]
            if holloway_tf_df is not None and not holloway_tf_df.empty:
                try:
                    # Align with the main dataframe's index and forward-fill
                    aligned_features = holloway_tf_df.reindex(df.index).fillna(method='ffill')
                    
                    # Join the aligned features
                    df = df.join(aligned_features, how='left')
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

    def _load_daily_price_file(self, pair: Optional[str] = None, timeframe_hint: str = 'Daily') -> pd.DataFrame:
        """Load the provided daily CSV for the specified pair if available."""
        pair = pair or self.pair
        
        # Dynamically create candidate filenames based on hint
        timeframe_variants = [timeframe_hint.upper(), timeframe_hint.lower()]
        if 'daily' in timeframe_hint.lower():
            timeframe_variants.extend(['D1', 'd1'])

        candidate_files = []
        for variant in set(timeframe_variants):
            candidate_files.append(self.data_dir / f"{pair}_{variant}.csv")
        
        # Add generic fallback
        if 'daily' in timeframe_hint.lower():
            candidate_files.append(self.data_dir / f"{pair}.csv")
            candidate_files.append(self.data_dir / "raw" / f"{pair}_Daily.csv")


        for csv_file in candidate_files:
            if not csv_file.exists():
                continue

            try:
                df = pd.read_csv(csv_file)

                if '<DATE>' in df.columns and '<TIME>' in df.columns:
                    df['Date'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])
                elif '<DATE>' in df.columns:
                    df['Date'] = pd.to_datetime(df['<DATE>'])
                elif 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                elif 'date' in df.columns:
                    df['Date'] = pd.to_datetime(df['date'])
                else:
                    logger.warning(f"Unable to identify date column in {csv_file}")
                    continue

                df = df.sort_values('Date').set_index('Date')

                rename_map = {
                    '<OPEN>': 'Open',
                    '<HIGH>': 'High',
                    '<LOW>': 'Low',
                    '<CLOSE>': 'Close',
                    '<VOL>': 'Volume',
                    '<TICKVOL>': 'TickVolume',
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume',
                    'tick_volume': 'TickVolume',
                    'spread': 'Spread'
                }
                available_map = {k: v for k, v in rename_map.items() if k in df.columns}
                if available_map:
                    df = df.rename(columns=available_map)

                if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
                    logger.warning(f"Daily file {csv_file} missing OHLC columns after normalization")
                    continue

                numeric_cols = [col for col in df.columns if col not in ['symbol', 'currency']]
                df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

                df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
                df = df[~df.index.duplicated(keep='last')]

                logger.info(
                    "%s daily coverage -> rows: %d, span: %s to %s from %s",
                    pair,
                    len(df),
                    df.index.min().date(),
                    df.index.max().date(),
                    csv_file.name
                )

                return df
            except Exception as e:
                logger.warning(f"Error loading daily data from {csv_file}: {e}")

        return pd.DataFrame()

    def _build_intraday_context(self, hourly: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Aggregate hourly candles into daily OHLC and derive intraday statistics."""
        if hourly.empty or not isinstance(hourly.index, pd.DatetimeIndex):
            return pd.DataFrame(), pd.DataFrame()

        hourly = hourly.sort_index()

        daily_ohlc = hourly[['Open', 'High', 'Low', 'Close']].resample('1D').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        }).dropna(how='all')

        intraday_features = pd.DataFrame(index=daily_ohlc.index)

        if 'Volume' in hourly.columns:
            intraday_features['intraday_volume_sum'] = hourly['Volume'].resample('1D').sum()
            intraday_features['intraday_volume_std'] = hourly['Volume'].resample('1D').std()

        if 'TickVolume' in hourly.columns:
            intraday_features['intraday_tick_volume_sum'] = hourly['TickVolume'].resample('1D').sum()

        if 'RealVolume' in hourly.columns:
            intraday_features['intraday_real_volume_sum'] = hourly['RealVolume'].resample('1D').sum()

        if 'Spread' in hourly.columns:
            intraday_features['intraday_spread_mean'] = hourly['Spread'].resample('1D').mean()

        intraday_returns = hourly['Close'].pct_change()
        intraday_range = (hourly['High'] - hourly['Low'])

        intraday_features['intraday_close_std'] = hourly['Close'].resample('1D').std()
        intraday_features['intraday_return_volatility'] = intraday_returns.resample('1D').std()
        intraday_features['intraday_return_sum'] = intraday_returns.resample('1D').sum()
        intraday_features['intraday_range_mean'] = intraday_range.resample('1D').mean()
        intraday_features['intraday_range_max'] = intraday_range.resample('1D').max()
        intraday_features['intraday_range_std'] = intraday_range.resample('1D').std()
        intraday_features['intraday_bar_count'] = hourly['Close'].resample('1D').count()

        intraday_features = intraday_features.replace([np.inf, -np.inf], np.nan)

        return daily_ohlc.dropna(subset=['Open', 'High', 'Low', 'Close'], how='any'), intraday_features

    def _combine_daily_sources(self, primary: pd.DataFrame, supplemental: pd.DataFrame) -> pd.DataFrame:
        """Merge provided daily candles with intraday-derived aggregates."""
        combined = primary.copy()
        for col in ['Open', 'High', 'Low', 'Close']:
            if col not in combined.columns:
                combined[col] = supplemental[col]
            else:
                combined[col] = combined[col].combine_first(supplemental[col])

        combined = combined.dropna(subset=['Open', 'High', 'Low', 'Close'])
        combined = combined[~combined.index.duplicated(keep='last')]

        return combined

    def _get_cross_pair(self) -> Optional[str]:
        """Identify the complementary pair used for cross-asset signals."""
        cross_map = {
            'EURUSD': 'XAUUSD',
            'XAUUSD': 'EURUSD'
        }
        return cross_map.get(self.pair)

    def _add_cross_pair_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Integrate cross-pair correlation features to enhance signal quality."""
        if df.empty:
            return df

        cross_pair = self.cross_pair
        if not cross_pair:
            return df

        cross_daily = self._load_daily_price_file(cross_pair)
        if cross_daily.empty:
            cross_intraday = self._load_intraday_data(cross_pair)
            if not cross_intraday.empty:
                cross_daily, _ = self._build_intraday_context(cross_intraday)

        if cross_daily.empty:
            logger.warning(f"Cross-pair data unavailable for {cross_pair}")
            return df

        cross_daily = cross_daily[['Close']].rename(columns={'Close': f'{cross_pair}_close'})
        cross_daily[f'{cross_pair}_returns'] = cross_daily[f'{cross_pair}_close'].pct_change()

        aligned_cross = cross_daily.reindex(df.index).fillna(method='ffill')

        enriched = df.join(aligned_cross, how='left')

        base_returns = enriched.get('returns')
        cross_returns = enriched.get(f'{cross_pair}_returns')

        if base_returns is not None and cross_returns is not None:
            enriched[f'corr_5_{cross_pair.lower()}'] = base_returns.rolling(5).corr(cross_returns)
            enriched[f'corr_20_{cross_pair.lower()}'] = base_returns.rolling(20).corr(cross_returns)
            enriched[f'return_spread_{cross_pair.lower()}'] = base_returns - cross_returns

        cross_prices = enriched.get(f'{cross_pair}_close')
        if cross_prices is not None:
            safe_cross = cross_prices.replace(0, np.nan)
            enriched[f'price_ratio_{cross_pair.lower()}'] = enriched['Close'] / safe_cross
            enriched[f'price_spread_{cross_pair.lower()}'] = enriched['Close'] - cross_prices

        return enriched

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

    def _calculate_holloway_for_timeframe(self, tf: str) -> Optional[pd.DataFrame]:
        """
        Helper function to calculate Holloway features for a single timeframe.
        This function is designed to be called in parallel.
        """
        try:
            # Load raw data for the specific timeframe
            tf_loader_map = {
                'H1': self._load_intraday_data,
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
        timeframes = ['H1', 'H4', 'Daily', 'Weekly', 'Monthly']
        
        results = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(self._calculate_holloway_for_timeframe)(tf) for tf in timeframes
        )

        for i, tf in enumerate(timeframes):
            holloway_tf_df = results[i]
            if holloway_tf_df is not None and not holloway_tf_df.empty:
                try:
                    # Align with the main dataframe's index and forward-fill
                    aligned_features = holloway_tf_df.reindex(df.index).fillna(method='ffill')
                    
                    # Join the aligned features
                    df = df.join(aligned_features, how='left')
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