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
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import joblib
import warnings
warnings.filterwarnings('ignore')

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

        # Configuration - adjust based on available data and timeframe
        if self.pair in ['EURUSD', 'XAUUSD']:
            # Check what data is available and set parameters accordingly
            monthly_file = self.data_dir / f"{self.pair}_Monthly.csv"
            h1_file = self.data_dir / f"{self.pair}_H1.csv"

            if monthly_file.exists():
                # Monthly data available - use monthly parameters
                self.forecast_horizon = 6  # 6 months ahead
                self.lookback_window = 60  # 5 years of monthly data
                self.timeframe = 'Monthly'
            elif h1_file.exists():
                # H1 data available - use H1 parameters
                self.forecast_horizon = 120  # 5 days * 24 hours = 120 hours ahead
                self.lookback_window = 1000  # ~6 weeks of H1 data
                self.timeframe = 'H1'
            else:
                # Fall back to daily parameters
                self.forecast_horizon = 5  # days ahead
                self.lookback_window = 200  # ~1 year of daily data
                self.timeframe = 'Daily'
        else:
            # Other pairs use daily data
            self.forecast_horizon = 5  # days ahead
            self.lookback_window = 200  # ~1 year of daily data
            self.timeframe = 'Daily'

        self.validation_splits = 5

        # Model configurations
        self.model_configs = self._get_model_configs()

        # Load data
        self.price_data = self._load_price_data()
        self.fundamental_data = self._load_fundamental_data()

        # Feature engineering
        self.feature_columns = []

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
        """Load historical price data for the currency pair."""
        # For EURUSD and XAUUSD, prefer Monthly > H1 > Daily data
        if self.pair in ['EURUSD', 'XAUUSD']:
            # Try Monthly data first
            csv_file = self.data_dir / f"{self.pair}_Monthly.csv"
            if not csv_file.exists():
                # Fall back to H1 data
                csv_file = self.data_dir / f"{self.pair}_H1.csv"
                if not csv_file.exists():
                    # Fall back to Daily data
                    csv_file = self.data_dir / "raw" / f"{self.pair}_Daily.csv"
        else:
            csv_file = self.data_dir / "raw" / f"{self.pair}_Daily.csv"

        if not csv_file.exists():
            logger.warning(f"Price data file not found: {csv_file}")
            return pd.DataFrame()

        try:
            # Handle different CSV formats
            if self.pair in ['EURUSD', 'XAUUSD'] and (csv_file.name.endswith('_Monthly.csv') or csv_file.name.endswith('_H1.csv')):
                # Monthly and H1 data are tab-separated
                df = pd.read_csv(csv_file, sep='\t')
            else:
                df = pd.read_csv(csv_file)

            # Handle different date/time column formats
            if '<DATE>' in df.columns and '<TIME>' in df.columns:
                # H1 format with separate DATE and TIME columns
                df['Date'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])
            elif '<DATE>' in df.columns:
                # Monthly format with just DATE column
                df['Date'] = pd.to_datetime(df['<DATE>'])
            else:
                # Handle both 'date' and 'Date' column names
                date_col = 'date' if 'date' in df.columns else 'Date'
                df['Date'] = pd.to_datetime(df[date_col])

            df = df.sort_values('Date').set_index('Date')

            # Ensure we have OHLC columns (handle both cases)
            required_cols = ['Open', 'High', 'Low', 'Close']
            actual_cols = ['open', 'high', 'low', 'close']
            ohlc_cols = ['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>']  # H1 format

            if not all(col in df.columns for col in required_cols):
                if all(col in df.columns for col in actual_cols):
                    # Rename lowercase columns to uppercase
                    df = df.rename(columns={
                        'open': 'Open',
                        'high': 'High',
                        'low': 'Low',
                        'close': 'Close'
                    })
                elif all(col in df.columns for col in ohlc_cols):
                    # Rename H1 format columns
                    df = df.rename(columns={
                        '<OPEN>': 'Open',
                        '<HIGH>': 'High',
                        '<LOW>': 'Low',
                        '<CLOSE>': 'Close'
                    })
                else:
                    logger.error(f"Missing required OHLC columns in {csv_file}")
                    return pd.DataFrame()

            logger.info(f"Loaded {len(df)} price observations for {self.pair}")
            return df

        except Exception as e:
            logger.error(f"Error loading price data: {e}")
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

        logger.info(f"Loaded fundamental data with {len(fundamental_df)} observations")
        return fundamental_df

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

        # Add fundamental features if available
        if not self.fundamental_data.empty:
            # Merge fundamental data
            df = df.join(self.fundamental_data, how='left')

            # Forward fill missing fundamental data
            fundamental_cols = [col for col in df.columns if col in self.fundamental_data.columns]
            df[fundamental_cols] = df[fundamental_cols].fillna(method='ffill')

            # Calculate changes in fundamental indicators
            for col in fundamental_cols:
                # Use robust percentage change calculation to avoid division by zero
                pct_change_1m = df[col].pct_change(20)
                pct_change_3m = df[col].pct_change(60)
                
                # Cap extreme values to prevent infinity
                pct_change_1m = pct_change_1m.clip(-10, 10)  # Cap at -1000% to +1000%
                pct_change_3m = pct_change_3m.clip(-10, 10)
                
                df[f'{col}_change_1m'] = pct_change_1m
                df[f'{col}_change_3m'] = pct_change_3m

        # Drop rows with NaN values, but be more lenient with technical indicators
        # Only drop rows where essential columns (OHLC) have NaN
        essential_cols = ['Open', 'High', 'Low', 'Close', 'returns']
        df_clean = df.dropna(subset=essential_cols)
        
        # Replace infinite values with NaN, then fill
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values in technical indicators with forward/backward fill or zeros
        df_clean = df_clean.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        logger.info(f"Engineered {len(df_clean.columns)} features from {len(df_clean)} observations")
        return df_clean

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

    def _calculate_holloway_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Holloway Algorithm features for trend analysis."""
        if df.empty:
            return df

        df = df.copy()

        periods = [5, 7, 10, 14, 20, 28, 50, 56, 100, 112, 200, 225]
        emas: Dict[int, pd.Series] = {}
        smas: Dict[int, pd.Series] = {}

        for period in periods:
            emas[period] = df['Close'].ewm(span=period, min_periods=max(3, period // 2)).mean()
            smas[period] = df['Close'].rolling(period, min_periods=max(3, period // 2)).mean()

            df[f'ema_{period}'] = emas[period]
            df[f'sma_{period}'] = smas[period]

        bull_signals: List[pd.Series] = []
        bear_signals: List[pd.Series] = []

        for period in periods:
            bull_signals.append(df['Close'] > emas[period])
            bull_signals.append(df['Close'] > smas[period])
            bear_signals.append(df['Close'] < emas[period])
            bear_signals.append(df['Close'] < smas[period])

        for i in range(len(periods) - 1):
            for j in range(i + 1, len(periods)):
                p1, p2 = periods[i], periods[j]
                bull_signals.append(emas[p1] > emas[p2])
                bear_signals.append(emas[p1] < emas[p2])
                bull_signals.append(smas[p1] > smas[p2])
                bear_signals.append(smas[p1] < smas[p2])

        for ep in periods:
            for sp in periods:
                bull_signals.append(emas[ep] > smas[sp])
                bear_signals.append(emas[ep] < smas[sp])

        for period in periods:
            bull_signals.append((df['Close'] > emas[period]) & (df['Close'].shift(1) <= emas[period].shift(1)))
            bear_signals.append((df['Close'] < emas[period]) & (df['Close'].shift(1) >= emas[period].shift(1)))
            bull_signals.append((df['Close'] > smas[period]) & (df['Close'].shift(1) <= smas[period].shift(1)))
            bear_signals.append((df['Close'] < smas[period]) & (df['Close'].shift(1) >= smas[period].shift(1)))

        for ep in periods[:5]:
            for sp in periods[:5]:
                bull_signals.append((emas[ep] > smas[sp]) & (emas[ep].shift(1) <= smas[sp].shift(1)))
                bear_signals.append((emas[ep] < smas[sp]) & (emas[ep].shift(1) >= smas[sp].shift(1)))

        df['holloway_bull_count'] = sum(bull_signals).astype(float)
        df['holloway_bear_count'] = sum(bear_signals).astype(float)

        df['holloway_bull_avg'] = df['holloway_bull_count'].ewm(span=27, min_periods=5).mean()
        df['holloway_bear_avg'] = df['holloway_bear_count'].ewm(span=27, min_periods=5).mean()

        df['holloway_count_diff'] = df['holloway_bull_count'] - df['holloway_bear_count']
        df['holloway_count_ratio'] = df['holloway_bull_count'] / (df['holloway_bear_count'] + 1)

        df['holloway_bull_max_20'] = df['holloway_bull_count'].rolling(20, min_periods=5).max()
        df['holloway_bull_min_20'] = df['holloway_bull_count'].rolling(20, min_periods=5).min()
        df['holloway_bear_max_20'] = df['holloway_bear_count'].rolling(20, min_periods=5).max()
        df['holloway_bear_min_20'] = df['holloway_bear_count'].rolling(20, min_periods=5).min()

        support_window = 40
        df['holloway_bull_support'] = df['holloway_bull_count'].rolling(support_window, min_periods=10).quantile(0.2)
        df['holloway_bull_resistance'] = df['holloway_bull_count'].rolling(support_window, min_periods=10).quantile(0.8)
        df['holloway_bear_support'] = df['holloway_bear_count'].rolling(support_window, min_periods=10).quantile(0.2)
        df['holloway_bear_resistance'] = df['holloway_bear_count'].rolling(support_window, min_periods=10).quantile(0.8)

        df['holloway_bull_dist_support'] = df['holloway_bull_count'] - df['holloway_bull_support']
        df['holloway_bull_dist_resistance'] = df['holloway_bull_resistance'] - df['holloway_bull_count']
        df['holloway_bear_dist_support'] = df['holloway_bear_count'] - df['holloway_bear_support']
        df['holloway_bear_dist_resistance'] = df['holloway_bear_resistance'] - df['holloway_bear_count']

        df['holloway_bull_pct_of_range'] = df['holloway_bull_dist_support'] / (
            (df['holloway_bull_resistance'] - df['holloway_bull_support']).replace(0, np.nan)
        )
        df['holloway_bear_pct_of_range'] = df['holloway_bear_dist_support'] / (
            (df['holloway_bear_resistance'] - df['holloway_bear_support']).replace(0, np.nan)
        )

        for lookback in [3, 5, 10]:
            df[f'holloway_bull_momentum_{lookback}'] = df['holloway_bull_count'].diff(lookback)
            df[f'holloway_bear_momentum_{lookback}'] = df['holloway_bear_count'].diff(lookback)

        df['holloway_bull_zscore_40'] = (
            df['holloway_bull_count'] - df['holloway_bull_count'].rolling(support_window, min_periods=10).mean()
        ) / (df['holloway_bull_count'].rolling(support_window, min_periods=10).std() + 1e-6)
        df['holloway_bear_zscore_40'] = (
            df['holloway_bear_count'] - df['holloway_bear_count'].rolling(support_window, min_periods=10).mean()
        ) / (df['holloway_bear_count'].rolling(support_window, min_periods=10).std() + 1e-6)

        df['holloway_bull_cross_up'] = (df['holloway_bull_count'] > df['holloway_bull_avg']) & (
            df['holloway_bull_count'].shift(1) <= df['holloway_bull_avg'].shift(1)
        )
        df['holloway_bull_cross_down'] = (df['holloway_bull_count'] < df['holloway_bull_avg']) & (
            df['holloway_bull_count'].shift(1) >= df['holloway_bull_avg'].shift(1)
        )
        df['holloway_bear_cross_up'] = (df['holloway_bear_count'] > df['holloway_bear_avg']) & (
            df['holloway_bear_count'].shift(1) <= df['holloway_bear_avg'].shift(1)
        )
        df['holloway_bear_cross_down'] = (df['holloway_bear_count'] < df['holloway_bear_avg']) & (
            df['holloway_bear_count'].shift(1) >= df['holloway_bear_avg'].shift(1)
        )

        if 'rsi_14' not in df.columns:
            df['rsi_14'] = self._calculate_rsi(df['Close'], 14)

        df['rsi_overbought'] = df['rsi_14'] > 70
        df['rsi_oversold'] = df['rsi_14'] < 30
        df['rsi_bounce_resistance'] = (df['rsi_14'] >= 51) & (df['rsi_14'].shift(1) < 51)
        df['rsi_bounce_support'] = (df['rsi_14'] <= 49) & (df['rsi_14'].shift(1) > 49)

        df['holloway_bull_signal'] = df['holloway_bull_cross_up'] & ~df['rsi_overbought']
        df['holloway_bear_signal'] = df['holloway_bear_cross_up'] & ~df['rsi_oversold']

        return df

    def _engineer_candle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features specifically for candle size prediction.
        
        This is separate from directional prediction to avoid interference.
        Focuses on ATR, ADX, pivot points, and other candle size dependent features.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with candle size features
        """
        if df.empty:
            return df

        df = df.copy()

        # Candle size features
        df['candle_range'] = df['High'] - df['Low']
        df['candle_body'] = abs(df['Close'] - df['Open'])
        df['upper_wick'] = df['High'] - np.maximum(df['Open'], df['Close'])
        df['lower_wick'] = np.minimum(df['Open'], df['Close']) - df['Low']
        df['candle_body_ratio'] = df['candle_body'] / (df['candle_range'] + 0.0001)  # Avoid division by zero
        
        # ATR (Average True Range) - candle size dependent
        df['atr_14'] = self._calculate_atr(df, 14)
        df['atr_20'] = self._calculate_atr(df, 20)
        df['atr_50'] = self._calculate_atr(df, 50)
        
        # ADX (Average Directional Index) - uses True Range
        df['adx_14'] = self._calculate_adx(df, 14)
        df['adx_20'] = self._calculate_adx(df, 20)
        
        # Pivot points - candle size dependent
        df['pivot_point'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['r1'] = 2 * df['pivot_point'] - df['Low']
        df['s1'] = 2 * df['pivot_point'] - df['High']
        df['r2'] = df['pivot_point'] + (df['High'] - df['Low'])
        df['s2'] = df['pivot_point'] - (df['High'] - df['Low'])
        
        # Rolling statistics of candle sizes
        for period in [5, 10, 20, 50]:
            df[f'candle_range_sma_{period}'] = df['candle_range'].rolling(period).mean()
            df[f'candle_body_sma_{period}'] = df['candle_body'].rolling(period).mean()
            df[f'candle_range_std_{period}'] = df['candle_range'].rolling(period).std()
            df[f'candle_body_std_{period}'] = df['candle_body'].rolling(period).std()
        
        # Volatility ratios
        df['atr_ratio_14'] = df['atr_14'] / df['Close']
        df['atr_ratio_20'] = df['atr_20'] / df['Close']
        
        # Candle type classification
        df['is_doji'] = df['candle_body'] / (df['candle_range'] + 0.0001) < 0.1
        df['is_marubozu'] = df['candle_body'] / (df['candle_range'] + 0.0001) > 0.8
        df['is_hammer'] = (df['lower_wick'] > 2 * df['candle_body']) & (df['upper_wick'] < df['candle_body'])
        df['is_shooting_star'] = (df['upper_wick'] > 2 * df['candle_body']) & (df['lower_wick'] < df['candle_body'])
        
        # Target for candle size prediction (next candle's range)
        df['target_candle_range_1d'] = df['candle_range'].shift(-1)
        df['target_candle_body_1d'] = df['candle_body'].shift(-1)
        
        # Clean up
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df

    def train_candle_size_model(self) -> Dict[str, float]:
        """
        Train a separate model for candle size prediction.
        
        This is independent of the directional prediction model.
        """
        logger.info("Training candle size prediction model")
        
        # Engineer candle features
        candle_df = self._engineer_candle_features(self.price_data)
        
        if len(candle_df) < 100:
            raise ValueError("Insufficient data for candle size training")
        
        # Define candle feature columns (exclude directional features)
        candle_feature_cols = [
            col for col in candle_df.columns 
            if col.startswith(('candle_', 'atr_', 'adx_', 'pivot_', 'r1', 's1', 'r2', 's2'))
            and not col.startswith('target_')
        ]
        
        # Add some basic price features that don't depend on direction
        basic_cols = ['Close', 'returns', 'log_returns', 'volatility_20', 'volatility_50']
        candle_feature_cols.extend([col for col in basic_cols if col in candle_df.columns])
        
        # Prepare data
        X = candle_df[candle_feature_cols].values
        y_range = candle_df['target_candle_range_1d'].values
        y_body = candle_df['target_candle_body_1d'].values
        
        # Remove NaN targets
        valid_idx = ~np.isnan(y_range) & ~np.isnan(y_body)
        X = X[valid_idx]
        y_range = y_range[valid_idx]
        y_body = y_body[valid_idx]
        
        if len(X) < 50:
            raise ValueError("Insufficient valid data for candle size training")
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_range_train, y_range_test = y_range[:split_idx], y_range[split_idx:]
        y_body_train, y_body_test = y_body[:split_idx], y_body[split_idx:]
        
        # Scale features
        self.scalers['candle_features'] = StandardScaler()
        X_train_scaled = self.scalers['candle_features'].fit_transform(X_train)
        X_test_scaled = self.scalers['candle_features'].transform(X_test)
        
        # Scale targets
        self.scalers['candle_range_target'] = StandardScaler()
        self.scalers['candle_body_target'] = StandardScaler()
        
        y_range_train_scaled = self.scalers['candle_range_target'].fit_transform(y_range_train.reshape(-1, 1)).ravel()
        y_body_train_scaled = self.scalers['candle_body_target'].fit_transform(y_body_train.reshape(-1, 1)).ravel()
        
        # Train models for range and body separately
        candle_models = {}
        
        # Range prediction model
        range_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        range_model.fit(X_train_scaled, y_range_train_scaled)
        candle_models['candle_range_model'] = range_model
        
        # Body prediction model
        body_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        body_model.fit(X_train_scaled, y_body_train_scaled)
        candle_models['candle_body_model'] = body_model
        
        # Save candle models
        self.models.update(candle_models)
        self._save_models()
        
        # Calculate metrics
        y_range_pred_scaled = range_model.predict(X_test_scaled)
        y_body_pred_scaled = body_model.predict(X_test_scaled)
        
        y_range_pred = self.scalers['candle_range_target'].inverse_transform(y_range_pred_scaled.reshape(-1, 1)).ravel()
        y_body_pred = self.scalers['candle_body_target'].inverse_transform(y_body_pred_scaled.reshape(-1, 1)).ravel()
        
        range_mae = np.mean(np.abs(y_range_pred - y_range_test))
        body_mae = np.mean(np.abs(y_body_pred - y_body_test))
        
        metrics = {
            'candle_range_mae': float(range_mae),
            'candle_body_mae': float(body_mae),
            'candle_range_mape': float(np.mean(np.abs((y_range_pred - y_range_test) / (y_range_test + 1e-8))) * 100),
            'candle_body_mape': float(np.mean(np.abs((y_body_pred - y_body_test) / (y_body_test + 1e-8))) * 100)
        }
        
        logger.info(f"Candle size model training completed. Range MAE: {range_mae:.6f}, Body MAE: {body_mae:.6f}")
        
        return metrics

    def _train_statistical_models(self, train_df: pd.DataFrame) -> Dict:
        """Train statistical forecasting models."""
        models = {}

        # Prepare data for statistical models
        ts_data = train_df[['Close']].reset_index()
        ts_data.columns = ['ds', 'y'] if 'prophet' in self.model_configs else ['ds', 'y']

        # Prophet
        if self.model_configs['prophet']['enabled']:
            try:
                logger.info("Training Prophet model")
                model = Prophet(**self.model_configs['prophet']['params'])
                model.fit(ts_data)
                models['prophet'] = model
            except Exception as e:
                logger.error(f"Error training Prophet: {e}")

        # StatsForecast models
        if StatsForecast is not None:
            try:
                logger.info("Training statistical forecasting models")
                sf_models = []

                if self.model_configs['auto_arima']['enabled']:
                    sf_models.append(AutoARIMA(**self.model_configs['auto_arima']['params']))

                if self.model_configs['ets']['enabled']:
                    sf_models.append(ETS(**self.model_configs['ets']['params']))

                if self.model_configs['theta']['enabled']:
                    sf_models.append(Theta(**self.model_configs['theta']['params']))

                if sf_models:
                    sf = StatsForecast(models=sf_models, freq='D')
                    models['statsforecast'] = sf

            except Exception as e:
                logger.error(f"Error training StatsForecast models: {e}")

        return models

    def _train_ml_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Train machine learning models with advanced regularization and early stopping."""
        models = {}
        
        # Split training data for validation
        from sklearn.model_selection import train_test_split
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, shuffle=False
        )

        # LightGBM with Early Stopping
        if self.model_configs['lightgbm']['enabled']:
            try:
                logger.info("Training LightGBM model with early stopping")
                config = self.model_configs['lightgbm']
                model = LGBMClassifier(**config['params'])
                
                if config.get('early_stopping', {}).get('enabled', False):
                    rounds = config['early_stopping']['rounds']
                    eval_set = [(X_val_split, y_val_split)]
                    # Try the common sklearn-wrapper argument first, then fallback to
                    # lightgbm callback API if the sklearn wrapper/version doesn't support it.
                    try:
                        model.fit(
                            X_train_split, y_train_split,
                            eval_set=eval_set,
                            eval_metric=config['early_stopping'].get('metric', None),
                            early_stopping_rounds=rounds,
                            verbose=False
                        )
                    except TypeError as e:
                        try:
                            import lightgbm as lgb
                            cb = [lgb.callback.early_stopping(rounds, verbose=False)]
                            model.fit(
                                X_train_split, y_train_split,
                                eval_set=eval_set,
                                eval_metric=config['early_stopping'].get('metric', None),
                                callbacks=cb,
                                verbose=False
                            )
                        except Exception as e2:
                            logger.warning(f"LightGBM early stopping not available: {e2}. Training without early stopping.")
                            model.fit(X_train, y_train)
                else:
                    model.fit(X_train, y_train)
                    
                models['lightgbm'] = model
                logger.info(f"LightGBM trained. Best iteration: {getattr(model, 'best_iteration_', 'N/A')}")
            except Exception as e:
                logger.error(f"Error training LightGBM: {e}")

        # XGBoost with Early Stopping
        if self.model_configs['xgboost']['enabled']:
            try:
                logger.info("Training XGBoost model with early stopping")
                config = self.model_configs['xgboost']
                model = XGBClassifier(**config['params'])
                
                if config.get('early_stopping', {}).get('enabled', False):
                    rounds = config['early_stopping']['rounds']
                    try:
                        model.fit(
                            X_train_split, y_train_split,
                            eval_set=[(X_val_split, y_val_split)],
                            eval_metric=config['early_stopping'].get('metric', 'rmse'),
                            early_stopping_rounds=rounds,
                            verbose=False
                        )
                    except TypeError:
                        # Some xgboost versions expect callbacks or don't accept the
                        # early_stopping_rounds kwarg on the sklearn wrapper. Try callback API,
                        # otherwise fallback to basic fit.
                        try:
                            import xgboost as xgb
                            cb = [xgb.callback.EarlyStopping(rounds)]
                            model.fit(
                                X_train_split, y_train_split,
                                eval_set=[(X_val_split, y_val_split)],
                                callbacks=cb,
                                verbose=False
                            )
                        except Exception as e:
                            logger.warning(f"XGBoost early stopping not available: {e}. Training without early stopping.")
                            model.fit(X_train, y_train)
                else:
                    model.fit(X_train, y_train)
                    
                models['xgboost'] = model
                logger.info(f"XGBoost trained. Best iteration: {getattr(model, 'best_iteration', 'N/A')}")
            except Exception as e:
                logger.error(f"Error training XGBoost: {e}")

        # Random Forest with Enhanced Regularization
        if self.model_configs['random_forest']['enabled']:
            try:
                logger.info("Training Random Forest model with regularization")
                config = self.model_configs['random_forest']
                model = RandomForestClassifier(**config['params'])
                model.fit(X_train, y_train)
                
                # Check OOB score if available
                if hasattr(model, 'oob_score_') and config.get('validation', {}).get('use_oob', False):
                    oob_score = model.oob_score_
                    logger.info(f"Random Forest OOB Score: {oob_score:.4f}")
                    
                    # Early termination if target reached
                    target_score = config.get('validation', {}).get('target_score', 0.85)
                    if oob_score >= target_score:
                        logger.info(f"Random Forest reached target score: {oob_score:.4f}")
                        
                models['random_forest'] = model
            except Exception as e:
                logger.error(f"Error training Random Forest: {e}")

        # Extra Trees with Enhanced Regularization
        if self.model_configs['extra_trees']['enabled']:
            try:
                logger.info("Training Extra Trees model with regularization")
                config = self.model_configs['extra_trees']
                model = ExtraTreesClassifier(**config['params'])
                model.fit(X_train, y_train)
                
                # Evaluate performance for early termination
                if len(X_val_split) > 0:
                    val_score = model.score(X_val_split, y_val_split)
                    logger.info(f"Extra Trees Validation Score: {val_score:.4f}")
                    
                models['extra_trees'] = model
            except Exception as e:
                logger.error(f"Error training Extra Trees: {e}")

        return models

    def _train_dl_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Train deep learning models."""
        models = {}

        if tf is None:
            return models

        # Prepare data for LSTM (3D: samples, timesteps, features)
        # Reshape for sequence modeling
        sequence_length = 30  # 30-day lookback
        X_seq = []
        y_seq = []

        for i in range(sequence_length, len(X_train)):
            X_seq.append(X_train[i-sequence_length:i])
            y_seq.append(y_train[i])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        if len(X_seq) == 0:
            logger.warning("Not enough data for deep learning models")
            return models

        # LSTM with Advanced Regularization
        if self.model_configs['lstm']['enabled']:
            try:
                logger.info("Training LSTM model with advanced regularization")
                config = self.model_configs['lstm']
                params = config['params']
                
                # Import regularizers
                from tensorflow.keras import regularizers
                from tensorflow.keras.callbacks import ReduceLROnPlateau
                
                model = Sequential([
                    LSTM(params['units'],
                         input_shape=(X_seq.shape[1], X_seq.shape[2]),
                         dropout=params['dropout'],
                         recurrent_dropout=params['recurrent_dropout'],
                         kernel_regularizer=regularizers.l1_l2(
                             l1=params['l1_reg'], l2=params['l2_reg']
                         ),
                         return_sequences=False),
                    Dropout(0.4),  # Additional dropout layer
                    Dense(64, activation='relu',
                          kernel_regularizer=regularizers.l1_l2(
                              l1=params['l1_reg']/2, l2=params['l2_reg']/2
                          )),
                    Dropout(0.3),
                    Dense(32, activation='relu'),
                    Dense(1, activation='sigmoid')  # Binary classification output
                ])

                model.compile(
                    optimizer=Adam(learning_rate=params['learning_rate']),
                    loss=params.get('loss', 'binary_crossentropy'),
                    metrics=params.get('metrics', ['accuracy', 'binary_accuracy'])
                )

                # Setup callbacks
                callbacks = []
                
                # Early Stopping
                if config.get('early_stopping', {}).get('enabled', False):
                    es_config = config['early_stopping']
                    early_stop = EarlyStopping(
                        monitor=es_config['monitor'],
                        patience=es_config['patience'],
                        min_delta=es_config['min_delta'],
                        restore_best_weights=es_config['restore_best_weights'],
                        verbose=1
                    )
                    callbacks.append(early_stop)
                
                # Learning Rate Reduction
                if config.get('callbacks', {}).get('reduce_lr', {}).get('enabled', False):
                    lr_config = config['callbacks']['reduce_lr']
                    reduce_lr = ReduceLROnPlateau(
                        monitor=lr_config['monitor'],
                        factor=lr_config['factor'],
                        patience=lr_config['patience'],
                        min_lr=lr_config['min_lr'],
                        verbose=1
                    )
                    callbacks.append(reduce_lr)

                # Train model
                history = model.fit(
                    X_seq, y_seq,
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    validation_split=0.25,  # Increased validation split
                    callbacks=callbacks,
                    verbose=1
                )

                models['lstm'] = model
                logger.info(f"LSTM training completed. Final val_loss: {min(history.history['val_loss']):.6f}")

            except Exception as e:
                logger.error(f"Error training LSTM: {e}")

        # BiLSTM with Advanced Regularization
        if self.model_configs['bilstm']['enabled']:
            try:
                logger.info("Training BiLSTM model with advanced regularization")
                config = self.model_configs['bilstm']
                params = config['params']
                
                # Import regularizers if not already imported
                if tf is not None:
                    from tensorflow.keras import regularizers
                    from tensorflow.keras.callbacks import ReduceLROnPlateau
                
                model = Sequential([
                    Bidirectional(
                        LSTM(params['units'],
                             dropout=params['dropout'],
                             recurrent_dropout=params['recurrent_dropout'],
                             kernel_regularizer=regularizers.l1_l2(
                                 l1=params['l1_reg'], l2=params['l2_reg']
                             )),
                        input_shape=(X_seq.shape[1], X_seq.shape[2])
                    ),
                    Dropout(0.4),  # Additional dropout
                    Dense(64, activation='relu',
                          kernel_regularizer=regularizers.l1_l2(
                              l1=params['l1_reg']/2, l2=params['l2_reg']/2
                          )),
                    Dropout(0.3),
                    Dense(32, activation='relu'),
                    Dense(1, activation='sigmoid')  # Binary classification output
                ])

                model.compile(
                    optimizer=Adam(learning_rate=params['learning_rate']),
                    loss=params.get('loss', 'binary_crossentropy'),
                    metrics=params.get('metrics', ['accuracy', 'binary_accuracy'])
                )

                # Setup callbacks
                callbacks = []
                
                # Early Stopping
                if config.get('early_stopping', {}).get('enabled', False):
                    es_config = config['early_stopping']
                    early_stop = EarlyStopping(
                        monitor=es_config['monitor'],
                        patience=es_config['patience'],
                        min_delta=es_config['min_delta'],
                        restore_best_weights=es_config['restore_best_weights'],
                        verbose=1
                    )
                    callbacks.append(early_stop)
                
                # Learning Rate Reduction
                if config.get('callbacks', {}).get('reduce_lr', {}).get('enabled', False):
                    lr_config = config['callbacks']['reduce_lr']
                    reduce_lr = ReduceLROnPlateau(
                        monitor=lr_config['monitor'],
                        factor=lr_config['factor'],
                        patience=lr_config['patience'],
                        min_lr=lr_config['min_lr'],
                        verbose=1
                    )
                    callbacks.append(reduce_lr)

                # Train model
                history = model.fit(
                    X_seq, y_seq,
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    validation_split=0.25,  # Increased validation split
                    callbacks=callbacks,
                    verbose=1
                )

                models['bilstm'] = model
                logger.info(f"BiLSTM training completed. Final val_loss: {min(history.history['val_loss']):.6f}")

            except Exception as e:
                logger.error(f"Error training BiLSTM: {e}")

        return models

    def _train_meta_model(self, predictions: np.ndarray, y_true: np.ndarray):
        """Train meta-model for ensemble stacking."""
        try:
            logger.info("Training meta-model for ensemble stacking")
            self.meta_model = LogisticRegression(random_state=42, max_iter=1000)
            self.meta_model.fit(predictions, y_true)
        except Exception as e:
            logger.error(f"Error training meta-model: {e}")

    def train_full_ensemble(self) -> Dict[str, float]:
        """
        Train the complete hybrid ensemble system.

        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Starting full ensemble training for {self.pair}")

        if self.price_data.empty:
            raise ValueError(f"No price data available for {self.pair}")

        # Engineer features
        feature_df = self._engineer_features(self.price_data)

        if len(feature_df) < self.lookback_window:
            raise ValueError(f"Insufficient data for training: {len(feature_df)} observations")

        # Prepare training data
        target_col = 'target_1d'  # 1-day ahead returns
        feature_cols = [col for col in feature_df.columns
                       if not col.startswith('target_') and col not in ['Close', 'date', 'Date']]

        self.feature_columns = feature_cols

        # Split data
        train_size = int(len(feature_df) * 0.8)
        train_df = feature_df.iloc[:train_size]
        val_df = feature_df.iloc[train_size:]

        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        X_val = val_df[feature_cols].values
        y_val = val_df[target_col].values

        # Scale features
        self.scalers['features'] = StandardScaler()
        X_train_scaled = self.scalers['features'].fit_transform(X_train)
        X_val_scaled = self.scalers['features'].transform(X_val)

        # Train base models
        logger.info("Training statistical models")
        self.models.update(self._train_statistical_models(train_df))

        logger.info("Training machine learning models")
        ml_models = self._train_ml_models(X_train_scaled, y_train)
        self.models.update(ml_models)

        logger.info("Training deep learning models")
        dl_models = self._train_dl_models(X_train_scaled, y_train)
        self.models.update(dl_models)

        # Generate predictions for meta-model training
        logger.info("Generating base model predictions for meta-learning")
        train_predictions = self._generate_base_predictions(train_df, X_train_scaled, single_point=False)
        val_predictions = self._generate_base_predictions(val_df, X_val_scaled, single_point=False)

        # Train meta-model
        if len(train_predictions) > 0:
            self._train_meta_model(train_predictions, y_train)

        # Calculate ensemble performance
        ensemble_predictions = self._generate_ensemble_predictions_batch(val_df, X_val_scaled)

        metrics = {}
        if len(ensemble_predictions) > 0:
            # For binary classification, threshold predictions at 0.5
            binary_predictions = (ensemble_predictions > 0.5).astype(int)
            metrics['accuracy'] = accuracy_score(y_val, binary_predictions)
            metrics['directional_accuracy'] = metrics['accuracy']  # Same for binary classification

            logger.info(f"Ensemble validation metrics: Accuracy={metrics['accuracy']:.6f}, "
                       f"Directional Accuracy={metrics['directional_accuracy']:.3f}")

        # Save models
        self._save_models()

        return metrics

    def _generate_base_predictions(self, df: pd.DataFrame, X_scaled: np.ndarray, single_point: bool = False) -> np.ndarray:
        """Generate predictions from all trained base models."""
        if single_point:
            return self._generate_single_predictions(df, X_scaled)
        else:
            return self._generate_batch_predictions(df, X_scaled)

    def _generate_single_predictions(self, df: pd.DataFrame, X_scaled: np.ndarray) -> np.ndarray:
        """Generate predictions for a single data point."""
        predictions = []

        # Use the same order as training to ensure consistency
        model_order = ['prophet', 'lightgbm', 'xgboost', 'random_forest', 'extra_trees', 'lstm', 'bilstm']

        for model_name in model_order:
            if model_name in self.models:
                try:
                    if model_name == 'prophet':
                        future = self.models['prophet'].make_future_dataframe(periods=1)
                        forecast = self.models['prophet'].predict(future)
                        pred = forecast['yhat'].iloc[-1] / df['Close'].iloc[-1] - 1
                    elif model_name in ['lightgbm', 'xgboost', 'random_forest', 'extra_trees']:
                        pred = self.models[model_name].predict(X_scaled[-1].reshape(1, -1))[0]
                    elif model_name in ['lstm', 'bilstm']:
                        sequence_length = 30
                        if len(X_scaled) >= sequence_length:
                            X_seq = X_scaled[-sequence_length:].reshape(1, sequence_length, -1)
                            pred = self.models[model_name].predict(X_seq, verbose=0)[0][0]
                        else:
                            pred = 0.0  # Fallback
                    predictions.append(pred)
                except Exception as e:
                    logger.warning(f"Error generating {model_name} prediction: {e}")
                    predictions.append(0.0)  # Consistent fallback
            else:
                predictions.append(0.0)  # Model not available

        return np.array(predictions).reshape(1, -1)

    def _generate_batch_predictions(self, df: pd.DataFrame, X_scaled: np.ndarray) -> np.ndarray:
        """Generate predictions for a batch of data points."""
        n_samples = len(X_scaled)
        predictions = []

        # Use the same order as training to ensure consistency
        model_order = ['prophet', 'lightgbm', 'xgboost', 'random_forest', 'extra_trees', 'lstm', 'bilstm']

        for model_name in model_order:
            if model_name in self.models:
                try:
                    if model_name == 'prophet':
                        # Prophet predictions (single point, repeat for all samples)
                        future = self.models['prophet'].make_future_dataframe(periods=1)
                        forecast = self.models['prophet'].predict(future)
                        prophet_pred = forecast['yhat'].iloc[-1] / df['Close'].iloc[-1] - 1
                        predictions.append(np.full(n_samples, prophet_pred))
                    elif model_name in ['lightgbm', 'xgboost', 'random_forest', 'extra_trees']:
                        preds = self.models[model_name].predict(X_scaled)
                        predictions.append(preds)
                    elif model_name in ['lstm', 'bilstm']:
                        # DL model predictions (sequence-based)
                        dl_preds = []
                        sequence_length = 30
                        for i in range(n_samples):
                            if i >= sequence_length:
                                X_seq = X_scaled[i-sequence_length:i].reshape(1, sequence_length, -1)
                                pred = self.models[model_name].predict(X_seq, verbose=0)[0][0]
                            else:
                                # For early samples, use the first available prediction
                                X_seq = X_scaled[:sequence_length].reshape(1, sequence_length, -1)
                                pred = self.models[model_name].predict(X_seq, verbose=0)[0][0]
                            dl_preds.append(pred)
                        predictions.append(np.array(dl_preds))
                except Exception as e:
                    logger.warning(f"Error generating {model_name} batch predictions: {e}")
                    predictions.append(np.zeros(n_samples))  # Fallback
            else:
                predictions.append(np.zeros(n_samples))  # Model not available

        # Stack predictions: shape should be (n_samples, n_models)
        if predictions:
            return np.column_stack(predictions)
        else:
            return np.array([])

    def _generate_ensemble_predictions_batch(self, df: pd.DataFrame, X_scaled: np.ndarray) -> np.ndarray:
        """Generate ensemble predictions for a batch using meta-model."""
        base_predictions = self._generate_base_predictions(df, X_scaled, single_point=False)

        if len(base_predictions) == 0 or self.meta_model is None:
            return np.array([])

        try:
            return self.meta_model.predict(base_predictions)
        except Exception as e:
            logger.error(f"Error generating ensemble batch predictions: {e}")
            return np.array([])

    def _generate_ensemble_predictions(self, df: pd.DataFrame, X_scaled: np.ndarray) -> np.ndarray:
        """Generate ensemble predictions for single data point using meta-model."""
        base_predictions = self._generate_base_predictions(df, X_scaled, single_point=True)

        if len(base_predictions) == 0 or self.meta_model is None:
            return np.array([])

        try:
            return self.meta_model.predict(base_predictions)
        except Exception as e:
            logger.error(f"Error generating ensemble predictions: {e}")
            return np.array([])

    def generate_forecast(self, days_ahead: int = 5) -> Dict:
        """
        Generate multi-day ahead forecast.

        Args:
            days_ahead: Number of days to forecast

        Returns:
            Dictionary with forecast results
        """
        if not self.models:
            raise ValueError("Models not trained. Call train_full_ensemble() first.")

        # Prepare latest data
        feature_df = self._engineer_features(self.price_data)
        if len(feature_df) < 50:
            raise ValueError("Insufficient data for forecasting")

        latest_features = feature_df[self.feature_columns].iloc[-1:].values
        latest_features_scaled = self.scalers['features'].transform(latest_features)

        # Generate forecast
        forecast = {
            'pair': self.pair,
            'timestamp': datetime.now().isoformat(),
            'current_price': float(feature_df['Close'].iloc[-1]),
            'forecasts': []
        }

        # For simplicity, use the 1-day model repeatedly
        # In production, you'd want proper multi-step forecasting
        current_price = forecast['current_price']

        for day in range(1, days_ahead + 1):
            base_predictions = self._generate_single_predictions(
                feature_df.iloc[-1:], latest_features_scaled
            )

            if len(base_predictions) > 0 and self.meta_model is not None:
                prediction = self.meta_model.predict(base_predictions)
                predicted_return = prediction[0] if hasattr(prediction, '__len__') else prediction
            else:
                predicted_return = 0.0  # Fallback

            predicted_price = current_price * (1 + predicted_return)

            forecast['forecasts'].append({
                'day': day,
                'predicted_return': float(predicted_return),
                'predicted_price': float(predicted_price),
                'confidence': self._estimate_confidence(np.array([predicted_return]))
            })

            current_price = predicted_price

        return forecast

    def _estimate_confidence(self, prediction: np.ndarray) -> float:
        """Estimate prediction confidence based on model agreement."""
        if len(prediction) < 2:
            return 0.5

        # Simple confidence based on prediction variance
        # Lower variance = higher confidence
        variance = np.var(prediction)
        confidence = 1 / (1 + variance)  # Scale to 0-1 range

        return float(confidence)

    def get_trading_signal(self) -> Dict:
        """
        Generate trading signal with confidence score.

        Returns:
            Dictionary with signal information
        """
        try:
            forecast = self.generate_forecast(days_ahead=1)

            if not forecast['forecasts']:
                return {
                    'pair': self.pair,
                    'signal': 'no_signal',
                    'confidence': 0.0,
                    'reason': 'No forecast available'
                }

            day1_forecast = forecast['forecasts'][0]
            predicted_return = day1_forecast['predicted_return']
            confidence = day1_forecast['confidence']

            # Determine signal based on prediction and confidence
            if abs(predicted_return) < 0.001:  # Less than 0.1% move
                signal = 'hold'
            elif predicted_return > 0.002 and confidence > 0.6:  # >0.2% bullish with good confidence
                signal = 'bullish'
            elif predicted_return < -0.002 and confidence > 0.6:  # <-0.2% bearish with good confidence
                signal = 'bearish'
            else:
                signal = 'no_signal'

            return {
                'pair': self.pair,
                'signal': signal,
                'confidence': confidence,
                'predicted_return': predicted_return,
                'current_price': forecast['current_price'],
                'timestamp': forecast['timestamp']
            }

        except Exception as e:
            logger.error(f"Error generating trading signal: {e}")
            return {
                'pair': self.pair,
                'signal': 'error',
                'confidence': 0.0,
                'reason': str(e)
            }

    def _save_models(self):
        """Save trained models to disk."""
        try:
            model_file = self.models_dir / f"{self.pair}_ensemble.joblib"
            joblib.dump({
                'models': self.models,
                'scalers': self.scalers,
                'meta_model': self.meta_model,
                'feature_columns': self.feature_columns,
                'model_configs': self.model_configs
            }, model_file)
            logger.info(f"Models saved to {model_file}")
        except Exception as e:
            logger.error(f"Error saving models: {e}")

    def load_models(self):
        """Load trained models from disk."""
        try:
            model_file = self.models_dir / f"{self.pair}_ensemble.joblib"
            if model_file.exists():
                data = joblib.load(model_file)
                self.models = data['models']
                self.scalers = data['scalers']
                self.meta_model = data['meta_model']
                self.feature_columns = data['feature_columns']
                self.model_configs = data.get('model_configs', self._get_model_configs())
                logger.info(f"Models loaded from {model_file}")
                return True
            else:
                logger.warning(f"Model file not found: {model_file}")
                return False
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False

    def backtest_ensemble(self, start_date: str = None, end_date: str = None) -> Dict:
        """
        Backtest the ensemble model performance.

        Args:
            start_date: Start date for backtesting (YYYY-MM-DD)
            end_date: End date for backtesting (YYYY-MM-DD)

        Returns:
            Dictionary with backtesting results
        """
        if not self.models:
            raise ValueError("Models not trained. Call train_full_ensemble() first.")

        # Prepare data
        feature_df = self._engineer_features(self.price_data)

        if start_date:
            feature_df = feature_df[feature_df.index >= start_date]
        if end_date:
            feature_df = feature_df[feature_df.index <= end_date]

        if len(feature_df) < 100:
            raise ValueError("Insufficient data for backtesting")

        # Generate predictions
        predictions = []
        actuals = []

        for i in range(len(feature_df) - 1):
            current_data = feature_df.iloc[:i+1]
            if len(current_data) < 50:
                continue

            try:
                X = current_data[self.feature_columns].iloc[-1:].values
                X_scaled = self.scalers['features'].transform(X)

                pred = self._generate_ensemble_predictions(current_data.iloc[-1:], X_scaled)
                if len(pred) > 0:
                    predictions.append(pred[0])
                    actuals.append(feature_df['target_1d'].iloc[i])
            except Exception as e:
                logger.warning(f"Error in backtest prediction {i}: {e}")

        if not predictions:
            return {'error': 'No predictions generated'}

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # Calculate metrics
        results = {
            'total_trades': len(predictions),
            'accuracy': float(accuracy_score(actuals, (predictions > 0.5).astype(int))),
            'directional_accuracy': float(accuracy_score(actuals, (predictions > 0.5).astype(int))),  # Same for binary
            'profit_factor': self._calculate_profit_factor(predictions, actuals),
            'max_drawdown': self._calculate_max_drawdown(predictions, actuals),
            'sharpe_ratio': self._calculate_sharpe_ratio(predictions, actuals)
        }

        logger.info(f"Backtest results: {results}")
        return results

    def _calculate_profit_factor(self, predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        returns = predictions * actuals  # Simulated P&L
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())

        return float(gross_profit / gross_loss) if gross_loss > 0 else float('inf')

    def _calculate_max_drawdown(self, predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        returns = predictions * actuals
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        return float(np.min(drawdown))

    def _calculate_sharpe_ratio(self, predictions: np.ndarray, actuals: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        returns = predictions * actuals
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate

        if np.std(excess_returns) == 0:
            return 0.0

        return float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252))


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description='Hybrid Price Forecasting Ensemble')
    parser.add_argument('--pair', required=True, help='Currency pair (EURUSD, XAUUSD)')
    parser.add_argument('--train', action='store_true', help='Train the ensemble')
    parser.add_argument('--forecast', type=int, default=5, help='Generate forecast for N days')
    parser.add_argument('--signal', action='store_true', help='Generate trading signal')
    parser.add_argument('--backtest', action='store_true', help='Run backtesting')
    parser.add_argument('--start-date', help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='Backtest end date (YYYY-MM-DD)')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--models-dir', default='models', help='Models directory')

    args = parser.parse_args()

    # Initialize ensemble
    ensemble = HybridPriceForecastingEnsemble(
        pair=args.pair,
        data_dir=args.data_dir,
        models_dir=args.models_dir
    )

    if args.train:
        # Train ensemble
        metrics = ensemble.train_full_ensemble()
        print(f"Training completed. Metrics: {metrics}")

    elif args.backtest:
        # Load models and run backtest
        if ensemble.load_models():
            results = ensemble.backtest_ensemble(args.start_date, args.end_date)
            print(f"Backtest Results: {results}")
        else:
            print("Error: Models not found. Run --train first.")

    elif args.forecast:
        # Load models and generate forecast
        if ensemble.load_models():
            forecast = ensemble.generate_forecast(args.forecast)
            print(f"Forecast: {forecast}")
        else:
            print("Error: Models not found. Run --train first.")

    elif args.signal:
        # Load models and generate signal
        if ensemble.load_models():
            signal = ensemble.get_trading_signal()
            print(f"Signal: {signal}")
        else:
            print("Error: Models not found. Run --train first.")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()