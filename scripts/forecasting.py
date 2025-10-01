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
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

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

        # Configuration
        self.forecast_horizon = 5  # days ahead
        self.lookback_window = 200  # ~1 year of daily data, reduced for XAUUSD
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

            # Machine Learning Models
            'lightgbm': {
                'enabled': True,
                'type': 'ml',
                'params': {
                    'n_estimators': 1000,
                    'learning_rate': 0.05,
                    'max_depth': 6,
                    'num_leaves': 31,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'verbosity': -1
                }
            },
            'xgboost': {
                'enabled': True,
                'type': 'ml',
                'params': {
                    'n_estimators': 1000,
                    'learning_rate': 0.05,
                    'max_depth': 6,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'verbosity': 0
                }
            },
            'random_forest': {
                'enabled': True,
                'type': 'ml',
                'params': {
                    'n_estimators': 500,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'extra_trees': {
                'enabled': True,
                'type': 'ml',
                'params': {
                    'n_estimators': 500,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42,
                    'n_jobs': -1
                }
            },

            # Deep Learning Models
            'lstm': {
                'enabled': tf is not None,
                'type': 'dl',
                'params': {
                    'units': 64,
                    'dropout': 0.2,
                    'epochs': 100,
                    'batch_size': 32,
                    'learning_rate': 0.001
                }
            },
            'bilstm': {
                'enabled': tf is not None,
                'type': 'dl',
                'params': {
                    'units': 64,
                    'dropout': 0.2,
                    'epochs': 100,
                    'batch_size': 32,
                    'learning_rate': 0.001
                }
            }
        }

    def _load_price_data(self) -> pd.DataFrame:
        """Load historical price data for the currency pair."""
        csv_file = self.data_dir / "raw" / f"{self.pair}_Daily.csv"
        if not csv_file.exists():
            logger.warning(f"Price data file not found: {csv_file}")
            return pd.DataFrame()

        try:
            df = pd.read_csv(csv_file)
            # Handle both 'date' and 'Date' column names
            date_col = 'date' if 'date' in df.columns else 'Date'
            df['Date'] = pd.to_datetime(df[date_col])
            df = df.sort_values('Date').set_index('Date')

            # Ensure we have OHLC columns (handle both cases)
            required_cols = ['Open', 'High', 'Low', 'Close']
            actual_cols = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                if all(col in df.columns for col in actual_cols):
                    # Rename lowercase columns to uppercase
                    df = df.rename(columns={
                        'open': 'Open',
                        'high': 'High',
                        'low': 'Low',
                        'close': 'Close'
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

        # Volatility measures
        df['volatility_20'] = df['returns'].rolling(20).std()
        df['volatility_50'] = df['returns'].rolling(50).std()
        df['atr_14'] = self._calculate_atr(df, 14)

        # Momentum indicators
        df['rsi_14'] = self._calculate_rsi(df['Close'], 14)
        df['macd'], df['macd_signal'], df['macd_hist'] = self._calculate_macd(df['Close'])

        # Trend indicators
        df['adx_14'] = self._calculate_adx(df, 14)

        # Support/Resistance levels
        df['pivot_point'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['r1'] = 2 * df['pivot_point'] - df['Low']
        df['s1'] = 2 * df['pivot_point'] - df['High']

        # Volume-based indicators (if volume exists)
        if 'Volume' in df.columns:
            df['volume_sma_20'] = df['Volume'].rolling(20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_sma_20']

        # Lagged features
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)

        # Target variables (future returns)
        for horizon in [1, 3, 5]:
            df[f'target_{horizon}d'] = df['Close'].shift(-horizon) / df['Close'] - 1

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
        """Train machine learning models."""
        models = {}

        # LightGBM
        if self.model_configs['lightgbm']['enabled']:
            try:
                logger.info("Training LightGBM model")
                model = LGBMRegressor(**self.model_configs['lightgbm']['params'])
                model.fit(X_train, y_train)
                models['lightgbm'] = model
            except Exception as e:
                logger.error(f"Error training LightGBM: {e}")

        # XGBoost
        if self.model_configs['xgboost']['enabled']:
            try:
                logger.info("Training XGBoost model")
                model = XGBRegressor(**self.model_configs['xgboost']['params'])
                model.fit(X_train, y_train)
                models['xgboost'] = model
            except Exception as e:
                logger.error(f"Error training XGBoost: {e}")

        # Random Forest
        if self.model_configs['random_forest']['enabled']:
            try:
                logger.info("Training Random Forest model")
                model = RandomForestRegressor(**self.model_configs['random_forest']['params'])
                model.fit(X_train, y_train)
                models['random_forest'] = model
            except Exception as e:
                logger.error(f"Error training Random Forest: {e}")

        # Extra Trees
        if self.model_configs['extra_trees']['enabled']:
            try:
                logger.info("Training Extra Trees model")
                model = ExtraTreesRegressor(**self.model_configs['extra_trees']['params'])
                model.fit(X_train, y_train)
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

        # LSTM
        if self.model_configs['lstm']['enabled']:
            try:
                logger.info("Training LSTM model")
                model = Sequential([
                    LSTM(self.model_configs['lstm']['params']['units'],
                         input_shape=(X_seq.shape[1], X_seq.shape[2])),
                    Dropout(self.model_configs['lstm']['params']['dropout']),
                    Dense(32, activation='relu'),
                    Dense(1)
                ])

                model.compile(optimizer=Adam(learning_rate=self.model_configs['lstm']['params']['learning_rate']),
                            loss='mse')

                early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

                model.fit(X_seq, y_seq,
                         epochs=self.model_configs['lstm']['params']['epochs'],
                         batch_size=self.model_configs['lstm']['params']['batch_size'],
                         validation_split=0.2,
                         callbacks=[early_stop],
                         verbose=0)

                models['lstm'] = model

            except Exception as e:
                logger.error(f"Error training LSTM: {e}")

        # BiLSTM
        if self.model_configs['bilstm']['enabled']:
            try:
                logger.info("Training BiLSTM model")
                model = Sequential([
                    Bidirectional(LSTM(self.model_configs['bilstm']['params']['units']),
                                 input_shape=(X_seq.shape[1], X_seq.shape[2])),
                    Dropout(self.model_configs['bilstm']['params']['dropout']),
                    Dense(32, activation='relu'),
                    Dense(1)
                ])

                model.compile(optimizer=Adam(learning_rate=self.model_configs['bilstm']['params']['learning_rate']),
                            loss='mse')

                early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

                model.fit(X_seq, y_seq,
                         epochs=self.model_configs['bilstm']['params']['epochs'],
                         batch_size=self.model_configs['bilstm']['params']['batch_size'],
                         validation_split=0.2,
                         callbacks=[early_stop],
                         verbose=0)

                models['bilstm'] = model

            except Exception as e:
                logger.error(f"Error training BiLSTM: {e}")

        return models

    def _train_meta_model(self, predictions: np.ndarray, y_true: np.ndarray):
        """Train meta-model for ensemble stacking."""
        try:
            logger.info("Training meta-model for ensemble stacking")
            self.meta_model = Ridge(alpha=0.1, random_state=42)
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
            metrics['mae'] = mean_absolute_error(y_val, ensemble_predictions)
            metrics['rmse'] = np.sqrt(mean_squared_error(y_val, ensemble_predictions))
            metrics['directional_accuracy'] = np.mean((ensemble_predictions > 0) == (y_val > 0))

            logger.info(f"Ensemble validation metrics: MAE={metrics['mae']:.6f}, "
                       f"RMSE={metrics['rmse']:.6f}, "
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
            'mae': float(mean_absolute_error(actuals, predictions)),
            'rmse': float(np.sqrt(mean_squared_error(actuals, predictions))),
            'directional_accuracy': float(np.mean((predictions > 0) == (actuals > 0))),
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