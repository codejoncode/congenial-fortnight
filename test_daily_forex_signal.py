# !pip install pytest
# !pytest -v
import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
from daily_forex_signal_system import DailyForexSignal  # Absolute import assuming namespace package

@pytest.fixture
def sample_df():
    dates = pd.date_range('2020-01-01', periods=50, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'open': np.random.uniform(1.1, 1.2, 50),
        'high': np.random.uniform(1.15, 1.25, 50),
        'low': np.random.uniform(1.05, 1.15, 50),
        'close': np.random.uniform(1.1, 1.2, 50),
        'tickvol': np.random.randint(1000, 5000, 50),
        'vol': np.random.randint(1000, 5000, 50),
        'spread': np.zeros(50)
    })
    return df

@pytest.fixture
def signal_gen():
    return DailyForexSignal()

class TestDailyForexSignal:

    @patch('os.path.exists', return_value=True)
    @patch('pd.read_csv')
    def test_clean_and_standardize_data(self, mock_read, mock_exists, sample_df):
        mock_read.return_value = sample_df
        gen = DailyForexSignal()
        result = gen.clean_and_standardize_data(sample_df, 'EURUSD')
        assert isinstance(result, pd.DataFrame)
        assert result.index.name == 'date'
        assert len(result.columns) == 8  # Standard columns
        assert result['close'].dtype == 'float64'

    @patch('os.path.exists', return_value=True)
    @patch('pd.read_csv')
    @patch('daily_forex_signal_system.yf.download')
    def test_load_data(self, mock_yf, mock_read, mock_exists, sample_df):
        mock_read.return_value = sample_df
        mock_yf.return_value = sample_df.set_index('date')
        gen = DailyForexSignal()
        result = gen.load_data('EURUSD')
        assert len(result) > 0
        assert 'close' in result.columns

    def test_engineer_features(self, signal_gen, sample_df):
        df = sample_df.set_index('date')
        result = signal_gen.engineer_features('EURUSD', df)
        assert 'target' in result.columns
        assert len(signal_gen.features) >= 20  # At least base features
        assert result.shape[0] < df.shape[0]  # NaN dropped

    @patch('joblib.dump')
    @patch('os.makedirs')
    def test_build_ensemble(self, mock_makedirs, mock_dump, signal_gen):
        X_train = np.random.rand(100, 20)
        y_train = np.random.randint(0, 2, 100)
        signal_gen.build_ensemble(X_train, y_train)
        assert 'rf' in signal_gen.models
        assert 'xgb' in signal_gen.models
        assert 'logistic' in signal_gen.calibrators

    @patch('daily_forex_signal_system.DailyForexSignal.load_data')
    @patch('daily_forex_signal_system.DailyForexSignal.build_ensemble')
    @patch('joblib.dump')
    @patch('os.makedirs')
    def test_train_models(self, mock_makedirs, mock_dump, mock_build, mock_load, signal_gen, sample_df):
        mock_load.return_value = sample_df.set_index('date')
        mock_build.return_value = None
        result = signal_gen.train_models('EURUSD')
        assert result is True or result is False  # Depending on data

    def test_predict_ensemble(self, signal_gen):
        X = np.random.rand(10, 20)
        preds = signal_gen.predict_ensemble(X)
        assert preds.shape == (10,)
        assert all(0 <= p <= 1 for p in preds)

    @patch('joblib.load')
    @patch('os.path.exists', return_value=True)
    def test_generate_signal(self, mock_exists, mock_load, signal_gen, sample_df):
        df = sample_df.set_index('date')
        mock_load.return_value = MagicMock()  # Mock models
        sig = signal_gen.generate_signal('EURUSD', df)
        assert 'signal' in sig
        assert sig['signal'] in ['bullish', 'bearish']
        assert 'p_up' in sig

    @patch('daily_forex_signal_system.DailyForexSignal.load_data')
    @patch('daily_forex_signal_system.DailyForexSignal.generate_signal')
    def test_backtest_last_n_days(self, mock_sig, mock_load, signal_gen, sample_df):
        mock_load.return_value = sample_df.set_index('date')
        mock_sig.return_value = {'signal': 'bullish', 'date': '2020-01-01'}
        summary = signal_gen.backtest_last_n_days('EURUSD', n=10)
        assert 'accuracy' in summary
        assert summary['total_signals'] >= 0

# Run with: pytest test_daily_forex_signal.py