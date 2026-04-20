"""
tests/test_signal_engine.py

Tests for signals/signal_engine.py:
  - Train/predict contract
  - Determinism: same data → same signal
  - Predict output schema
  - SL/TP are actual prices (not pips)
  - models_exist reflects disk state
  - No model → FileNotFoundError
"""
import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from trading.features import FEATURE_COLS
from signals.signal_engine import SignalEngine, _model_paths


# ── Synthetic data helpers ─────────────────────────────────────────────────────

def _make_ohlcv(n=400, pair='EURUSD', seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = 1.1000 if pair == 'EURUSD' else 2000.0
    step = 0.0005 if pair == 'EURUSD' else 2.0
    close = base + np.cumsum(rng.normal(0, step, n))
    spread = rng.uniform(step * 0.2, step * 2, n)
    high  = close + spread
    low   = close - spread
    open_ = close + rng.normal(0, step * 0.4, n)
    idx   = pd.date_range('2022-01-01', periods=n, freq='h')
    return pd.DataFrame({'open': open_, 'high': high, 'low': low,
                          'close': close, 'volume': rng.integers(1000, 5000, n)},
                         index=idx)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def tmp_model_dir(tmp_path_factory):
    """Temporary models/ directory; patch MODEL_DIR so artifacts don't pollute repo."""
    return tmp_path_factory.mktemp('models')


@pytest.fixture(scope='module')
def trained_engine(tmp_model_dir):
    """Train a SignalEngine on synthetic EURUSD data once for the module."""
    import signals.signal_engine as se_module
    original_dir = se_module.MODEL_DIR

    se_module.MODEL_DIR = tmp_model_dir
    engine = SignalEngine()
    # Monkey-patch the module-level MODEL_DIR so _model_paths resolves correctly
    with patch('signals.signal_engine.MODEL_DIR', tmp_model_dir):
        df = _make_ohlcv(n=400)
        meta = engine.train('EURUSD', df)

    se_module.MODEL_DIR = original_dir
    return engine, meta, tmp_model_dir, df


# ── Training contract ──────────────────────────────────────────────────────────

class TestTrain:
    def test_train_returns_meta_dict(self, tmp_path):
        with patch('signals.signal_engine.MODEL_DIR', tmp_path):
            engine = SignalEngine()
            meta = engine.train('EURUSD', _make_ohlcv(n=400))
        assert isinstance(meta, dict)

    def test_meta_has_required_keys(self, tmp_path):
        required = {'pair', 'features', 'n_features', 'n_samples',
                    'cv_accuracy', 'cv_auc', 'trained_at'}
        with patch('signals.signal_engine.MODEL_DIR', tmp_path):
            engine = SignalEngine()
            meta = engine.train('EURUSD', _make_ohlcv(n=400))
        assert required.issubset(meta.keys())

    def test_meta_features_equals_feature_cols(self, tmp_path):
        with patch('signals.signal_engine.MODEL_DIR', tmp_path):
            engine = SignalEngine()
            meta = engine.train('EURUSD', _make_ohlcv(n=400))
        assert meta['features'] == FEATURE_COLS

    def test_artifacts_saved_to_disk(self, tmp_path):
        with patch('signals.signal_engine.MODEL_DIR', tmp_path):
            engine = SignalEngine()
            engine.train('EURUSD', _make_ohlcv(n=400))
            paths = _model_paths('EURUSD')
        for name, path in paths.items():
            # Reconstruct path under tmp_path
            p = tmp_path / path.name
            assert p.exists(), f'Missing artifact: {name} at {p}'

    def test_meta_json_readable(self, tmp_path):
        with patch('signals.signal_engine.MODEL_DIR', tmp_path):
            engine = SignalEngine()
            engine.train('EURUSD', _make_ohlcv(n=400))
        meta_path = tmp_path / 'EURUSD_meta.json'
        assert meta_path.exists()
        data = json.loads(meta_path.read_text())
        assert data['pair'] == 'EURUSD'

    def test_cv_accuracy_reasonable(self, tmp_path):
        with patch('signals.signal_engine.MODEL_DIR', tmp_path):
            engine = SignalEngine()
            meta = engine.train('EURUSD', _make_ohlcv(n=400))
        # CV accuracy should be between 40% and 80% for any reasonable dataset
        assert 0.40 <= meta['cv_accuracy'] <= 0.80

    def test_insufficient_data_raises(self, tmp_path):
        with patch('signals.signal_engine.MODEL_DIR', tmp_path):
            engine = SignalEngine()
            with pytest.raises(ValueError, match='usable rows'):
                engine.train('EURUSD', _make_ohlcv(n=20))

    def test_xauusd_trains_without_error(self, tmp_path):
        with patch('signals.signal_engine.MODEL_DIR', tmp_path):
            engine = SignalEngine()
            meta = engine.train('XAUUSD', _make_ohlcv(n=400, pair='XAUUSD'))
        assert meta['pair'] == 'XAUUSD'


# ── Predict contract ───────────────────────────────────────────────────────────

class TestPredict:
    def test_predict_returns_dict(self, tmp_path):
        with patch('signals.signal_engine.MODEL_DIR', tmp_path):
            engine = SignalEngine()
            df = _make_ohlcv(n=400)
            engine.train('EURUSD', df)
            result = engine.predict('EURUSD', df)
        assert isinstance(result, dict)

    def test_predict_required_keys(self, tmp_path):
        required = {'pair', 'signal', 'probability', 'confidence',
                    'entry', 'stop_loss', 'take_profit', 'risk_reward',
                    'atr', 'date', 'model_info'}
        with patch('signals.signal_engine.MODEL_DIR', tmp_path):
            engine = SignalEngine()
            df = _make_ohlcv(n=400)
            engine.train('EURUSD', df)
            result = engine.predict('EURUSD', df)
        assert required.issubset(result.keys())

    def test_signal_valid_value(self, tmp_path):
        with patch('signals.signal_engine.MODEL_DIR', tmp_path):
            engine = SignalEngine()
            df = _make_ohlcv(n=400)
            engine.train('EURUSD', df)
            result = engine.predict('EURUSD', df)
        assert result['signal'] in ('bullish', 'bearish', 'no_signal')

    def test_probability_in_range(self, tmp_path):
        with patch('signals.signal_engine.MODEL_DIR', tmp_path):
            engine = SignalEngine()
            df = _make_ohlcv(n=400)
            engine.train('EURUSD', df)
            result = engine.predict('EURUSD', df)
        assert 0.0 <= result['probability'] <= 1.0

    def test_confidence_in_range(self, tmp_path):
        with patch('signals.signal_engine.MODEL_DIR', tmp_path):
            engine = SignalEngine()
            df = _make_ohlcv(n=400)
            engine.train('EURUSD', df)
            result = engine.predict('EURUSD', df)
        assert 0.0 <= result['confidence'] <= 1.0

    def test_sl_tp_are_price_levels_not_pips(self, tmp_path):
        """SL/TP must be close to entry price, not tiny pip values."""
        with patch('signals.signal_engine.MODEL_DIR', tmp_path):
            engine = SignalEngine()
            df = _make_ohlcv(n=400)
            engine.train('EURUSD', df)
            result = engine.predict('EURUSD', df)
        entry = result['entry']
        sl = result['stop_loss']
        tp = result['take_profit']
        # For EURUSD ~1.10, SL/TP should be within 5% of entry
        assert abs(sl - entry) / entry < 0.05, f'SL={sl} seems like pips, not price'
        assert abs(tp - entry) / entry < 0.05, f'TP={tp} seems like pips, not price'

    def test_bullish_sl_below_entry(self, tmp_path):
        """For bullish signals, SL must be below entry and TP above."""
        with patch('signals.signal_engine.MODEL_DIR', tmp_path):
            engine = SignalEngine()
            # Build a strongly trending up dataset to force a bullish signal
            n = 400
            close = np.linspace(1.05, 1.15, n)
            idx = pd.date_range('2022-01-01', periods=n, freq='h')
            df = pd.DataFrame({
                'open':  close - 0.0001,
                'high':  close + 0.0003,
                'low':   close - 0.0003,
                'close': close,
            }, index=idx)
            engine.train('EURUSD', df)
            result = engine.predict('EURUSD', df)
        if result['signal'] == 'bullish':
            assert result['stop_loss'] < result['entry']
            assert result['take_profit'] > result['entry']

    def test_bearish_sl_above_entry(self, tmp_path):
        """For bearish signals, SL must be above entry and TP below."""
        with patch('signals.signal_engine.MODEL_DIR', tmp_path):
            engine = SignalEngine()
            n = 400
            close = np.linspace(1.15, 1.05, n)
            idx = pd.date_range('2022-01-01', periods=n, freq='h')
            df = pd.DataFrame({
                'open':  close + 0.0001,
                'high':  close + 0.0003,
                'low':   close - 0.0003,
                'close': close,
            }, index=idx)
            engine.train('EURUSD', df)
            result = engine.predict('EURUSD', df)
        if result['signal'] == 'bearish':
            assert result['stop_loss'] > result['entry']
            assert result['take_profit'] < result['entry']

    def test_risk_reward_positive(self, tmp_path):
        with patch('signals.signal_engine.MODEL_DIR', tmp_path):
            engine = SignalEngine()
            df = _make_ohlcv(n=400)
            engine.train('EURUSD', df)
            result = engine.predict('EURUSD', df)
        assert result['risk_reward'] >= 0.0

    def test_predict_deterministic(self, tmp_path):
        """Same model + same data must always return same signal."""
        with patch('signals.signal_engine.MODEL_DIR', tmp_path):
            engine = SignalEngine()
            df = _make_ohlcv(n=400)
            engine.train('EURUSD', df)
            r1 = engine.predict('EURUSD', df)
            r2 = engine.predict('EURUSD', df)
        assert r1['signal'] == r2['signal']
        assert r1['probability'] == r2['probability']
        assert r1['entry'] == r2['entry']

    def test_predict_no_model_raises(self, tmp_path):
        """Predicting without training must raise FileNotFoundError."""
        with patch('signals.signal_engine.MODEL_DIR', tmp_path):
            engine = SignalEngine()
            df = _make_ohlcv(n=400)
            with pytest.raises(FileNotFoundError):
                engine.predict('EURUSD', df)


# ── models_exist ──────────────────────────────────────────────────────────────

class TestModelsExist:
    def test_returns_false_before_training(self, tmp_path):
        with patch('signals.signal_engine.MODEL_DIR', tmp_path):
            engine = SignalEngine()
            assert not engine.models_exist('EURUSD')

    def test_returns_true_after_training(self, tmp_path):
        with patch('signals.signal_engine.MODEL_DIR', tmp_path):
            engine = SignalEngine()
            engine.train('EURUSD', _make_ohlcv(n=400))
            assert engine.models_exist('EURUSD')

    def test_returns_false_if_one_artifact_missing(self, tmp_path):
        with patch('signals.signal_engine.MODEL_DIR', tmp_path):
            engine = SignalEngine()
            engine.train('EURUSD', _make_ohlcv(n=400))
            (tmp_path / 'EURUSD_meta.json').unlink()
            # Must invalidate cache since we deleted a file
            engine.invalidate_cache('EURUSD')
            assert not engine.models_exist('EURUSD')
