"""
tests/test_health.py — Unit tests for signals/health.py

Covers:
  - _file_age_minutes(): None for missing, positive for existing
  - _count_rows(): 0 for missing, correct for real CSV
  - get_system_health(): required keys, overall logic (RED/YELLOW/GREEN)
"""

import json
import time
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import django
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'forex_signal.settings')

from signals.health import (
    _file_age_minutes,
    _count_rows,
    _data_health,
    _model_health,
    get_system_health,
    DATA_DIR,
    MODELS_DIR,
)


# ── _file_age_minutes ─────────────────────────────────────────────────────────

class TestFileAgeMinutes:
    def test_missing_file_returns_none(self, tmp_path):
        missing = tmp_path / 'nonexistent.csv'
        assert _file_age_minutes(missing) is None

    def test_existing_file_returns_positive_float(self, tmp_path):
        f = tmp_path / 'test.csv'
        f.write_text('header\nrow1\n')
        result = _file_age_minutes(f)
        assert result is not None
        assert result >= 0.0

    def test_just_created_file_is_very_young(self, tmp_path):
        f = tmp_path / 'fresh.csv'
        f.write_text('data')
        result = _file_age_minutes(f)
        # Should be well under 1 minute old
        assert result < 1.0

    def test_returns_float(self, tmp_path):
        f = tmp_path / 'test.csv'
        f.write_text('x')
        result = _file_age_minutes(f)
        assert isinstance(result, float)


# ── _count_rows ───────────────────────────────────────────────────────────────

class TestCountRows:
    def test_missing_file_returns_zero(self, tmp_path):
        missing = tmp_path / 'nope.csv'
        assert _count_rows(missing) == 0

    def test_empty_file_returns_zero(self, tmp_path):
        f = tmp_path / 'empty.csv'
        f.write_text('')
        assert _count_rows(f) == 0

    def test_header_only_returns_zero(self, tmp_path):
        f = tmp_path / 'header.csv'
        f.write_text('col1,col2,col3\n')
        assert _count_rows(f) == 0

    def test_one_data_row(self, tmp_path):
        f = tmp_path / 'one.csv'
        f.write_text('header\nrow1\n')
        assert _count_rows(f) == 1

    def test_multiple_rows(self, tmp_path):
        f = tmp_path / 'multi.csv'
        lines = ['header'] + [f'row{i}' for i in range(10)]
        f.write_text('\n'.join(lines) + '\n')
        assert _count_rows(f) == 10

    def test_returns_int(self, tmp_path):
        f = tmp_path / 'test.csv'
        f.write_text('h\na\nb\n')
        assert isinstance(_count_rows(f), int)


# ── _data_health ──────────────────────────────────────────────────────────────

class TestDataHealth:
    def test_returns_dict(self):
        result = _data_health()
        assert isinstance(result, dict)

    def test_has_eurusd_entry(self):
        result = _data_health()
        assert any('EURUSD' in k for k in result)

    def test_has_xauusd_entry(self):
        result = _data_health()
        assert any('XAUUSD' in k for k in result)

    def test_each_entry_has_required_keys(self):
        result = _data_health()
        for key, val in result.items():
            assert 'exists' in val
            assert 'rows' in val
            assert 'fresh' in val

    def test_missing_data_dir_returns_exists_false(self, tmp_path):
        with patch('signals.health.DATA_DIR', tmp_path / 'no_data'):
            result = _data_health()
            for val in result.values():
                assert val['exists'] is False


# ── _model_health ─────────────────────────────────────────────────────────────

class TestModelHealth:
    def test_returns_dict(self):
        result = _model_health()
        assert isinstance(result, dict)

    def test_has_eurusd_and_xauusd(self):
        result = _model_health()
        assert 'EURUSD' in result
        assert 'XAUUSD' in result

    def test_each_entry_has_ready_key(self):
        result = _model_health()
        for pair, val in result.items():
            assert 'ready' in val

    def test_missing_models_returns_not_ready(self, tmp_path):
        with patch('signals.health.MODELS_DIR', tmp_path / 'no_models'):
            result = _model_health()
            for pair, val in result.items():
                assert val['ready'] is False

    def test_complete_models_returns_ready(self, tmp_path):
        # Create all 4 artifacts for EURUSD
        for name in ['EURUSD_rf.joblib', 'EURUSD_xgb.joblib', 'EURUSD_scaler.joblib']:
            (tmp_path / name).write_bytes(b'fake')
        meta = {'trained_at': '2024-01-01T00:00:00', 'cv_accuracy': 0.65}
        (tmp_path / 'EURUSD_meta.json').write_text(json.dumps(meta))

        with patch('signals.health.MODELS_DIR', tmp_path):
            result = _model_health()
            # EURUSD should be ready (XAUUSD still missing → not ready)
            assert result['EURUSD']['ready'] is True
            assert result['EURUSD']['cv_accuracy'] == pytest.approx(0.65)

    def test_corrupt_meta_does_not_crash(self, tmp_path):
        for name in ['EURUSD_rf.joblib', 'EURUSD_xgb.joblib', 'EURUSD_scaler.joblib']:
            (tmp_path / name).write_bytes(b'fake')
        (tmp_path / 'EURUSD_meta.json').write_text('NOT JSON {{{')

        with patch('signals.health.MODELS_DIR', tmp_path):
            result = _model_health()
            # Should not raise; ready=True (files exist), cv_accuracy=None
            assert result['EURUSD']['ready'] is True
            assert result['EURUSD']['cv_accuracy'] is None


# ── get_system_health ─────────────────────────────────────────────────────────

class TestGetSystemHealth:
    def _call(self, mock_signals=None, mock_positions=None):
        if mock_signals is None:
            mock_signals = {'today_count': 0, 'today_pairs': [], 'last_generated': None, 'total_in_db': 0}
        if mock_positions is None:
            mock_positions = {'open_count': 0, 'total_pnl': 0.0}

        with patch('signals.health._signals_health', return_value=mock_signals), \
             patch('signals.health._positions_health', return_value=mock_positions):
            return get_system_health()

    def test_returns_dict(self):
        result = self._call()
        assert isinstance(result, dict)

    def test_has_required_top_level_keys(self):
        result = self._call()
        for key in ('overall', 'timestamp', 'data', 'models', 'signals', 'positions'):
            assert key in result

    def test_overall_is_valid_status(self):
        result = self._call()
        assert result['overall'] in ('RED', 'YELLOW', 'GREEN')

    def test_timestamp_is_iso_string(self):
        result = self._call()
        ts = result['timestamp']
        assert isinstance(ts, str)
        assert 'T' in ts  # ISO 8601 format

    def test_red_when_no_data_files(self, tmp_path):
        with patch('signals.health.DATA_DIR', tmp_path / 'empty'), \
             patch('signals.health.MODELS_DIR', tmp_path / 'empty'), \
             patch('signals.health._signals_health', return_value={}), \
             patch('signals.health._positions_health', return_value={'open_count': 0, 'total_pnl': 0.0}):
            result = get_system_health()
            assert result['overall'] == 'RED'

    def test_red_when_no_models(self, tmp_path):
        # Create fresh data but no models
        csv = tmp_path / 'EURUSD_H1.csv'
        csv.write_text('header\n' + 'row\n' * 100)

        with patch('signals.health.DATA_DIR', tmp_path), \
             patch('signals.health.MODELS_DIR', tmp_path / 'no_models'), \
             patch('signals.health._signals_health', return_value={}), \
             patch('signals.health._positions_health', return_value={'open_count': 0, 'total_pnl': 0.0}):
            result = get_system_health()
            assert result['overall'] == 'RED'

    def test_yellow_when_data_stale(self, tmp_path):
        # Fresh data + models present but data age > 360 min → YELLOW
        fresh_data = {
            'EURUSD_H1': {'exists': True, 'age_minutes': 400, 'rows': 1000, 'fresh': True},
            'XAUUSD_H1': {'exists': True, 'age_minutes': 30, 'rows': 1000, 'fresh': True},
        }
        model_ok = {
            'EURUSD': {'ready': True, 'trained_at': '2024-01-01', 'cv_accuracy': 0.65},
            'XAUUSD': {'ready': True, 'trained_at': '2024-01-01', 'cv_accuracy': 0.65},
        }
        with patch('signals.health._data_health', return_value=fresh_data), \
             patch('signals.health._model_health', return_value=model_ok), \
             patch('signals.health._signals_health', return_value={}), \
             patch('signals.health._positions_health', return_value={'open_count': 0, 'total_pnl': 0.0}):
            result = get_system_health()
            assert result['overall'] == 'YELLOW'

    def test_green_when_all_fresh_and_ready(self):
        fresh_data = {
            'EURUSD_H1': {'exists': True, 'age_minutes': 30, 'rows': 1000, 'fresh': True},
            'XAUUSD_H1': {'exists': True, 'age_minutes': 30, 'rows': 1000, 'fresh': True},
        }
        model_ok = {
            'EURUSD': {'ready': True, 'trained_at': '2024-01-01', 'cv_accuracy': 0.65},
            'XAUUSD': {'ready': True, 'trained_at': '2024-01-01', 'cv_accuracy': 0.65},
        }
        with patch('signals.health._data_health', return_value=fresh_data), \
             patch('signals.health._model_health', return_value=model_ok), \
             patch('signals.health._signals_health', return_value={'today_count': 2, 'total_in_db': 10}), \
             patch('signals.health._positions_health', return_value={'open_count': 1, 'total_pnl': 5.0}):
            result = get_system_health()
            assert result['overall'] == 'GREEN'

    def test_signals_key_propagated(self):
        sig_data = {'today_count': 3, 'today_pairs': ['EURUSD', 'XAUUSD'], 'total_in_db': 50}
        result = self._call(mock_signals=sig_data)
        assert result['signals']['today_count'] == 3

    def test_positions_key_propagated(self):
        pos_data = {'open_count': 2, 'total_pnl': -12.5}
        result = self._call(mock_positions=pos_data)
        assert result['positions']['open_count'] == 2
        assert result['positions']['total_pnl'] == pytest.approx(-12.5)
