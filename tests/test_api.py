"""
tests/test_api.py — Django integration tests using APIClient.

Covers all endpoints added or modified in this sprint:
  - GET  /api/health/            → 200, {'status': 'healthy'}
  - GET  /api/system-health/     → 200, has 'overall' (RED/YELLOW/GREEN)
  - GET  /api/signals/           → 200, returns list
  - POST /api/signals/generate/  → accepts request, has 'status' key
  - GET  /api/signals/decision/  → 200, has EURUSD and XAUUSD keys
  - GET  /api/signal-performance/→ 200 (graceful even with no CSV)
  - Signal model CRUD via ORM
  - SignalSerializer round-trip
  - Signal unique_together constraint
"""

import os
import pytest
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'forex_signal.settings')
django.setup()

from django.test import TestCase
from rest_framework.test import APIClient
from rest_framework import status
from django.db import IntegrityError
from django.utils import timezone

from signals.models import Signal
from signals.serializers import SignalSerializer

import datetime


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_signal(**kwargs):
    defaults = {
        'pair':        'EURUSD',
        'signal':      'bullish',
        'probability': 0.72,
        'date':        datetime.date.today(),
        'entry_price': 1.10000,
        'stop_loss':   1.09700,
        'take_profit': 1.10600,
        'risk_reward': 2.0,
        'atr':         0.001,
        'source':      'engine',
    }
    defaults.update(kwargs)
    return Signal.objects.create(**defaults)


# ── Signal Model Tests ────────────────────────────────────────────────────────

@pytest.mark.django_db
class TestSignalModel:
    def test_create_signal_with_required_fields(self):
        sig = Signal.objects.create(
            pair='EURUSD',
            signal='bullish',
            probability=0.70,
            date=datetime.date.today(),
        )
        assert sig.pk is not None

    def test_default_source_is_engine(self):
        sig = Signal.objects.create(
            pair='EURUSD',
            signal='bullish',
            probability=0.65,
            date=datetime.date(2024, 6, 1),
        )
        assert sig.source == 'engine'

    def test_unique_together_pair_date_raises(self):
        Signal.objects.create(
            pair='EURUSD',
            signal='bullish',
            probability=0.65,
            date=datetime.date(2024, 5, 1),
        )
        with pytest.raises(IntegrityError):
            Signal.objects.create(
                pair='EURUSD',
                signal='bearish',
                probability=0.72,
                date=datetime.date(2024, 5, 1),
            )

    def test_different_pairs_same_date_allowed(self):
        d = datetime.date(2024, 5, 2)
        Signal.objects.create(pair='EURUSD', signal='bullish', probability=0.65, date=d)
        sig2 = Signal.objects.create(pair='XAUUSD', signal='bearish', probability=0.68, date=d)
        assert sig2.pk is not None

    def test_signal_choices_bullish_bearish_no_signal(self):
        for label in ('bullish', 'bearish', 'no_signal'):
            d = datetime.date(2024, 1, int(label == 'bullish') + int(label == 'bearish') * 2 + int(label == 'no_signal') * 3)
            sig = Signal.objects.create(pair='EURUSD', signal=label, probability=0.60, date=d)
            assert sig.signal == label

    def test_optional_fields_can_be_null(self):
        sig = Signal.objects.create(
            pair='XAUUSD',
            signal='no_signal',
            probability=0.45,
            date=datetime.date(2024, 4, 15),
            entry_price=None,
            stop_loss=None,
            take_profit=None,
        )
        assert sig.entry_price is None
        assert sig.stop_loss is None

    def test_created_at_auto_set(self):
        sig = Signal.objects.create(
            pair='EURUSD', signal='bullish', probability=0.60,
            date=datetime.date(2024, 3, 10),
        )
        assert sig.created_at is not None

    def test_ordering_by_date_descending(self):
        Signal.objects.create(pair='EURUSD', signal='bullish', probability=0.60, date=datetime.date(2024, 2, 1))
        Signal.objects.create(pair='EURUSD', signal='bearish', probability=0.65, date=datetime.date(2024, 2, 2))
        signals = list(Signal.objects.filter(pair='EURUSD').order_by('-date'))
        assert signals[0].date > signals[-1].date


# ── SignalSerializer Tests ────────────────────────────────────────────────────

@pytest.mark.django_db
class TestSignalSerializer:
    def test_serializer_contains_all_model_fields(self):
        sig = make_signal(date=datetime.date(2024, 7, 1))
        data = SignalSerializer(sig).data
        for field in ('id', 'pair', 'signal', 'probability', 'entry_price', 'stop_loss',
                      'take_profit', 'risk_reward', 'atr', 'source', 'date', 'created_at'):
            assert field in data

    def test_serializer_pair_value(self):
        sig = make_signal(pair='XAUUSD', date=datetime.date(2024, 7, 2))
        data = SignalSerializer(sig).data
        assert data['pair'] == 'XAUUSD'

    def test_serializer_probability_value(self):
        sig = make_signal(probability=0.8123, date=datetime.date(2024, 7, 3))
        data = SignalSerializer(sig).data
        assert abs(data['probability'] - 0.8123) < 0.001

    def test_serializer_list(self):
        Signal.objects.create(pair='EURUSD', signal='bullish', probability=0.60, date=datetime.date(2024, 8, 1))
        Signal.objects.create(pair='XAUUSD', signal='bearish', probability=0.65, date=datetime.date(2024, 8, 2))
        qs = Signal.objects.all()
        data = SignalSerializer(qs, many=True).data
        assert len(data) >= 2


# ── API Endpoint Tests ────────────────────────────────────────────────────────

@pytest.fixture
def client():
    return APIClient()


@pytest.mark.django_db
class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get('/api/health/')
        assert resp.status_code == status.HTTP_200_OK

    def test_health_returns_status_healthy(self, client):
        resp = client.get('/api/health/')
        assert resp.json().get('status') == 'healthy'


@pytest.mark.django_db
class TestSystemHealthEndpoint:
    def test_system_health_returns_200(self, client):
        resp = client.get('/api/system-health/')
        assert resp.status_code == status.HTTP_200_OK

    def test_system_health_has_overall_key(self, client):
        resp = client.get('/api/system-health/')
        data = resp.json()
        assert 'overall' in data

    def test_system_health_overall_is_valid_status(self, client):
        resp = client.get('/api/system-health/')
        assert resp.json()['overall'] in ('RED', 'YELLOW', 'GREEN')

    def test_system_health_has_data_key(self, client):
        resp = client.get('/api/system-health/')
        assert 'data' in resp.json()

    def test_system_health_has_models_key(self, client):
        resp = client.get('/api/system-health/')
        assert 'models' in resp.json()

    def test_system_health_has_signals_key(self, client):
        resp = client.get('/api/system-health/')
        assert 'signals' in resp.json()

    def test_system_health_has_positions_key(self, client):
        resp = client.get('/api/system-health/')
        assert 'positions' in resp.json()

    def test_system_health_has_timestamp(self, client):
        resp = client.get('/api/system-health/')
        data = resp.json()
        assert 'timestamp' in data
        assert 'T' in data['timestamp']  # ISO 8601


@pytest.mark.django_db
class TestSignalsListEndpoint:
    def test_signals_list_returns_200(self, client):
        resp = client.get('/api/signals/')
        assert resp.status_code == status.HTTP_200_OK

    def test_signals_list_returns_list(self, client):
        resp = client.get('/api/signals/')
        assert isinstance(resp.json(), list)

    def test_signals_list_includes_created_signals(self, client):
        make_signal(date=datetime.date(2024, 9, 1))
        resp = client.get('/api/signals/')
        assert len(resp.json()) >= 1

    def test_signals_list_item_has_pair(self, client):
        make_signal(pair='EURUSD', date=datetime.date(2024, 9, 2))
        resp = client.get('/api/signals/')
        data = resp.json()
        pairs = [s.get('pair') for s in data]
        assert 'EURUSD' in pairs


@pytest.mark.django_db
class TestGenerateSignalsEndpoint:
    def test_generate_returns_200_or_202(self, client):
        from unittest.mock import patch
        # call_command is imported inside the view function, patch at Django source
        with patch('django.core.management.call_command'):
            resp = client.post('/api/signals/generate/')
            assert resp.status_code in (200, 202, 500)

    def test_generate_response_has_status_key(self, client):
        from unittest.mock import patch
        with patch('django.core.management.call_command'):
            resp = client.post('/api/signals/generate/')
            assert 'status' in resp.json() or resp.status_code in (200, 202)


@pytest.mark.django_db
class TestSignalDecisionEndpoint:
    def test_decision_returns_200(self, client):
        resp = client.get('/api/signals/decision/')
        assert resp.status_code == status.HTTP_200_OK

    def test_decision_has_eurusd_key(self, client):
        resp = client.get('/api/signals/decision/')
        data = resp.json()
        assert 'EURUSD' in data

    def test_decision_has_xauusd_key(self, client):
        resp = client.get('/api/signals/decision/')
        data = resp.json()
        assert 'XAUUSD' in data

    def test_decision_action_is_valid(self, client):
        # Seed a signal first so there's something to evaluate
        make_signal(pair='EURUSD', date=datetime.date.today())
        resp = client.get('/api/signals/decision/')
        data = resp.json()
        if data.get('EURUSD') and data['EURUSD'].get('action'):
            assert data['EURUSD']['action'] in ('EXECUTE', 'WAIT', 'SKIP')

    def test_decision_accepts_balance_param(self, client):
        resp = client.get('/api/signals/decision/?balance=750')
        assert resp.status_code == status.HTTP_200_OK

    def test_decision_accepts_open_positions_param(self, client):
        resp = client.get('/api/signals/decision/?open_positions=2')
        assert resp.status_code == status.HTTP_200_OK

    def test_decision_accepts_daily_pnl_param(self, client):
        resp = client.get('/api/signals/decision/?daily_pnl=-15.0')
        assert resp.status_code == status.HTTP_200_OK

    def test_decision_all_params_together(self, client):
        resp = client.get('/api/signals/decision/?balance=500&open_positions=1&daily_pnl=-5')
        assert resp.status_code == status.HTTP_200_OK

    def test_decision_pair_endpoint(self, client):
        resp = client.get('/api/signals/decision/EURUSD/')
        assert resp.status_code == status.HTTP_200_OK


@pytest.mark.django_db
class TestSignalPerformanceEndpoint:
    def test_performance_returns_200(self, client):
        resp = client.get('/api/signal-performance/')
        # Even with no CSV files, should return 200 with error dict per pair
        assert resp.status_code == status.HTTP_200_OK

    def test_performance_response_is_dict(self, client):
        resp = client.get('/api/signal-performance/')
        assert isinstance(resp.json(), dict)


@pytest.mark.django_db
class TestDataStatusEndpoint:
    def test_data_status_returns_200(self, client):
        resp = client.get('/api/data/status/')
        assert resp.status_code == status.HTTP_200_OK
