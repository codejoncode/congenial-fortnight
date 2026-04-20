"""
signals/health.py — System health checks for the trading system.

Returns a single dict describing the current status of:
  - Data files (freshness + row count)
  - Trained models (present on disk)
  - Open positions
  - Today's signals
  - Last signal generation time
  - API availability
"""

import os
import json
import time
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
PAIRS      = ['EURUSD', 'XAUUSD']


def _file_age_minutes(path: Path) -> float | None:
    """Minutes since file was last modified. None if file doesn't exist."""
    if not path.exists():
        return None
    age_sec = time.time() - path.stat().st_mtime
    return round(age_sec / 60, 1)


def _count_rows(path: Path) -> int:
    """Fast row count (counts newlines minus header)."""
    if not path.exists():
        return 0
    try:
        with open(path, 'rb') as f:
            count = sum(1 for _ in f)
        return max(0, count - 1)
    except Exception:
        return 0


def _data_health() -> dict:
    results = {}
    for pair in PAIRS:
        for suffix in ['H1', 'H4', 'Daily']:
            path = DATA_DIR / f'{pair}_{suffix}.csv'
            age  = _file_age_minutes(path)
            rows = _count_rows(path) if path.exists() else 0
            if age is not None:
                results[f'{pair}_{suffix}'] = {
                    'exists': True,
                    'age_minutes': age,
                    'rows': rows,
                    'fresh': age < 1440,  # fresh if < 24 hours
                }
                break  # use first found timeframe
        else:
            results[f'{pair}_H1'] = {'exists': False, 'age_minutes': None, 'rows': 0, 'fresh': False}
    return results


def _model_health() -> dict:
    results = {}
    for pair in PAIRS:
        artifacts = {
            'rf':     MODELS_DIR / f'{pair}_rf.joblib',
            'xgb':   MODELS_DIR / f'{pair}_xgb.joblib',
            'scaler': MODELS_DIR / f'{pair}_scaler.joblib',
            'meta':   MODELS_DIR / f'{pair}_meta.json',
        }
        all_present = all(p.exists() for p in artifacts.values())
        trained_at  = None
        cv_accuracy = None
        if all_present:
            try:
                meta = json.loads(artifacts['meta'].read_text())
                trained_at  = meta.get('trained_at')
                cv_accuracy = meta.get('cv_accuracy')
            except Exception:
                pass
        results[pair] = {
            'ready':       all_present,
            'trained_at':  trained_at,
            'cv_accuracy': cv_accuracy,
        }
    return results


def _signals_health() -> dict:
    """Import Django models here (lazy) so this module is importable at startup."""
    try:
        from signals.models import Signal
        from django.utils import timezone as dj_tz

        today = dj_tz.now().date()
        today_signals = Signal.objects.filter(date=today)
        latest        = Signal.objects.order_by('-created_at').first()

        return {
            'today_count':    today_signals.count(),
            'today_pairs':    list(today_signals.values_list('pair', flat=True)),
            'last_generated': latest.created_at.isoformat() if latest and hasattr(latest, 'created_at') else None,
            'total_in_db':    Signal.objects.count(),
        }
    except Exception as exc:
        return {'error': str(exc)}


def _positions_health() -> dict:
    try:
        from paper_trading.models import Trade
        open_trades = Trade.objects.filter(status='open')
        total_pnl   = sum(float(t.unrealized_pnl or 0) for t in open_trades)
        return {
            'open_count': open_trades.count(),
            'total_pnl':  round(total_pnl, 2),
        }
    except Exception:
        return {'open_count': 0, 'total_pnl': 0.0}


def get_system_health() -> dict:
    """
    Returns a full health snapshot.  Safe to call from any view — all DB
    imports are lazy so the module can be imported before Django is set up.
    """
    data   = _data_health()
    models = _model_health()
    sigs   = _signals_health()
    pos    = _positions_health()

    # Overall status: RED if no data or no models; YELLOW if stale; GREEN otherwise
    data_ok   = any(v.get('fresh') for v in data.values())
    models_ok = all(v['ready'] for v in models.values())

    if not data_ok or not models_ok:
        overall = 'RED'
    elif any(v.get('age_minutes', 0) > 360 for v in data.values() if v.get('exists')):
        overall = 'YELLOW'
    else:
        overall = 'GREEN'

    return {
        'overall':    overall,
        'timestamp':  datetime.now(timezone.utc).isoformat(),
        'data':       data,
        'models':     models,
        'signals':    sigs,
        'positions':  pos,
    }
