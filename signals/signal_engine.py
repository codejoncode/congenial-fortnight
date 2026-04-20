"""
signals/signal_engine.py — Clean, self-contained trading signal engine.

Architecture:
  - Uses trading.features.build_features() as the ONLY feature source
  - Trains RF (isotonic-calibrated) + XGBoost ensemble
  - Saves all artifacts + meta.json (feature list, thresholds, accuracy)
  - Inference always reads the feature list from meta.json → training/inference parity guaranteed

Usage (training):
    engine = SignalEngine()
    meta = engine.train('EURUSD', df)      # df: OHLCV DataFrame

Usage (inference):
    engine = SignalEngine()
    result = engine.predict('EURUSD', df)  # returns dict with entry/SL/TP/RR
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

from trading.features import FEATURE_COLS, build_features, get_latest_atr

logger = logging.getLogger(__name__)

MODEL_DIR = Path('models')

# ── Per-pair inference parameters ─────────────────────────────────────────────
# These become defaults; trained values are stored in meta.json and override these.
_DEFAULTS = {
    'EURUSD': dict(bull_threshold=0.54, bear_threshold=0.46, sl_mult=1.5, tp_mult=2.5),
    'XAUUSD': dict(bull_threshold=0.55, bear_threshold=0.45, sl_mult=2.0, tp_mult=3.0),
}
_FALLBACK = dict(bull_threshold=0.54, bear_threshold=0.46, sl_mult=1.5, tp_mult=2.5)


def _model_paths(pair: str) -> dict:
    MODEL_DIR.mkdir(exist_ok=True)
    return {
        'rf':     MODEL_DIR / f'{pair}_rf.joblib',
        'xgb':    MODEL_DIR / f'{pair}_xgb.joblib',
        'scaler': MODEL_DIR / f'{pair}_scaler.joblib',
        'meta':   MODEL_DIR / f'{pair}_meta.json',
    }


class SignalEngine:
    """
    Production-grade signal engine.
    Instance-level model cache: load once per process, predict many times.
    """

    def __init__(self):
        MODEL_DIR.mkdir(exist_ok=True)
        self._cache: dict = {}

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, pair: str, df: pd.DataFrame) -> dict:
        """
        Train RF + XGB ensemble on an OHLCV DataFrame.
        Saves all model artifacts and returns accuracy metrics.

        df must be sorted chronologically and have: open, high, low, close columns.
        Minimum ~300 rows recommended for reliable cross-validation.
        """
        logger.info(f'[{pair}] Building features on {len(df)} rows…')
        features = build_features(df)
        target = (df['close'].shift(-1) > df['close']).astype(int)

        data = pd.concat([features, target.rename('target')], axis=1).dropna()
        X = data[FEATURE_COLS].values
        y = data['target'].values

        n = len(X)
        if n < 100:
            raise ValueError(
                f'[{pair}] Only {n} usable rows after feature engineering — need ≥ 100. '
                'Run: python manage.py run_daily_signal --fetch-data to update market data.'
            )

        logger.info(f'[{pair}] Training on {n} samples × {len(FEATURE_COLS)} features')

        # ── Walk-forward CV for honest accuracy estimate ───────────────────────
        tscv = TimeSeriesSplit(n_splits=5)
        oof_probs = np.zeros(n)
        fold_accs, fold_aucs = [], []

        for fold_num, (tr_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            sc_fold = StandardScaler().fit(X_tr)
            X_tr_s  = sc_fold.transform(X_tr)
            X_val_s = sc_fold.transform(X_val)

            rf_fold = RandomForestClassifier(
                n_estimators=200, max_depth=6, min_samples_leaf=15,
                max_features='sqrt', class_weight='balanced',
                random_state=42, n_jobs=-1,
            ).fit(X_tr_s, y_tr)

            xgb_fold = xgb.XGBClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.08,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric='logloss', random_state=42, verbosity=0,
            ).fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], verbose=False)

            rf_p  = rf_fold.predict_proba(X_val_s)[:, 1]
            xgb_p = xgb_fold.predict_proba(X_val_s)[:, 1]
            ens_p = 0.5 * rf_p + 0.5 * xgb_p
            oof_probs[val_idx] = ens_p

            preds = (ens_p >= 0.5).astype(int)
            fold_accs.append(float(accuracy_score(y_val, preds)))
            try:
                fold_aucs.append(float(roc_auc_score(y_val, ens_p)))
            except Exception:
                fold_aucs.append(0.5)

            logger.info(
                f'  Fold {fold_num + 1}: acc={fold_accs[-1]:.4f}  auc={fold_aucs[-1]:.4f}'
            )

        cv_accuracy = float(np.mean(fold_accs))
        cv_auc      = float(np.mean(fold_aucs))
        logger.info(f'[{pair}] Mean CV acc={cv_accuracy:.4f}  auc={cv_auc:.4f}')

        # ── Final model trained on ALL data ───────────────────────────────────
        scaler = StandardScaler().fit(X)
        X_s = scaler.transform(X)

        rf_base = RandomForestClassifier(
            n_estimators=500, max_depth=6, min_samples_leaf=15,
            max_features='sqrt', class_weight='balanced',
            random_state=42, n_jobs=-1,
        )
        rf_final = CalibratedClassifierCV(rf_base, method='isotonic', cv=5)
        rf_final.fit(X_s, y)

        xgb_final = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='logloss', random_state=42, verbosity=0,
        ).fit(X_s, y)

        # ── Persist artifacts ─────────────────────────────────────────────────
        paths = _model_paths(pair)
        joblib.dump(rf_final,   paths['rf'])
        joblib.dump(xgb_final,  paths['xgb'])
        joblib.dump(scaler,     paths['scaler'])

        defaults = _DEFAULTS.get(pair, _FALLBACK)
        meta = {
            'pair':           pair,
            'features':       FEATURE_COLS,
            'n_features':     len(FEATURE_COLS),
            'n_samples':      n,
            'cv_accuracy':    cv_accuracy,
            'cv_auc':         cv_auc,
            'fold_accuracies': fold_accs,
            'fold_aucs':       fold_aucs,
            'bull_threshold': defaults['bull_threshold'],
            'bear_threshold': defaults['bear_threshold'],
            'sl_mult':        defaults['sl_mult'],
            'tp_mult':        defaults['tp_mult'],
            'trained_at':     datetime.utcnow().isoformat() + 'Z',
        }
        with open(paths['meta'], 'w') as f:
            json.dump(meta, f, indent=2)

        logger.info(f'[{pair}] Artifacts saved to {MODEL_DIR}/')
        self._cache.pop(pair, None)   # invalidate in-memory cache
        return meta

    # ── Inference ─────────────────────────────────────────────────────────────

    def _load(self, pair: str) -> dict:
        if pair not in self._cache:
            paths = _model_paths(pair)
            missing = [k for k, p in paths.items() if not p.exists()]
            if missing:
                raise FileNotFoundError(
                    f'Missing model artifacts for {pair}: {missing}. '
                    'Run:  python manage.py train_models'
                )
            self._cache[pair] = {
                'rf':     joblib.load(paths['rf']),
                'xgb':    joblib.load(paths['xgb']),
                'scaler': joblib.load(paths['scaler']),
                'meta':   json.loads(paths['meta'].read_text()),
            }
        return self._cache[pair]

    def predict(self, pair: str, df: pd.DataFrame) -> dict:
        """
        Generate a trading signal for the latest bar in df.

        Returns
        -------
        {
          pair, signal, probability, confidence,
          entry, stop_loss, take_profit, risk_reward, atr,
          date, model_info
        }

        signal is one of: 'bullish' | 'bearish' | 'no_signal'
        entry / stop_loss / take_profit are ACTUAL PRICES (not pips/distances).
        """
        mdl  = self._load(pair)
        meta = mdl['meta']
        cols = meta['features']   # always consistent with training

        feat = build_features(df)
        latest_feat = feat.iloc[[-1]][cols]

        nan_cols = latest_feat.columns[latest_feat.isnull().any()].tolist()
        if nan_cols:
            raise ValueError(
                f'[{pair}] NaN in latest features: {nan_cols}. '
                'Need more bars in data file (≥ 60 recommended).'
            )

        X = mdl['scaler'].transform(latest_feat.values)

        rf_prob  = float(mdl['rf'].predict_proba(X)[0, 1])
        xgb_prob = float(mdl['xgb'].predict_proba(X)[0, 1])
        prob = 0.5 * rf_prob + 0.5 * xgb_prob

        bull_thr = meta.get('bull_threshold', 0.54)
        bear_thr = meta.get('bear_threshold', 0.46)
        sl_mult  = meta.get('sl_mult', 1.5)
        tp_mult  = meta.get('tp_mult', 2.5)

        if prob >= bull_thr:
            signal = 'bullish'
        elif prob <= bear_thr:
            signal = 'bearish'
        else:
            signal = 'no_signal'

        atr   = get_latest_atr(df, 14)
        entry = float(df['close'].iloc[-1])

        if signal == 'bullish':
            stop_loss   = round(entry - sl_mult * atr, 5)
            take_profit = round(entry + tp_mult * atr, 5)
        elif signal == 'bearish':
            stop_loss   = round(entry + sl_mult * atr, 5)
            take_profit = round(entry - tp_mult * atr, 5)
        else:
            # No-signal: still compute levels for display (directional bias from prob)
            if prob >= 0.5:
                stop_loss   = round(entry - sl_mult * atr, 5)
                take_profit = round(entry + tp_mult * atr, 5)
            else:
                stop_loss   = round(entry + sl_mult * atr, 5)
                take_profit = round(entry - tp_mult * atr, 5)

        risk   = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        rr     = round(reward / risk, 2) if risk > 1e-10 else 0.0

        date_idx = df.index[-1]
        date_str = date_idx.date().isoformat() if hasattr(date_idx, 'date') else str(date_idx)[:10]

        return {
            'pair':        pair,
            'signal':      signal,
            'probability': round(prob, 4),
            'confidence':  round(abs(prob - 0.5) * 2, 4),   # 0 = uncertain, 1 = max confidence
            'entry':       round(entry, 5),
            'stop_loss':   stop_loss,
            'take_profit': take_profit,
            'risk_reward': rr,
            'atr':         round(atr, 5),
            'date':        date_str,
            'model_info': {
                'trained_at':   meta.get('trained_at'),
                'cv_accuracy':  meta.get('cv_accuracy'),
                'cv_auc':       meta.get('cv_auc'),
                'rf_prob':      round(rf_prob, 4),
                'xgb_prob':     round(xgb_prob, 4),
            },
        }

    def models_exist(self, pair: str) -> bool:
        """Return True only if ALL artifacts for pair are present on disk."""
        return all(p.exists() for p in _model_paths(pair).values())

    def invalidate_cache(self, pair: str = None):
        """Force reload on next predict() call. Pass None to invalidate all pairs."""
        if pair:
            self._cache.pop(pair, None)
        else:
            self._cache.clear()
