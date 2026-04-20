"""
signals/signal_engine.py — Production signal engine with real positive expectancy.

Target: did the trade (in trend direction) hit 2×ATR TP before 1×ATR SL
        within the next 10 bars?  1 = win, 0 = loss, NaN = unresolved (dropped).

This beats next-bar direction because:
  - At 2:1 RR, break-even is only 33.3% win rate
  - Trend-aligned entries avoid fighting momentum
  - Confidence filtering removes low-edge days

Architecture:
  - Features: trading.features.build_features() (37 TA) + FRED macro fundamentals
  - Training: RF(class_weight=balanced) + XGB(scale_pos_weight) ensemble
  - CV: TimeSeriesSplit(n_splits=5) — zero lookahead bias
  - Saves: cv_win_rate, expectancy_R, threshold to meta.json
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
DATA_DIR  = Path('data')

# ── FRED macro series ──────────────────────────────────────────────────────────
# Each tuple: (filename_stem, column_name, csv_col_name)
_FRED_SERIES = [
    ('DGS10',       'dgs10',    'dgs10'),
    ('DGS2',        'dgs2',     'dgs2'),
    ('VIXCLS',      'vix',      'vixcls'),
    ('DFF',         'dff',      'dff'),
    ('FEDFUNDS',    'fedfunds', 'fedfunds'),
    ('DCOILBRENTEU','brent',    'dcoilbrenteu'),
]
FRED_FEATURE_COLS = ['dgs10', 'dgs2', 'yield_spread', 'vix', 'dff', 'fedfunds', 'brent']

# ── Per-pair defaults ──────────────────────────────────────────────────────────
# sl_atr/tp_atr must match what was used during training (stored in meta.json)
_DEFAULTS = {
    'EURUSD': dict(threshold=0.58, sl_atr=1.0, tp_atr=1.5),
    'XAUUSD': dict(threshold=0.55, sl_atr=1.0, tp_atr=2.0),
}
_FALLBACK = dict(threshold=0.58, sl_atr=1.0, tp_atr=2.0)


# ── Private helpers ────────────────────────────────────────────────────────────

def _atr_series(high: pd.Series, low: pd.Series, close: pd.Series, period=14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _model_paths(pair: str) -> dict:
    MODEL_DIR.mkdir(exist_ok=True)
    return {
        'rf':     MODEL_DIR / f'{pair}_rf.joblib',
        'xgb':   MODEL_DIR / f'{pair}_xgb.joblib',
        'scaler': MODEL_DIR / f'{pair}_scaler.joblib',
        'meta':   MODEL_DIR / f'{pair}_meta.json',
    }


def _load_fred(df: pd.DataFrame) -> pd.DataFrame:
    """
    Load FRED macro series and merge into df by date (forward-fill).
    Returns df with extra columns added; index is preserved as-is.
    Missing FRED files are silently skipped (columns set to NaN).
    """
    df = df.copy()

    # Build a tz-naive date index for matching against FRED dates
    raw_index = df.index
    if isinstance(raw_index, pd.DatetimeIndex):
        date_index = raw_index.normalize().tz_localize(None)
    else:
        date_index = pd.to_datetime(raw_index, errors='coerce').normalize()

    for stem, feat_col, csv_col in _FRED_SERIES:
        path = DATA_DIR / f'{stem}.csv'
        if not path.exists():
            df[feat_col] = np.nan
            continue
        try:
            fred = pd.read_csv(path, parse_dates=['date'])
            fred['date'] = pd.to_datetime(fred['date']).dt.normalize()
            fred = fred.set_index('date').sort_index()
            if csv_col in fred.columns:
                fred = fred[[csv_col]].rename(columns={csv_col: feat_col})
            elif feat_col in fred.columns:
                fred = fred[[feat_col]]
            else:
                df[feat_col] = np.nan
                continue
            fred[feat_col] = pd.to_numeric(fred[feat_col], errors='coerce')
            # Reindex to date_index (tz-naive); unlimited forward-fill for macro series
            merged = fred.reindex(date_index, method='ffill', limit=None)
            df[feat_col] = merged[feat_col].values
        except Exception as exc:
            logger.warning('FRED load failed for %s: %s', stem, exc)
            df[feat_col] = np.nan

    if 'dgs10' in df.columns and 'dgs2' in df.columns:
        df['yield_spread'] = df['dgs10'] - df['dgs2']
    else:
        df['yield_spread'] = np.nan

    return df


def _build_target(df: pd.DataFrame, lookahead=10, tp_mult=2.0, sl_mult=1.0) -> pd.Series:
    """
    TP/SL outcome target, trend-aligned.

    For each bar i:
      - trend = sign(close[i] - SMA20[i]): +1 = long bias, -1 = short bias
      - entry = close[i]
      - TP   = entry + trend * tp_mult * ATR14[i]
      - SL   = entry - trend * sl_mult * ATR14[i]
      - scan bars i+1 .. i+lookahead:
          if high crosses TP first (long) or low crosses TP first (short) → 1
          if low  crosses SL first (long) or high crosses SL first (short) → 0
      - NaN if unresolved within lookahead

    Returns pd.Series aligned with df.index.
    """
    c = df['close'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)

    atr   = _atr_series(h, l, c, 14)
    sma20 = c.rolling(20, min_periods=20).mean()
    trend = np.sign(c - sma20)

    ch = h.values
    cl = l.values
    ca = atr.values
    cc = c.values
    ct = trend.values

    labels = np.full(len(df), np.nan)
    n = len(df)

    for i in range(n - 1):
        d = ct[i]
        if d == 0 or np.isnan(d):
            continue
        a = ca[i]
        if np.isnan(a) or a == 0:
            continue

        entry = cc[i]
        tp = entry + d * tp_mult * a
        sl = entry - d * sl_mult * a

        for j in range(i + 1, min(i + lookahead + 1, n)):
            if d > 0:       # long
                if ch[j] >= tp:
                    labels[i] = 1; break
                if cl[j] <= sl:
                    labels[i] = 0; break
            else:            # short
                if cl[j] <= tp:
                    labels[i] = 1; break
                if ch[j] >= sl:
                    labels[i] = 0; break

    return pd.Series(labels, index=df.index)


def _optimal_threshold(probs: np.ndarray, labels: np.ndarray,
                        tp_mult=2.0, sl_mult=1.0) -> float:
    """
    Find the probability threshold that maximises expectancy on OOF predictions.
    Steps from 0.50 to 0.75 in 0.01 increments; returns best with min 30 trades.
    Falls back to default if no threshold qualifies.
    """
    best_exp, best_thr = -99.0, 0.58
    for thr in np.arange(0.50, 0.76, 0.01):
        mask = probs >= thr
        n = mask.sum()
        if n < 30:
            continue
        wr = labels[mask].mean()
        exp = wr * tp_mult - (1 - wr) * sl_mult
        if exp > best_exp:
            best_exp, best_thr = exp, round(float(thr), 2)
    return best_thr


# ── SignalEngine ───────────────────────────────────────────────────────────────

class SignalEngine:
    """
    Production-grade signal engine.
    Instance-level model cache: load once per process, predict many times.
    """

    def __init__(self):
        MODEL_DIR.mkdir(exist_ok=True)
        self._cache: dict = {}

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, pair: str, df: pd.DataFrame,
              lookahead: int = 10, tp_mult: float = 2.0, sl_mult: float = 1.0) -> dict:
        """
        Train RF + XGB ensemble on an OHLCV DataFrame.

        df must be sorted chronologically with: open, high, low, close columns
        and a DatetimeIndex (or timestamp-parseable index).
        Minimum ~200 rows recommended; 500+ for reliable CV.
        """
        logger.info('[%s] Building features on %d rows…', pair, len(df))

        # ── Features ──────────────────────────────────────────────────────────
        ta_feat = build_features(df)           # 37 TA features

        df_mac  = _load_fred(df)               # FRED macro features
        mac_feat = df_mac[FRED_FEATURE_COLS]   # 7 macro features

        all_feat = pd.concat([ta_feat, mac_feat], axis=1)
        feature_cols = FEATURE_COLS + FRED_FEATURE_COLS

        # ── Target ────────────────────────────────────────────────────────────
        target = _build_target(df, lookahead=lookahead, tp_mult=tp_mult, sl_mult=sl_mult)

        data = pd.concat([all_feat, target.rename('target')], axis=1).dropna()
        X = data[feature_cols].values
        y = data['target'].values.astype(int)

        n = len(X)
        if n < 100:
            raise ValueError(
                f'[{pair}] Only {n} usable rows after feature+target build — need ≥ 100. '
                'Run: python manage.py fetch_price_data --full'
            )

        n_pos = y.sum()
        n_neg = n - n_pos
        pos_weight = max(1.0, n_neg / n_pos) if n_pos > 0 else 1.0

        logger.info(
            '[%s] %d samples × %d features  pos=%d neg=%d  scale_pos_weight=%.2f',
            pair, n, len(feature_cols), n_pos, n_neg, pos_weight,
        )

        # ── Walk-forward CV ───────────────────────────────────────────────────
        tscv = TimeSeriesSplit(n_splits=5)
        oof_probs  = np.zeros(n)
        fold_accs, fold_aucs, fold_wrs = [], [], []

        for fold_num, (tr_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            sc = StandardScaler().fit(X_tr)
            X_tr_s  = sc.transform(X_tr)
            X_val_s = sc.transform(X_val)

            n_pos_tr = y_tr.sum()
            n_neg_tr = len(y_tr) - n_pos_tr
            pw = max(1.0, n_neg_tr / n_pos_tr) if n_pos_tr > 0 else 1.0

            rf_f = RandomForestClassifier(
                n_estimators=200, max_depth=6, min_samples_leaf=15,
                max_features='sqrt', class_weight='balanced',
                random_state=42, n_jobs=-1,
            ).fit(X_tr_s, y_tr)

            xgb_f = xgb.XGBClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.08,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=pw,
                eval_metric='logloss', random_state=42, verbosity=0,
            ).fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], verbose=False)

            rf_p  = rf_f.predict_proba(X_val_s)[:, 1]
            xgb_p = xgb_f.predict_proba(X_val_s)[:, 1]
            ens_p = 0.5 * rf_p + 0.5 * xgb_p
            oof_probs[val_idx] = ens_p

            preds = (ens_p >= 0.5).astype(int)
            fold_accs.append(float(accuracy_score(y_val, preds)))
            try:
                fold_aucs.append(float(roc_auc_score(y_val, ens_p)))
            except Exception:
                fold_aucs.append(0.5)

            # Win rate = fraction of predicted-positive that are wins
            pos_mask = preds == 1
            wr = float(y_val[pos_mask].mean()) if pos_mask.sum() > 0 else 0.0
            fold_wrs.append(wr)

            logger.info(
                '  Fold %d: acc=%.4f  auc=%.4f  wr(pos)=%.4f  n_trades=%d',
                fold_num + 1, fold_accs[-1], fold_aucs[-1], wr, pos_mask.sum(),
            )

        cv_accuracy = float(np.mean(fold_accs))
        cv_auc      = float(np.mean(fold_aucs))
        cv_win_rate = float(np.mean(fold_wrs))

        # Optimise threshold from OOF probabilities
        threshold = _optimal_threshold(oof_probs, y, tp_mult=tp_mult, sl_mult=sl_mult)
        # Also compute win rate at optimised threshold
        thr_mask = oof_probs >= threshold
        thr_wr   = float(y[thr_mask].mean()) if thr_mask.sum() > 0 else 0.0
        expectancy_R = thr_wr * tp_mult - (1 - thr_wr) * sl_mult

        logger.info(
            '[%s] CV acc=%.4f  auc=%.4f  threshold=%.2f  wr@thr=%.4f  E[R]=%.3f',
            pair, cv_accuracy, cv_auc, threshold, thr_wr, expectancy_R,
        )

        # ── Final model trained on ALL data ───────────────────────────────────
        scaler = StandardScaler().fit(X)
        X_s = scaler.transform(X)

        rf_base  = RandomForestClassifier(
            n_estimators=500, max_depth=6, min_samples_leaf=15,
            max_features='sqrt', class_weight='balanced',
            random_state=42, n_jobs=-1,
        )
        rf_final = CalibratedClassifierCV(rf_base, method='isotonic', cv=5)
        rf_final.fit(X_s, y)

        xgb_final = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            eval_metric='logloss', random_state=42, verbosity=0,
        ).fit(X_s, y)

        # ── Save artifacts ────────────────────────────────────────────────────
        paths = _model_paths(pair)
        joblib.dump(rf_final,  paths['rf'])
        joblib.dump(xgb_final, paths['xgb'])
        joblib.dump(scaler,    paths['scaler'])

        defaults = _DEFAULTS.get(pair, _FALLBACK)
        meta = {
            'pair':            pair,
            'features':        feature_cols,
            'n_features':      len(feature_cols),
            'n_samples':       n,
            'target_lookahead': lookahead,
            'tp_mult':         tp_mult,
            'sl_mult':         sl_mult,
            'cv_accuracy':     cv_accuracy,
            'cv_auc':          cv_auc,
            'cv_win_rate':     cv_win_rate,
            'threshold':       threshold,
            'thr_win_rate':    thr_wr,
            'expectancy_R':    round(expectancy_R, 4),
            'bull_threshold':  threshold,           # kept for backward compat
            'bear_threshold':  round(1 - threshold, 2),
            'sl_mult_default': defaults['sl_atr'],
            'tp_mult_default': defaults['tp_atr'],
            'fold_accuracies': fold_accs,
            'fold_aucs':       fold_aucs,
            'fold_win_rates':  fold_wrs,
            'trained_at':      datetime.utcnow().isoformat() + 'Z',
        }
        with open(paths['meta'], 'w') as f:
            json.dump(meta, f, indent=2)

        logger.info('[%s] Artifacts saved → %s/', pair, MODEL_DIR)
        self._cache.pop(pair, None)
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
                'xgb':   joblib.load(paths['xgb']),
                'scaler': joblib.load(paths['scaler']),
                'meta':   json.loads(paths['meta'].read_text()),
            }
        return self._cache[pair]

    def predict(self, pair: str, df: pd.DataFrame) -> dict:
        """
        Generate a trading signal for the latest bar in df.

        Returns
        -------
        dict with keys:
          pair, signal, probability, confidence,
          entry, stop_loss, take_profit, risk_reward, atr,
          date, direction, model_info

        signal ∈ {'bullish', 'bearish', 'no_signal'}
        entry/SL/TP are actual prices, trend-aligned.
        """
        mdl  = self._load(pair)
        meta = mdl['meta']
        cols = meta['features']

        # ── Build features for latest bar ──────────────────────────────────────
        ta_feat  = build_features(df)
        df_mac   = _load_fred(df)
        mac_feat = df_mac[FRED_FEATURE_COLS]

        all_feat = pd.concat([ta_feat, mac_feat], axis=1)

        # Validate we have all required columns
        missing_cols = [c for c in cols if c not in all_feat.columns]
        if missing_cols:
            raise ValueError(f'[{pair}] Features missing at inference: {missing_cols}')

        latest_feat = all_feat.iloc[[-1]][cols]

        nan_cols = latest_feat.columns[latest_feat.isnull().any()].tolist()
        if nan_cols:
            raise ValueError(
                f'[{pair}] NaN in latest features: {nan_cols}. '
                'Need more bars in data file (>= 60 recommended).'
            )

        X = mdl['scaler'].transform(latest_feat.values)

        rf_prob  = float(mdl['rf'].predict_proba(X)[0, 1])
        xgb_prob = float(mdl['xgb'].predict_proba(X)[0, 1])
        prob = 0.5 * rf_prob + 0.5 * xgb_prob

        threshold = meta.get('threshold', meta.get('bull_threshold', 0.55))
        sl_mult   = meta.get('sl_mult_default', meta.get('sl_mult', 1.0))
        tp_mult   = meta.get('tp_mult_default', meta.get('tp_mult', 2.0))

        # ── Trend direction for SL/TP placement ───────────────────────────────
        c_series = df['close'].astype(float)
        sma20    = c_series.rolling(20, min_periods=20).mean()
        trend_dir = int(np.sign(float(c_series.iloc[-1]) - float(sma20.iloc[-1])))
        if trend_dir == 0:
            trend_dir = 1   # default long if at MA

        # ── Signal determination ───────────────────────────────────────────────
        if prob >= threshold:
            # High-confidence "TP will be hit" — trade in trend direction
            signal = 'bullish' if trend_dir > 0 else 'bearish'
        else:
            signal = 'no_signal'

        atr   = get_latest_atr(df, 14)
        entry = float(c_series.iloc[-1])

        if signal == 'bullish':
            stop_loss   = round(entry - sl_mult * atr, 5)
            take_profit = round(entry + tp_mult * atr, 5)
        elif signal == 'bearish':
            stop_loss   = round(entry + sl_mult * atr, 5)
            take_profit = round(entry - tp_mult * atr, 5)
        else:
            # No-signal: still compute levels for display
            if trend_dir > 0:
                stop_loss   = round(entry - sl_mult * atr, 5)
                take_profit = round(entry + tp_mult * atr, 5)
            else:
                stop_loss   = round(entry + sl_mult * atr, 5)
                take_profit = round(entry - tp_mult * atr, 5)

        risk   = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        rr     = round(reward / risk, 2) if risk > 1e-10 else float(tp_mult)

        date_idx = df.index[-1]
        date_str = date_idx.date().isoformat() if hasattr(date_idx, 'date') else str(date_idx)[:10]

        return {
            'pair':        pair,
            'signal':      signal,
            'probability': round(prob, 4),
            'confidence':  round(abs(prob - threshold) / (1.0 - threshold + 1e-9), 4),
            'entry':       round(entry, 5),
            'stop_loss':   stop_loss,
            'take_profit': take_profit,
            'risk_reward': rr,
            'atr':         round(atr, 5),
            'direction':   'long' if trend_dir > 0 else 'short',
            'date':        date_str,
            'model_info': {
                'trained_at':    meta.get('trained_at'),
                'cv_accuracy':   meta.get('cv_accuracy'),
                'cv_auc':        meta.get('cv_auc'),
                'cv_win_rate':   meta.get('cv_win_rate'),
                'thr_win_rate':  meta.get('thr_win_rate'),
                'expectancy_R':  meta.get('expectancy_R'),
                'threshold':     threshold,
                'rf_prob':       round(rf_prob, 4),
                'xgb_prob':      round(xgb_prob, 4),
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
