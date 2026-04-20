"""
signals/management/commands/evaluate_signals.py

Compute per-feature signal accuracy and write {PAIR}_signal_evaluation.csv.
These CSVs power the /api/signal-performance/ endpoint and the
SignalPerformanceView in the frontend.

Methodology:
  For each feature in FEATURE_COLS:
    - Split the full H1 dataset into 5 folds (TimeSeriesSplit)
    - In each fold, threshold the feature at its in-sample median
    - Count how often the resulting direction == actual next-bar direction
  This gives an honest out-of-sample accuracy estimate for each individual feature.

Usage:
    python manage.py evaluate_signals
    python manage.py evaluate_signals --pair EURUSD
"""

import os
import pandas as pd
import numpy as np
from django.core.management.base import BaseCommand
from sklearn.model_selection import TimeSeriesSplit
from trading.features import build_features, FEATURE_COLS

PAIRS    = ['EURUSD', 'XAUUSD']
DATA_DIR = 'data'


class Command(BaseCommand):
    help = 'Compute per-feature accuracy and update {PAIR}_signal_evaluation.csv files.'

    def add_arguments(self, parser):
        parser.add_argument('--pair', type=str, default=None)

    def handle(self, *args, **options):
        pairs = [options['pair'].upper()] if options.get('pair') else PAIRS

        for pair in pairs:
            self.stdout.write('\n[{}] Loading data...'.format(pair))
            df = self._load_data(pair)
            if df is None or len(df) < 100:
                self.stdout.write(self.style.ERROR(
                    '[{}] Insufficient data ({} rows). Need >= 100.'.format(
                        pair, len(df) if df is not None else 0
                    )
                ))
                continue

            self.stdout.write('[{}] Computing features on {} rows...'.format(pair, len(df)))
            try:
                feat = build_features(df)
                target = (df['close'].shift(-1) > df['close']).astype(int)
                combined = pd.concat([feat, target.rename('target')], axis=1).dropna()
            except Exception as exc:
                self.stdout.write(self.style.ERROR(
                    '[{}] Feature build failed: {}'.format(pair, exc)
                ))
                continue

            X_data = combined[FEATURE_COLS]
            y_data = combined['target']

            rows = []
            tscv = TimeSeriesSplit(n_splits=5)

            for col in FEATURE_COLS:
                col_vals = X_data[col].values
                y_vals   = y_data.values
                fold_accs, hit_rates, corrs = [], [], []

                for tr_idx, val_idx in tscv.split(col_vals):
                    col_tr  = col_vals[tr_idx]
                    col_val = col_vals[val_idx]
                    y_val   = y_vals[val_idx]

                    # Direction prediction: above in-sample median → bullish
                    median = np.nanmedian(col_tr)
                    preds  = (col_val > median).astype(int)

                    acc      = float(np.mean(preds == y_val))
                    hit_rate = float(y_val[preds == 1].mean()) if preds.sum() > 0 else 0.5
                    fold_accs.append(acc)
                    hit_rates.append(hit_rate)

                # Full-sample Pearson correlation (directional signal quality)
                try:
                    corr = float(pd.Series(col_vals).corr(pd.Series(y_vals)))
                    corr = 0.0 if np.isnan(corr) else corr
                except Exception:
                    corr = 0.0

                rows.append({
                    'feature':    col,
                    'label':      col.replace('_', ' ').title(),
                    'accuracy':   round(float(np.mean(fold_accs)), 4),
                    'hit_rate':   round(float(np.mean(hit_rates)), 4),
                    'correlation': round(corr, 4),
                })

            df_out = (
                pd.DataFrame(rows)
                .sort_values('accuracy', ascending=False)
                .reset_index(drop=True)
            )

            out_path = '{}_signal_evaluation.csv'.format(pair)
            df_out.to_csv(out_path, index=False)

            top = df_out.iloc[0]
            self.stdout.write(self.style.SUCCESS(
                '[{}] Saved {} features to {}\n'
                '       Top feature: {} ({:.1%} accuracy, corr={:.3f})'.format(
                    pair, len(df_out), out_path,
                    top['feature'], top['accuracy'], top['correlation'],
                )
            ))

    @staticmethod
    def _load_data(pair):
        for suffix in ['H1', 'H4', 'Daily']:
            path = os.path.join(DATA_DIR, '{}_{}.csv'.format(pair, suffix))
            if not os.path.exists(path):
                continue

            df = pd.read_csv(path)
            df.columns = [c.lower().strip() for c in df.columns]

            for ts_col in ['timestamp', 'datetime', 'date']:
                if ts_col in df.columns:
                    df.index = pd.to_datetime(df[ts_col], errors='coerce')
                    break

            df = df.sort_index()
            col_map = {'<open>': 'open', '<high>': 'high', '<low>': 'low', '<close>': 'close'}
            df.rename(columns=col_map, inplace=True)

            required = {'open', 'high', 'low', 'close'}
            if not required.issubset(df.columns):
                continue

            for col in required:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            return df.dropna(subset=list(required))

        return None
