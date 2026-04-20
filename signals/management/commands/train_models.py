"""
signals/management/commands/train_models.py

Train RF + XGB ensemble models for EURUSD and XAUUSD using the clean SignalEngine.
Uses Daily CSV as primary training source (longest history) with FRED fundamentals.
Must be run BEFORE run_daily_signal can generate signals.

Usage:
    python manage.py train_models                 # train all pairs
    python manage.py train_models --pair EURUSD   # train one pair
    python manage.py train_models --fetch-data    # update data first, then train

Artifacts saved to models/:
    {PAIR}_rf.joblib       — calibrated RandomForestClassifier
    {PAIR}_xgb.joblib      — XGBClassifier
    {PAIR}_scaler.joblib   — StandardScaler
    {PAIR}_meta.json       — feature list + thresholds + accuracy metrics
"""

import os
import pandas as pd

from django.core.management.base import BaseCommand
from django.core.management import call_command

from signals.signal_engine import SignalEngine

PAIRS    = ['EURUSD', 'XAUUSD']
DATA_DIR = 'data'

# Per-pair target parameters optimised for daily bars:
#   EURUSD: 1.5:1 RR (breakeven=40%), 15-day window — EURUSD often ranges, 2:1 in 10 days too tight
#   XAUUSD: 2:1 RR (breakeven=33%), 10-day window — gold trends strongly, higher RR is achievable
PAIR_TRAIN_PARAMS = {
    'EURUSD': {'lookahead': 15, 'tp_mult': 1.5, 'sl_mult': 1.0},
    'XAUUSD': {'lookahead': 10, 'tp_mult': 2.0, 'sl_mult': 1.0},
}


class Command(BaseCommand):
    help = 'Train RF + XGB signal models for EURUSD and XAUUSD.'

    def add_arguments(self, parser):
        parser.add_argument('--pair', type=str, default=None,
                            help='Specific pair to train (EURUSD or XAUUSD).')
        parser.add_argument('--fetch-data', action='store_true',
                            help='Fetch fresh price data (all timeframes) before training.')
        parser.add_argument('--full', action='store_true',
                            help='Fetch maximum history before training (implies --fetch-data).')

    def handle(self, *args, **options):
        pairs  = [options['pair'].upper()] if options.get('pair') else PAIRS
        engine = SignalEngine()

        if options['fetch_data'] or options.get('full'):
            self.stdout.write('  Fetching fresh price data...')
            kwargs = {'full': True} if options.get('full') else {}
            if options.get('pair'):
                kwargs['pairs'] = [options['pair'].upper()]
            try:
                call_command('fetch_price_data', **kwargs)
            except Exception as exc:
                self.stdout.write(self.style.WARNING(f'  fetch_price_data failed: {exc}'))

        self.stdout.write('\n{}\n  MODEL TRAINING\n{}\n'.format('=' * 60, '=' * 60))

        for pair in pairs:
            self.stdout.write('\n[{}] Loading data...'.format(pair))
            df = self._load_data(pair)

            if df is None:
                self.stdout.write(self.style.ERROR(
                    '[{}] No data file found in {}/.\n'
                    '  Run:  python manage.py train_models --fetch-data'.format(pair, DATA_DIR)
                ))
                continue

            self.stdout.write('[{}] {} rows loaded'.format(pair, len(df)))

            if len(df) < 200:
                self.stdout.write(self.style.WARNING(
                    '[{}] Only {} rows — need ≥200. Run --fetch-data --full.'.format(pair, len(df))
                ))

            try:
                train_params = PAIR_TRAIN_PARAMS.get(pair, {'lookahead': 10, 'tp_mult': 2.0, 'sl_mult': 1.0})
                self.stdout.write('[{}] Target: {:.1f}:1 RR, {}-bar lookahead (breakeven={:.0%})'.format(
                    pair, train_params['tp_mult'], train_params['lookahead'],
                    1 / (1 + train_params['tp_mult']),
                ))
                meta = engine.train(pair, df, **train_params)
                exp_r   = meta.get('expectancy_R', 0)
                thr_wr  = meta.get('thr_win_rate', 0)
                thr     = meta.get('threshold', 0)
                breakeven = 1 / (1 + meta.get('tp_mult', 2.0))

                positive = exp_r > 0
                style = self.style.SUCCESS if positive else self.style.WARNING

                self.stdout.write(style(
                    '[{}] Training complete\n'
                    '       CV accuracy  : {:.1%}\n'
                    '       CV AUC       : {:.4f}\n'
                    '       CV win rate  : {:.1%}\n'
                    '       Threshold    : {:.2f}\n'
                    '       Win@threshold: {:.1%}  (breakeven={:.1%})\n'
                    '       Expectancy   : {:.3f}R  {}\n'
                    '       Samples      : {:,}\n'
                    '       Features     : {}\n'
                    '       Trained at   : {}'.format(
                        pair,
                        meta['cv_accuracy'],
                        meta.get('cv_auc', 0),
                        meta.get('cv_win_rate', 0),
                        thr,
                        thr_wr, breakeven,
                        exp_r, '[POSITIVE EDGE]' if positive else '[NO EDGE]',
                        meta['n_samples'],
                        meta['n_features'],
                        meta['trained_at'],
                    )
                ))
            except ValueError as exc:
                self.stdout.write(self.style.ERROR('[{}] Training failed: {}'.format(pair, exc)))
            except Exception:
                import traceback
                self.stdout.write(self.style.ERROR(
                    '[{}] Unexpected training error:\n{}'.format(pair, traceback.format_exc())
                ))

        self.stdout.write('\n' + '=' * 60)
        self.stdout.write('Next step:  python manage.py run_daily_signal --force')
        self.stdout.write('=' * 60 + '\n')

    @staticmethod
    def _load_data(pair):
        # Prefer Daily (most history) → H4 → H1 for training
        for suffix in ['Daily', 'H4', 'H1', 'Weekly']:
            path = os.path.join(DATA_DIR, '{}_{}.csv'.format(pair, suffix))
            if not os.path.exists(path):
                continue

            df = pd.read_csv(path)
            df.columns = [c.lower().strip() for c in df.columns]

            for ts_col in ['timestamp', 'datetime', 'date', '<date>']:
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

