"""
signals/management/commands/train_models.py

Train RF + XGB ensemble models for EURUSD and XAUUSD using the clean SignalEngine.
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

from signals.signal_engine import SignalEngine

PAIRS    = ['EURUSD', 'XAUUSD']
DATA_DIR = 'data'
TICKER_MAP = {'EURUSD': 'EURUSD=X', 'XAUUSD': 'GC=F'}


class Command(BaseCommand):
    help = 'Train RF + XGB signal models for EURUSD and XAUUSD.'

    def add_arguments(self, parser):
        parser.add_argument('--pair', type=str, default=None,
                            help='Specific pair to train (EURUSD or XAUUSD).')
        parser.add_argument('--fetch-data', action='store_true',
                            help='Fetch latest 120-day H1 data from yfinance before training.')

    def handle(self, *args, **options):
        pairs  = [options['pair'].upper()] if options.get('pair') else PAIRS
        engine = SignalEngine()

        if options['fetch_data']:
            self._fetch_data(pairs)

        self.stdout.write('\n{}\n  MODEL TRAINING\n{}\n'.format('=' * 60, '=' * 60))

        for pair in pairs:
            self.stdout.write('\n[{}] Loading data...'.format(pair))
            df = self._load_data(pair)

            if df is None:
                self.stdout.write(self.style.ERROR(
                    '[{}] No data file found in {}/.\n'
                    '  Run with --fetch-data or:\n'
                    '    python manage.py run_daily_signal --fetch-data'.format(pair, DATA_DIR)
                ))
                continue

            self.stdout.write('[{}] {} rows loaded from {}'.format(pair, len(df), DATA_DIR))

            if len(df) < 200:
                self.stdout.write(self.style.WARNING(
                    '[{}] Only {} rows — training on sparse data. '
                    'Recommend >= 300 rows (run --fetch-data).'.format(pair, len(df))
                ))

            try:
                meta = engine.train(pair, df)
                self.stdout.write(self.style.SUCCESS(
                    '[{}] Training complete\n'
                    '       CV accuracy : {:.1%}\n'
                    '       CV AUC      : {:.4f}\n'
                    '       Samples     : {:,}\n'
                    '       Features    : {}\n'
                    '       Trained at  : {}'.format(
                        pair,
                        meta['cv_accuracy'],
                        meta.get('cv_auc', 0),
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
        for suffix in ['H1', 'H4', 'Daily', 'Weekly']:
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

    @staticmethod
    def _fetch_data(pairs):
        try:
            import yfinance as yf
        except ImportError:
            print('yfinance not installed — skipping')
            return

        os.makedirs(DATA_DIR, exist_ok=True)
        for pair in pairs:
            ticker = TICKER_MAP.get(pair, pair)
            try:
                print('[{}] Fetching {} (120d H1)...'.format(pair, ticker))
                raw = yf.download(ticker, period='120d', interval='1h',
                                  progress=False, auto_adjust=True)
                if raw is None or raw.empty:
                    print('[{}] No data returned'.format(pair))
                    continue

                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)

                raw = raw.reset_index()
                ts_col = 'Datetime' if 'Datetime' in raw.columns else 'Date'
                raw['timestamp'] = pd.to_datetime(raw[ts_col])
                raw = raw.rename(columns={
                    'Open': 'open', 'High': 'high',
                    'Low': 'low', 'Close': 'close', 'Volume': 'volume',
                })
                out_cols = [c for c in ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                            if c in raw.columns]
                df_out = raw[out_cols].dropna()
                out_path = os.path.join(DATA_DIR, '{}_H1.csv'.format(pair))
                df_out.to_csv(out_path, index=False)
                print('[{}] Saved {} rows -> {}'.format(pair, len(df_out), out_path))
            except Exception as exc:
                print('[{}] Failed: {}'.format(pair, exc))
