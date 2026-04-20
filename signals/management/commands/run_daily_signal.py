"""
signals/management/commands/run_daily_signal.py

Daily signal generation using the clean SignalEngine pipeline.
Replaces the old broken implementation that had:
  - Missing calibrator artifacts (FileNotFoundError on every run)
  - Feature set mismatch between training and inference
  - --force-signals that was registered but never wired in add_arguments

Usage:
    python manage.py run_daily_signal                          # generate if not yet today
    python manage.py run_daily_signal --force                  # always regenerate
    python manage.py run_daily_signal --fetch-data             # update market data first
    python manage.py run_daily_signal --fetch-data --force     # full refresh
    python manage.py run_daily_signal --pair EURUSD --force    # single pair
    python manage.py run_daily_signal --dry-run                # print without saving
"""

import logging
import os
from datetime import datetime

import pandas as pd

from django.core.management.base import BaseCommand

from signals.models import Signal
from signals.signal_engine import SignalEngine

logger = logging.getLogger(__name__)

PAIRS    = ['EURUSD', 'XAUUSD']
DATA_DIR = 'data'
TICKER_MAP = {'EURUSD': 'EURUSD=X', 'XAUUSD': 'GC=F'}


class Command(BaseCommand):
    help = 'Generate today\'s trading signals using the clean SignalEngine (RF + XGB ensemble).'

    def add_arguments(self, parser):
        parser.add_argument('--pair', type=str, default=None,
                            help='Specific pair (EURUSD or XAUUSD). Defaults to all.')
        parser.add_argument('--force', action='store_true',
                            help='Regenerate even if today\'s signal already exists.')
        parser.add_argument('--fetch-data', action='store_true',
                            help='Fetch fresh H1 market data from yfinance before generating.')
        parser.add_argument('--dry-run', action='store_true',
                            help='Print signal without saving to database.')

    def handle(self, *args, **options):
        pairs  = [options['pair'].upper()] if options.get('pair') else PAIRS
        today  = datetime.utcnow().date()
        engine = SignalEngine()

        if options['fetch_data']:
            self._fetch_data(pairs)

        generated = []
        for pair in pairs:
            result = self._process_pair(pair, today, engine, options)
            if result:
                generated.append(result)

        if generated:
            self.stdout.write(self.style.SUCCESS(
                '\nDone -- {} signal(s) generated: {}'.format(
                    len(generated),
                    ', '.join('{pair}={signal}'.format(**r) for r in generated),
                )
            ))
        else:
            self.stdout.write('No new signals generated (use --force to override).')

    # ─────────────────────────────────────────────────────────────────────────

    def _process_pair(self, pair, today, engine, options):
        dry   = options['dry_run']
        force = options['force']

        if not force and Signal.objects.filter(pair=pair, date=today).exists():
            self.stdout.write(
                '[{}] Signal for {} already in DB — skipped (use --force to regenerate).'
                .format(pair, today)
            )
            return None

        if not engine.models_exist(pair):
            self.stdout.write(self.style.ERROR(
                '[{pair}] Model artifacts missing.\n'
                '  Run:  python manage.py train_models\n'
                '  Or full refresh:  python manage.py daily_workflow --train'.format(pair=pair)
            ))
            return None

        df = self._load_data(pair)
        if df is None:
            self.stdout.write(self.style.ERROR(
                '[{}] No data file in {}/. Run with --fetch-data.'.format(pair, DATA_DIR)
            ))
            return None

        if len(df) < 60:
            self.stdout.write(self.style.ERROR(
                '[{}] Only {} rows — need >= 60.'.format(pair, len(df))
            ))
            return None

        try:
            result = engine.predict(pair, df)
        except FileNotFoundError as exc:
            self.stdout.write(self.style.ERROR(str(exc)))
            return None
        except ValueError as exc:
            self.stdout.write(self.style.ERROR('[{}] Prediction error: {}'.format(pair, exc)))
            return None
        except Exception:
            logger.exception('[%s] Unexpected error during prediction', pair)
            return None

        arrow  = 'UP' if result['signal'] == 'bullish' else ('DN' if result['signal'] == 'bearish' else '--')
        cv_acc = result['model_info'].get('cv_accuracy')
        cv_str = '  model_cv={:.1%}'.format(cv_acc) if cv_acc else ''

        self.stdout.write(
            '[{pair}] {arrow} {sig:<10s} prob={prob:.3f}  conf={conf:.3f}{cv}\n'
            '         entry={entry}  SL={sl}  TP={tp}  RR={rr}'.format(
                pair=pair, arrow=arrow,
                sig=result['signal'].upper(),
                prob=result['probability'], conf=result['confidence'], cv=cv_str,
                entry=result['entry'], sl=result['stop_loss'],
                tp=result['take_profit'], rr=result['risk_reward'],
            )
        )

        if not dry:
            Signal.objects.update_or_create(
                pair=pair,
                date=today,
                defaults={
                    'signal':      result['signal'],
                    'probability': result['probability'],
                    'confidence':  result['confidence'],
                    'entry_price': result['entry'],
                    'stop_loss':   result['stop_loss'],
                    'take_profit': result['take_profit'],
                    'risk_reward': result['risk_reward'],
                    'atr':         result.get('atr'),
                    'source':      'engine',
                },
            )
            self.stdout.write(self.style.SUCCESS('         [saved] Signal written to DB'))

            # Fire email alert (non-fatal — never crashes signal generation)
            if result['signal'] != 'no_signal':
                try:
                    from signals.notifications import send_signal_alert
                    from signals.decision_engine import evaluate
                    decision = evaluate(result)
                    sent = send_signal_alert(result, decision)
                    if sent:
                        self.stdout.write('         [email] Alert sent')
                except Exception:
                    pass

        return result

    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_data(pair):
        for suffix in ['H1', 'H4', 'Daily', 'Weekly']:
            path = os.path.join(DATA_DIR, '{pair}_{suffix}.csv'.format(pair=pair, suffix=suffix))
            if not os.path.exists(path):
                continue

            df = pd.read_csv(path)
            df.columns = [c.lower().strip() for c in df.columns]

            for ts_col in ['timestamp', 'datetime', 'date', '<date>']:
                if ts_col in df.columns:
                    df.index = pd.to_datetime(df[ts_col], errors='coerce')
                    break

            df = df.sort_index()

            # MT4-style column rename
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
            print('yfinance not installed — skipping data fetch')
            return

        os.makedirs(DATA_DIR, exist_ok=True)

        for pair in pairs:
            ticker = TICKER_MAP.get(pair, pair)
            try:
                print('[{}] Fetching from Yahoo Finance ({})...'.format(pair, ticker))
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
                print('[{}] Saved {} rows to {}'.format(pair, len(df_out), out_path))

            except Exception as exc:
                print('[{}] Data fetch failed: {}'.format(pair, exc))
