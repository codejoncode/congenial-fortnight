"""
signals/management/commands/backtest_signal.py

Run a walk-forward backtest for EURUSD and/or XAUUSD using the trained SignalEngine.
Outputs a human-readable performance report and optionally saves JSON.

Usage:
    python manage.py backtest_signal
    python manage.py backtest_signal --pair EURUSD
    python manage.py backtest_signal --pair XAUUSD --test-pct 0.30 --save
    python manage.py backtest_signal --pair EURUSD --signal-every 12 --max-hold 48

The --test-pct flag controls how much of the end of the dataset is used as the
test window (default 0.20 = last 20% of bars, ~1000 bars on H1 = 42 days).
"""

import json
import os
from datetime import datetime

import pandas as pd

from django.core.management.base import BaseCommand

from signals.signal_engine import SignalEngine
from trading.backtest import run_backtest

PAIRS    = ['EURUSD', 'XAUUSD']
DATA_DIR = 'data'


class Command(BaseCommand):
    help = 'Walk-forward backtest using the trained SignalEngine. Reports win rate, PF, drawdown.'

    def add_arguments(self, parser):
        parser.add_argument('--pair', type=str, default=None,
                            help='Specific pair (default: all)')
        parser.add_argument('--test-pct', type=float, default=0.20,
                            help='Fraction of data to use as test window (default: 0.20)')
        parser.add_argument('--signal-every', type=int, default=24,
                            help='Generate a signal every N bars (default: 24 = daily on H1)')
        parser.add_argument('--max-hold', type=int, default=24,
                            help='Max bars to hold a position (default: 24)')
        parser.add_argument('--save', action='store_true',
                            help='Save results to backtest_{pair}_{date}.json')

    def handle(self, *args, **options):
        pairs  = [options['pair'].upper()] if options.get('pair') else PAIRS
        engine = SignalEngine()

        self.stdout.write('\n' + '=' * 65)
        self.stdout.write('  WALK-FORWARD BACKTEST REPORT')
        self.stdout.write('  Test window: last {:.0f}% of H1 bars'.format(options['test_pct'] * 100))
        self.stdout.write('  Signal interval: every {} bars  |  Max hold: {} bars'.format(
            options['signal_every'], options['max_hold']))
        self.stdout.write('=' * 65 + '\n')

        all_results = {}

        for pair in pairs:
            if not engine.models_exist(pair):
                self.stdout.write(self.style.ERROR(
                    '[{}] Models not found. Run: python manage.py train_models'.format(pair)
                ))
                continue

            df = self._load_data(pair)
            if df is None:
                self.stdout.write(self.style.ERROR(
                    '[{}] No data file found in {}/'.format(pair, DATA_DIR)
                ))
                continue

            self.stdout.write('[{}] Running backtest on {} bars...'.format(pair, len(df)))

            result = run_backtest(
                pair=pair,
                df=df,
                engine=engine,
                test_from_pct=1.0 - options['test_pct'],
                signal_every=options['signal_every'],
                max_hold=options['max_hold'],
            )

            if result.error:
                self.stdout.write(self.style.ERROR(
                    '[{}] Backtest error: {}'.format(pair, result.error)
                ))
                continue

            all_results[pair] = result.to_dict()
            self._print_report(pair, result)

            if options['save']:
                date_str  = datetime.utcnow().strftime('%Y%m%d')
                out_path  = 'backtest_{}_{}.json'.format(pair, date_str)
                with open(out_path, 'w') as f:
                    json.dump(result.to_dict(), f, indent=2, default=str)
                self.stdout.write(self.style.SUCCESS(
                    '  Saved: {}'.format(out_path)
                ))

        self.stdout.write('\n' + '=' * 65)

        # Overall verdict
        profitable = [p for p, r in all_results.items() if r.get('is_profitable')]
        if len(profitable) == len(all_results) and all_results:
            self.stdout.write(self.style.SUCCESS(
                'VERDICT: ALL PAIRS PROFITABLE -- signal engine is ready for live use.'
            ))
        elif profitable:
            self.stdout.write(self.style.WARNING(
                'VERDICT: MIXED -- profitable on {} of {} pairs.'.format(
                    len(profitable), len(all_results)
                )
            ))
        else:
            self.stdout.write(self.style.ERROR(
                'VERDICT: NOT PROFITABLE -- retrain with more data or adjust thresholds.'
            ))
        self.stdout.write('=' * 65 + '\n')

    def _print_report(self, pair, result):
        verdict_fn = {
            'PROFITABLE':   self.style.SUCCESS,
            'MARGINAL':     self.style.WARNING,
            'UNPROFITABLE': self.style.ERROR,
        }.get(result.verdict, lambda x: x)

        self.stdout.write('\n  [{pair}]  {verdict}'.format(
            pair=pair, verdict=verdict_fn(result.verdict)
        ))
        self.stdout.write('  {}'.format('-' * 55))
        self.stdout.write('  Trades        : {n_trades}  ({n_wins} wins / {n_losses} losses)'.format(
            **result.__dict__))
        self.stdout.write('  Win Rate      : {:.1f}%'.format(result.win_rate * 100))
        self.stdout.write('  Profit Factor : {:.3f}  (>1.0 = profitable, >1.5 = strong)'.format(
            result.profit_factor))
        self.stdout.write('  Net Pips      : {:+.1f}  (gross win: {:.1f}  loss: {:.1f})'.format(
            result.net_pips, result.gross_profit_pips, result.gross_loss_pips))
        self.stdout.write('  Avg Win/Loss  : +{:.1f} / -{:.1f} pips'.format(
            result.avg_win_pips, result.avg_loss_pips))
        self.stdout.write('  Max Drawdown  : {:.1f} pips'.format(result.max_drawdown_pips))
        self.stdout.write('  Sharpe Ratio  : {:.3f}'.format(result.sharpe_ratio))
        self.stdout.write('  Expectancy    : {:+.2f} pips/trade'.format(result.expectancy_pips))

        if result.trades:
            self.stdout.write('\n  Last 5 trades:')
            for t in result.trades[-5:]:
                icon = '[W]' if t.outcome == 'win' else '[L]'
                self.stdout.write('    {} {dir:8s} prob={prob:.2f}  {pips:+.1f} pips'.format(
                    icon, dir=t.direction.upper(), prob=t.probability, pips=t.pips))

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
