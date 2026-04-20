"""
signals/management/commands/daily_workflow.py

Master command that orchestrates the full daily pipeline:
  1. Fetch latest H1 market data
  2. Generate signals (using trained models)
  3. Evaluate per-feature accuracy (update CSVs)

Optionally also retrain models (use --train for bi-weekly retraining).

Usage:
    python manage.py daily_workflow                  # update data + signals + evaluate
    python manage.py daily_workflow --train          # also retrain models (slow)
    python manage.py daily_workflow --skip-evaluate  # skip evaluation step
    python manage.py daily_workflow --dry-run        # print without saving

This is the command invoked by scheduler.py and start.sh.
"""

import time
from django.core.management.base import BaseCommand
from django.core.management import call_command


class Command(BaseCommand):
    help = 'Master daily workflow: fetch data → generate signals → evaluate accuracy.'

    def add_arguments(self, parser):
        parser.add_argument('--train', action='store_true',
                            help='Retrain models before generating signals (takes 5–15 min).')
        parser.add_argument('--skip-evaluate', action='store_true',
                            help='Skip the evaluate_signals step.')
        parser.add_argument('--dry-run', action='store_true',
                            help='Pass --dry-run to signal generation (no DB writes).')
        parser.add_argument('--pair', type=str, default=None,
                            help='Limit to one pair.')

    def handle(self, *args, **options):
        t0 = time.time()
        self.stdout.write('\n' + '=' * 60)
        self.stdout.write('  FOREX SIGNAL DAILY WORKFLOW')
        self.stdout.write('=' * 60 + '\n')

        pair_args = {}
        if options.get('pair'):
            pair_args['pair'] = options['pair']

        # ── Step 1: Fetch data ────────────────────────────────────────────
        self.stdout.write('\n[1/{}] Fetching latest market data...'.format(
            3 if not options['train'] and not options['skip_evaluate'] else 4
        ))
        call_command('run_daily_signal', fetch_data=True, dry_run=True, **pair_args)

        step = 2

        # ── Step 2 (optional): Retrain models ────────────────────────────
        if options['train']:
            self.stdout.write('\n[{}/{}] Retraining models (this takes a few minutes)...'.format(
                step, step + 2
            ))
            call_command('train_models', **pair_args)
            step += 1

        # ── Step 3: Generate signals ──────────────────────────────────────
        self.stdout.write('\n[{}/{}] Generating signals...'.format(step, step + 1))
        gen_kwargs = dict(force=True, fetch_data=False, dry_run=options['dry_run'])
        gen_kwargs.update(pair_args)
        call_command('run_daily_signal', **gen_kwargs)
        step += 1

        # ── Step 4: Evaluate ──────────────────────────────────────────────
        if not options['skip_evaluate']:
            self.stdout.write('\n[{}/{}] Evaluating signal accuracy...'.format(step, step))
            call_command('evaluate_signals', **pair_args)

        elapsed = time.time() - t0
        self.stdout.write('\n' + '=' * 60)
        self.stdout.write(self.style.SUCCESS(
            'Daily workflow complete in {:.1f}s'.format(elapsed)
        ))
        self.stdout.write('=' * 60 + '\n')
