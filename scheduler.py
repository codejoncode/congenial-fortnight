"""
scheduler.py — Background scheduler for the Forex Signal System.

Runs as a standalone process alongside Django.  Uses APScheduler to:
  • Every day at 22:00 UTC   → fetch data + generate signals + evaluate
  • On 1st & 15th at 22:30   → also retrain models (bi-weekly)
  • Every 30 min (daytime)   → update open paper-trade positions (SL/TP check)

Usage:
    python scheduler.py                 # run with scheduled times
    python scheduler.py --run-now       # run daily workflow immediately (for testing)

Start this AFTER Django is running (it calls management commands via subprocess).
"""

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger('scheduler')

try:
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger
except ImportError:
    logger.error(
        'APScheduler not installed.\n'
        'Run:  pip install "apscheduler>=3.10,<4"'
    )
    sys.exit(1)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON      = sys.executable
MANAGE      = [PYTHON, os.path.join(PROJECT_DIR, 'manage.py')]


def _run(*args: str, label: str = '') -> bool:
    """Run a manage.py command, stream output, return True on success."""
    cmd = MANAGE + list(args)
    tag = label or args[0] if args else 'cmd'
    logger.info('[%s] Starting: %s', tag, ' '.join(args))
    try:
        proc = subprocess.run(
            cmd, cwd=PROJECT_DIR, capture_output=False, text=True
        )
        if proc.returncode == 0:
            logger.info('[%s] Completed successfully', tag)
        else:
            logger.warning('[%s] Exited with code %d', tag, proc.returncode)
        return proc.returncode == 0
    except Exception as exc:
        logger.error('[%s] Exception: %s', tag, exc)
        return False


# ── Jobs ──────────────────────────────────────────────────────────────────────

def daily_workflow():
    """Fetch data → generate signals → evaluate. Runs every day at 22:00 UTC."""
    logger.info('=' * 55)
    logger.info('DAILY WORKFLOW START — %s UTC', datetime.utcnow().strftime('%Y-%m-%d %H:%M'))
    logger.info('=' * 55)
    _run('run_daily_signal', '--fetch-data', '--force', label='signals')
    _run('evaluate_signals', label='evaluate')
    logger.info('DAILY WORKFLOW DONE')


def retrain_workflow():
    """Fetch data → retrain models → generate → evaluate. Runs bi-weekly."""
    logger.info('=' * 55)
    logger.info('RETRAIN WORKFLOW START — %s UTC', datetime.utcnow().strftime('%Y-%m-%d %H:%M'))
    logger.info('=' * 55)
    _run('run_daily_signal', '--fetch-data', '--force', label='fetch-data')
    _run('train_models', label='train')
    _run('run_daily_signal', '--force', label='signals-post-train')
    _run('evaluate_signals', label='evaluate')
    logger.info('RETRAIN WORKFLOW DONE')


def update_positions():
    """Check SL/TP on open paper trades. Runs every 30 minutes during market hours."""
    logger.info('[positions] Checking open paper trade positions...')
    try:
        import django
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'forex_signal.settings')
        if not django.conf.settings.configured:
            django.setup()

        import requests
        try:
            resp = requests.post(
                'http://localhost:8000/api/paper/positions/update/',
                timeout=15,
            )
            if resp.ok:
                data = resp.json()
                closed = data.get('positions_closed', 0)
                logger.info('[positions] Updated: %d position(s) closed', closed)
            else:
                logger.warning('[positions] HTTP %d from position update', resp.status_code)
        except requests.ConnectionError:
            logger.debug('[positions] Django not reachable — skipping position update')
    except Exception as exc:
        logger.error('[positions] Error: %s', exc)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Forex Signal System Scheduler')
    parser.add_argument('--run-now', action='store_true',
                        help='Execute daily_workflow immediately, then start scheduler')
    args = parser.parse_args()

    os.chdir(PROJECT_DIR)

    scheduler = BlockingScheduler(timezone='UTC')

    # Daily signal generation — 10 PM UTC (after US session, before Asian open)
    scheduler.add_job(
        daily_workflow,
        CronTrigger(hour=22, minute=0, timezone='UTC'),
        id='daily_workflow',
        name='Daily: fetch data + generate signals + evaluate',
        misfire_grace_time=3600,  # allow up to 1h late if system was down
    )

    # Bi-weekly model retraining — 1st and 15th of each month at 22:30 UTC
    scheduler.add_job(
        retrain_workflow,
        CronTrigger(day='1,15', hour=22, minute=30, timezone='UTC'),
        id='retrain_workflow',
        name='Bi-weekly: retrain models',
        misfire_grace_time=7200,
    )

    # Position SL/TP monitoring — every 30 minutes, 8am–10pm UTC (market hours)
    scheduler.add_job(
        update_positions,
        CronTrigger(hour='8-21', minute='0,30', timezone='UTC'),
        id='update_positions',
        name='Every 30min: check SL/TP on open positions',
    )

    if args.run_now:
        logger.info('--run-now flag: executing daily_workflow immediately...')
        daily_workflow()

    logger.info('')
    logger.info('Scheduler started. Jobs:')
    logger.info('  • Daily signals  : 22:00 UTC every day')
    logger.info('  • Model retrain  : 22:30 UTC on 1st and 15th of each month')
    logger.info('  • Position check : every 30 min, 08:00–21:30 UTC')
    logger.info('')
    logger.info('Press Ctrl+C to stop.')

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info('Scheduler stopped.')


if __name__ == '__main__':
    main()
