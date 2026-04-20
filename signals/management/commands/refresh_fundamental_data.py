"""
management command: refresh_fundamental_data

Refreshes all FRED economic indicator CSV files using the public FRED CSV
endpoint - no API key required.  Falls back to the FRED API (if FRED_API_KEY
is set) for series that fail the public endpoint.

Usage:
    python manage.py refresh_fundamental_data
    python manage.py refresh_fundamental_data --series VIXCLS DGS10
    python manage.py refresh_fundamental_data --dry-run
"""

import os
import time
import requests
import pandas as pd
from io import StringIO
from pathlib import Path
from django.core.management.base import BaseCommand

DATA_DIR = Path(__file__).resolve().parents[3] / 'data'

# All FRED series the pipeline uses, with canonical column names
FRED_SERIES = {
    'DEXUSEU':            'dexuseu',
    'DEXJPUS':            'dexjpus',
    'DEXCHUS':            'dexchus',
    'FEDFUNDS':           'fedfunds',
    'DFF':                'dff',
    'CPIAUCSL':           'cpiaucsl',
    'CPALTT01USM661S':    'cpaltt01usm661s',
    'UNRATE':             'unrate',
    'PAYEMS':             'payems',
    'INDPRO':             'indpro',
    'DGORDER':            'dgorder',
    'ECBDFR':             'ecbdfr',
    'ECBRR':              'ecbrr',
    'CP0000EZ19M086NEST': 'cp0000ez19m086nest',
    'LRHUTTTTDEM156S':    'lrhuttttdem156s',
    'GOLDAMGBD228NLBM':   'goldamgbd228nlbm',
    'DCOILWTICO':         'dcoilwtico',
    'DCOILBRENTEU':       'dcoilbrenteu',
    'VIXCLS':             'vixcls',
    'DGS10':              'dgs10',
    'DGS2':               'dgs2',
    'DGS3MO':             'dgs3mo',
    'BOPGSTB':            'bopgstb',
    'T10YIE':             't10yie',
}

FRED_PUBLIC_URL = 'https://fred.stlouisfed.org/graph/fredgraph.csv'
FRED_API_URL    = 'https://api.stlouisfed.org/fred/series/observations'


def _fetch_public(series_id: str) -> pd.DataFrame | None:
    """Fetch via FRED public CSV endpoint (no key required)."""
    try:
        resp = requests.get(
            FRED_PUBLIC_URL,
            params={'id': series_id},
            timeout=20,
            headers={'User-Agent': 'congenial-fortnight/1.0 (fundamental data refresh)'},
        )
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        df.columns = [c.lower() for c in df.columns]
        # FRED public endpoint returns: observation_date, <SERIES_ID>
        if 'observation_date' in df.columns:
            df = df.rename(columns={'observation_date': 'date'})
        # Drop rows where value is '.' (FRED missing value marker)
        val_col = [c for c in df.columns if c != 'date'][0]
        df = df[df[val_col] != '.'].copy()
        df[val_col] = pd.to_numeric(df[val_col], errors='coerce')
        df = df.dropna(subset=[val_col])
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        return df.reset_index(drop=True)
    except Exception:
        return None


def _fetch_api(series_id: str, api_key: str) -> pd.DataFrame | None:
    """Fetch via FRED REST API (requires key)."""
    try:
        resp = requests.get(
            FRED_API_URL,
            params={
                'series_id':   series_id,
                'api_key':     api_key,
                'file_type':   'json',
                'observation_start': '2000-01-01',
            },
            timeout=20,
        )
        resp.raise_for_status()
        obs = resp.json().get('observations', [])
        if not obs:
            return None
        rows = [(o['date'], o['value']) for o in obs if o['value'] != '.']
        col = series_id.lower()
        df = pd.DataFrame(rows, columns=['date', col])
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=[col])
        return df
    except Exception:
        return None


class Command(BaseCommand):
    help = 'Refresh FRED fundamental data CSV files (no API key required)'

    def add_arguments(self, parser):
        parser.add_argument(
            '--series', nargs='+', metavar='SERIES_ID',
            help='Specific FRED series to refresh (default: all)',
        )
        parser.add_argument(
            '--dry-run', action='store_true',
            help='Show what would be updated without writing files',
        )
        parser.add_argument(
            '--delay', type=float, default=1.0,
            help='Seconds between requests (default 1.0 to respect FRED rate limits)',
        )

    def handle(self, *args, **options):
        DATA_DIR.mkdir(exist_ok=True)
        api_key   = os.getenv('FRED_API_KEY', '')
        dry_run   = options['dry_run']
        delay     = options['delay']
        requested = options['series']

        target = {k: v for k, v in FRED_SERIES.items()
                  if not requested or k in requested}

        self.stdout.write(f'\nRefreshing {len(target)} FRED series into {DATA_DIR}/\n')

        ok = failed = skipped = 0

        for series_id, col_name in target.items():
            self.stdout.write(f'  [{series_id}] ', ending='')

            df = _fetch_public(series_id)
            source = 'public'

            if df is None and api_key:
                df = _fetch_api(series_id, api_key)
                source = 'api'

            if df is None:
                self.stdout.write(self.style.WARNING('FAILED - skipping'))
                failed += 1
                continue

            # Normalise column name to match existing files
            val_col = [c for c in df.columns if c != 'date'][0]
            if val_col != col_name:
                df = df.rename(columns={val_col: col_name})

            out_path = DATA_DIR / f'{series_id}.csv'

            if dry_run:
                self.stdout.write(
                    self.style.SUCCESS(f'OK ({source}) - {len(df)} rows [dry-run, not saved]')
                )
                skipped += 1
            else:
                df.to_csv(out_path, index=False)
                self.stdout.write(
                    self.style.SUCCESS(f'OK ({source}) - {len(df)} rows -> {out_path.name}')
                )
                ok += 1

            time.sleep(delay)

        self.stdout.write(
            f'\nDone: {ok} updated, {failed} failed, {skipped} skipped (dry-run)\n'
        )
        if failed:
            self.stderr.write(
                f'{failed} series could not be fetched. '
                'Set FRED_API_KEY for better coverage or check network.\n'
            )
