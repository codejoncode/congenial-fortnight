"""
management command: fetch_price_data

Fetches current OHLCV price data for EURUSD and XAUUSD across all timeframes
(H1, H4, Daily).  Uses multiple sources with automatic fallback:

  1. yfinance   — free, no key, H1/H4/Daily
  2. Alpha Vantage FX_DAILY — free (500 calls/day), key: ALPHA_VANTAGE_API_KEY

Appends new rows to existing CSVs so historical data is preserved.
Files written: data/EURUSD_H1.csv, EURUSD_H4.csv, EURUSD_Daily.csv,
               data/XAUUSD_H1.csv, XAUUSD_H4.csv, XAUUSD_Daily.csv

Usage:
    python manage.py fetch_price_data
    python manage.py fetch_price_data --pairs EURUSD
    python manage.py fetch_price_data --full      # fetch max history
"""

import os
import logging
from pathlib import Path
from django.core.management.base import BaseCommand

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[3] / 'data'

TICKER_MAP = {
    'EURUSD': 'EURUSD=X',
    'XAUUSD': 'GC=F',
}

# Alpha Vantage from/to symbols for forex API
AV_SYMBOLS = {
    'EURUSD': ('EUR', 'USD'),
    'XAUUSD': ('XAU', 'USD'),
}

# yfinance period + interval for each timeframe
YF_CONFIG = {
    'H1':    {'interval': '1h',  'period': '120d'},
    'H4':    {'interval': '1h',  'period': '120d'},   # resample to H4
    'Daily': {'interval': '1d',  'period': '730d'},
}


def _load_yfinance(ticker: str, interval: str, period: str):
    import yfinance as yf
    df = yf.download(ticker, period=period, interval=interval,
                     progress=False, auto_adjust=True)
    if df is None or df.empty:
        return None
    if hasattr(df.columns, 'get_level_values'):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    ts_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
    df = df.rename(columns={
        ts_col: 'timestamp',
        'Open': 'open', 'High': 'high', 'Low': 'low',
        'Close': 'close', 'Volume': 'volume',
    })
    df['timestamp'] = df['timestamp'].astype(str)
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].dropna()


def _resample_h4(df_h1):
    """Resample H1 DataFrame to H4."""
    import pandas as pd
    df = df_h1.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    h4 = df.resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    h4 = h4.reset_index()
    h4['timestamp'] = h4['timestamp'].astype(str)
    return h4


def _load_av_daily(from_sym: str, to_sym: str, api_key: str):
    """Alpha Vantage FX_DAILY — free tier, daily bars only."""
    import requests
    resp = requests.get('https://www.alphavantage.co/query', params={
        'function':    'FX_DAILY',
        'from_symbol': from_sym,
        'to_symbol':   to_sym,
        'outputsize':  'full',
        'apikey':      api_key,
    }, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    ts = data.get('Time Series FX (Daily)', {})
    if not ts:
        return None
    import pandas as pd
    rows = []
    for date_str, bar in sorted(ts.items()):
        rows.append({
            'timestamp': date_str,
            'open':   float(bar['1. open']),
            'high':   float(bar['2. high']),
            'low':    float(bar['3. low']),
            'close':  float(bar['4. close']),
            'volume': 0,
        })
    return pd.DataFrame(rows)


def _merge_into_csv(path: Path, new_df, timeframe: str) -> int:
    """Append new rows not already in the CSV.  Returns count of new rows added."""
    import pandas as pd
    if path.exists():
        existing = pd.read_csv(path, dtype=str)
        # Normalize timestamp column name
        if 'timestamp' not in existing.columns and 'date' in existing.columns:
            existing = existing.rename(columns={'date': 'timestamp'})
        existing_ts = set(existing['timestamp'].astype(str).str[:16])
        new_df['timestamp'] = new_df['timestamp'].astype(str).str[:16]
        new_rows = new_df[~new_df['timestamp'].isin(existing_ts)].copy()
        if new_rows.empty:
            return 0
        combined = pd.concat([existing, new_rows], ignore_index=True)
        combined = combined.sort_values('timestamp').drop_duplicates('timestamp')
    else:
        combined = new_df.copy()

    combined.to_csv(path, index=False)
    return len(new_df[~new_df['timestamp'].isin(
        set(combined['timestamp'].astype(str).str[:16]) - set(new_df['timestamp'].astype(str).str[:16])
    )]) if path.exists() else len(new_df)


def _save_csv(path: Path, df, timeframe: str) -> int:
    """Write/merge CSV, return rows added."""
    import pandas as pd
    path.parent.mkdir(exist_ok=True)

    # Normalize to full timestamp strings (YYYY-MM-DD HH:MM:SS)
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed').dt.strftime('%Y-%m-%d %H:%M:%S')

    if path.exists():
        existing = pd.read_csv(path, dtype=str)
        ts_col = 'timestamp' if 'timestamp' in existing.columns else existing.columns[0]
        # Normalize existing timestamps too
        try:
            existing[ts_col] = pd.to_datetime(existing[ts_col], format='mixed').dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            pass
        existing_ts = set(existing[ts_col].astype(str))
        new_rows = df[~df['timestamp'].isin(existing_ts)].copy()
        if new_rows.empty:
            return 0
        combined = pd.concat([existing, new_rows], ignore_index=True)
        combined = combined.sort_values(ts_col).drop_duplicates(ts_col)
        combined.to_csv(path, index=False)
        return len(new_rows)
    else:
        df.to_csv(path, index=False)
        return len(df)


class Command(BaseCommand):
    help = 'Fetch current OHLCV price data for EURUSD and XAUUSD (all timeframes)'

    def add_arguments(self, parser):
        parser.add_argument(
            '--pairs', nargs='+', default=['EURUSD', 'XAUUSD'],
            metavar='PAIR', help='Pairs to fetch (default: EURUSD XAUUSD)',
        )
        parser.add_argument(
            '--full', action='store_true',
            help='Fetch maximum available history instead of recent data',
        )

    def handle(self, *args, **options):
        DATA_DIR.mkdir(exist_ok=True)
        pairs   = [p.upper() for p in options['pairs']]
        full    = options['full']
        av_key  = os.getenv('ALPHA_VANTAGE_API_KEY', '')

        if full:
            YF_CONFIG['H1']['period']    = '730d'
            YF_CONFIG['Daily']['period'] = '5y'

        self.stdout.write(f'\nFetching price data for: {", ".join(pairs)}\n')

        total_new = 0
        for pair in pairs:
            ticker = TICKER_MAP.get(pair)
            if not ticker:
                self.stdout.write(self.style.WARNING(f'  [{pair}] Unknown pair — skipping'))
                continue

            self.stdout.write(f'  [{pair}]')

            # ── H1 ──
            try:
                h1_df = _load_yfinance(ticker, '1h', YF_CONFIG['H1']['period'])
                if h1_df is not None:
                    n = _save_csv(DATA_DIR / f'{pair}_H1.csv', h1_df, 'H1')
                    self.stdout.write(
                        self.style.SUCCESS(f'    H1:    +{n} new rows (yfinance)')
                    )
                    total_new += n
                else:
                    self.stdout.write(self.style.WARNING('    H1:    yfinance returned no data'))
            except Exception as e:
                self.stdout.write(self.style.WARNING(f'    H1:    FAILED — {e}'))
                h1_df = None

            # ── H4 (resample from H1) ──
            if h1_df is not None:
                try:
                    h4_df = _resample_h4(h1_df)
                    n = _save_csv(DATA_DIR / f'{pair}_H4.csv', h4_df, 'H4')
                    self.stdout.write(
                        self.style.SUCCESS(f'    H4:    +{n} new rows (resampled)')
                    )
                    total_new += n
                except Exception as e:
                    self.stdout.write(self.style.WARNING(f'    H4:    resample FAILED — {e}'))

            # ── Daily ──
            daily_df = None
            try:
                daily_df = _load_yfinance(ticker, '1d', YF_CONFIG['Daily']['period'])
            except Exception:
                pass

            # Fallback to Alpha Vantage Daily
            if (daily_df is None or daily_df.empty) and av_key:
                try:
                    from_s, to_s = AV_SYMBOLS[pair]
                    daily_df = _load_av_daily(from_s, to_s, av_key)
                    source = 'AlphaVantage'
                except Exception as e:
                    self.stdout.write(self.style.WARNING(f'    Daily: AlphaVantage FAILED — {e}'))
            else:
                source = 'yfinance'

            if daily_df is not None and not daily_df.empty:
                n = _save_csv(DATA_DIR / f'{pair}_Daily.csv', daily_df, 'Daily')
                self.stdout.write(
                    self.style.SUCCESS(f'    Daily: +{n} new rows ({source})')
                )
                total_new += n
            else:
                self.stdout.write(self.style.WARNING('    Daily: no data from any source'))

        self.stdout.write(
            self.style.SUCCESS(f'\nDone: {total_new} total new rows added across all files.\n')
        )
