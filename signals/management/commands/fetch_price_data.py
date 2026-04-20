"""
management command: fetch_price_data

Fetches current OHLCV price data for EURUSD and XAUUSD across all timeframes
(H1, H4, Daily, Weekly).  Uses multiple sources with automatic fallback:

  1. yfinance   — free, no key, H1/H4/Daily/Weekly
  2. Alpha Vantage FX_DAILY — free (500 calls/day), key: ALPHA_VANTAGE_API_KEY

H4 is resampled from H1; Weekly is resampled from Daily.
Appends new rows to existing CSVs so historical data is preserved.

Files written: data/EURUSD_H1.csv, EURUSD_H4.csv, EURUSD_Daily.csv, EURUSD_Weekly.csv
               data/XAUUSD_H1.csv, XAUUSD_H4.csv, XAUUSD_Daily.csv, XAUUSD_Weekly.csv

Usage:
    python manage.py fetch_price_data
    python manage.py fetch_price_data --pairs EURUSD
    python manage.py fetch_price_data --full      # fetch max history
"""

import os
import logging
from pathlib import Path
import pandas as pd
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

# yfinance period + interval for H1 and Daily base fetches
YF_CONFIG = {
    'H1':    {'interval': '1h', 'period': '120d'},
    'Daily': {'interval': '1d', 'period': '730d'},
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
    result = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    # Drop yfinance ticker-name artifact rows (non-numeric close)
    result = result[pd.to_numeric(result['close'], errors='coerce').notna()]
    return result.dropna(subset=['timestamp']).reset_index(drop=True)


def _resample(df_src, rule: str):
    """Resample a higher-frequency DataFrame to a lower frequency."""
    df = df_src.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    out = df.resample(rule).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    out = out.reset_index()
    out['timestamp'] = out['timestamp'].astype(str)
    return out


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


def _save_csv(path: Path, df, timeframe: str) -> int:
    """Write/merge CSV, return rows added."""
    path.parent.mkdir(exist_ok=True)

    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
    df = df.dropna(subset=['timestamp'])

    if path.exists():
        existing = pd.read_csv(path, dtype=str)
        ts_col = 'timestamp' if 'timestamp' in existing.columns else existing.columns[0]
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
    help = 'Fetch current OHLCV price data for EURUSD and XAUUSD (H1, H4, Daily, Weekly)'

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
        pairs  = [p.upper() for p in options['pairs']]
        full   = options['full']
        av_key = os.getenv('ALPHA_VANTAGE_API_KEY', '')

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
            h1_df = None
            try:
                h1_df = _load_yfinance(ticker, '1h', YF_CONFIG['H1']['period'])
                if h1_df is not None:
                    n = _save_csv(DATA_DIR / f'{pair}_H1.csv', h1_df, 'H1')
                    self.stdout.write(self.style.SUCCESS(f'    H1:     +{n} new rows (yfinance)'))
                    total_new += n
                else:
                    self.stdout.write(self.style.WARNING('    H1:     yfinance returned no data'))
            except Exception as e:
                self.stdout.write(self.style.WARNING(f'    H1:     FAILED — {e}'))

            # ── H4 (resampled from H1) ──
            if h1_df is not None:
                try:
                    h4_df = _resample(h1_df, '4h')
                    n = _save_csv(DATA_DIR / f'{pair}_H4.csv', h4_df, 'H4')
                    self.stdout.write(self.style.SUCCESS(f'    H4:     +{n} new rows (resampled from H1)'))
                    total_new += n
                except Exception as e:
                    self.stdout.write(self.style.WARNING(f'    H4:     resample FAILED — {e}'))

            # ── Daily ──
            daily_df = None
            try:
                daily_df = _load_yfinance(ticker, '1d', YF_CONFIG['Daily']['period'])
            except Exception:
                pass

            if (daily_df is None or daily_df.empty) and av_key:
                try:
                    from_s, to_s = AV_SYMBOLS[pair]
                    daily_df = _load_av_daily(from_s, to_s, av_key)
                    source = 'AlphaVantage'
                except Exception as e:
                    self.stdout.write(self.style.WARNING(f'    Daily:  AlphaVantage FAILED — {e}'))
            else:
                source = 'yfinance'

            if daily_df is not None and not daily_df.empty:
                n = _save_csv(DATA_DIR / f'{pair}_Daily.csv', daily_df, 'Daily')
                self.stdout.write(self.style.SUCCESS(f'    Daily:  +{n} new rows ({source})'))
                total_new += n
            else:
                self.stdout.write(self.style.WARNING('    Daily:  no data from any source'))

            # ── Weekly (resampled from Daily) ──
            if daily_df is not None and not daily_df.empty:
                try:
                    weekly_df = _resample(daily_df, 'W-MON')
                    n = _save_csv(DATA_DIR / f'{pair}_Weekly.csv', weekly_df, 'Weekly')
                    self.stdout.write(self.style.SUCCESS(f'    Weekly: +{n} new rows (resampled from Daily)'))
                    total_new += n
                except Exception as e:
                    self.stdout.write(self.style.WARNING(f'    Weekly: resample FAILED — {e}'))
            else:
                self.stdout.write(self.style.WARNING('    Weekly: skipped (no Daily data)'))

        self.stdout.write(
            self.style.SUCCESS(f'\nDone: {total_new} total new rows added across all files.\n')
        )
