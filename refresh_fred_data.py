#!/usr/bin/env python3
"""
Standalone FRED data refresh — no Django, no pandas required.

Uses only stdlib (urllib, csv) + optional requests if available.
Downloads all 24 FRED series to data/ using the public CSV endpoint.

Usage:
    python refresh_fred_data.py
    python refresh_fred_data.py VIXCLS DGS10 FEDFUNDS
"""

import csv
import os
import sys
import time
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent / 'data'

FRED_SERIES = [
    'DEXUSEU', 'DEXJPUS', 'DEXCHUS',
    'FEDFUNDS', 'DFF',
    'CPIAUCSL', 'CPALTT01USM661S',
    'UNRATE', 'PAYEMS', 'INDPRO', 'DGORDER',
    'ECBDFR', 'ECBRR',
    'CP0000EZ19M086NEST', 'LRHUTTTTDEM156S',
    'GOLDAMGBD228NLBM', 'DCOILWTICO', 'DCOILBRENTEU',
    'VIXCLS', 'DGS10', 'DGS2', 'DGS3MO',
    'BOPGSTB', 'T10YIE',
]

FRED_URL = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}'


def fetch_series(series_id: str) -> tuple[int, str] | None:
    """Return (row_count, csv_content) or None on failure."""
    url = FRED_URL.format(series=series_id)
    req = urllib.request.Request(
        url, headers={'User-Agent': 'congenial-fortnight/1.0 data-refresh'}
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read().decode('utf-8')
    except Exception as e:
        return None, str(e)

    lines = [l for l in raw.strip().splitlines() if l]
    if len(lines) < 2:
        return None, 'empty response'

    # Normalize header: observation_date,SERIES_ID -> date,series_id
    header = lines[0].lower().replace('observation_date', 'date')
    col_name = series_id.lower()
    # Replace raw series name with canonical lowercase
    parts = header.split(',')
    if len(parts) >= 2:
        parts[1] = col_name
    header = ','.join(parts)

    # Filter out FRED missing-value rows ('.')
    good_rows = [header]
    for line in lines[1:]:
        if '.' not in line.split(',')[1:]:  # value column not '.'
            good_rows.append(line)
        else:
            # Check if the value field is literally '.'
            row_parts = line.split(',')
            if len(row_parts) >= 2 and row_parts[1].strip() != '.':
                good_rows.append(line)

    return len(good_rows) - 1, '\n'.join(good_rows) + '\n'


def main():
    DATA_DIR.mkdir(exist_ok=True)
    targets = sys.argv[1:] or FRED_SERIES

    print(f'Refreshing {len(targets)} FRED series -> {DATA_DIR}/')
    print()

    ok = failed = 0
    for series_id in targets:
        print(f'  [{series_id}] ', end='', flush=True)
        count, content = fetch_series(series_id)
        if count is None:
            print(f'FAILED — {content}')
            failed += 1
        else:
            out = DATA_DIR / f'{series_id}.csv'
            out.write_text(content, encoding='utf-8')
            print(f'OK — {count} rows → {out.name}')
            ok += 1
        time.sleep(0.3)

    print()
    print(f'Done: {ok} updated, {failed} failed')
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
