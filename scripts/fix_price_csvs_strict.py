"""
Strict fixer for price CSVs.
- Detects files with inconsistent row column counts (trailing commas).
- Chooses best source for each target: prefer *_complete_holloway.csv with numeric OHLC, else latest backup that contains numeric OHLC, else the file itself.
- Normalizes headers to canonical cols and writes CSV with pandas.to_csv(index=False) to avoid trailing commas.
- Creates a .orig backup of any overwritten main file if not present.
- Does NOT modify data/update_metadata.json (it will call the existing refresh script instead).

Run from repo root: python scripts/fix_price_csvs_strict.py
"""
from pathlib import Path
import glob
import pandas as pd
import shutil
import sys
import re
import os
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
CANONICAL = ['id','timestamp','time','open','high','low','close','volume','spread']

# helper to normalize column names
def normalize_cols(cols):
    mapping = {}
    for c in cols:
        if c is None:
            continue
        c0 = c.strip()
        # remove angle brackets and whitespace
        c0 = re.sub(r"[<>]", "", c0)
        c0 = c0.replace('\r','').replace('\n','')
        low = c0.lower()
        if low in ('id','index'):
            mapping[c] = 'id'
        elif low in ('timestamp','date','datetime'):
            mapping[c] = 'timestamp'
        elif low in ('time','hh:mm:ss'):
            mapping[c] = 'time'
        elif 'open' in low:
            mapping[c] = 'open'
        elif 'high' in low:
            mapping[c] = 'high'
        elif 'low' in low:
            mapping[c] = 'low'
        elif 'close' in low:
            mapping[c] = 'close'
        elif 'volume' in low:
            mapping[c] = 'volume'
        elif 'spread' in low:
            mapping[c] = 'spread'
        else:
            mapping[c] = low
    return mapping

# detect extra commas (max columns per row vs header length)
def detect_trailing_commas(path, sample_lines=2000):
    header_len = None
    max_len = 0
    rows = 0
    with open(path, 'r', encoding='utf-8', errors='replace') as fh:
        for i,line in enumerate(fh):
            if i==0:
                header_len = len(line.rstrip('\n').split(','))
            parts = line.rstrip('\n').split(',')
            max_len = max(max_len, len(parts))
            rows += 1
            if rows>=sample_lines:
                break
    return header_len, max_len

# count non-null numeric OHLC in a pandas-readable file
def count_ohlc(path):
    try:
        df = pd.read_csv(path, nrows=2000)
    except Exception:
        try:
            df = pd.read_csv(path, nrows=2000, engine='python')
        except Exception:
            return 0,0
    open_cnt = 0
    close_cnt = 0
    for c in df.columns:
        if c.lower()=='open':
            open_cnt = int(df[c].notna().sum())
        if c.lower()=='close':
            close_cnt = int(df[c].notna().sum())
    return open_cnt, close_cnt

# find best backup/holloway source for a file
def find_best_source(target_path:Path):
    base = target_path.stem  # e.g., EURUSD_H1
    # candidate holloway
    hollow = DATA / f"{base.lower().replace('_','_')}.csv"
    # but better search for *_complete_holloway for same symbol and timeframe
    symbol = None
    for prefix in ('EURUSD','XAUUSD'):
        if target_path.name.startswith(prefix):
            symbol = prefix
            break
    timeframe = target_path.name.replace(symbol+'_','') if symbol else None
    hollow_name = None
    if timeframe:
        hollow_name = f"{symbol}_{timeframe.lower()}_complete_holloway.csv".replace('__','_')
        hollow_cand = DATA / hollow_name
        if hollow_cand.exists():
            o,c = count_ohlc(hollow_cand)
            if o>0 or c>0:
                return hollow_cand
    # else look for backups next to file
    candidates = sorted([Path(x) for x in glob.glob(str(target_path) + '.*') if ('.backup' in x or x.endswith('.orig') or x.endswith('.backup'))], key=lambda p: p.stat().st_mtime, reverse=True)
    for b in candidates:
        o,c = count_ohlc(b)
        if o>0 or c>0:
            return b
    # fallback: look for any *_complete_holloway for same symbol
    if symbol:
        pattern = DATA / f"{symbol}_*complete_holloway.csv"
        found = sorted(glob.glob(str(pattern)), key=os.path.getmtime, reverse=True)
        for f in found:
            o,c = count_ohlc(f)
            if o>0 or c>0:
                return Path(f)
    # final fallback: return the target itself
    return target_path

# main
if __name__ == '__main__':
    csvs = sorted([p for p in DATA.glob('*.csv') if not p.name.endswith('.backup') and not p.name.endswith('.orig')])
    problem_files = []
    summary = []
    for p in csvs:
        header_len, max_len = detect_trailing_commas(p)
        if header_len is None:
            continue
        extra = max_len - header_len
        summary.append((p.name, header_len, max_len, extra))
        if extra>0:
            problem_files.append(p)
    print('Detected files with mismatch counts (header_len, max_len, extra):')
    for s in summary:
        print(' ',s)

    # repair problem files (and also non-problem mains per user's instruction)
    repaired = []
    for p in csvs:
        # ignore files that are clearly auxiliary (complete_holloway files) - but user said backups look correct; we will not overwrite holloway files
        if p.name.endswith('_complete_holloway.csv'):
            continue
        # find best source
        src = find_best_source(p)
        src_used = src
        print('\nProcessing', p.name, '-> source:', src_used.name)
        # read source robustly
        try:
            df = pd.read_csv(src_used, dtype=str)
        except Exception:
            df = pd.read_csv(src_used, dtype=str, engine='python')
        # normalize columns
        mapping = normalize_cols(df.columns.tolist())
        df = df.rename(columns=mapping)
        # ensure canonical columns exist
        for col in CANONICAL:
            if col not in df.columns:
                df[col] = pd.NA
        # generate id if missing or all null
        if df['id'].isna().all():
            df['id'] = range(1, len(df)+1)
        # fill time column default
        if df['time'].isna().all():
            df['time'] = '00:00:00'
        # coerce numeric for ohlc/volume/spread
        for c in ['open','high','low','close','volume','spread']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        # select columns in canonical order
        out_df = df[CANONICAL]
        # backup existing main if needed
        main_path = p
        orig_backup = main_path.with_suffix(main_path.suffix + '.orig')
        if main_path.exists() and not orig_backup.exists():
            shutil.copy2(main_path, orig_backup)
            print('Backed up', main_path.name, '->', orig_backup.name)
        # write CSV strictly (no index) - this will remove trailing comma columns
        out_df.to_csv(main_path, index=False)
        repaired.append(main_path.name)
        print('Wrote cleaned', main_path.name)

    # call metadata refresh (if script exists)
    refresh = ROOT / 'scripts' / 'refresh_metadata_for_prices.py'
    if refresh.exists():
        print('\nRefreshing metadata...')
        try:
            os.system(f'"{sys.executable}" "{refresh}"')
        except Exception as e:
            print('metadata refresh failed', e)

    print('\nRepaired files:')
    for r in repaired:
        print(' ', r)

    print('\nDone at', datetime.utcnow().isoformat()+'Z')
