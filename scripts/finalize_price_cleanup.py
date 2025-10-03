"""
Finalize price CSV cleanup:
- Ensure main EURUSD/XAUUSD timeframe CSVs have numeric OHLC; if missing, restore from *_complete_holloway.csv or from latest backup that contains OHLC.
- Normalize headers to canonical set and write via pandas.to_csv(index=False) (removes trailing commas).
- Create .orig backups before overwriting mains if not present.
- Archive nested .backup* files to data/backups_archive/<timestamp>/ to avoid accidental overwrites.
- Merge and refresh `data/update_metadata.json` with rows and last_data_date for mains, backing up the original metadata file first.

Run: python scripts/finalize_price_cleanup.py
"""
from pathlib import Path
import glob
import pandas as pd
import shutil
import os
import json
import re
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
BACKUP_ARCHIVE = DATA / 'backups_archive'
CANONICAL = ['id','timestamp','time','open','high','low','close','volume','spread']
SYMBOLS = ['EURUSD','XAUUSD']

# helpers

def normalize_cols(cols):
    mapping = {}
    for c in cols:
        if c is None:
            continue
        c0 = c.strip()
        c0 = re.sub(r"[<>]", "", c0)
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


def find_holloway_for(main_path:Path):
    name = main_path.name
    for sym in SYMBOLS:
        if name.startswith(sym):
            symbol = sym
            break
    else:
        return None
    # timeframe portion after symbol_
    timeframe = name[len(symbol)+1:]
    timeframe = timeframe.replace('.csv','')
    # try common casing
    candidates = [DATA / f"{symbol}_{timeframe}_complete_holloway.csv",
                  DATA / f"{symbol}_{timeframe.lower()}_complete_holloway.csv",
                  DATA / f"{symbol}_{timeframe.capitalize()}_complete_holloway.csv"]
    for c in candidates:
        if c.exists():
            o,cnt = count_ohlc(c)
            if o>0 or cnt>0:
                return c
    # fallback: any *_complete_holloway for symbol
    pattern = DATA / f"{symbol}_*complete_holloway.csv"
    found = sorted(glob.glob(str(pattern)), key=os.path.getmtime, reverse=True)
    for f in found:
        o,cnt = count_ohlc(f)
        if o>0 or cnt>0:
            return Path(f)
    return None


def find_best_backup(main_path:Path):
    # look for any adjacent file with .backup or .orig that contains ohlc
    candidates = sorted([Path(x) for x in glob.glob(str(main_path) + '.*') if ('.backup' in x or x.endswith('.orig'))], key=lambda p: p.stat().st_mtime, reverse=True)
    for b in candidates:
        o,c = count_ohlc(b)
        if o>0 or c>0:
            return b
    return None


def normalize_and_write(main_path:Path, src_path:Path, make_backup=True):
    # make .orig of main if not exists
    if main_path.exists() and make_backup:
        orig = main_path.with_suffix(main_path.suffix + '.orig')
        if not orig.exists():
            shutil.copy2(main_path, orig)
    # read source robustly
    try:
        df = pd.read_csv(src_path, dtype=str)
    except Exception:
        df = pd.read_csv(src_path, dtype=str, engine='python')
    # normalize names
    mapping = normalize_cols(df.columns.tolist())
    df = df.rename(columns=mapping)
    # ensure canonical
    for col in CANONICAL:
        if col not in df.columns:
            df[col] = pd.NA
    # generate id if missing or all null
    if df['id'].isna().all():
        df['id'] = range(1, len(df)+1)
    # fill time column default if empty
    if df['time'].isna().all():
        df['time'] = '00:00:00'
    # coerce numeric
    for c in ['open','high','low','close','volume','spread']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    out = df[CANONICAL]
    out.to_csv(main_path, index=False)


# Main processing
if __name__ == '__main__':
    mains = sorted([p for p in DATA.glob('*.csv') if (not p.name.endswith('.backup')) and (not p.name.endswith('.orig')) and (not p.name.endswith('_complete_holloway.csv'))])
    changed = []
    for m in mains:
        if not any(str(m.name).startswith(sym) for sym in SYMBOLS):
            continue
        o,c = count_ohlc(m)
        print('Checking', m.name, 'open/close:', o, c)
        if o>0 or c>0:
            # still normalize and ensure canonical (in case previous writes left issue)
            normalize_and_write(m, m, make_backup=False)
            continue
        # try holloway
        hollow = find_holloway_for(m)
        if hollow:
            print(' Restoring from holloway', hollow.name)
            normalize_and_write(m, hollow)
            changed.append((m.name, 'holloway', hollow.name))
            continue
        # try backups
        b = find_best_backup(m)
        if b:
            print(' Restoring from backup', b.name)
            normalize_and_write(m, b)
            changed.append((m.name, 'backup', b.name))
            continue
        print(' No source found for', m.name)

    # archive nested .backup* files into backups_archive
    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    archive_dir = BACKUP_ARCHIVE / ts
    archive_dir.mkdir(parents=True, exist_ok=True)
    moved = []
    for p in DATA.glob('*'):
        if '.backup' in p.name and p.exists():
            try:
                target = archive_dir / p.name
                shutil.move(str(p), str(target))
                moved.append(p.name)
            except Exception as e:
                print('Failed to move', p, e)
    print('Archived nested backups to', archive_dir)

    # merge/update metadata
    meta_file = DATA / 'update_metadata.json'
    meta = {}
    if meta_file.exists():
        try:
            with meta_file.open('r', encoding='utf-8') as fh:
                meta = json.load(fh)
        except Exception as e:
            print('Could not read existing metadata, will overwrite. Error:', e)
            meta = {}
        # backup user's metadata
        shutil.copy2(meta_file, meta_file.with_suffix('.orig'))
    # update entries for mains
    for m in mains:
        if not any(str(m.name).startswith(sym) for sym in SYMBOLS):
            continue
        try:
            df = pd.read_csv(m, usecols=['timestamp'], parse_dates=['timestamp'], infer_datetime_format=True)
            # drop NA
            if 'timestamp' in df.columns and not df['timestamp'].isnull().all():
                last_date = df['timestamp'].dropna().max().strftime('%Y-%m-%d')
                rows = int(len(df))
            else:
                last_date = datetime.utcfromtimestamp(m.stat().st_mtime).strftime('%Y-%m-%d')
                rows = int(sum(1 for _ in open(m, 'r', encoding='utf-8', errors='replace')))-1
        except Exception:
            # fallback to mtime
            last_date = datetime.utcfromtimestamp(m.stat().st_mtime).strftime('%Y-%m-%d')
            rows = int(sum(1 for _ in open(m, 'r', encoding='utf-8', errors='replace')))-1
        meta_key = m.stem
        meta_entry = meta.get(meta_key, {})
        meta_entry.update({'rows': rows, 'last_data_date': last_date, 'last_inspected': datetime.utcnow().isoformat()+'Z'})
        meta[meta_key] = meta_entry

    # write metadata
    with meta_file.open('w', encoding='utf-8') as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)

    print('\nChanged files:')
    for c in changed:
        print(' ', c)
    print('\nArchived backups:', moved)
    print('Metadata written to', meta_file)
    print('Done')
