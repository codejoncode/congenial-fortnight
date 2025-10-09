import glob
import pandas as pd
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
print('Repo root:', ROOT)

# scan for EURUSD/XAUUSD csvs
patterns = [str(ROOT / 'data' / 'EURUSD_*.csv'), str(ROOT / 'data' / 'XAUUSD_*.csv')]
files = sorted(sum([glob.glob(p) for p in patterns], []))

candidates = []
for f in files:
    try:
        df = pd.read_csv(f, nrows=2000)
    except Exception:
        df = pd.read_csv(f, nrows=2000, engine='python', error_bad_lines=False)
    # detect ohlc columns
    has_open = any(c.lower()=='open' for c in df.columns)
    has_close = any(c.lower()=='close' for c in df.columns)
    open_nonnull = 0
    close_nonnull = 0
    for c in df.columns:
        if c.lower()=='open':
            open_nonnull = int(df[c].notna().sum())
        if c.lower()=='close':
            close_nonnull = int(df[c].notna().sum())
    if open_nonnull==0 and close_nonnull==0:
        candidates.append(f)

print('Files with empty OHLC:', candidates)
restored = []

for orig in candidates:
    p = Path(orig)
    # find backup files matching prefix
    backups = sorted([Path(x) for x in glob.glob(str(p) + '*') if 'backup' in x or x.endswith('.orig')], key=lambda x: x.stat().st_mtime, reverse=True)
    print(f'\nChecking backups for {p} -> found {len(backups)} backups')
    found = None
    for b in backups:
        print(' trying', b.name)
        try:
            df = pd.read_csv(b, nrows=2000)
            open_nonnull = 0
            close_nonnull = 0
            for c in df.columns:
                if c.lower()=='open':
                    open_nonnull = int(df[c].notna().sum())
                if c.lower()=='close':
                    close_nonnull = int(df[c].notna().sum())
            if open_nonnull>0 or close_nonnull>0:
                found = b
                print('  -> candidate backup with data:', b)
                break
        except Exception as e:
            # try reading with python engine
            try:
                df = pd.read_csv(b, nrows=2000, engine='python')
                open_nonnull = 0
                close_nonnull = 0
                for c in df.columns:
                    if c.lower()=='open':
                        open_nonnull = int(df[c].notna().sum())
                    if c.lower()=='close':
                        close_nonnull = int(df[c].notna().sum())
                if open_nonnull>0 or close_nonnull>0:
                    found = b
                    print('  -> candidate backup with data (python engine):', b)
                    break
            except Exception:
                continue
    if found:
        backup_path = found
        # create .orig if not exists
        orig_orig = p.with_suffix(p.suffix + '.orig')
        if not orig_orig.exists():
            print('  creating orig backup:', orig_orig.name)
            shutil.copy2(p, orig_orig)
        # copy backup to original path
        print('  restoring', backup_path.name, '->', p.name)
        shutil.copy2(backup_path, p)
        restored.append((p.name, backup_path.name))
        # run cleaner script
        try:
            print('  running cleaner script...')
            subprocess.run([sys.executable, str(ROOT / 'scripts' / 'clean_trailing_commas.py')], check=True)
        except Exception as e:
            print('  cleaner failed:', e)
    else:
        print('  no usable backup found for', p.name)

# refresh metadata
try:
    print('\nRefreshing metadata...')
    subprocess.run([sys.executable, str(ROOT / 'scripts' / 'refresh_metadata_for_prices.py')], check=True)
except Exception as e:
    print('refresh metadata failed:', e)

print('\nRestored backups:')
for a,b in restored:
    print(' ', a, '<- from', b)

# final quick scan
print('\nFinal OHLC non-null scan:')
for f in files:
    try:
        df = pd.read_csv(f, nrows=1000)
        open_nonnull = 0
        close_nonnull = 0
        for c in df.columns:
            if c.lower()=='open':
                open_nonnull = int(df[c].notna().sum())
            if c.lower()=='close':
                close_nonnull = int(df[c].notna().sum())
        print(f, open_nonnull, close_nonnull)
    except Exception as e:
        print(f, 'ERROR', e)
