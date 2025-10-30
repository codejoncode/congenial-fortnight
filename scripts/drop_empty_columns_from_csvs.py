"""
Drop columns that are entirely empty from price CSVs and their backups.
Targets: data/EURUSD_*.csv, data/XAUUSD_*.csv and their .backup / .orig variants.
Behavior:
- For each target file, read with pandas (dtype=str), trim whitespace.
- Identify columns where every value is empty/blank/NA.
- If any such columns exist, drop them and write file back using to_csv(index=False) which prevents trailing commas.
- Create a .orig backup of any file before modifying if it doesn't already exist.
- Preserve column order: canonical order (id,timestamp,time,open,high,low,close,volume,spread) for present columns, then any remaining.
- Print summary of changes.
"""
from pathlib import Path
import pandas as pd
import glob
import shutil
import os

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
CANONICAL = ['id','timestamp','time','open','high','low','close','volume','spread']

patterns = [str(DATA / 'EURUSD_*.csv'), str(DATA / 'XAUUSD_*.csv')]
all_files = []
for pat in patterns:
    all_files += glob.glob(pat)
    # include backups/orig
    all_files += glob.glob(pat + '.*')

all_files = sorted(set(all_files))
changes = []
for fp in all_files:
    p = Path(fp)
    if not p.exists():
        continue
    # skip complete_holloway files (we won't touch those unless named explicitly)
    if p.name.endswith('_complete_holloway.csv'):
        continue
    try:
        df = pd.read_csv(p, dtype=str)
    except Exception:
        try:
            df = pd.read_csv(p, dtype=str, engine='python')
        except Exception:
            print(' Cannot read', p)
            continue
    # normalize whitespace in strings
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    # consider empty if NaN or ''
    empty_cols = [c for c in df.columns if df[c].astype(str).replace('nan','').replace('None','').apply(lambda x: x=='' or x.lower()=='nan').all()]
    if not empty_cols:
        continue
    # drop empty columns
    print('Dropping empty cols from', p.name, ':', empty_cols)
    df = df.drop(columns=empty_cols)
    # reorder columns: canonical first then others
    cols = [c for c in CANONICAL if c in df.columns] + [c for c in df.columns if c not in CANONICAL]
    df = df[cols]
    # backup original if not exists
    orig = p.with_suffix(p.suffix + '.orig')
    if not orig.exists():
        shutil.copy2(p, orig)
        print(' Backed up', p.name, '->', orig.name)
    # write clean csv (to_csv removes trailing comma problem)
    df.to_csv(p, index=False)
    changes.append((p.name, empty_cols))

print('\nDone. Files changed:')
for name, cols in changes:
    print(' ', name, '-> dropped', cols)
