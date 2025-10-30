"""
Clean trailing empty commas from price CSVs by enforcing a canonical header.

This script:
 - Backs up the original file to <filename>.orig if not already present
 - Loads the CSV with pandas (robust to delimiters)
 - Writes back only the canonical columns: id,timestamp,time,open,high,low,close,volume,spread
 - This removes trailing commas caused by extra empty fields

Use: python -m scripts.clean_trailing_commas
"""
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('clean_trailing_commas')

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'

CANONICAL = ['id', 'timestamp', 'time', 'open', 'high', 'low', 'close', 'volume', 'spread']

def clean_file(p: Path):
    logger.info(f"Cleaning {p}")
    orig = p.with_suffix(p.suffix + '.orig')
    if not orig.exists():
        p.replace(orig)
        # Read from orig and rewrite to p
        source = orig
    else:
        source = orig

    # Try to read robustly
    df = None
    for kwargs in [{'sep': ',', 'engine': 'c'}, {'sep': '\t', 'engine': 'python'}, {'sep': None, 'engine': 'python'}]:
        try:
            df_try = pd.read_csv(source, **kwargs, low_memory=False)
            if df_try.shape[1] >= 1:
                df = df_try
                break
        except Exception:
            continue

    if df is None:
        logger.error(f"Failed to read {source}")
        return False

    # Ensure canonical columns exist
    for c in CANONICAL:
        if c not in df.columns:
            df[c] = pd.NA

    # Select canonical columns only
    out = df.loc[:, CANONICAL]

    # Write back without index — pandas will not write extra trailing commas
    out.to_csv(p, index=False)
    logger.info(f"Wrote cleaned file {p}")
    return True

def main():
    patterns = ['EURUSD_*.csv', 'XAUUSD_*.csv']
    files = []
    for pat in patterns:
        files.extend(sorted(DATA.glob(pat)))

    if not files:
        logger.warning('No EURUSD/XAUUSD CSVs found to clean')
        return 0

    count = 0
    for f in files:
        if clean_file(f):
            count += 1

    logger.info(f"Cleaned {count}/{len(files)} files")
    return count

if __name__ == '__main__':
    main()
