"""
Standardize price CSV files in data/ so all timeframes share a uniform schema:
 - Ensure columns: id, timestamp (datetime), time (HH:MM:SS), open, high, low, close, volume, spread
 - If a file has separate `time` column, combine it with `timestamp` date to create a proper datetime
 - If time is missing (Daily/Weekly/Monthly), set time to '00:00:00'
 - Add an `id` column (monotonic integer) if missing
 - Create a backup copy before overwriting: <filename>.backup

Usage: python -m scripts.standardize_price_files
"""
import pandas as pd
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('standardize')

DATA_DIR = Path('data')
PATTERNS = ['EURUSD_*', 'XAUUSD_*']

def standardize_file(path: Path):
    logger.info(f"Processing {path}")
    backup = path.with_suffix(path.suffix + '.backup')
    try:
        # Read CSV with pandas
        df = pd.read_csv(path)
    except Exception as e:
        logger.error(f"Failed to read {path}: {e}")
        return False

    # Detect a date-like column (timestamp/date/datetime) or fallback to first column
    date_candidates = ['timestamp', 'date', 'datetime', 'Date', 'Timestamp', 'DATE']
    date_col = None
    for c in date_candidates:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        # fallback to first column name if it looks like a date series
        date_col = df.columns[0]
        logger.info(f"No standard date column found; using first column '{date_col}' as date")

    # Rename date_col to 'timestamp' for uniformity if needed
    if date_col != 'timestamp':
        df = df.rename(columns={date_col: 'timestamp'})

    # If there's a separate 'time' column, combine it with date where needed
    if 'time' in df.columns:
        # Some files have timestamp as date only; combine date + time
        try:
            # Coerce both parts to strings and combine
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(str).str.slice(0, 19) + ' ' + df['time'].astype(str), errors='coerce')
            # If there are still NaT values, try a second approach
            if df['timestamp'].isna().all():
                df['timestamp'] = pd.to_datetime(df['timestamp'].astype(str) + ' ' + df['time'].astype(str), errors='coerce')
        except Exception:
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(str) + ' ' + df['time'].astype(str), errors='coerce')
    else:
        # No time column - create one with default midnight
        df['time'] = '00:00:00'
        # Combine
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(str) + ' ' + df['time'].astype(str), errors='coerce')

    # Ensure id column exists
    if 'id' not in df.columns:
        df.insert(0, 'id', range(1, len(df) + 1))

    # Ensure core numeric columns exist and are numeric
    for col in ['open', 'high', 'low', 'close', 'volume', 'spread']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = pd.NA

    # Reorder columns: id, timestamp, time, open, high, low, close, volume, spread, ...rest
    rest = [c for c in df.columns if c not in ['id', 'timestamp', 'time', 'open', 'high', 'low', 'close', 'volume', 'spread']]
    new_order = ['id', 'timestamp', 'time', 'open', 'high', 'low', 'close', 'volume', 'spread'] + rest
    df = df.reindex(columns=new_order)

    # Backup original file
    try:
        if not backup.exists():
            path.replace(backup)
            # Write standardized df to original path
            df.to_csv(path, index=False)
        else:
            # If backup exists, just overwrite original
            df.to_csv(path, index=False)
    except Exception as e:
        logger.error(f"Failed to backup/write {path}: {e}")
        return False

    logger.info(f"Standardized and saved {path} (backup: {backup.name})")
    return True

def main():
    paths = []
    for pattern in PATTERNS:
        paths.extend(DATA_DIR.glob(pattern))

    if not paths:
        logger.warning("No matching price files found in data/")
        return 0

    successes = 0
    for p in sorted(paths):
        if standardize_file(p):
            successes += 1

    logger.info(f"Finished standardizing {successes}/{len(paths)} files")
    return successes

if __name__ == '__main__':
    main()
