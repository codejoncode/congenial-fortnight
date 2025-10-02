import os
import json
from datetime import datetime
import pandas as pd

METADATA_PATH = os.path.join('data', 'update_metadata.json')

def _make_key_from_filename(path: str) -> str:
    name = os.path.basename(path)
    key = os.path.splitext(name)[0]
    return key

def update_metadata(file_path: str, last_data_date: str = None, rows: int = None, status: str = 'success') -> bool:
    """Update data/update_metadata.json for the supplied CSV file.

    If last_data_date or rows are not provided, they will be inferred from the CSV.
    """
    try:
        key = _make_key_from_filename(file_path)
        # Infer details if missing
        if (last_data_date is None) or (rows is None):
            # Use pandas to safely parse CSV and get last index
            try:
                df = pd.read_csv(file_path)
                if rows is None:
                    rows = len(df)
                if last_data_date is None and 'date' in df.columns:
                    try:
                        last = pd.to_datetime(df['date'].iloc[-1])
                        # Normalize to ISO date (no time) to match existing format
                        last_data_date = last.strftime('%Y-%m-%dT00:00:00')
                    except Exception:
                        last_data_date = None
            except Exception:
                # Fall back to simple file read
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = [l for l in f.read().splitlines() if l.strip()]
                        if rows is None:
                            rows = len(lines) - 1 if len(lines) > 0 else 0
                        if last_data_date is None and len(lines) > 1:
                            last_line = lines[-1]
                            parts = last_line.split(',')
                            if parts:
                                last_data_date = parts[0]
                except Exception:
                    pass

        # Load metadata file
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, 'r', encoding='utf-8') as f:
                try:
                    meta = json.load(f)
                except Exception:
                    meta = {}
        else:
            meta = {}

        now = datetime.now().isoformat()

        entry = meta.get(key, {})
        entry['last_update'] = now
        entry['status'] = status
        if last_data_date:
            # Normalize datetime-like strings to YYYY-MM-DDT00:00:00 if possible
            try:
                t = pd.to_datetime(last_data_date)
                entry['last_data_date'] = t.strftime('%Y-%m-%dT00:00:00')
            except Exception:
                entry['last_data_date'] = last_data_date
        if rows is not None:
            entry['rows'] = int(rows)

        meta[key] = entry

        # Write back
        os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
        with open(METADATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, sort_keys=False)

        return True
    except Exception:
        return False
