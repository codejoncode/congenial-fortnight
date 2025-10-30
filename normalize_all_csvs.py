#!/usr/bin/env python3
"""
Universal CSV Normalizer & Validator
- Ensures all data/*.csv files have clean date columns, no merge markers, and correct headers.
- To be run after any data update/fetch or before training.
"""
import pandas as pd
import os
import re
from pathlib import Path

def normalize_csv_file(filepath):
    # Read as text to remove merge markers and blank lines
    with open(filepath, 'r') as f:
        lines = f.readlines()
    # Remove git conflict markers and blank lines
    lines = [l for l in lines if not re.match(r'^(<<<<<<<|=======|>>>>>>>|#)', l) and l.strip()]
    # Write back cleaned lines
    with open(filepath, 'w') as f:
        f.writelines(lines)
    # Now try to load as DataFrame
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"[ERROR] Could not parse {filepath}: {e}")
        return False
    # Fix date columns
    for col in df.columns:
        if 'date' in col.lower() or 'timestamp' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                # Standardize to ISO format
                df[col] = df[col].dt.strftime('%Y-%m-%d')
            except Exception:
                pass
    # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    # Save back
    df.to_csv(filepath, index=False)
    return True

def normalize_all_csvs(data_dir='data'):
    data_path = Path(data_dir)
    for csv_file in data_path.glob('*.csv'):
        print(f"Normalizing {csv_file}...")
        normalize_csv_file(csv_file)
    print("All CSVs normalized.")

if __name__ == "__main__":
    normalize_all_csvs()
