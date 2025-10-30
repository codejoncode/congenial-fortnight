#!/usr/bin/env python3
"""
Fix EURUSD_Weekly.csv date format issues for all rows.
Ensures all 'timestamp' values are in '%Y-%m-%d' format (hyphens, not dots).
"""
import pandas as pd
from pathlib import Path

def fix_weekly_csv_date_format(csv_path):
    df = pd.read_csv(csv_path)
    # Fix any dots in the timestamp to hyphens
    df['timestamp'] = df['timestamp'].astype(str).str.replace('.', '-', regex=False)
    # Save back to CSV
    df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    fix_weekly_csv_date_format('data/EURUSD_Weekly.csv')
    print("EURUSD_Weekly.csv date format fixed.")
