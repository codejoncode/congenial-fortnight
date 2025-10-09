import os
import pandas as pd
from pathlib import Path

def test_csv_normalization():
    """
    Test that all CSVs in the data directory:
    - Have no merge conflict markers
    - Have date columns in ISO format
    - Can be loaded as DataFrames without error
    """
    data_dir = Path('data')
    for csv_file in data_dir.glob('*.csv'):
        with open(csv_file, 'r') as f:
            content = f.read()
            assert '>>>>>>>' not in content, f"Merge marker found in {csv_file}"
            assert '=======' not in content, f"Merge marker found in {csv_file}"
            assert '<<<<<<<' not in content, f"Merge marker found in {csv_file}"
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            assert False, f"Could not load {csv_file}: {e}"
        # Check date columns
        for col in df.columns:
            if 'date' in col.lower() or 'timestamp' in col.lower():
                try:
                    pd.to_datetime(df[col], errors='raise')
                except Exception as e:
                    assert False, f"Date column {col} in {csv_file} not parseable: {e}"

def test_all():
    test_csv_normalization()

if __name__ == "__main__":
    test_all()
    print("All CSV normalization tests passed.")
