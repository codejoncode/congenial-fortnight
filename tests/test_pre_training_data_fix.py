import os
import pandas as pd
from scripts.data_issue_fixes import pre_training_data_fix


def test_pre_training_data_fix_ok(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    # create the essential CSV files expected by pre_training_data_fix
    filenames = [
        'EURUSD_H1.csv',
        'XAUUSD_H1.csv',
        'EURUSD_Monthly.csv',
        'XAUUSD_Monthly.csv'
    ]
    for name in filenames:
        df = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01', periods=60, freq='D'),
            'open': range(60),
            'high': range(60),
            'low': range(60),
            'close': range(60),
        })
        p = d / name
        df.to_csv(p, index=False)

    assert pre_training_data_fix(data_dir=str(d), min_rows=50) is True


def test_pre_training_data_fix_missing(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    # no csv files
    assert pre_training_data_fix(data_dir=str(d), min_rows=1) is False
