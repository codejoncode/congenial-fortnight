import os
import sys
import pandas as pd

# Ensure repo root on sys.path for test discovery
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.holloway_algorithm import load_data_file

DATA_DIR = os.path.join(os.getcwd(), 'data')


def test_load_weekly_eurusd():
    df = load_data_file('EURUSD', 'weekly')
    assert not df.empty
    # basic columns
    assert set(['open','high','low','close']).issubset(set(df.columns))
    assert isinstance(df.index, pd.DatetimeIndex)


def test_provenance_is_set():
    df = load_data_file('EURUSD','daily')
    assert not df.empty
    # ensure provenance attribute exists
    assert hasattr(df, '_source_file')
    assert isinstance(df._source_file, str)
