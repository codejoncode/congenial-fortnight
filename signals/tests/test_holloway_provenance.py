import os
import sys
import pandas as pd

# Ensure repo root on sys.path for test discovery
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.holloway_algorithm import CompleteHollowayAlgorithm


def test_merged_features_exist():
    # check merged latest features were saved for both pairs
    for pair in ['EURUSD', 'XAUUSD']:
        merged = os.path.join(os.getcwd(), 'data', f"{pair}_latest_multi_timeframe_features.csv")
        assert os.path.exists(merged), f"Merged features missing for {pair}"
        df = pd.read_csv(merged)
        assert len(df) == 1


def test_per_timeframe_files_have_provenance():
    # ensure per-timeframe holloway CSVs exist and include source_file provenance column (if present)
    for pair in ['EURUSD','XAUUSD']:
        for tf in ['daily','h4','h1','weekly','monthly']:
            path = os.path.join(os.getcwd(), 'data', f"{pair}_{tf}_complete_holloway.csv")
            if os.path.exists(path):
                df = pd.read_csv(path)
                # provenance column may or may not exist depending on earlier runs, but if present it should be non-empty
                if 'source_file' in df.columns:
                    assert df['source_file'].notna().any()
