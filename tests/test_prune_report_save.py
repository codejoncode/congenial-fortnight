import os
from scripts.report_utils import save_prune_report


def test_save_prune_report_writes_file(tmp_path):
    report = {
        'dropped_na_pct': ['col1'],
        'dropped_zero_variance': ['col2'],
        'initial_cols': ['col1', 'col2', 'col3'],
        'final_cols': ['col3'],
        'n_initial_cols': 3,
        'n_final_cols': 1,
    }

    out_dir = str(tmp_path)
    path = save_prune_report(report, pair='EURUSD', out_dir=out_dir)
    assert path.endswith('prune_report_EURUSD.json')
    assert os.path.exists(path)
    # Read and verify
    import json
    with open(path, 'r') as f:
        loaded = json.load(f)
    assert loaded['n_final_cols'] == 1
