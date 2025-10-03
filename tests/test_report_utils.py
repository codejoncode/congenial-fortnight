import os
import pandas as pd
from scripts.report_utils import generate_schema_report


def test_generate_schema_report_writes_file(tmp_path):
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(['2025-09-01', '2025-09-02', '2025-09-03']),
        'open': [1.0, 1.1, 1.2],
        'close': [1.05, 1.08, 1.1],
        'fund_gdp': [0.5, 0.6, 0.55]
    })

    out_dir = str(tmp_path)
    report = generate_schema_report(df, pair='EURUSD', pairs=['EURUSD', 'XAUUSD'], out_dir=out_dir)
    assert report['pair'] == 'EURUSD'
    assert report['cols'] == 4
    # confirm file exists
    path = os.path.join(out_dir, 'schema_report_EURUSD.json')
    assert os.path.exists(path)
