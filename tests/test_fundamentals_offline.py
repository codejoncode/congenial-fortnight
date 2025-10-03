import json
import os
from scripts.fundamental_pipeline import FundamentalDataPipeline


def test_load_sample_fundamentals():
    path = os.path.join(os.getcwd(), 'data', 'tests', 'fundamentals', 'sample_fundamentals_series.json')
    assert os.path.exists(path), "Sample fundamentals JSON should exist for offline tests"

    with open(path, 'r') as f:
        data = json.load(f)

    # The pipeline expects a specific structure; we'll simulate loading into a DataFrame via the pipeline helper
    pipeline = FundamentalDataPipeline(data_dir=os.path.join(os.getcwd(), 'data', 'tests', 'fundamentals'))
    series = pipeline._load_series_from_json_file(path)

    # basic assertions
    assert 'cpi' in series.columns
    assert len(series) == 3
