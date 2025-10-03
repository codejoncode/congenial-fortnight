#!/usr/bin/env python3
"""Guarded runner for live API tests.

This script will only perform live API calls if the required environment
variables are set. It's intended to be run in a CI job where secrets are
provided (see .github/workflows/live_api_tests.yml).
"""
import os
import sys
from scripts.fundamental_pipeline import FundamentalDataPipeline


def main():
    fred_key = os.getenv('FRED_API_KEY')
    if not fred_key:
        print('FRED_API_KEY not set - skipping live API tests')
        return 0

    pipeline = FundamentalDataPipeline(data_dir='data/live_tests', fred_api_key=fred_key)
    # Run a quick fetch for a small set
    try:
        ok = pipeline.update_fred_series('CPIAUCSL')
        print('Update CPIAUCSL:', ok)
        return 0 if ok else 2
    except Exception as e:
        print('Live API test failed:', e)
        return 3


if __name__ == '__main__':
    sys.exit(main())
