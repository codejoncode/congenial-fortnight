#!/usr/bin/env python3
"""
test_fundamentals_endtoend.py

End-to-end smoke test for fundamental data ingestion:
- Fetches one ticker from each source
- Validates non-emptiness and schema
- Persists raw JSON and processed Parquet
- Reloads processed data
"""
import os
import sys
import json
import pandas as pd
import logging
from pathlib import Path

try:
    from fundamentals import fetch_fundamental_features
except Exception:
    # If fundamentals module isn't present yet, provide a helpful error
    raise

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load API keys from environment
REQUIRED_KEYS = ['AV_API_KEY', 'FINNHUB_API_KEY', 'FMP_API_KEY']
OPTIONAL_KEYS = ['API_NINJA_KEY']

# Fundamental schema to test minimal fields
FUND_SCHEMA_KEYS = ['pe_ratio', 'ebitda', 'debt_to_equity']

# Test configuration
TEST_TICKER = 'EURUSD'
RAW_DIR = Path('data/raw/fundamentals')
PROC_DIR = Path('data/processed/fundamentals')


def check_env_keys():
    missing = [key for key in REQUIRED_KEYS if not os.getenv(key)]
    if missing:
        logger.error(f"Missing required API keys: {missing}")
        sys.exit(1)
    logger.info("All required API keys found.")


def save_raw(source: str, ticker: str, data: dict) -> Path:
    path = RAW_DIR / source / f"{ticker}_{pd.Timestamp.now().strftime('%Y%m%d')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved raw data to {path}")
    return path


def save_processed(source: str, ticker: str, data: dict) -> Path:
    df = pd.DataFrame([data])
    path = PROC_DIR / source / f"{ticker}_{pd.Timestamp.now().strftime('%Y%m%d')}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"Saved processed data to {path}")
    return path


def validate_schema(data: dict):
    for key in FUND_SCHEMA_KEYS:
        if key not in data:
            raise AssertionError(f"Missing schema key: {key}")
        if not isinstance(data[key], (int, float)):
            raise AssertionError(f"Invalid type for {key}: {type(data[key])}")


def main():
    check_env_keys()
    sources = ['alpha_vantage', 'finnhub', 'fmp', 'api_ninja']
    success = True

    for src in sources:
        logger.info(f"Testing source: {src}")
        try:
            data = fetch_fundamental_features(src, TEST_TICKER)
            assert isinstance(data, dict), "Returned data is not a dict"
            assert data, "Returned data is empty"
            validate_schema(data)
            raw_path = save_raw(src, TEST_TICKER, data)
            proc_path = save_processed(src, TEST_TICKER, data)

            # Reload processed data
            df = pd.read_parquet(proc_path)
            assert not df.empty, "Processed DataFrame is empty"
            logger.info(f"Source {src} passed end-to-end test.")

        except Exception as e:
            logger.error(f"Source {src} failed: {e}")
            success = False

    if not success:
        logger.error("End-to-end fundamental data ingestion tests FAILED")
        sys.exit(1)

    logger.info("All fundamental sources passed end-to-end tests.")


if __name__ == '__main__':
    main()
