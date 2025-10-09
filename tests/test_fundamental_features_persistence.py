# tests/test_fundamental_features_schema.py

import pytest
import jsonschema
import os
import json

# Define the schema that all fundamental features must adhere to
FUNDAMENTAL_SCHEMA = {
    "type": "object",
    "properties": {
        "PE_ratio": {"type": "number"},
        "debtToEquity": {"type": "number"},
        "ebitda": {"type": "number"},
        "returnOnEquity": {"type": "number"},
        "marketCap": {"type": "number"},
        "priceEarningsRatio": {"type": "number"},
        "debtEquityRatio": {"type": "number"},
        # Add more fields as needed
    },
    "required": [
        "PE_ratio",
        "debtToEquity",
        "ebitda",
        "returnOnEquity", 
        "marketCap",
        "priceEarningsRatio",
        "debtEquityRatio"
    ],
    "additionalProperties": False
}

# Define a helper function to load sample features for a source and ticker
def load_sample_features(source, ticker):
    # Path to either raw or processed data, assuming structured as:
    # data/test_samples/{source}_{ticker}.json
    filename = f"data/test_samples/{source}_{ticker}.json"
    try:
        with open(filename, 'r') as f:
            features = json.load(f)
        return features
    except Exception as e:
        pytest.fail(f"Failed to load sample features from {filename}: {e}")

# List of sources and tickers to test
test_cases = [
    ("alpha_vantage", "EURUSD"),
    ("finnhub", "EURUSD"),
    ("fmp", "EURUSD"),
    ("api_ninja", "EURUSD"),
    # Add more source/ticker combinations as needed
]

@pytest.mark.parametrize("source,ticker", test_cases)
def test_schema_adherence(source, ticker):
    """Test that fetched fundamental features match the schema."""
    features = load_sample_features(source, ticker)
    # Validate schema
    jsonschema.validate(features, FUNDAMENTAL_SCHEMA)

@pytest.mark.parametrize("source,ticker", test_cases)
def test_no_extra_fields(source, ticker):
    """Test that features do not contain unexpected fields."""
    features = load_sample_features(source, ticker)
    for key in features.keys():
        assert key in FUNDAMENTAL_SCHEMA['properties'], \
            f"Field '{key}' in {source} {ticker} not declared in schema"

@pytest.mark.parametrize("source,ticker", test_cases)
def test_feature_types_and_values(source, ticker):
    """Test that feature values are of correct type (numbers)."""
    features = load_sample_features(source, ticker)
    for field, field_type in FUNDAMENTAL_SCHEMA['properties'].items():
        if field in features:
            value = features[field]
            assert isinstance(value, (int, float)), \
                f"{field} in {source} {ticker} is not a number"

# Optional: Test for presence of required fields explicitly
@pytest.mark.parametrize("source,ticker", test_cases)
def test_required_fields_present(source, ticker):
    """Ensure all required fields are present."""
    features = load_sample_features(source, ticker)
    for req_field in FUNDAMENTAL_SCHEMA['required']:
        assert req_field in features, \
            f"Missing required field '{req_field}' in {source} {ticker}"

# Notes to remember
# This test suite:
#  - Validates that each sample feature JSON conforms to the schema, with correct types.
#  - Checks that no unexpected fields are present.
#  - Ensures all required fields are included.
# Next steps:
#  - Populate your `data/test_samples/` directory with sample JSON files for each source/ticker
#    combination, generated from actual API responses or mocks.
#  - Run the tests with `pytest`.

# add file stop for another
