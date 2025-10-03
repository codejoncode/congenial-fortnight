import pytest
import jsonschema
from fundamentals import fetch_fundamental_features

# Define the required schema for fundamental features
FUND_SCHEMA = {
    "type": "object",
    "properties": {
        "pe_ratio": {"type": "number"},
        "ebitda": {"type": "number"},
        "debt_to_equity": {"type": "number"},
    },
    "required": ["pe_ratio", "ebitda", "debt_to_equity"],
    "additionalProperties": False
}

# Parameterize over all data sources and a test ticker
@pytest.mark.parametrize("source", ["alpha_vantage", "finnhub", "fmp", "api_ninja"])
@pytest.mark.parametrize("ticker", ["EURUSD", "AAPL", "GOOG"])
def test_schema_adherence(source, ticker):
    """
    For each source and ticker, fetch the fundamental features and validate against schema.
    """
    features = fetch_fundamental_features(source, ticker)
    assert isinstance(features, dict), f"{source} did not return a dict for {ticker}"
    # Validate schema
    jsonschema.validate(instance=features, schema=FUND_SCHEMA)
    
    # Ensure fields are non-null and finite
    for key in FUND_SCHEMA['required']:
        assert features[key] is not None, f"{source}:{ticker} - {key} is None"
        assert features[key] == features[key], f"{source}:{ticker} - {key} is NaN"

# Test error handling for missing API key
@pytest.mark.parametrize("source", ["alpha_vantage", "finnhub", "fmp"])
def test_missing_api_key(monkeypatch, source):
    monkeypatch.delenv(source.upper() + "_API_KEY", raising=False)
    with pytest.raises(EnvironmentError):
        fetch_fundamental_features(source, "EURUSD")

# NOTE:
# - Replace the import path if `fetch_fundamental_features` is in a different module.
# - Extend the schema to include additional fields as needed for your pipeline.
