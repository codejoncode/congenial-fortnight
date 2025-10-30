import pytest
import jsonschema
import fundamentals

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
def test_schema_adherence(monkeypatch, source, ticker):
    """
    For each source and ticker, fetch the fundamental features and validate against schema.

    This test is offline: it monkeypatches the unified `fetch_fundamental_features`
    with a deterministic fixture so CI does not require network access or API keys.
    """
    # Provide canned fixture responses for each source
    fixture = {
        'pe_ratio': 15.0,
        'ebitda': 1_000_000.0,
        'debt_to_equity': 0.5,
    }

    # Monkeypatch the fetch_fundamental_features function to return the fixture
    monkeypatch.setattr(fundamentals, 'fetch_fundamental_features', lambda s, t: fixture)

    features = fundamentals.fetch_fundamental_features(source, ticker)
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
        fundamentals.fetch_fundamental_features(source, "EURUSD")

# NOTE:
# - Replace the import path if `fetch_fundamental_features` is in a different module.
# - Extend the schema to include additional fields as needed for your pipeline.
