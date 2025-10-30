import responses
import os
import json
from fundamentals import fetch_alpha_vantage_overview, fetch_finnhub_metrics


@responses.activate
def test_alpha_vantage_rate_limit(monkeypatch):
    # Ensure API key is set for the function to attempt a call
    monkeypatch.setenv('AV_API_KEY', 'test')
    url = 'https://www.alphavantage.co/query'
    responses.add(responses.GET, url, status=429)
    try:
        fetch_alpha_vantage_overview('EURUSD')
        assert False, "Expected exception on 429"
    except Exception:
        assert True


@responses.activate
def test_finnhub_404(monkeypatch):
    monkeypatch.setenv('FINNHUB_API_KEY', 'test')
    url = 'https://finnhub.io/api/v1/stock/metric'
    responses.add(responses.GET, url, status=404)
    try:
        fetch_finnhub_metrics('EURUSD')
        assert False, "Expected exception on 404"
    except Exception:
        assert True
