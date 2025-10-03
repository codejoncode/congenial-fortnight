import os
import requests


class ApiKeyError(EnvironmentError):
    pass


def fetch_alpha_vantage_overview(symbol):
    key = os.getenv('AV_API_KEY')
    if not key:
        raise ApiKeyError('Missing Alpha Vantage API key')
    url = 'https://www.alphavantage.co/query'
    params = {'function': 'OVERVIEW', 'symbol': symbol, 'apikey': key}
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    return {
        'pe_ratio': float(data.get('PERatio', 0)),
        'ebitda': float(data.get('EBITDA', 0)) if data.get('EBITDA') not in ('N/A', '') else 0.0,
        'debt_to_equity': float(data.get('DebtToEquity', 0)),
    }


def fetch_finnhub_metrics(symbol):
    key = os.getenv('FINNHUB_API_KEY')
    if not key:
        raise ApiKeyError('Missing Finnhub API key')
    url = f'https://finnhub.io/api/v1/stock/metric'
    params = {'symbol': symbol, 'metric': 'all', 'token': key}
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    m = resp.json().get('metric', {})
    return {
        'pe_ratio': m.get('peBasicExclExtraTTM', 0.0),
        'ebitda': m.get('ebitda', 0.0),
        'debt_to_equity': m.get('debt/asset', 0.0),
    }


def fetch_fmp_ratios(symbol):
    key = os.getenv('FMP_API_KEY')
    if not key:
        raise ApiKeyError('Missing FMP API key')
    url = f'https://financialmodelingprep.com/api/v3/ratios/{symbol}'
    params = {'apikey': key}
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    arr = resp.json()
    if not arr:
        return {'pe_ratio': 0, 'ebitda': 0, 'debt_to_equity': 0}
    d = arr[0]
    return {
        'pe_ratio': d.get('priceEarningsRatio', 0.0),
        'ebitda': d.get('ebitda', 0.0),
        'debt_to_equity': d.get('debtEquityRatio', 0.0),
    }


def fetch_api_ninja_currency(symbol):
    # symbol like 'EURUSD' -> pair 'EUR_USD'
    key = os.getenv('API_NINJA_KEY', '')
    headers = {'X-Api-Key': key} if key else {}
    pair = f'{symbol[:3]}_{symbol[3:]}'
    url = 'https://api.api-ninjas.com/v1/currency'
    params = {'pair': pair}
    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    d = resp.json()
    return {
        'pe_ratio': 0.0,
        'ebitda': 0.0,
        'debt_to_equity': 0.0,
    }


def fetch_fundamental_features(source, symbol):
    """Unified interface to get fundamental features."""
    if source == 'alpha_vantage':
        return fetch_alpha_vantage_overview(symbol)
    if source == 'finnhub':
        return fetch_finnhub_metrics(symbol)
    if source == 'fmp':
        return fetch_fmp_ratios(symbol)
    if source == 'api_ninja':
        return fetch_api_ninja_currency(symbol)
    raise ValueError(f"Unknown source: {source}")
