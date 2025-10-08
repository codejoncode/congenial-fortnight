"""
Fundamentals Data Fetching Module

This module provides a unified interface for fetching fundamental data from various sources.
It's designed to be used by tests and can be extended to support multiple data providers.

Note: This is a compatibility module to support existing tests.
The main fundamental data pipeline is in scripts/fundamental_pipeline.py
"""

import os
from typing import Dict, Optional


def fetch_fundamental_features(source: str, ticker: str) -> Dict[str, float]:
    """
    Unified interface for fetching fundamental features from various sources.
    
    Args:
        source: Data source name ('alpha_vantage', 'finnhub', 'fmp', 'api_ninja')
        ticker: Ticker symbol (e.g., 'EURUSD', 'AAPL')
    
    Returns:
        Dictionary with fundamental features
        
    Raises:
        EnvironmentError: If required API key is missing
        NotImplementedError: If source is not supported
    """
    if source == 'alpha_vantage':
        return fetch_alpha_vantage_overview(ticker)
    elif source == 'finnhub':
        return fetch_finnhub_metrics(ticker)
    elif source == 'fmp':
        return fetch_fmp_data(ticker)
    elif source == 'api_ninja':
        return fetch_api_ninja_data(ticker)
    else:
        raise NotImplementedError(f"Source '{source}' is not supported")


def fetch_alpha_vantage_overview(ticker: str) -> Dict[str, float]:
    """
    Fetch fundamental data from Alpha Vantage.
    
    Args:
        ticker: Ticker symbol
        
    Returns:
        Dictionary with fundamental features
        
    Raises:
        EnvironmentError: If ALPHA_VANTAGE_API_KEY is not set
    """
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY') or os.getenv('AV_API_KEY')
    if not api_key:
        raise EnvironmentError(
            "ALPHA_VANTAGE_API_KEY environment variable is required"
        )
    
    # Import here to avoid circular dependencies
    import requests
    
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'OVERVIEW',
        'symbol': ticker,
        'apikey': api_key
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    
    data = response.json()
    
    # Extract and normalize fundamental features
    return {
        'pe_ratio': float(data.get('PERatio', 0) or 0),
        'ebitda': float(data.get('EBITDA', 0) or 0),
        'debt_to_equity': float(data.get('DebtToEquity', 0) or 0),
    }


def fetch_finnhub_metrics(ticker: str) -> Dict[str, float]:
    """
    Fetch fundamental data from Finnhub.
    
    Args:
        ticker: Ticker symbol
        
    Returns:
        Dictionary with fundamental features
        
    Raises:
        EnvironmentError: If FINNHUB_API_KEY is not set
    """
    api_key = os.getenv('FINNHUB_API_KEY')
    if not api_key:
        raise EnvironmentError(
            "FINNHUB_API_KEY environment variable is required"
        )
    
    import requests
    
    url = 'https://finnhub.io/api/v1/stock/metric'
    params = {
        'symbol': ticker,
        'metric': 'all',
        'token': api_key
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    
    data = response.json()
    metric = data.get('metric', {})
    
    return {
        'pe_ratio': float(metric.get('peBasicExclExtraTTM', 0) or 0),
        'ebitda': float(metric.get('ebitdaTTM', 0) or 0),
        'debt_to_equity': float(metric.get('totalDebt/totalEquityAnnual', 0) or 0),
    }


def fetch_fmp_data(ticker: str) -> Dict[str, float]:
    """
    Fetch fundamental data from Financial Modeling Prep.
    
    Args:
        ticker: Ticker symbol
        
    Returns:
        Dictionary with fundamental features
        
    Raises:
        EnvironmentError: If FMP_API_KEY is not set
    """
    api_key = os.getenv('FMP_API_KEY')
    if not api_key:
        raise EnvironmentError(
            "FMP_API_KEY environment variable is required"
        )
    
    import requests
    
    url = f'https://financialmodelingprep.com/api/v3/ratios/{ticker}'
    params = {'apikey': api_key}
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    
    data = response.json()
    if data:
        latest = data[0]
        return {
            'pe_ratio': float(latest.get('priceEarningsRatio', 0) or 0),
            'ebitda': 0.0,  # Not directly available in ratios endpoint
            'debt_to_equity': float(latest.get('debtEquityRatio', 0) or 0),
        }
    
    return {
        'pe_ratio': 0.0,
        'ebitda': 0.0,
        'debt_to_equity': 0.0,
    }


def fetch_api_ninja_data(ticker: str) -> Dict[str, float]:
    """
    Fetch fundamental data from API Ninjas.
    
    Args:
        ticker: Ticker symbol
        
    Returns:
        Dictionary with fundamental features
        
    Raises:
        EnvironmentError: If API_NINJA_API_KEY is not set
    """
    api_key = os.getenv('API_NINJA_API_KEY')
    if not api_key:
        raise EnvironmentError(
            "API_NINJA_API_KEY environment variable is required"
        )
    
    import requests
    
    url = f'https://api.api-ninjas.com/v1/stockfundamentals'
    params = {'ticker': ticker}
    headers = {'X-Api-Key': api_key}
    
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    
    data = response.json()
    
    return {
        'pe_ratio': float(data.get('pe_ratio', 0) or 0),
        'ebitda': float(data.get('ebitda', 0) or 0),
        'debt_to_equity': float(data.get('debt_to_equity', 0) or 0),
    }


if __name__ == '__main__':
    # Simple test
    print("Fundamentals module loaded successfully")
    print("\nAvailable functions:")
    print("  - fetch_fundamental_features(source, ticker)")
    print("  - fetch_alpha_vantage_overview(ticker)")
    print("  - fetch_finnhub_metrics(ticker)")
    print("  - fetch_fmp_data(ticker)")
    print("  - fetch_api_ninja_data(ticker)")
    print("\nSupported sources: alpha_vantage, finnhub, fmp, api_ninja")
