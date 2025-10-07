#!/usr/bin/env python3
"""Fetch comprehensive gold price data from multiple sources"""
import os
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

def fetch_gold_from_fmp():
    """Fetch gold from Financial Modeling Prep"""
    fmp_key = os.getenv('FMP_API_KEY')
    
    print('Fetching Gold (XAUUSD) from Financial Modeling Prep...')
    url = f'https://financialmodelingprep.com/api/v3/historical-price-full/XAUUSD'
    params = {
        'from': '2000-01-01',
        'to': '2025-10-06',
        'apikey': fmp_key
    }
    
    response = requests.get(url, params=params, timeout=30)
    data = response.json()
    
    if 'historical' in data:
        df = pd.DataFrame(data['historical'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        df_out = df[['date', 'close']].copy()
        df_out.columns = ['date', 'gold_price']
        
        df_out.to_csv('data/GOLD_PRICE_FMP.csv', index=False)
        print(f'✅ Saved {len(df_out)} gold price observations')
        print(f'   Date range: {df_out["date"].min()} to {df_out["date"].max()}')
        return df_out
    else:
        print(f'❌ FMP Error: {data}')
        return pd.DataFrame()

def fetch_gold_from_xauusd_data():
    """Extract gold from existing XAUUSD_Daily.csv"""
    print('Extracting gold from XAUUSD_Daily.csv...')
    
    df = pd.read_csv('data/XAUUSD_Daily.csv')
    df['date'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('date')
    
    df_out = df[['date', 'close']].copy()
    df_out.columns = ['date', 'gold_price']
    
    df_out.to_csv('data/GOLD_PRICE_MT.csv', index=False)
    print(f'✅ Saved {len(df_out)} gold price observations from MetaTrader')
    print(f'   Date range: {df_out["date"].min()} to {df_out["date"].max()}')
    return df_out

if __name__ == "__main__":
    # Try FMP first
    gold_fmp = fetch_gold_from_fmp()
    
    # Also get from our existing MT data
    gold_mt = fetch_gold_from_xauusd_data()
    
    print('\n✅ Gold data collection complete')
