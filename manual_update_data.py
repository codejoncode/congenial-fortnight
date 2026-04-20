#!/usr/bin/env python3
"""
Manually update market data CSV files
"""
import os
import sys
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def update_market_data():
    """Update EURUSD and XAUUSD data files"""
    data_dir = Path('data')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    symbol_mappings = [
        ('EURUSD=X', 'EURUSD'),
        ('GC=F', 'XAUUSD'),
    ]
    
    for symbol, filename in symbol_mappings:
        print(f"\n{'='*60}")
        print(f"Updating {filename} ({symbol})")
        print(f"{'='*60}")
        
        try:
            # Fetch H1 data (60 days)
            print(f"Fetching H1 data (60 days, 1h interval)...")
            df_h1 = yf.download(symbol, period='60d', interval='1h', progress=True)
            
            if not df_h1.empty:
                df_h1 = df_h1.reset_index()
                
                # Handle column names
                timestamp_col = 'Datetime' if 'Datetime' in df_h1.columns else 'Date'
                df_h1['timestamp'] = pd.to_datetime(df_h1[timestamp_col])
                df_h1['date'] = df_h1['timestamp'].dt.date
                df_h1['time'] = df_h1['timestamp'].dt.time
                
                # Rename columns
                df_h1 = df_h1.rename(columns={
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                
                # Add required columns
                df_h1['id'] = range(1, len(df_h1) + 1)
                df_h1['spread'] = 2
                
                # Select columns
                columns = ['id', 'timestamp', 'date', 'time', 'open', 'high', 'low', 'close', 'volume', 'spread']
                df_h1 = df_h1[columns]
                
                # Save
                h1_file = data_dir / f'{filename}_H1.csv'
                df_h1.to_csv(h1_file, index=False)
                
                latest_date = df_h1['timestamp'].max()
                print(f"✅ Saved {h1_file}")
                print(f"   Records: {len(df_h1)}")
                print(f"   Latest: {latest_date}")
            
            # Fetch Daily data (1 year)
            print(f"\nFetching Daily data (1 year, 1d interval)...")
            df_daily = yf.download(symbol, period='1y', interval='1d', progress=True)
            
            if not df_daily.empty:
                df_daily = df_daily.reset_index()
                df_daily['timestamp'] = pd.to_datetime(df_daily['Date'])
                df_daily['date'] = df_daily['timestamp'].dt.date
                df_daily['time'] = pd.to_datetime('00:00:00').time()
                
                # Rename columns
                df_daily = df_daily.rename(columns={
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                
                # Add required columns
                df_daily['id'] = range(1, len(df_daily) + 1)
                df_daily['spread'] = 2
                
                # Select columns
                columns = ['id', 'timestamp', 'date', 'time', 'open', 'high', 'low', 'close', 'volume', 'spread']
                df_daily = df_daily[columns]
                
                # Save
                daily_file = data_dir / f'{filename}_Daily.csv'
                df_daily.to_csv(daily_file, index=False)
                
                latest_date = df_daily['timestamp'].max()
                print(f"✅ Saved {daily_file}")
                print(f"   Records: {len(df_daily)}")
                print(f"   Latest: {latest_date}")
                
        except Exception as e:
            print(f"❌ Error updating {filename}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"✅ Data update complete!")
    print(f"{'='*60}")

if __name__ == '__main__':
    update_market_data()
