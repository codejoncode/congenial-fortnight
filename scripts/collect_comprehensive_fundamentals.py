#!/usr/bin/env python3
"""
Comprehensive Fundamental Data Collection
Fetches all available macro data for EURUSD and XAUUSD from 2000-2025
Uses FRED, ECB, Alpha Vantage, Finnhub, FMP APIs
"""
import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from fredapi import Fred
import logging
from pathlib import Path
from dotenv import load_dotenv
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ComprehensiveFundamentalCollector:
    """Collect comprehensive fundamental data from multiple sources"""
    
    def __init__(self):
        self.fred_key = os.getenv('FRED_API_KEY')
        self.alpha_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.finnhub_key = os.getenv('FINNHUB_API_KEY')
        self.fmp_key = os.getenv('FMP_API_KEY')
        self.api_ninjas_key = os.getenv('API_NINJAS_API_KEY')
        
        self.fred = Fred(api_key=self.fred_key) if self.fred_key else None
        self.data_dir = Path('data')
        self.data_dir.mkdir(exist_ok=True)
        
        self.start_date = '2000-01-01'
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        
    def collect_fred_series(self, series_id: str, name: str) -> pd.DataFrame:
        """Collect a single FRED series"""
        try:
            logger.info(f"Fetching FRED series: {series_id} ({name})")
            data = self.fred.get_series(series_id, observation_start=self.start_date)
            df = pd.DataFrame(data, columns=[series_id.lower()])
            df.index.name = 'date'
            df = df.reset_index()
            df['date'] = pd.to_datetime(df['date'])
            
            # Save to CSV
            filepath = self.data_dir / f"{series_id}.csv"
            df.to_csv(filepath, index=False)
            logger.info(f"  ✅ Saved {len(df)} observations to {filepath}")
            
            return df
            
        except Exception as e:
            logger.error(f"  ❌ Error fetching {series_id}: {e}")
            return pd.DataFrame()
    
    def collect_ecb_eurusd(self) -> pd.DataFrame:
        """Collect EUR/USD from ECB Data Portal"""
        try:
            logger.info("Fetching EUR/USD from ECB Data Portal")
            url = "https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A"
            params = {
                'startPeriod': self.start_date,
                'endPeriod': self.end_date,
                'format': 'csvdata'
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse CSV response
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            # Clean and format
            if 'TIME_PERIOD' in df.columns and 'OBS_VALUE' in df.columns:
                df = df[['TIME_PERIOD', 'OBS_VALUE']].copy()
                df.columns = ['date', 'ecb_eurusd']
                df['date'] = pd.to_datetime(df['date'])
                
                filepath = self.data_dir / "ECB_EURUSD.csv"
                df.to_csv(filepath, index=False)
                logger.info(f"  ✅ Saved {len(df)} observations to {filepath}")
                return df
            
        except Exception as e:
            logger.error(f"  ❌ Error fetching ECB EUR/USD: {e}")
        
        return pd.DataFrame()
    
    def collect_alpha_vantage_fx(self, from_symbol: str, to_symbol: str) -> pd.DataFrame:
        """Collect FX daily data from Alpha Vantage"""
        try:
            logger.info(f"Fetching {from_symbol}/{to_symbol} from Alpha Vantage")
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'FX_DAILY',
                'from_symbol': from_symbol,
                'to_symbol': to_symbol,
                'outputsize': 'full',
                'apikey': self.alpha_key
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'Time Series FX (Daily)' in data:
                ts = data['Time Series FX (Daily)']
                df = pd.DataFrame.from_dict(ts, orient='index')
                df.index = pd.to_datetime(df.index)
                df.index.name = 'date'
                df = df.reset_index()
                
                # Rename columns
                df.columns = ['date', f'av_{from_symbol.lower()}{to_symbol.lower()}_open',
                             f'av_{from_symbol.lower()}{to_symbol.lower()}_high',
                             f'av_{from_symbol.lower()}{to_symbol.lower()}_low',
                             f'av_{from_symbol.lower()}{to_symbol.lower()}_close']
                
                # Keep only close
                df = df[['date', f'av_{from_symbol.lower()}{to_symbol.lower()}_close']].copy()
                
                filepath = self.data_dir / f"AV_{from_symbol}{to_symbol}.csv"
                df.to_csv(filepath, index=False)
                logger.info(f"  ✅ Saved {len(df)} observations to {filepath}")
                return df
                
            else:
                logger.warning(f"  ⚠️ No data in response: {data.get('Note', data.get('Error Message', 'Unknown error'))}")
                
        except Exception as e:
            logger.error(f"  ❌ Error fetching Alpha Vantage {from_symbol}/{to_symbol}: {e}")
        
        return pd.DataFrame()
    
    def collect_all_fred_fundamentals(self) -> dict:
        """Collect all FRED fundamental series"""
        
        fred_series = {
            # EURUSD Fundamentals
            'CPIAUCSL': 'US Consumer Price Index',
            'CP0000EZ19M086NEST': 'Euro Area HICP',
            'GDPC1': 'US Real GDP',
            'NAEXKP01EZQ657S': 'Euro Area Real GDP',
            'FEDFUNDS': 'US Federal Funds Rate',
            'UNRATE': 'US Unemployment Rate',
            'LRUNTTTTEZM156S': 'Euro Area Unemployment Rate',
            'DTWEXBGS': 'US Dollar Index (DXY)',
            'DGS3MO': '3-Month US Treasury',
            'DGS10': '10-Year US Treasury',
            'DGS2': '2-Year US Treasury',
            'DFF': 'Effective Federal Funds Rate',
            
            # XAUUSD Fundamentals
            'GOLDAMGBD228NLBM': 'Gold Price London PM Fix',
            'GOLDPMGBD228NLBM': 'Gold Price London PM Fix (alternative)',
            'T10YIE': '10-Year Breakeven Inflation Rate',
            'VIXCLS': 'VIX Volatility Index',
            
            # Energy (affects both)
            'DCOILWTICO': 'WTI Crude Oil Price',
            'DCOILBRENTEU': 'Brent Crude Oil Price',
            
            # Additional Economic Indicators
            'INDPRO': 'US Industrial Production',
            'PAYEMS': 'US Total Nonfarm Payrolls',
            'DGORDER': 'US Durable Goods Orders',
            'BOPGSTB': 'US Trade Balance',
            'DEXUSEU': 'US/Euro FX Rate',
            'DEXJPUS': 'Japan/US FX Rate',
            'DEXCHUS': 'China/US FX Rate',
            
            # Interest Rate Differentials
            'ECBDFR': 'ECB Deposit Facility Rate',
            'ECBRR': 'ECB Refinancing Rate',
        }
        
        collected = {}
        for series_id, name in fred_series.items():
            df = self.collect_fred_series(series_id, name)
            if not df.empty:
                collected[series_id] = df
            time.sleep(0.5)  # Rate limiting
        
        return collected
    
    def calculate_dxy_exy_cross(self) -> pd.DataFrame:
        """Calculate DXY/EXY and EXY/DXY cross indicators"""
        try:
            logger.info("Calculating DXY and EXY cross indicators")
            
            # Load DXY (US Dollar Index)
            dxy_path = self.data_dir / "DTWEXBGS.csv"
            if not dxy_path.exists():
                logger.warning("  ⚠️ DXY data not found, collecting first...")
                self.collect_fred_series('DTWEXBGS', 'US Dollar Index')
            
            dxy = pd.read_csv(dxy_path)
            dxy['date'] = pd.to_datetime(dxy['date'])
            
            # Load EURUSD to calculate EXY (Euro Index proxy)
            eurusd_path = self.data_dir / "DEXUSEU.csv"
            if not eurusd_path.exists():
                logger.warning("  ⚠️ EURUSD data not found, collecting first...")
                self.collect_fred_series('DEXUSEU', 'US/Euro FX Rate')
            
            eurusd = pd.read_csv(eurusd_path)
            eurusd['date'] = pd.to_datetime(eurusd['date'])
            
            # Merge on date
            df = pd.merge(dxy, eurusd, on='date', how='outer', suffixes=('_dxy', '_eurusd'))
            df = df.sort_values('date')
            
            # Forward fill missing values
            df['dtwexbgs'] = df['dtwexbgs'].fillna(method='ffill')
            df['dexuseu'] = df['dexuseu'].fillna(method='ffill')
            
            # Calculate EXY (Euro Index) as inverse of DXY weighted by EUR
            # EXY is approximated as 1/EURUSD normalized to DXY scale
            df['exy'] = (1 / df['dexuseu']) * 100  # Scale to index
            
            # Calculate DXY/EXY ratio (strength differential)
            df['dxy_exy_ratio'] = df['dtwexbgs'] / df['exy']
            
            # Calculate EXY/DXY ratio (inverse)
            df['exy_dxy_ratio'] = df['exy'] / df['dtwexbgs']
            
            # Calculate spread
            df['dxy_exy_spread'] = df['dtwexbgs'] - df['exy']
            
            # Calculate momentum (7-day change)
            df['dxy_exy_momentum'] = df['dxy_exy_ratio'].pct_change(7)
            df['exy_dxy_momentum'] = df['exy_dxy_ratio'].pct_change(7)
            
            # Save
            output_cols = ['date', 'dtwexbgs', 'exy', 'dxy_exy_ratio', 'exy_dxy_ratio', 
                          'dxy_exy_spread', 'dxy_exy_momentum', 'exy_dxy_momentum']
            df_output = df[output_cols].dropna()
            
            filepath = self.data_dir / "DXY_EXY_CROSS.csv"
            df_output.to_csv(filepath, index=False)
            logger.info(f"  ✅ Saved {len(df_output)} DXY/EXY cross indicators to {filepath}")
            
            return df_output
            
        except Exception as e:
            logger.error(f"  ❌ Error calculating DXY/EXY cross: {e}")
            return pd.DataFrame()
    
    def collect_all(self):
        """Collect all fundamental data"""
        logger.info("="*60)
        logger.info("COMPREHENSIVE FUNDAMENTAL DATA COLLECTION")
        logger.info("="*60)
        logger.info(f"Date Range: {self.start_date} to {self.end_date}")
        logger.info("")
        
        # 1. FRED fundamentals
        logger.info("1. Collecting FRED fundamentals...")
        fred_data = self.collect_all_fred_fundamentals()
        logger.info(f"   ✅ Collected {len(fred_data)} FRED series")
        logger.info("")
        
        # 2. ECB EUR/USD
        logger.info("2. Collecting ECB EUR/USD...")
        ecb_data = self.collect_ecb_eurusd()
        logger.info("")
        
        # 3. Alpha Vantage FX (with rate limiting)
        if self.alpha_key:
            logger.info("3. Collecting Alpha Vantage FX data...")
            logger.info("   Note: Alpha Vantage has strict rate limits (5 calls/min)")
            
            # EUR/USD
            self.collect_alpha_vantage_fx('EUR', 'USD')
            time.sleep(15)  # Wait to avoid rate limit
            
            # USD/JPY
            self.collect_alpha_vantage_fx('USD', 'JPY')
            time.sleep(15)
            
            # USD/CHF
            self.collect_alpha_vantage_fx('USD', 'CHF')
            logger.info("")
        
        # 4. DXY/EXY Cross Indicators
        logger.info("4. Calculating DXY/EXY cross indicators...")
        dxy_exy = self.calculate_dxy_exy_cross()
        logger.info("")
        
        # Summary
        logger.info("="*60)
        logger.info("COLLECTION COMPLETE")
        logger.info("="*60)
        logger.info("Data saved to data/ directory")
        logger.info("")
        logger.info("Available for EURUSD:")
        logger.info("  - US CPI, Euro HICP")
        logger.info("  - US GDP, Euro GDP")
        logger.info("  - Fed Funds, ECB rates")
        logger.info("  - Unemployment (US, Euro)")
        logger.info("  - DXY, EUR/USD, USD/JPY, USD/CHF")
        logger.info("  - DXY/EXY ratio and momentum")
        logger.info("")
        logger.info("Available for XAUUSD:")
        logger.info("  - Gold prices (FRED)")
        logger.info("  - US CPI, DXY")
        logger.info("  - Fed Funds, Treasury rates")
        logger.info("  - VIX, Oil prices")
        logger.info("  - DXY/EXY ratio (correlation test)")
        logger.info("")
        logger.info("Next: Update fundamental_pipeline.py to load all series")


if __name__ == "__main__":
    collector = ComprehensiveFundamentalCollector()
    collector.collect_all()
