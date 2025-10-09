#!/usr/bin/env python3
"""
FRED API Integration Fix for Forex Trading System

Provides a small, robust FRED API client and helpers to fetch and merge
economic series used as fundamentals in the forecasting pipeline.
"""

import os
import requests
import pandas as pd
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class FredAPIClient:
    """
    Fixed FRED API client for economic data integration
    Addresses common API key and request issues
    """
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('FRED_API_KEY')
        self.base_url = 'https://api.stlouisfed.org/fred'
        
        if not self.api_key:
            raise ValueError("FRED API key is required. Set FRED_API_KEY environment variable or pass api_key parameter.")
        
        logger.info(f"✅ FRED API initialized with key: {self.api_key[:8]}...")
    
    def get_series_data(self, series_id, start_date=None, end_date=None, limit=None):
        """
        Fetch economic data series from FRED API
        
        Args:
            series_id (str): FRED series ID (e.g., 'UNRATE', 'GDP', 'FEDFUNDS')
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format  
            limit (int): Maximum number of observations to return
        
        Returns:
            pandas.DataFrame: Time series data with date and value columns
        """
        
        try:
            # Build request parameters
            params = {
                'series_id': series_id,
                'api_key': self.api_key,
                'file_type': 'json'
            }
            
            if start_date:
                params['start_date'] = start_date
            if end_date:
                params['end_date'] = end_date
            if limit:
                params['limit'] = limit
            
            # Make API request
            url = f"{self.base_url}/series/observations"
            logger.info(f"📊 Fetching FRED series: {series_id}")
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            if 'observations' not in data:
                logger.error(f"❌ No observations found for series {series_id}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            observations = data['observations']
            df = pd.DataFrame(observations)
            
            # Clean and format data
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            # Remove missing values (marked as '.' in FRED)
            df = df.dropna(subset=['value'])
            
            # Sort by date
            df = df.sort_values('date').reset_index(drop=True)
            
            logger.info(f"✅ Retrieved {len(df)} observations for {series_id}")
            logger.info(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
            
            return df[['date', 'value']]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ FRED API request failed: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"❌ FRED data processing error: {str(e)}")
            return pd.DataFrame()
    
    def get_multiple_series(self, series_dict, start_date=None, end_date=None):
        """
        Fetch multiple economic series and merge them
        
        Args:
            series_dict (dict): Dictionary mapping column names to FRED series IDs
                               e.g., {'unemployment': 'UNRATE', 'gdp': 'GDP'}
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
        
        Returns:
            pandas.DataFrame: Combined data with date and multiple value columns
        """
        
        combined_df = None
        
        for column_name, series_id in series_dict.items():
            logger.info(f"📈 Fetching {column_name} ({series_id})")
            
            df = self.get_series_data(series_id, start_date, end_date)
            
            if df.empty:
                logger.warning(f"⚠️  No data for {column_name}, skipping")
                continue
            
            # Rename value column
            df = df.rename(columns={'value': column_name})
            
            # Merge with combined data
            if combined_df is None:
                combined_df = df.copy()
            else:
                combined_df = pd.merge(combined_df, df, on='date', how='outer')
        
        if combined_df is not None:
            # Sort by date and forward fill missing values
            combined_df = combined_df.sort_values('date').reset_index(drop=True)
            combined_df = combined_df.fillna(method='ffill')
            
            logger.info(f"✅ Combined dataset: {len(combined_df)} rows, {len(combined_df.columns)-1} series")
        
        return combined_df or pd.DataFrame()
    
    def test_connection(self):
        """Test FRED API connection and key validity"""
        
        logger.info("🧪 Testing FRED API connection...")
        
        try:
            # Test with unemployment rate (always available)
            test_df = self.get_series_data('UNRATE', limit=5)
            
            if not test_df.empty:
                logger.info(f"✅ FRED API connection successful!")
                logger.info(f"   Test data: {len(test_df)} unemployment rate observations")
                logger.info(f"   Latest value: {test_df['value'].iloc[-1]}% ({test_df['date'].iloc[-1].date()})")
                return True
            else:
                logger.error("❌ FRED API connection failed - no data returned")
                return False
                
        except Exception as e:
            logger.error(f"❌ FRED API test failed: {str(e)}")
            return False


def get_economic_fundamentals_for_forex(fred_api_key=None, start_date=None):
    """
    Get key economic indicators relevant for forex trading
    """
    if not start_date:
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    
    # Initialize FRED client
    fred_client = FredAPIClient(fred_api_key)
    
    # Test connection first
    if not fred_client.test_connection():
        logger.error("❌ FRED API connection failed - check your API key")
        return pd.DataFrame()
    
    # Define economic series relevant for forex trading
    economic_series = {
        'unemployment_rate': 'UNRATE',           # US Unemployment Rate
        'gdp_growth': 'GDP',                     # US GDP
        'inflation_cpi': 'CPIAUCSL',            # Consumer Price Index
        'fed_funds_rate': 'FEDFUNDS',           # Federal Funds Rate
        'treasury_10y': 'GS10',                 # 10-Year Treasury Rate
        'dollar_index': 'DTWEXBGS',             # Trade Weighted US Dollar Index
        'consumer_sentiment': 'UMCSENT',         # Consumer Sentiment
        'industrial_production': 'INDPRO'       # Industrial Production Index
    }
    
    logger.info("📊 Fetching forex-relevant economic fundamentals...")
    
    # Get combined economic data
    fundamentals_df = fred_client.get_multiple_series(
        economic_series, 
        start_date=start_date
    )
    
    if fundamentals_df.empty:
        logger.error("❌ No economic fundamentals data retrieved")
        return pd.DataFrame()
    
    # Rename date column to match forex data format
    fundamentals_df = fundamentals_df.rename(columns={'date': 'timestamp'})
    
    # Ensure timestamp is datetime
    fundamentals_df['timestamp'] = pd.to_datetime(fundamentals_df['timestamp'])
    
    logger.info(f"✅ Economic fundamentals retrieved:")
    logger.info(f"   {len(fundamentals_df)} observations from {fundamentals_df['timestamp'].min().date()}")
    logger.info(f"   Latest data: {fundamentals_df['timestamp'].max().date()}")
    logger.info(f"   Series: {list(fundamentals_df.columns)[1:]}")  # Exclude timestamp
    
    return fundamentals_df


def integrate_fundamentals_with_forex(forex_df, fundamentals_df):
    """
    Integrate economic fundamentals with forex price data
    """
    logger.info("🔗 Integrating economic fundamentals with forex data...")
    
    # Ensure both have timestamp columns
    if 'timestamp' not in forex_df.columns:
        logger.error("❌ Forex data missing 'timestamp' column")
        return forex_df
    
    if 'timestamp' not in fundamentals_df.columns:
        logger.error("❌ Fundamentals data missing 'timestamp' column")  
        return forex_df
    
    # Convert to datetime
    forex_df['timestamp'] = pd.to_datetime(forex_df['timestamp'])
    fundamentals_df['timestamp'] = pd.to_datetime(fundamentals_df['timestamp'])
    
    # Merge using forward fill for fundamentals (they update less frequently)
    combined_df = pd.merge_asof(
        forex_df.sort_values('timestamp'),
        fundamentals_df.sort_values('timestamp'),
        on='timestamp',
        direction='backward'  # Use latest available fundamental data
    )
    
    logger.info(f"✅ Integration complete:")
    logger.info(f"   {len(combined_df)} forex observations")
    logger.info(f"   {len([col for col in combined_df.columns if col not in forex_df.columns])} fundamental features added")
    
    return combined_df


if __name__ == "__main__":
    print("🔧 FRED API INTEGRATION TEST")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv('FRED_API_KEY')
    if not api_key:
        print("❌ FRED_API_KEY environment variable not set")
        print("   Get your key from: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("   Set it with: export FRED_API_KEY='your_key_here'")
    else:
        print(f"✅ FRED API key found: {api_key[:8]}...")
        
        # Test the integration
        try:
            fundamentals = get_economic_fundamentals_for_forex(api_key)
            
            if not fundamentals.empty:
                print(f"✅ Successfully retrieved economic data:")
                print(f"   {len(fundamentals)} observations")
                print(f"   Columns: {list(fundamentals.columns)}")
                
                # Show sample data
                print("\n📊 Sample economic data:")
                print(fundamentals.head())
                
                # Save to CSV for testing
                fundamentals.to_csv('economic_fundamentals_sample.csv', index=False)
                print("\n💾 Sample data saved to: economic_fundamentals_sample.csv")
                
            else:
                print("❌ No economic data retrieved")
                
        except Exception as e:
            print(f"❌ FRED integration test failed: {str(e)}")
    
    print("\n" + "=" * 50)
