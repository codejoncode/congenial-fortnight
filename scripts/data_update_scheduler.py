#!/usr/bin/env python3
"""
Data Update Scheduler for Trading System
Fetches price data at US market close and manages fundamental data updates
Optimized to stay within free API tier limits

Schedule:
- Price data (Yahoo Finance): 9 PM UTC daily (US market close at 4 PM ET)
- Fundamental data (FRED): Twice daily at 6 AM and 2 PM UTC (within 100 calls/day limit)
"""

import os
import sys
import schedule
import time
import yfinance as yf
import pandas as pd
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataUpdateScheduler:
    """
    Scheduler for trading data updates
    
    Features:
    - Price data updates at US market close (9 PM UTC / 4 PM ET)
    - Fundamental data updates twice daily (within FRED limits)
    - Avoids redundant API calls
    - Free tier optimization
    """
    
    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path(project_root) / 'data'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Trading pairs to update
        self.forex_pairs = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X']
        self.commodities = ['GC=F', 'CL=F']  # Gold, Oil
        
        # Map Yahoo Finance symbols to our file names
        self.symbol_mapping = {
            'EURUSD=X': 'EURUSD',
            'GBPUSD=X': 'GBPUSD',
            'USDJPY=X': 'USDJPY',
            'AUDUSD=X': 'AUDUSD',
            'GC=F': 'XAUUSD',  # Gold
            'CL=F': 'USOIL'    # Crude Oil
        }
        
        logger.info(f"DataUpdateScheduler initialized")
        logger.info(f"Data directory: {self.data_dir}")
    
    def fetch_price_data(self, symbol: str, period: str = '60d', interval: str = '1h') -> pd.DataFrame:
        """
        Fetch price data from Yahoo Finance (unlimited free tier)
        
        Args:
            symbol: Yahoo Finance symbol (e.g., 'EURUSD=X', 'GC=F')
            period: Data period ('60d', '1y', etc.)
            interval: Data interval ('1h', '1d', etc.)
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Fetching {symbol} data ({interval} interval, {period} period)...")
            
            df = yf.download(
                symbol,
                period=period,
                interval=interval,
                progress=False,
                show_errors=False
            )
            
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Standardize column names
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Reset index to make datetime a column
            df = df.reset_index()
            if 'Date' in df.columns:
                df = df.rename(columns={'Date': 'timestamp'})
            elif 'Datetime' in df.columns:
                df = df.rename(columns={'Datetime': 'timestamp'})
            
            # Add required columns if missing
            if 'id' not in df.columns:
                df['id'] = range(len(df))
            if 'time' not in df.columns:
                df['time'] = df['timestamp']
            if 'spread' not in df.columns:
                df['spread'] = 0
            
            # Reorder columns to match expected schema
            expected_columns = ['id', 'timestamp', 'time', 'open', 'high', 'low', 'close', 'volume', 'spread']
            df = df[[col for col in expected_columns if col in df.columns]]
            
            logger.info(f"✅ Successfully fetched {len(df)} rows for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error fetching {symbol}: {e}")
            return pd.DataFrame()
    
    def update_price_data_at_us_close(self):
        """
        Update price data at US market close (9 PM UTC / 4 PM ET)
        This runs once per day and fetches H1 data for all pairs
        """
        logger.info("=" * 60)
        logger.info("🕒 US Market Close - Updating Price Data")
        logger.info(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        logger.info("=" * 60)
        
        updated_count = 0
        failed_count = 0
        
        # Update forex pairs
        all_symbols = self.forex_pairs + self.commodities
        
        for symbol in all_symbols:
            try:
                # Fetch H1 (1-hour) data for last 60 days
                df = self.fetch_price_data(symbol, period='60d', interval='1h')
                
                if df.empty:
                    logger.warning(f"⏭️  Skipping {symbol} - no data available")
                    failed_count += 1
                    continue
                
                # Save to file
                file_name = self.symbol_mapping.get(symbol, symbol.replace('=X', '').replace('=F', ''))
                file_path = self.data_dir / f"{file_name}_H1.csv"
                
                df.to_csv(file_path, index=False)
                logger.info(f"✅ Updated {file_path.name} - {len(df)} rows")
                updated_count += 1
                
                # Also update D1 (daily) data for longer-term analysis
                df_daily = self.fetch_price_data(symbol, period='1y', interval='1d')
                if not df_daily.empty:
                    file_path_daily = self.data_dir / f"{file_name}_D1.csv"
                    df_daily.to_csv(file_path_daily, index=False)
                    logger.info(f"✅ Updated {file_path_daily.name} - {len(df_daily)} rows")
                
                # Small delay to avoid rate limiting (even though Yahoo is unlimited)
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Failed to update {symbol}: {e}")
                failed_count += 1
        
        logger.info("=" * 60)
        logger.info(f"✅ Price data update complete: {updated_count} succeeded, {failed_count} failed")
        logger.info("=" * 60)
        
        return updated_count, failed_count
    
    def check_fundamental_data_freshness(self) -> bool:
        """
        Check if fundamental data needs updating
        Avoid redundant API calls to stay within FRED limits
        
        Returns:
            True if data needs updating, False otherwise
        """
        try:
            # Check timestamp of a key fundamental file
            sample_file = self.data_dir / 'CPIAUCSL.csv'
            
            if not sample_file.exists():
                logger.info("Fundamental data not found - needs update")
                return True
            
            # Get file modification time
            file_time = datetime.fromtimestamp(sample_file.stat().st_mtime)
            time_since_update = datetime.now() - file_time
            
            # Update if older than 12 hours
            if time_since_update > timedelta(hours=12):
                logger.info(f"Fundamental data is {time_since_update.total_seconds()/3600:.1f} hours old - needs update")
                return True
            
            logger.info(f"Fundamental data is recent ({time_since_update.total_seconds()/3600:.1f} hours old) - skipping")
            return False
            
        except Exception as e:
            logger.error(f"Error checking fundamental data freshness: {e}")
            return True  # Update on error to be safe
    
    def update_fundamental_data(self):
        """
        Update fundamental data from FRED API
        This is scheduled twice daily (6 AM and 2 PM UTC) within free tier limits
        """
        if not self.check_fundamental_data_freshness():
            logger.info("⏭️  Skipping fundamental data update - data is recent")
            return
        
        logger.info("=" * 60)
        logger.info("📊 Updating Fundamental Data")
        logger.info(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        logger.info("=" * 60)
        
        try:
            # Import fundamental pipeline
            from scripts.fundamental_pipeline import FundamentalDataPipeline
            
            # Initialize and run pipeline
            pipeline = FundamentalDataPipeline()
            fundamental_df = pipeline.run()
            
            if fundamental_df is not None and not fundamental_df.empty:
                logger.info(f"✅ Fundamental data updated successfully - {len(fundamental_df)} rows")
            else:
                logger.warning("⚠️  Fundamental data update returned empty DataFrame")
                
        except Exception as e:
            logger.error(f"❌ Error updating fundamental data: {e}")
        
        logger.info("=" * 60)
    
    def run_scheduler(self):
        """
        Main scheduler loop
        Schedules:
        - Price data: Daily at 21:00 UTC (US market close)
        - Fundamental data: Daily at 06:00 and 14:00 UTC
        """
        logger.info("=" * 60)
        logger.info("🚀 Starting Data Update Scheduler")
        logger.info("=" * 60)
        logger.info("")
        logger.info("Schedule:")
        logger.info("  📈 Price Data:        21:00 UTC daily (US close)")
        logger.info("  📊 Fundamental Data:  06:00, 14:00 UTC daily")
        logger.info("")
        logger.info("Free Tier Limits:")
        logger.info("  - Yahoo Finance:  Unlimited (price data)")
        logger.info("  - FRED API:       100 calls/day (conservative)")
        logger.info("")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 60)
        
        # Schedule price data updates at US market close
        schedule.every().day.at("21:00").do(self.update_price_data_at_us_close)
        
        # Schedule fundamental data updates (twice daily)
        schedule.every().day.at("06:00").do(self.update_fundamental_data)
        schedule.every().day.at("14:00").do(self.update_fundamental_data)
        
        # Run immediately on startup to check/update data
        logger.info("\n🔄 Running initial data check...")
        self.update_price_data_at_us_close()
        self.update_fundamental_data()
        
        # Main scheduler loop
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("\n⏹️  Scheduler stopped by user")
        except Exception as e:
            logger.error(f"\n❌ Scheduler error: {e}")


def main():
    """
    Run the data update scheduler
    """
    scheduler = DataUpdateScheduler()
    scheduler.run_scheduler()


if __name__ == "__main__":
    main()
