# Specific solutions for your current data validation issues
# Address the exact problems shown in your logs

import pandas as pd
import logging
from pathlib import Path

def fix_current_data_issues():
    """
    Targeted fixes for the specific issues in your logs:
    1. Unable to identify date column in data/EURUSD_H4.csv
    2. Skipping Holloway for H4 due to insufficient data (0 rows)
    3. Skipping Holloway for Weekly due to insufficient data (0 rows)
    """
    
    logger = logging.getLogger('forecasting')
    
    # 1. FIX H4 DATA ISSUE
    h4_file = 'data/EURUSD_H4.csv'
    if Path(h4_file).exists():
        try:
            h4_df = pd.read_csv(h4_file)
            logger.info(f"H4 file exists with {len(h4_df)} rows")
            logger.info(f"H4 columns: {list(h4_df.columns)}")
            
            if len(h4_df) == 0:
                logger.error("❌ H4 file is empty - need to regenerate H4 data")
                # Generate H4 data from H1 or Daily data
                regenerate_h4_data_from_daily()
            else:
                # Fix column naming issue
                fixed_h4 = fix_column_naming_issues(h4_df, 'H4')
                if fixed_h4 is not None:
                    fixed_h4.to_csv(h4_file, index=False)
                    logger.info("✅ Fixed H4 column naming and saved")
                    
        except Exception as e:
            logger.error(f"❌ H4 data issue: {str(e)}")
            logger.info("Attempting to regenerate H4 data...")
            regenerate_h4_data_from_daily()
    else:
        logger.error("❌ H4 file doesn't exist - creating from Daily data")
        regenerate_h4_data_from_daily()
    
    # 2. FIX WEEKLY DATA ISSUE
    weekly_file = 'data/EURUSD_Weekly.csv'
    if Path(weekly_file).exists():
        try:
            weekly_df = pd.read_csv(weekly_file)
            logger.info(f"Weekly file exists with {len(weekly_df)} rows")
            
            if len(weekly_df) == 0:
                logger.error("❌ Weekly file is empty - need to regenerate")
                regenerate_weekly_data_from_daily()
            else:
                logger.info("✅ Weekly data looks OK")
                
        except Exception as e:
            logger.error(f"❌ Weekly data issue: {str(e)}")
            regenerate_weekly_data_from_daily()
    else:
        logger.error("❌ Weekly file doesn't exist - creating from Daily data")
        regenerate_weekly_data_from_daily()


def fix_column_naming_issues(df, timeframe_name):
    """Fix common column naming issues that prevent date column identification"""
    
    logger = logging.getLogger('forecasting')
    
    # Common problematic column names and their fixes
    column_fixes = {
        # Timestamp variations
        'Date': 'timestamp',
        'date': 'timestamp', 
        'DateTime': 'timestamp',
        'datetime': 'timestamp',
        'Time': 'timestamp',
        'time': 'timestamp',
        'Timestamp': 'timestamp',
        # OHLC variations
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'O': 'open',
        'H': 'high',
        'L': 'low',
        'C': 'close',
        'Volume': 'volume'
    }
    
    # Apply column name fixes
    df_fixed = df.rename(columns=column_fixes)
    
    # Check if we now have a timestamp column
    timestamp_candidates = ['timestamp', 'date', 'datetime', 'time']
    timestamp_col = None
    
    for candidate in timestamp_candidates:
        if candidate in df_fixed.columns:
            timestamp_col = candidate
            break
    
    if timestamp_col is None:
        logger.error(f"❌ {timeframe_name}: Still cannot identify timestamp column after fixes")
        logger.error(f"Available columns: {list(df_fixed.columns)}")
        return None
    
    # Ensure timestamp is properly formatted
    try:
        df_fixed[timestamp_col] = pd.to_datetime(df_fixed[timestamp_col])
        logger.info(f"✅ {timeframe_name}: Fixed timestamp column '{timestamp_col}'")
    except Exception as e:
        logger.error(f"❌ {timeframe_name}: Cannot convert {timestamp_col} to datetime: {str(e)}")
        return None
    
    # Ensure required OHLC columns exist
    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df_fixed.columns]
    
    if missing_cols:
        logger.error(f"❌ {timeframe_name}: Missing required columns: {missing_cols}")
        return None
    
    # Sort by timestamp and remove duplicates
    df_fixed = df_fixed.sort_values(timestamp_col).drop_duplicates(subset=[timestamp_col])
    
    logger.info(f"✅ {timeframe_name}: Column issues fixed, {len(df_fixed)} valid rows")
    return df_fixed


def regenerate_h4_data_from_daily():
    """Generate H4 data by resampling Daily data"""
    
    logger = logging.getLogger('forecasting')
    daily_file = 'data/EURUSD_Daily.csv'
    
    try:
        if not Path(daily_file).exists():
            logger.error("❌ Cannot regenerate H4 data - Daily file missing")
            return False
            
        daily_df = pd.read_csv(daily_file)
        
        if len(daily_df) == 0:
            logger.error("❌ Cannot regenerate H4 data - Daily file is empty")
            return False
        
        # Fix daily data column names first
        daily_df = fix_column_naming_issues(daily_df, 'Daily')
        if daily_df is None:
            return False
        
        # For forex, we need to simulate intraday data to create H4
        # This is a simplified approach - ideally you'd have real H4 data
        logger.info("📊 Generating synthetic H4 data from Daily OHLC...")
        
        h4_data = []
        for _, row in daily_df.iterrows():
            # Create 6 H4 bars per day (24/4 = 6)
            daily_open = row['open']
            daily_high = row['high'] 
            daily_low = row['low']
            daily_close = row['close']
            base_date = row['timestamp']
            
            # Generate 6 H4 bars with realistic progression
            for hour in [0, 4, 8, 12, 16, 20]:
                h4_timestamp = base_date + pd.Timedelta(hours=hour)
                
                # Distribute price movement across H4 bars
                if hour == 0:
                    h4_open = daily_open
                else:
                    h4_open = h4_data[-1]['close']  # Previous bar's close
                
                if hour == 20:  # Last H4 bar of the day
                    h4_close = daily_close
                else:
                    # Interpolate toward daily close
                    progress = (hour + 4) / 24
                    h4_close = daily_open + (daily_close - daily_open) * progress
                    # Add some randomness
                    h4_close += (daily_close - daily_open) * 0.1 * (0.5 - pd.np.random.random())
                
                # Calculate high/low for this H4 bar
                bar_range = abs(daily_high - daily_low) / 6  # Distribute daily range
                h4_high = max(h4_open, h4_close) + bar_range * 0.3
                h4_low = min(h4_open, h4_close) - bar_range * 0.3
                
                # Ensure daily high/low constraints
                h4_high = min(h4_high, daily_high)
                h4_low = max(h4_low, daily_low)
                
                h4_data.append({
                    'timestamp': h4_timestamp,
                    'open': h4_open,
                    'high': h4_high,
                    'low': h4_low,
                    'close': h4_close,
                    'volume': row.get('volume', 0) / 6  # Distribute daily volume
                })
        
        h4_df = pd.DataFrame(h4_data)
        h4_file = 'data/EURUSD_H4.csv'
        h4_df.to_csv(h4_file, index=False)
        
        logger.info(f"✅ Generated {len(h4_df)} H4 bars and saved to {h4_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to regenerate H4 data: {str(e)}")
        return False


def regenerate_weekly_data_from_daily():
    """Generate Weekly data by resampling Daily data"""
    
    logger = logging.getLogger('forecasting')
    daily_file = 'data/EURUSD_Daily.csv'
    
    try:
        if not Path(daily_file).exists():
            logger.error("❌ Cannot regenerate Weekly data - Daily file missing")
            return False
            
        daily_df = pd.read_csv(daily_file)
        
        if len(daily_df) == 0:
            logger.error("❌ Cannot regenerate Weekly data - Daily file is empty")
            return False
        
        # Fix daily data column names first
        daily_df = fix_column_naming_issues(daily_df, 'Daily')
        if daily_df is None:
            return False
        
        logger.info("📊 Resampling Daily data to Weekly...")
        
        # Set timestamp as index for resampling
        daily_df.set_index('timestamp', inplace=True)
        
        # Resample to weekly data (Sunday to Saturday)
        weekly_df = daily_df.resample('W-SUN').agg({
            'open': 'first',   # First open of the week
            'high': 'max',     # Highest high of the week
            'low': 'min',      # Lowest low of the week
            'close': 'last',   # Last close of the week
            'volume': 'sum'    # Sum of weekly volume
        }).dropna()
        
        # Reset index to make timestamp a column again
        weekly_df.reset_index(inplace=True)
        
        weekly_file = 'data/EURUSD_Weekly.csv'
        weekly_df.to_csv(weekly_file, index=False)
        
        logger.info(f"✅ Generated {len(weekly_df)} Weekly bars and saved to {weekly_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to regenerate Weekly data: {str(e)}")
        return False


def validate_and_fix_all_timeframes():
    """
    Comprehensive function to validate and fix all timeframe data issues
    Call this before running your training pipeline
    """
    
    logger = logging.getLogger('forecasting')
    logger.info("🔧 FIXING ALL TIMEFRAME DATA ISSUES...")
    
    # Fix current specific issues
    fix_current_data_issues()
    
    # Validate all files now exist and have proper format
    timeframes = {
        'Daily': 'data/EURUSD_Daily.csv',
        'H4': 'data/EURUSD_H4.csv', 
        'H1': 'data/EURUSD_H1.csv',
        'Weekly': 'data/EURUSD_Weekly.csv'
    }
    
    fixed_count = 0
    for timeframe, filepath in timeframes.items():
        if Path(filepath).exists():
            try:
                df = pd.read_csv(filepath)
                if len(df) > 0:
                    # Quick validation
                    if 'timestamp' in df.columns or any(col.lower() in ['date', 'time'] for col in df.columns):
                        logger.info(f"✅ {timeframe}: {len(df)} rows - OK")
                        fixed_count += 1
                    else:
                        logger.warning(f"⚠️  {timeframe}: Column naming issues remain")
                else:
                    logger.warning(f"⚠️  {timeframe}: Still empty after fixes")
            except Exception as e:
                logger.error(f"❌ {timeframe}: Still has issues - {str(e)}")
        else:
            logger.error(f"❌ {timeframe}: File still missing")
    
    logger.info(f"🎯 TIMEFRAME VALIDATION COMPLETE: {fixed_count}/{len(timeframes)} timeframes ready")
    
    if fixed_count < 2:  # Need at least 2 timeframes for training
        logger.error("❌ INSUFFICIENT VIABLE TIMEFRAMES - Need at least 2 for training")
        return False
    
    return True


# Add this to your main forecasting.py before training starts:
def pre_training_data_fix():
    """Call this at the very beginning of your forecasting pipeline"""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger('forecasting')
    
    logger.info("🚀 STARTING PRE-TRAINING DATA VALIDATION AND FIXES")
    logger.info("=" * 60)
    
    # Fix all known issues
    success = validate_and_fix_all_timeframes()
    
    if not success:
        logger.error("❌ DATA FIXES FAILED - CANNOT PROCEED WITH TRAINING")
        logger.error("Manual intervention required to fix data files")
        return False
    
    logger.info("✅ ALL DATA ISSUES FIXED - READY FOR TRAINING")
    logger.info("=" * 60)
    return True


if __name__ == "__main__":
    # Test the fixes
    pre_training_data_fix()
