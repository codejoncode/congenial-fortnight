import os
from robust_data_loader import robust_data_loading_pipeline, ForexDataLoader, emergency_data_recovery
import pandas as pd
import numpy as np
import logging
import sys

# Setup logging for integration
logger = logging.getLogger('forecasting')

def main():
    """
    Main forecasting function - replace your current main() with this structure
    """
    
    logger.info("🚀 STARTING FORECASTING SYSTEM WITH ROBUST DATA LOADING")
    logger.info("=" * 70)
    
    try:
        # STEP 1: ROBUST DATA LOADING (replaces your current data loading)
        logger.info("STEP 1: Loading and validating all timeframe data...")
        
        # This replaces all your current CSV loading logic
        datasets = robust_data_loading_pipeline()
        
        if not datasets:
            logger.error("❌ CRITICAL: No data could be loaded - system cannot proceed")
            sys.exit(1)
        
        logger.info(f"✅ Data loading successful for timeframes: {list(datasets.keys())}")
        
        # STEP 2: DATA PREPROCESSING WITH LOADED DATA
        logger.info("\nSTEP 2: Preprocessing loaded data...")
        
        processed_datasets = {}
        for timeframe, df in datasets.items():
            logger.info(f"Processing {timeframe}...")
            
            # Your existing preprocessing logic goes here
            # But now you're guaranteed to have clean data with proper columns
            
            # Example preprocessing:
            processed_df = preprocess_timeframe_data(df, timeframe)
            
            if processed_df is not None and len(processed_df) > 0:
                processed_datasets[timeframe] = processed_df
                logger.info(f"✅ {timeframe}: Preprocessing complete ({len(processed_df)} rows)")
            else:
                logger.warning(f"⚠️  {timeframe}: Preprocessing failed, excluding from training")
        
        if not processed_datasets:
            logger.error("❌ CRITICAL: No data survived preprocessing")
            sys.exit(1)
        
        # STEP 3: FEATURE ENGINEERING
        logger.info(f"\nSTEP 3: Feature engineering for {len(processed_datasets)} timeframes...")
        
        featured_datasets = {}
        for timeframe, df in processed_datasets.items():
            
            # Your existing feature engineering logic
            featured_df = create_features_for_timeframe(df, timeframe)
            
            if featured_df is not None:
                featured_datasets[timeframe] = featured_df
                logger.info(f"✅ {timeframe}: Feature engineering complete")
        
        # STEP 4: HOLLOWAY FEATURES (if needed)
        logger.info(f"\nSTEP 4: Adding Holloway features...")
        
        final_datasets = {}
        for timeframe, df in featured_datasets.items():
            
            # Add Holloway features with proper error handling
            try:
                holloway_df = add_holloway_features_safe(df, timeframe)
                final_datasets[timeframe] = holloway_df
                logger.info(f"✅ {timeframe}: Holloway features added")
            except Exception as e:
                logger.warning(f"⚠️  {timeframe}: Holloway features failed ({str(e)}), using data without Holloway")
                final_datasets[timeframe] = df

        # Generate schema reports for each final timeframe dataset
        try:
            from scripts.report_utils import generate_schema_report
            for timeframe, dff in final_datasets.items():
                try:
                    generate_schema_report(dff, pair=timeframe, out_dir=os.path.join(os.getcwd(), 'output'))
                except Exception:
                    logger.debug(f"Failed to generate schema report for {timeframe}")
        except Exception:
            logger.debug("report_utils not available to generate reports")
        
        # STEP 5: MODEL TRAINING
        logger.info(f"\nSTEP 5: Training models...")
        
        # Now use the robust LightGBM training we discussed earlier
        from robust_lightgbm_config import enhanced_lightgbm_training_pipeline
        
        trained_models = enhanced_lightgbm_training_pipeline(final_datasets)
        
        if trained_models:
            logger.info(f"✅ TRAINING COMPLETE: {len(trained_models)} models successfully trained")
            
            # Your model evaluation and prediction logic here
            evaluate_models(trained_models, final_datasets)
            
        else:
            logger.error("❌ MODEL TRAINING FAILED")
            sys.exit(1)
        
        logger.info("🎉 FORECASTING SYSTEM COMPLETED SUCCESSFULLY")
        
    except Exception as e:
        logger.error(f"❌ SYSTEM FAILURE: {str(e)}")
        logger.error("Check logs above for detailed error information")
        sys.exit(1)

def preprocess_timeframe_data(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Preprocess individual timeframe data
    Replace this with your existing preprocessing logic
    """
    
    try:
        # Ensure timestamp is datetime
        if 'timestamp' not in df.columns:
            logger.error(f"❌ {timeframe}: No timestamp column found")
            return None
            
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Remove duplicates
        initial_len = len(df)
        df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        if len(df) < initial_len:
            logger.info(f"   Removed {initial_len - len(df)} duplicate rows")
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"❌ {timeframe}: Missing required columns: {missing_cols}")
            return None
        
        # Basic data cleaning
        # Remove rows where high < low (data error)
        df = df[df['high'] >= df['low']]
        
        # Remove extreme outliers
        for col in required_cols:
            Q1 = df[col].quantile(0.01)
            Q99 = df[col].quantile(0.99)
            df = df[(df[col] >= Q1) & (df[col] <= Q99)]
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"   {timeframe}: Cleaned to {len(df)} rows")
        return df
        
    except Exception as e:
        logger.error(f"❌ {timeframe}: Preprocessing failed - {str(e)}")
        return None

def create_features_for_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Create technical indicators and features
    Replace this with your existing feature creation logic
    """
    
    try:
        # Enhanced technical indicators
        # Simple & Exponential Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()

        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # RSI (robust calculation)
        def calculate_rsi(prices, window=14):
            delta = prices.diff()
            gain = delta.clip(lower=0).ewm(alpha=1/window, adjust=False).mean()
            loss = (-delta).clip(lower=0).ewm(alpha=1/window, adjust=False).mean()
            rs = gain / (loss + 1e-12)
            return 100 - (100 / (1 + rs))

        df['rsi_14'] = calculate_rsi(df['close'], window=14)

        # Bollinger Bands
        bb_window = 20
        bb_std = 2
        df['bb_middle'] = df['close'].rolling(window=bb_window).mean()
        bb_rolling_std = df['close'].rolling(window=bb_window).std()
        df['bb_upper'] = df['bb_middle'] + (bb_rolling_std * bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_rolling_std * bb_std)

        # ATR (reuse code similar to Holloway's ATR calculation)
        if set(['high', 'low', 'close']).issubset(df.columns):
            high_low = (df['high'] - df['low']).abs()
            high_close = (df['high'] - df['close'].shift(1)).abs()
            low_close = (df['low'] - df['close'].shift(1)).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['tr'] = tr
            df['atr_14'] = df['tr'].rolling(window=14).mean()

        # Returns and lagged returns
        df['return'] = df['close'].pct_change()
        df['log_return'] = np.log(df['close']).diff()
        for lag in range(1, 6):
            df[f'return_lag_{lag}'] = df['return'].shift(lag)

        # Momentum and volatility regime
        df['mom_10'] = df['close'].pct_change(periods=10)
        df['rv_20'] = df['log_return'].rolling(window=20).std() * np.sqrt(20)

        # Conservative cleanup: drop rows with NaNs in the core engineered features
        core_engineered = ['sma_20', 'sma_50', 'ema_12', 'ema_26', 'macd', 'rsi_14', 'bb_middle']
        core_present = [c for c in core_engineered if c in df.columns]
        df = df.dropna(subset=core_present).reset_index(drop=True)
        
        if len(df) == 0:
            logger.error(f"❌ {timeframe}: No data left after feature creation")
            return None
        
        logger.info(f"   {timeframe}: Created features, {len(df)} rows remaining")
        return df
        
    except Exception as e:
        logger.error(f"❌ {timeframe}: Feature creation failed - {str(e)}")
        return None

def add_holloway_features_safe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Add Holloway features with error handling
    Replace this with your actual Holloway calculation
    """
    try:
        # Work on a copy to avoid side effects
        if df is None:
            raise ValueError("Input dataframe is None")

        if len(df) < 50:  # minimum sensible window for rolling features
            logger.warning(f"⚠️  {timeframe}: Insufficient data for Holloway features ({len(df)} rows)")
            return df

        d = df.copy()

        # Ensure numeric columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in d.columns:
                d[col] = pd.to_numeric(d[col], errors='coerce')

        # Basic returns and log-returns
        d['return'] = d['close'].pct_change()
        d['log_return'] = np.log(d['close']).diff()

        # Realized volatility (rolling std of log returns scaled by sqrt(window))
        rv_window = 20
        d['rv_20'] = d['log_return'].rolling(window=rv_window).std() * np.sqrt(rv_window)

        # Longer-term realized vol for regime comparisons
        rv_long = 50
        d['rv_50'] = d['log_return'].rolling(window=rv_long).std() * np.sqrt(rv_long)

        # ATR (average true range)
        if set(['high', 'low', 'close']).issubset(d.columns):
            high_low = (d['high'] - d['low']).abs()
            high_close = (d['high'] - d['close'].shift(1)).abs()
            low_close = (d['low'] - d['close'].shift(1)).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            d['tr'] = tr
            d['atr_14'] = d['tr'].rolling(window=14).mean()

        # Rolling higher moments
        d['skew_20'] = d['return'].rolling(window=20).skew()
        d['kurt_20'] = d['return'].rolling(window=20).kurt()

        # Momentum & lagged returns
        d['mom_10'] = d['close'].pct_change(periods=10)
        for lag in range(1, 6):
            d[f'return_lag_{lag}'] = d['return'].shift(lag)

        # Volatility regime indicator
        d['rv_ratio'] = d['rv_20'] / (d['rv_50'] + 1e-12)

        # Conservative cleanup: drop rows that have NaNs from required engineered features
        engineered_cols = ['rv_20', 'rv_50', 'skew_20', 'kurt_20', 'mom_10']
        present_engineered = [c for c in engineered_cols if c in d.columns]
        if present_engineered:
            d = d.dropna(subset=present_engineered).reset_index(drop=True)

        if len(d) == 0:
            logger.warning(f"⚠️  {timeframe}: All rows dropped after Holloway feature engineering; returning original df")
            return df

        logger.info(f"   {timeframe}: Holloway features calculated ({len(d)} rows)")
        return d

    except Exception as e:
        logger.warning(f"⚠️  {timeframe}: Holloway calculation failed - {str(e)}")
        return df

def evaluate_models(models: dict, datasets: dict):
    """
    Evaluate trained models
    Replace with your existing evaluation logic
    """
    
    for timeframe, model in models.items():
        logger.info(f"📊 Evaluating {timeframe} model...")
        
        # Your model evaluation logic here
        logger.info(f"   {timeframe}: Model has {model.num_trees()} trees")

# Alternative: If you want to test data loading separately
def test_data_loading_only():
    """
    Test just the data loading functionality
    """
    
    logger.info("🧪 TESTING DATA LOADING ONLY")
    
    try:
        datasets = robust_data_loading_pipeline()
        
        print("\n📊 DATA LOADING TEST RESULTS:")
        print("=" * 50)
        
        for timeframe, df in datasets.items():
            print(f"\n✅ {timeframe}:")
            print(f"   Rows: {len(df):,}")
            print(f"   Columns: {list(df.columns)}")
            print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"   Memory usage: {df.memory_usage().sum() / 1024**2:.1f} MB")
            
            # Show sample data
            print(f"   Sample data:")
            print(df[['timestamp', 'open', 'high', 'low', 'close']].head(3).to_string(index=False))
        
        print(f"\n🎉 Data loading test successful for {len(datasets)} timeframes")
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {str(e)}")
        return False

# Emergency recovery function
def emergency_data_fix():
    """
    Emergency function to fix data issues
    Call this if your normal pipeline fails
    """
    
    logger.info("🚨 EMERGENCY DATA FIX MODE")
    
    try:
        # Create missing files with sample data
        emergency_data_recovery()
        
        # Try loading again
        return test_data_loading_only()
        
    except Exception as e:
        logger.error(f"❌ Emergency fix failed: {str(e)}")
        return False

if __name__ == "__main__":
    # You can run this to test the data loading
    if test_data_loading_only():
        print("✅ Data loading works - ready to integrate into forecasting.py")
    else:
        print("❌ Data loading failed - check the issues above")
        if input("Try emergency fix? (y/n): ").lower() == 'y':
            emergency_data_fix()
