#!/usr/bin/env python3
"""
Integrate Fundamental Data with Technical Features
Combines technical features with comprehensive fundamental data for model training
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the fundamental pipeline
from scripts.fundamental_pipeline import load_all_fundamentals

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_technical_features(pair: str) -> pd.DataFrame:
    """Load technical features from original data files"""
    data_dir = Path('data')
    
    # Try different file patterns - prefer complete historical data
    possible_files = [
        data_dir / f"{pair}_Daily.csv.orig",
        data_dir / f"{pair}_Daily.csv",
        data_dir / f"{pair}_daily_complete_holloway.csv.orig",
        data_dir / f"{pair}_daily_complete_holloway.csv",
        data_dir / f"{pair}_latest_multi_timeframe_features.csv.orig",
        data_dir / f"{pair}_latest_multi_timeframe_features.csv",
    ]
    
    for filepath in possible_files:
        if filepath.exists():
            logger.info(f"Loading technical features from: {filepath}")
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)[:10]}...")
            
            # Identify the time column - prioritize 'timestamp' over 'time' since 'time' is often just HH:MM:SS
            time_col = None
            for col in ['timestamp', 'date', 'Timestamp', 'Date', 'time', 'Time']:
                if col in df.columns:
                    # Check if this column actually contains dates (not just times)
                    sample_val = str(df[col].iloc[0])
                    if '-' in sample_val or '/' in sample_val or len(sample_val) > 8:
                        time_col = col
                        break
            
            if time_col:
                df['timestamp'] = pd.to_datetime(df[time_col])
                df = df.set_index('timestamp').sort_index()
                logger.info(f"Set index to timestamp from '{time_col}' column")
                logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
                return df
            else:
                logger.error(f"No time column found in {filepath}")
                logger.error(f"Available columns: {list(df.columns)}")
    
    raise FileNotFoundError(f"No technical features file found for {pair}")


def merge_fundamentals_with_technical(technical_df: pd.DataFrame, fundamentals_df: pd.DataFrame) -> pd.DataFrame:
    """Merge fundamental data with technical features using forward-fill"""
    logger.info("Merging fundamentals with technical features...")
    
    # Ensure both have datetime index
    if not isinstance(technical_df.index, pd.DatetimeIndex):
        technical_df.index = pd.to_datetime(technical_df.index)
    if not isinstance(fundamentals_df.index, pd.DatetimeIndex):
        fundamentals_df.index = pd.to_datetime(fundamentals_df.index)
    
    logger.info(f"Technical data: {len(technical_df)} rows, {technical_df.shape[1]} columns")
    logger.info(f"Fundamental data: {len(fundamentals_df)} rows, {fundamentals_df.shape[1]} columns")
    logger.info(f"Technical date range: {technical_df.index.min()} to {technical_df.index.max()}")
    logger.info(f"Fundamental date range: {fundamentals_df.index.min()} to {fundamentals_df.index.max()}")
    
    # Merge using outer join and forward-fill fundamentals
    merged_df = technical_df.join(fundamentals_df, how='left')
    
    # Forward-fill fundamental data (since it's lower frequency)
    fundamental_cols = fundamentals_df.columns
    merged_df[fundamental_cols] = merged_df[fundamental_cols].ffill()
    
    # Drop rows with missing fundamentals at the start
    initial_rows = len(merged_df)
    merged_df = merged_df.dropna(subset=fundamental_cols, how='all')
    dropped_rows = initial_rows - len(merged_df)
    
    logger.info(f"Merged result: {len(merged_df)} rows, {merged_df.shape[1]} columns")
    logger.info(f"Dropped {dropped_rows} rows with missing fundamental data")
    logger.info(f"Added fundamental features: {list(fundamental_cols)}")
    
    return merged_df


def integrate_pair(pair: str):
    """Integrate fundamentals for a specific pair"""
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing {pair}")
    logger.info(f"{'='*80}")
    
    try:
        # Load technical features
        technical_df = load_technical_features(pair)
        
        # Load fundamentals
        fundamentals_df = load_all_fundamentals()
        
        # Merge
        merged_df = merge_fundamentals_with_technical(technical_df, fundamentals_df)
        
        # Save to new file
        output_file = Path('data') / f"{pair}_with_fundamentals.csv"
        merged_df.to_csv(output_file)
        logger.info(f"✅ Saved integrated data to: {output_file}")
        logger.info(f"   Total rows: {len(merged_df)}")
        logger.info(f"   Total features: {merged_df.shape[1]}")
        logger.info(f"   Technical features: {technical_df.shape[1]}")
        logger.info(f"   Fundamental features: {fundamentals_df.shape[1]}")
        
        # Display sample
        logger.info("\nSample of fundamental features:")
        fundamental_cols = fundamentals_df.columns[:5]
        logger.info(merged_df[fundamental_cols].tail())
        
        return merged_df
        
    except Exception as e:
        logger.error(f"❌ Error processing {pair}: {e}", exc_info=True)
        return None


def main():
    """Main integration function"""
    logger.info("="*80)
    logger.info("FUNDAMENTAL DATA INTEGRATION")
    logger.info("="*80)
    
    pairs = ['EURUSD', 'XAUUSD']
    results = {}
    
    for pair in pairs:
        result = integrate_pair(pair)
        if result is not None:
            results[pair] = result
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("INTEGRATION SUMMARY")
    logger.info("="*80)
    
    for pair, df in results.items():
        logger.info(f"{pair}:")
        logger.info(f"  ✅ {len(df)} rows")
        logger.info(f"  ✅ {df.shape[1]} total features")
        logger.info(f"  ✅ Data from {df.index.min()} to {df.index.max()}")
    
    logger.info("\nNext step: Retrain models with:")
    logger.info("  python scripts/train_with_fundamentals.py")
    
    return results


if __name__ == "__main__":
    main()
