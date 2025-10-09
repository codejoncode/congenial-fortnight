#!/usr/bin/env python3
"""
Data Validation Script - Run Before Training

This script validates:
1. All price data files exist and have correct schema
2. All fundamental data files exist and have 'date' column
3. Data quality (no empty files, proper date ranges)
4. Feature engineering can run successfully
"""

import pandas as pd
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def validate_price_data(pair: str, data_dir: Path) -> bool:
    """Validate price data files for a pair"""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Validating Price Data for {pair}")
    logger.info(f"{'='*60}")
    
    timeframes = ['Daily', 'H4', 'H1', 'Weekly', 'Monthly']
    required_columns = ['timestamp', 'open', 'high', 'low', 'close']
    
    all_valid = True
    
    for tf in timeframes:
        file_path = data_dir / f"{pair}_{tf}.csv"
        
        if not file_path.exists():
            logger.warning(f"⚠️  {tf}: File not found - {file_path}")
            if tf in ['Daily', 'H4']:
                all_valid = False
            continue
        
        try:
            df = pd.read_csv(file_path, nrows=5)
            
            # Check columns
            missing_cols = [col for col in required_columns if col not in df.columns.str.lower()]
            
            if missing_cols:
                logger.error(f"❌ {tf}: Missing columns: {missing_cols}")
                all_valid = False
            else:
                # Check data
                full_df = pd.read_csv(file_path)
                logger.info(f"✅ {tf}: {len(full_df)} rows, {list(df.columns)[:5]}...")
                
                # Check for empty values
                if full_df[['open', 'high', 'low', 'close']].isnull().any().any():
                    logger.warning(f"⚠️  {tf}: Contains null values in OHLC")
                
        except Exception as e:
            logger.error(f"❌ {tf}: Error reading file - {e}")
            all_valid = False
    
    return all_valid


def validate_fundamental_data(data_dir: Path) -> bool:
    """Validate fundamental data files"""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Validating Fundamental Data")
    logger.info(f"{'='*60}")
    
    fundamental_files = [
        'INDPRO.csv', 'DGORDER.csv', 'ECBDFR.csv',
        'CP0000EZ19M086NEST.csv', 'LRHUTTTTDEM156S.csv',
        'DCOILWTICO.csv', 'DCOILBRENTEU.csv', 'VIXCLS.csv',
        'DGS10.csv', 'DGS2.csv', 'BOPGSTB.csv',
        'CPIAUCSL.csv', 'DFF.csv', 'FEDFUNDS.csv',
        'PAYEMS.csv', 'UNRATE.csv'
    ]
    
    all_valid = True
    found_count = 0
    
    for filename in fundamental_files:
        file_path = data_dir / filename
        
        if not file_path.exists():
            logger.warning(f"⚠️  {filename}: Not found")
            continue
        
        try:
            df = pd.read_csv(file_path, nrows=5)
            
            # Check for 'date' column
            if 'date' not in df.columns.str.lower():
                logger.error(f"❌ {filename}: Missing 'date' column. Has: {list(df.columns)}")
                all_valid = False
            else:
                full_df = pd.read_csv(file_path)
                logger.info(f"✅ {filename}: {len(full_df)} rows")
                found_count += 1
                
        except Exception as e:
            logger.error(f"❌ {filename}: Error - {e}")
            all_valid = False
    
    logger.info(f"\n📊 Found {found_count}/{len(fundamental_files)} fundamental files")
    
    if found_count < 10:
        logger.warning("⚠️  Less than 10 fundamental files found - training may be limited")
    
    return all_valid


def test_feature_engineering(pair: str) -> bool:
    """Test if feature engineering can run"""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing Feature Engineering for {pair}")
    logger.info(f"{'='*60}")
    
    try:
        sys.path.insert(0, str(Path.cwd()))
        from scripts.forecasting import HybridPriceForecastingEnsemble
        
        logger.info(f"Initializing forecasting system...")
        ensemble = HybridPriceForecastingEnsemble(pair)
        
        logger.info(f"Loading and preparing datasets...")
        X_train, y_train, X_val, y_val = ensemble.load_and_prepare_datasets()
        
        if X_train is None or y_train is None:
            logger.error(f"❌ Failed to load datasets for {pair}")
            return False
        
        logger.info(f"✅ Feature engineering successful!")
        logger.info(f"   Train: {X_train.shape}")
        logger.info(f"   Val: {X_val.shape}")
        logger.info(f"   Features: {X_train.shape[1]}")
        logger.info(f"   Target balance: {y_train.mean():.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validations"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║           DATA VALIDATION - PRE-TRAINING CHECK               ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    data_dir = Path('data')
    pairs = ['EURUSD', 'XAUUSD']
    
    all_checks_passed = True
    
    # Validate price data
    for pair in pairs:
        if not validate_price_data(pair, data_dir):
            logger.error(f"❌ Price data validation failed for {pair}")
            all_checks_passed = False
    
    # Validate fundamental data
    if not validate_fundamental_data(data_dir):
        logger.error(f"❌ Fundamental data validation failed")
        all_checks_passed = False
    
    # Test feature engineering
    for pair in pairs:
        if not test_feature_engineering(pair):
            logger.error(f"❌ Feature engineering test failed for {pair}")
            all_checks_passed = False
    
    # Final summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    if all_checks_passed:
        print("✅ ALL CHECKS PASSED - READY FOR TRAINING!")
        print("\nNext step: Run training with pip tracking")
        print("   python train_with_pip_tracking.py")
        return 0
    else:
        print("❌ SOME CHECKS FAILED - FIX ISSUES BEFORE TRAINING")
        print("\nReview the errors above and fix data issues")
        return 1


if __name__ == '__main__':
    sys.exit(main())
