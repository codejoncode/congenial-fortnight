#!/usr/bin/env python3
"""
Comprehensive Fundamental Data Validation Script

Validates all fundamental economic data CSV files to ensure:
1. Files exist and are readable
2. Correct schema (date column + value column)
3. No missing or corrupt data
4. Proper date formatting
5. Can be loaded by FundamentalDataPipeline

Usage:
    python validate_fundamental_data.py
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_fundamental_data(data_dir: str = "data") -> dict:
    """
    Validate all fundamental data CSV files.
    
    Returns:
        dict with validation results
    """
    
    # All fundamental data files that should exist
    fundamental_files = [
        'INDPRO.csv',           # Industrial Production
        'DGORDER.csv',          # Durable Goods Orders
        'ECBDFR.csv',           # ECB Deposit Facility Rate
        'CP0000EZ19M086NEST.csv',  # Euro Area CPI
        'LRHUTTTTDEM156S.csv',     # Germany Unemployment Rate
        'DCOILWTICO.csv',       # WTI Oil Price
        'DCOILBRENTEU.csv',     # Brent Oil Price
        'VIXCLS.csv',           # VIX Volatility Index
        'DGS10.csv',            # 10-Year Treasury Rate
        'DGS2.csv',             # 2-Year Treasury Rate
        'BOPGSTB.csv',          # Balance of Payments
        'CPIAUCSL.csv',         # US CPI
        'CPALTT01USM661S.csv',  # OECD CPI
        'DFF.csv',              # Federal Funds Rate (Daily)
        'DEXCHUS.csv',          # USD/CHF Exchange Rate
        'DEXJPUS.csv',          # USD/JPY Exchange Rate
        'DEXUSEU.csv',          # USD/EUR Exchange Rate
        'FEDFUNDS.csv',         # Federal Funds Rate (Monthly)
        'PAYEMS.csv',           # Nonfarm Payrolls
        'UNRATE.csv'            # Unemployment Rate
    ]
    
    data_path = Path(data_dir)
    results = {
        'validated': [],
        'missing': [],
        'errors': [],
        'warnings': []
    }
    
    logger.info(f"🔍 Validating {len(fundamental_files)} fundamental data files...\n")
    
    for filename in fundamental_files:
        filepath = data_path / filename
        
        # Check 1: File exists
        if not filepath.exists():
            results['missing'].append(filename)
            logger.warning(f"❌ {filename}: File not found")
            continue
        
        # Check 2: File not empty
        if filepath.stat().st_size == 0:
            results['errors'].append(f"{filename}: Empty file")
            logger.error(f"❌ {filename}: File is empty")
            continue
        
        try:
            # Check 3: Can be read as CSV
            df = pd.read_csv(filepath)
            
            # Check 4: Has 'date' column
            if 'date' not in df.columns:
                results['errors'].append(f"{filename}: Missing 'date' column (has: {list(df.columns)})")
                logger.error(f"❌ {filename}: Missing 'date' column. Found: {list(df.columns)}")
                continue
            
            # Check 5: Has at least 2 columns (date + value)
            if len(df.columns) < 2:
                results['errors'].append(f"{filename}: Only has {len(df.columns)} column(s)")
                logger.error(f"❌ {filename}: Expected at least 2 columns, found {len(df.columns)}")
                continue
            
            # Check 6: Date column can be parsed
            try:
                df['date'] = pd.to_datetime(df['date'], errors='raise')
            except Exception as e:
                results['errors'].append(f"{filename}: Date parsing failed - {str(e)}")
                logger.error(f"❌ {filename}: Cannot parse dates - {e}")
                continue
            
            # Check 7: Has data rows
            if len(df) == 0:
                results['warnings'].append(f"{filename}: No data rows")
                logger.warning(f"⚠️  {filename}: File has headers but no data")
                continue
            
            # Check 8: Value column exists and has data
            value_col = df.columns[1]  # Second column should be the value
            non_null_values = df[value_col].notna().sum()
            null_pct = (df[value_col].isna().sum() / len(df)) * 100
            
            if non_null_values == 0:
                results['errors'].append(f"{filename}: All values are null")
                logger.error(f"❌ {filename}: All values in '{value_col}' are null")
                continue
            
            # All checks passed!
            results['validated'].append(filename)
            
            # Log detailed info
            date_range = f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
            logger.info(f"✅ {filename}:")
            logger.info(f"   📊 {len(df)} rows, {len(df.columns)} columns")
            logger.info(f"   📅 Date range: {date_range}")
            logger.info(f"   📈 Value column: '{value_col}' ({non_null_values} non-null, {null_pct:.1f}% null)")
            
            if null_pct > 10:
                results['warnings'].append(f"{filename}: {null_pct:.1f}% null values")
                logger.warning(f"   ⚠️  High null percentage: {null_pct:.1f}%")
            
        except Exception as e:
            results['errors'].append(f"{filename}: {str(e)}")
            logger.error(f"❌ {filename}: Validation error - {e}")
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("📊 VALIDATION SUMMARY")
    logger.info("="*70)
    logger.info(f"✅ Validated:  {len(results['validated'])}/{len(fundamental_files)} files")
    logger.info(f"❌ Missing:    {len(results['missing'])} files")
    logger.info(f"❌ Errors:     {len(results['errors'])} files")
    logger.info(f"⚠️  Warnings:   {len(results['warnings'])} issues")
    
    if results['missing']:
        logger.info(f"\nMissing files: {', '.join(results['missing'])}")
    
    if results['errors']:
        logger.info("\n❌ Errors:")
        for error in results['errors']:
            logger.info(f"   - {error}")
    
    if results['warnings']:
        logger.info("\n⚠️  Warnings:")
        for warning in results['warnings']:
            logger.info(f"   - {warning}")
    
    # Final verdict
    logger.info("\n" + "="*70)
    if len(results['validated']) == len(fundamental_files):
        logger.info("🎉 ALL FUNDAMENTAL DATA FILES VALIDATED SUCCESSFULLY!")
        logger.info("✅ Ready for fundamental pipeline processing")
        return_code = 0
    elif len(results['errors']) > 0:
        logger.error("❌ VALIDATION FAILED - Critical errors found")
        logger.error("⛔ Cannot proceed with fundamental pipeline until errors are fixed")
        return_code = 1
    else:
        logger.warning("⚠️  VALIDATION COMPLETED WITH WARNINGS")
        logger.warning("📝 Review warnings but pipeline can proceed")
        return_code = 0
    
    logger.info("="*70)
    
    return results, return_code


def test_pipeline_loading():
    """Test that FundamentalDataPipeline can load the data"""
    logger.info("\n" + "="*70)
    logger.info("🔧 TESTING FUNDAMENTAL PIPELINE DATA LOADING")
    logger.info("="*70)
    
    try:
        sys.path.insert(0, '/workspaces/congenial-fortnight')
        from scripts.fundamental_pipeline import FundamentalDataPipeline
        
        pipeline = FundamentalDataPipeline(data_dir='data')
        
        # Test loading a few key series
        test_series = ['DGS10', 'VIXCLS', 'BOPGSTB', 'INDPRO']
        
        for series_id in test_series:
            df = pipeline.load_series_from_csv(series_id)
            if not df.empty:
                logger.info(f"✅ Pipeline loaded {series_id}: {len(df)} rows")
            else:
                logger.error(f"❌ Pipeline failed to load {series_id}")
                return False
        
        logger.info("\n✅ FundamentalDataPipeline can successfully load all test series")
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run validation
    results, return_code = validate_fundamental_data()
    
    # Test pipeline loading
    if return_code == 0:
        pipeline_ok = test_pipeline_loading()
        if not pipeline_ok:
            return_code = 1
    
    # Exit with appropriate code
    sys.exit(return_code)
