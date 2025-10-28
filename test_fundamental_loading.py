#!/usr/bin/env python3
"""
Test script to verify fundamental data CSV loading
Checks for 'date' column presence and proper data structure
"""

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_fundamental_csv_files():
    """Test all fundamental CSV files for proper schema"""
    
    data_dir = Path('data')
    
    # List of fundamental files that should exist
    fundamental_files = [
        'INDPRO.csv',
        'DGORDER.csv', 
        'ECBDFR.csv',
        'CP0000EZ19M086NEST.csv',
        'LRHUTTTTDEM156S.csv',
        'DCOILWTICO.csv',
        'DCOILBRENTEU.csv',
        'VIXCLS.csv',
        'DGS10.csv',
        'DGS2.csv',
        'BOPGSTB.csv',
        'CPIAUCSL.csv',
        'CPALTT01USM661S.csv',
        'DFF.csv',
        'DEXCHUS.csv',
        'DEXJPUS.csv',
        'DEXUSEU.csv',
        'FEDFUNDS.csv',
        'PAYEMS.csv',
        'UNRATE.csv'
    ]
    
    results = {
        'passed': [],
        'failed': [],
        'missing': []
    }
    
    for filename in fundamental_files:
        filepath = data_dir / filename
        
        if not filepath.exists():
            results['missing'].append(filename)
            logger.warning(f"❌ {filename}: File not found")
            continue
        
        try:
            # Try to load the CSV
            df = pd.read_csv(filepath)
            
            # Check for 'date' column
            if 'date' not in df.columns:
                results['failed'].append(f"{filename}: Missing 'date' column. Found: {list(df.columns)}")
                logger.error(f"❌ {filename}: Missing 'date' column. Found columns: {list(df.columns)}")
                continue
            
            # Check if date can be parsed
            try:
                pd.to_datetime(df['date'], errors='coerce')
            except Exception as e:
                results['failed'].append(f"{filename}: Date parsing failed: {e}")
                logger.error(f"❌ {filename}: Date parsing failed: {e}")
                continue
            
            # Check if there's at least one value column
            if len(df.columns) < 2:
                results['failed'].append(f"{filename}: No value column found")
                logger.error(f"❌ {filename}: No value column found. Columns: {list(df.columns)}")
                continue
            
            # Success!
            results['passed'].append(filename)
            logger.info(f"✅ {filename}: OK ({len(df)} rows, columns: {list(df.columns)})")
            
        except Exception as e:
            results['failed'].append(f"{filename}: Error loading: {e}")
            logger.error(f"❌ {filename}: Error loading: {e}")
    
    # Print summary
    print("\n" + "="*80)
    print("FUNDAMENTAL DATA CSV VALIDATION SUMMARY")
    print("="*80)
    print(f"✅ Passed:  {len(results['passed'])} files")
    print(f"❌ Failed:  {len(results['failed'])} files")
    print(f"⚠️  Missing: {len(results['missing'])} files")
    print("="*80)
    
    if results['failed']:
        print("\nFailed files:")
        for failure in results['failed']:
            print(f"  • {failure}")
    
    if results['missing']:
        print("\nMissing files:")
        for missing in results['missing']:
            print(f"  • {missing}")
    
    print()
    
    # Return True if all checks passed
    return len(results['failed']) == 0 and len(results['missing']) == 0


def test_fundamental_pipeline_loading():
    """Test the fundamental_pipeline.py loading functions"""
    
    print("\n" + "="*80)
    print("TESTING FUNDAMENTAL_PIPELINE.PY LOADING")
    print("="*80)
    
    try:
        from scripts.fundamental_pipeline import FundamentalDataPipeline, load_all_fundamentals
        
        # Test FundamentalDataPipeline
        logger.info("Testing FundamentalDataPipeline...")
        pipeline = FundamentalDataPipeline(data_dir="data")
        
        # Test loading a few series
        test_series = ['INDPRO', 'DGS10', 'VIXCLS', 'CPIAUCSL']
        
        for series_id in test_series:
            df = pipeline.load_series_from_csv(series_id)
            if df.empty:
                logger.warning(f"⚠️  {series_id}: No data loaded")
            else:
                logger.info(f"✅ {series_id}: Loaded {len(df)} rows, columns: {list(df.columns)}")
        
        # Test load_all_fundamentals
        logger.info("\nTesting load_all_fundamentals()...")
        all_data = load_all_fundamentals(data_dir="data")
        
        if all_data.empty:
            logger.error("❌ load_all_fundamentals() returned empty DataFrame")
            return False
        else:
            logger.info(f"✅ load_all_fundamentals() loaded {len(all_data)} rows, {len(all_data.columns)} columns")
            logger.info(f"   Date range: {all_data.index.min()} to {all_data.index.max()}")
            logger.info(f"   Columns: {list(all_data.columns[:10])}{'...' if len(all_data.columns) > 10 else ''}")
            return True
            
    except Exception as e:
        logger.error(f"❌ Error testing fundamental_pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Test CSV files
    csv_test_passed = test_fundamental_csv_files()
    
    # Test pipeline loading
    pipeline_test_passed = test_fundamental_pipeline_loading()
    
    # Final result
    print("\n" + "="*80)
    if csv_test_passed and pipeline_test_passed:
        print("✅ ALL TESTS PASSED - Fundamental data loading is working correctly")
        print("="*80)
        exit(0)
    else:
        print("❌ SOME TESTS FAILED - Please review the errors above")
        print("="*80)
        exit(1)
