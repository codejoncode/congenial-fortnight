#!/usr/bin/env python3
"""
Fix CSV column names for data loading
Adds 'date' column to price files while keeping 'timestamp'
"""
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_price_csv_columns():
    """Add 'date' column to price CSV files"""
    
    data_dir = Path('data')
    
    # Price files that need fixing
    price_files = [
        'EURUSD_H1.csv',
        'EURUSD_H4.csv',
        'EURUSD_Daily.csv',
        'EURUSD_Weekly.csv',
        'EURUSD_Monthly.csv',
        'XAUUSD_H1.csv',
        'XAUUSD_H4.csv',
        'XAUUSD_Daily.csv',
        'XAUUSD_Weekly.csv',
        'XAUUSD_Monthly.csv',
    ]
    
    fixed_count = 0
    
    for filename in price_files:
        filepath = data_dir / filename
        
        if not filepath.exists():
            logger.warning(f"⏭️  {filename}: File not found, skipping")
            continue
            
        try:
            # Read the CSV
            df = pd.read_csv(filepath)
            logger.info(f"📄 {filename}: Current columns: {list(df.columns)}")
            
            # Check if 'date' column already exists
            if 'date' in df.columns:
                logger.info(f"✅ {filename}: Already has 'date' column")
                continue
            
            # Check if 'timestamp' column exists
            if 'timestamp' not in df.columns:
                logger.warning(f"⚠️  {filename}: No 'timestamp' column found")
                continue
            
            # Create backup
            backup_path = filepath.with_suffix('.csv.before_date_fix')
            if not backup_path.exists():
                df.to_csv(backup_path, index=False)
                logger.info(f"💾 {filename}: Backup saved")
            
            # Add 'date' column as copy of 'timestamp'
            df['date'] = df['timestamp']
            
            # Reorder columns to put 'date' right after 'timestamp'
            cols = list(df.columns)
            # Remove 'date' from its current position
            cols.remove('date')
            # Find timestamp position
            timestamp_idx = cols.index('timestamp')
            # Insert 'date' right after 'timestamp'
            cols.insert(timestamp_idx + 1, 'date')
            df = df[cols]
            
            # Save the fixed file
            df.to_csv(filepath, index=False)
            logger.info(f"✅ {filename}: Added 'date' column - new columns: {list(df.columns)}")
            fixed_count += 1
            
        except Exception as e:
            logger.error(f"❌ {filename}: Error fixing file - {e}")
    
    logger.info(f"\n🎉 Fixed {fixed_count} price data files")
    return fixed_count

if __name__ == "__main__":
    print("=" * 70)
    print("FIXING DATA CSV COLUMNS")
    print("=" * 70)
    print("\nAdding 'date' column to price CSV files...")
    print("This will keep 'timestamp' and add 'date' as a copy.\n")
    
    fixed = fix_price_csv_columns()
    
    print("\n" + "=" * 70)
    print(f"✅ COMPLETE: Fixed {fixed} files")
    print("=" * 70)
    print("\nYou can now restart the Django server:")
    print("  python manage.py runserver")
