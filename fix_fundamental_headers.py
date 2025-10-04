#!/usr/bin/env python3
"""
Fix Fundamental Data CSV Headers
Standardizes FRED economic data CSV files to have proper 'date' column
"""
import os
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_fundamental_csv_headers():
    """Fix fundamental data CSV files to have proper date columns"""

    # FRED fundamental data files that need fixing
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

    data_dir = Path('data')
    fixed_count = 0

    for filename in fundamental_files:
        filepath = data_dir / filename

        if not filepath.exists():
            logger.info(f"⏭️  {filename}: File not found, skipping")
            continue

        try:
            # Read the current file
            df = pd.read_csv(filepath)
            original_columns = list(df.columns)
            logger.info(f"📄 {filename}: Original columns: {original_columns}")

            # Check if it already has a proper 'date' column
            if 'date' in df.columns:
                logger.info(f"✅ {filename}: Already has 'date' column")
                continue

            # Common FRED CSV patterns:
            # 1. First column is usually the date (might be named DATE, Date, or unnamed)
            # 2. Second column is usually the value

            # Backup original file
            backup_path = filepath.with_suffix('.csv.backup')
            if not backup_path.exists():
                df.to_csv(backup_path, index=False)
                logger.info(f"💾 {filename}: Backup saved as {backup_path.name}")

            # Fix the headers
            new_columns = list(df.columns)

            # If first column looks like a date column, rename it
            first_col = new_columns[0]
            if first_col.lower() in ['date', 'datetime', 'time', 'timestamp'] or first_col == '' or 'Unnamed' in first_col:
                new_columns[0] = 'date'
                logger.info(f"🔧 {filename}: Renamed '{first_col}' to 'date'")

            # If second column is the value, give it a meaningful name based on filename
            if len(new_columns) >= 2:
                second_col = new_columns[1]
                if second_col == '' or 'Unnamed' in second_col or second_col.lower() == 'value':
                    # Use the filename (without extension) as the value column name
                    value_name = filename.replace('.csv', '').lower()
                    new_columns[1] = value_name
                    logger.info(f"🔧 {filename}: Renamed '{second_col}' to '{value_name}'")

            # Apply new column names
            df.columns = new_columns

            # Ensure date column is properly formatted
            if 'date' in df.columns:
                try:
                    df['date'] = pd.to_datetime(df['date'])
                    logger.info(f"📅 {filename}: Date column converted to datetime")
                except:
                    logger.warning(f"⚠️  {filename}: Could not convert date column to datetime")

            # Save the fixed file
            df.to_csv(filepath, index=False)
            logger.info(f"✅ {filename}: Fixed and saved with columns: {list(df.columns)}")
            fixed_count += 1

        except Exception as e:
            logger.error(f"❌ {filename}: Error fixing file - {e}")

    logger.info(f"\n🎉 Fixed {fixed_count} fundamental data files")
    return fixed_count

if __name__ == "__main__":
    fix_fundamental_csv_headers()