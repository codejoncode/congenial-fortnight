#!/usr/bin/env python3
"""
Fix Fundamental Data Schema
Converts fundamental CSVs from price schema back to proper fundamental schema
"""
import os
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_fundamental_data_schema():
    """Fix fundamental data files that have wrong price schema"""

    # Fundamental files that need schema conversion
    fundamental_files = [
        'INDPRO.csv',       # Industrial Production
        'DGORDER.csv',      # Durable Goods Orders
        'ECBDFR.csv',       # ECB Deposit Facility Rate
        'CP0000EZ19M086NEST.csv',  # Euro Area CPI
        'LRHUTTTTDEM156S.csv',     # Germany Unemployment Rate
        'DCOILWTICO.csv',   # WTI Oil Price
        'DCOILBRENTEU.csv', # Brent Oil Price
        'VIXCLS.csv',       # VIX Volatility Index
        'DGS10.csv',        # 10-Year Treasury Rate
        'DGS2.csv',         # 2-Year Treasury Rate
        'BOPGSTB.csv',      # Balance of Payments
        'CPIAUCSL.csv',     # US CPI
        'CPALTT01USM661S.csv',  # OECD CPI
        'DFF.csv',          # Federal Funds Rate
        'DEXCHUS.csv',      # USD/CHF Exchange Rate
        'DEXJPUS.csv',      # USD/JPY Exchange Rate
        'DEXUSEU.csv',      # USD/EUR Exchange Rate
        'FEDFUNDS.csv',     # Federal Funds Rate
        'PAYEMS.csv',       # Nonfarm Payrolls
        'UNRATE.csv'        # Unemployment Rate
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
            logger.info(f"📄 {filename}: Current shape: {df.shape}, columns: {list(df.columns)}")

            # Check if this file has the wrong price schema
            price_schema_columns = ['id', 'timestamp', 'time', 'open', 'high', 'low', 'close', 'volume', 'spread']
            has_price_schema = all(col in df.columns for col in price_schema_columns)

            if not has_price_schema:
                logger.info(f"✅ {filename}: Already has proper schema, skipping")
                continue

            # Backup the file
            backup_path = filepath.with_suffix('.csv.price_schema_backup')
            if not backup_path.exists():
                df.to_csv(backup_path, index=False)
                logger.info(f"💾 {filename}: Backup saved as {backup_path.name}")

            # Create proper fundamental data structure
            # Use timestamp as date and assume we need to add actual fundamental values
            new_df = pd.DataFrame()

            # Convert timestamp to proper date column
            new_df['date'] = pd.to_datetime(df['timestamp'], errors='coerce')

            # Create value column name based on filename
            value_col_name = filename.replace('.csv', '').lower()

            # For now, create a placeholder value column (you'll need to populate with actual data)
            # This ensures the schema is correct for the fundamental pipeline
            new_df[value_col_name] = None  # or 0.0 for numeric values

            # Save the corrected file
            new_df.to_csv(filepath, index=False)
            logger.info(f"✅ {filename}: Fixed schema - columns: {list(new_df.columns)}")
            fixed_count += 1

        except Exception as e:
            logger.error(f"❌ {filename}: Error fixing schema - {e}")

    logger.info(f"\n🎉 Fixed schema for {fixed_count} fundamental files")
    logger.info("\n⚠️  Note: Value columns are placeholder - you need to populate with actual fundamental data")
    return fixed_count

if __name__ == "__main__":
    fix_fundamental_data_schema()