#!/usr/bin/env python3
"""
Restore Fundamental Data from Backups
Restores fundamental CSVs from .orig backups if available
"""
import os
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def restore_fundamental_from_backups():
    """Restore fundamental data from .orig backup files"""

    data_dir = Path('data')
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

    restored_count = 0

    for filename in fundamental_files:
        filepath = data_dir / filename
        orig_backup = data_dir / f"{filename}.orig"

        if orig_backup.exists():
            try:
                # Read the original backup
                orig_df = pd.read_csv(orig_backup)
                logger.info(f"📄 {filename}: Found .orig backup with columns: {list(orig_df.columns)}")

                # Ensure proper schema for fundamental data
                new_df = pd.DataFrame()

                # Find the date column (could be DATE, date, or first column)
                date_col = None
                for col in orig_df.columns:
                    if col.lower() in ['date', 'datetime', 'time', 'timestamp'] or col == orig_df.columns[0]:
                        date_col = col
                        break

                if date_col:
                    new_df['date'] = pd.to_datetime(orig_df[date_col], errors='coerce')

                # Find the value column (usually second column)
                value_col = None
                if len(orig_df.columns) >= 2:
                    # Use second column as value
                    value_col = orig_df.columns[1]
                    value_name = filename.replace('.csv', '').lower()
                    new_df[value_name] = pd.to_numeric(orig_df[value_col], errors='coerce')

                # Save the restored file
                new_df.to_csv(filepath, index=False)
                logger.info(f"✅ {filename}: Restored from backup - shape: {new_df.shape}")
                restored_count += 1

            except Exception as e:
                logger.error(f"❌ {filename}: Error restoring from backup - {e}")
        else:
            logger.info(f"⏭️  {filename}: No .orig backup found")

    logger.info(f"\n🎉 Restored {restored_count} fundamental files from backups")
    return restored_count

if __name__ == "__main__":
    restore_fundamental_from_backups()