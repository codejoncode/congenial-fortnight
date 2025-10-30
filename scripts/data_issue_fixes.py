import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def _count_rows_in_csv(path):
    try:
        df = pd.read_csv(path)
        return len(df)
    except Exception:
        return 0


def pre_training_data_fix(data_dir='data', min_rows=50):
    """Scan CSV files in data_dir for minimal row counts and basic sanity.

    Returns True if all checks pass, False otherwise.
    """
    missing = []
    too_small = []

    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith('.csv'):
                path = os.path.join(root, f)
                rows = _count_rows_in_csv(path)
                if rows == 0:
                    missing.append(path)
                elif rows < min_rows:
                    too_small.append((path, rows))

    if missing or too_small:
        if missing:
            logger.error(f"Missing or unreadable CSVs: {missing}")
        if too_small:
            logger.warning("Files with too few rows detected:")
            for p, r in too_small:
                logger.warning(f"  - {p}: {r} rows (min {min_rows})")
        return False

    logger.info("pre_training_data_fix: all CSV files passed basic checks")
    return True
import pandas as pd
import logging
import os

def pre_training_data_fix(data_dir='data', min_rows=100):
    """
    Validates that essential CSV files in the data directory have enough data.
    
    Args:
        data_dir (str): The directory containing the CSV files.
        min_rows (int): The minimum number of rows required for a file to be valid.
        
    Returns:
        bool: True if all essential files are valid, False otherwise.
    """
    logger = logging.getLogger('forecasting')
    logger.info("--- Running Pre-Training Data Fix/Validation ---")
    
    # List of essential files for training. Add to this list as needed.
    essential_files = [
        'EURUSD_H1.csv',
        'XAUUSD_H1.csv',
        'EURUSD_Monthly.csv',
        'XAUUSD_Monthly.csv',
        # Add other critical data files here
    ]
    
    all_valid = True
    
    for filename in essential_files:
        file_path = os.path.join(data_dir, filename)
        
        if not os.path.exists(file_path):
            logger.error(f"❌ CRITICAL: Essential data file not found: {file_path}")
            all_valid = False
            continue
            
        try:
            df = pd.read_csv(file_path)
            if len(df) < min_rows:
                logger.error(f"❌ CRITICAL: Insufficient data in {filename}. Found {len(df)} rows, but require at least {min_rows}.")
                all_valid = False
            else:
                logger.info(f"✅ Data file {filename} is valid with {len(df)} rows.")
        except Exception as e:
            logger.error(f"❌ CRITICAL: Could not read or process {filename}. Error: {e}")
            all_valid = False
            
    if all_valid:
        logger.info("✅ All essential data files passed validation.")
    else:
        logger.error("❌ One or more essential data files failed validation. Training cannot proceed.")
        
    return all_valid
