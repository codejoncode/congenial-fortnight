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
