"""
Orchestration wrapper to run pre-training data fixes and validation
before kicking off training. This ensures training will not start on
malformed or insufficient data.

Usage:
    from enhanced_forecasting import run
    run(dry_run=True)

This module expects the following functions to be available in the
repo:
 - data_validation.validate_data_before_training
 - data_issue_fixes.pre_training_data_fix

The run() function returns True when training may proceed, False otherwise.
"""

import logging
from pathlib import Path


try:
    from data_validation import validate_data_before_training, validate_data_before_training_auto
except Exception:
    validate_data_before_training = None
    validate_data_before_training_auto = None

try:
    from data_issue_fixes import pre_training_data_fix
except Exception:
    pre_training_data_fix = None


def run(dry_run=False):
    """Run pre-training fixes and validation.

    Parameters
    - dry_run: if True, do not start training even if validation passes.

    Returns: True if training can proceed, False otherwise.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('enhanced_forecasting')

    logger.info("🔎 Running pre-training data fix and validation")

    if pre_training_data_fix is None:
        logger.error("Missing data_issue_fixes.pre_training_data_fix - cannot run fixes")
        return False

    if validate_data_before_training is None:
        logger.error("Missing data_validation.validate_data_before_training - cannot validate data")
        return False

    # Run fixers first
    ok_fix = pre_training_data_fix()
    if not ok_fix:
        logger.error("Pre-training data fixes failed. Aborting.")
        return False

    # Then validate
    ok_valid = validate_data_before_training_auto()
    if not ok_valid:
        logger.error("Data validation failed after fixes. Aborting.")
        return False

    logger.info("✅ Data fixed and validated. Ready to train.")

    if dry_run:
        logger.info("Dry run requested - not starting training")
        return True

    # --- ACTUAL TRAINING ENTRYPOINT ---
    import os
    from pathlib import Path
    # Only require FRED API key if any fundamental CSV is missing or empty
    required_fundamentals = [
        'INDPRO','DGORDER','ECBDFR','CP0000EZ19M086NEST','LRHUTTTTDEM156S','DCOILWTICO','DCOILBRENTEU','VIXCLS','DGS10','DGS2','BOPGSTB','CPIAUCSL','CPALTT01USM661S','DFF','DEXCHUS','DEXJPUS','DEXUSEU','FEDFUNDS','PAYEMS','UNRATE'
    ]
    missing_or_empty = []
    for fname in required_fundamentals:
        fpath = Path('data') / f'{fname}.csv'
        if not fpath.exists() or fpath.stat().st_size == 0:
            missing_or_empty.append(fname)
    if missing_or_empty:
        fred_key = os.environ.get('FRED_API_KEY')
        if not fred_key or fred_key.strip() == '':
            logger.error(f"❌ FRED_API_KEY is missing and fundamental data missing/empty for: {missing_or_empty}. Fundamental data cannot be loaded.")
            logger.error("To fix: Add your FRED API key to your .env file as FRED_API_KEY=your_key_here and restart your environment.")
            logger.error("If you do not have a FRED key, sign up at https://fred.stlouisfed.org/docs/api/api_key.html")
            return False
    try:
        from scripts.automated_training import AutomatedTrainer
        trainer = AutomatedTrainer(target_accuracy=0.85, max_iterations=50)
        results = trainer.run_automated_training(pairs=["EURUSD", "XAUUSD"], dry_run=False)
        logger.info(f"Training results: {results}")
        logger.info("✅ Full training completed for all pairs.")
        return True
    except Exception as e:
        if 'FATAL: Fundamental data is missing or empty' in str(e):
            logger.error("❌ Fundamental data is missing or empty during feature preparation.\n")
            logger.error("How to fix:")
            logger.error("1. Ensure your .env file contains a valid FRED_API_KEY and restart your environment.")
            logger.error("2. If you have the key, but still see this error, check that your fundamental CSVs in the data/ folder are populated with real values (not just headers or placeholders).\n")
            logger.error("3. If you need a FRED key, get one at https://fred.stlouisfed.org/docs/api/api_key.html\n")
            logger.error("4. If you want to run without fundamentals, modify your pipeline to skip fundamental features.")
        else:
            logger.error(f"❌ Training failed: {e}")
        return False


if __name__ == '__main__':
    run(dry_run=False)
