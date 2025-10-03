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
    from data_validation import validate_data_before_training
except Exception:
    validate_data_before_training = None

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
    ok_valid = validate_data_before_training()
    if not ok_valid:
        logger.error("Data validation failed after fixes. Aborting.")
        return False

    logger.info("✅ Data fixed and validated. Ready to train.")

    if dry_run:
        logger.info("Dry run requested - not starting training")
        return True

    # Place holder for training start. Importing your existing training
    # entrypoint would happen here. For now, return True to indicate OK.
    logger.info("Starting training pipeline (placeholder) — replace with your training call")
    return True


if __name__ == '__main__':
    run(dry_run=True)
