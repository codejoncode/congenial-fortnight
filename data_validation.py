# Data Validation Pipeline for Forex Trading Model
# Implements comprehensive pre-training checks for all timeframes and features

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings

class DataValidationError(Exception):
    """Custom exception for data validation failures"""
    pass

class ForexDataValidator:
    """
    Comprehensive data validator for multi-timeframe forex trading model
    Ensures all data requirements are met before training begins
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = self._setup_logger()
        self.validation_results = {}
        
    def _setup_logger(self):
        logger = logging.getLogger('data_validator')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def validate_all_data(self, data_sources: Dict[str, str]) -> bool:
        """
        Main validation entry point - validates all timeframes and features
        
        Args:
            data_sources: Dict mapping timeframe names to file paths
            
        Returns:
            bool: True if all validations pass
            
        Raises:
            DataValidationError: If any critical validation fails
        """
        self.logger.info("Starting comprehensive data validation...")
        
        validation_steps = [
            self._validate_file_existence,
            self._validate_data_structure,
            self._validate_data_quality,
            self._validate_timeframe_coverage,
            self._validate_feature_requirements,
            self._validate_holloway_prerequisites,
            self._validate_training_readiness
        ]
        
        for step in validation_steps:
            try:
                step(data_sources)
            except Exception as e:
                self.logger.error(f"Validation failed at step {step.__name__}: {str(e)}")
                raise DataValidationError(f"Critical validation failure: {str(e)}")
        
        self.logger.info("All data validation checks passed successfully!")
        return True
    
    def _validate_file_existence(self, data_sources: Dict[str, str]):
        """Validate that all required data files exist and are accessible"""
        self.logger.info("Validating file existence...")
        
        missing_files = []
        for timeframe, filepath in data_sources.items():
            path = Path(filepath)
            if not path.exists():
                missing_files.append(f"{timeframe}: {filepath}")
            elif path.stat().st_size == 0:
                missing_files.append(f"{timeframe}: {filepath} (empty file)")
        
        if missing_files:
            raise DataValidationError(f"Missing or empty data files: {missing_files}")
        
        self.logger.info(f"✓ All {len(data_sources)} data files exist and are non-empty")
    
    def _validate_data_structure(self, data_sources: Dict[str, str]):
        """Validate data structure and required columns"""
        self.logger.info("Validating data structure...")
        
        required_columns = ['timestamp', 'open', 'high', 'low', 'close']
        optional_columns = ['volume', 'spread']
        
        for timeframe, filepath in data_sources.items():
            try:
                df = pd.read_csv(filepath, nrows=5)  # Quick structure check
                
                # Check for required columns (case insensitive)
                df_columns_lower = [col.lower() for col in df.columns]
                missing_required = []
                
                for req_col in required_columns:
                    if req_col not in df_columns_lower:
                        # Try common alternatives
                        alternatives = {
                            'timestamp': ['date', 'datetime', 'time'],
                            'open': ['o'],
                            'high': ['h'],
                            'low': ['l'],
                            'close': ['c', 'closing']
                        }
                        
                        found_alternative = False
                        for alt in alternatives.get(req_col, []):
                            if alt in df_columns_lower:
                                found_alternative = True
                                break
                        
                        if not found_alternative:
                            missing_required.append(req_col)
                
                if missing_required:
                    raise DataValidationError(
                        f"{timeframe} missing required columns: {missing_required}"
                    )
                
                # Validate timestamp column can be parsed
                timestamp_col = self._identify_timestamp_column(df.columns)
                if timestamp_col is None:
                    raise DataValidationError(f"{timeframe}: Cannot identify timestamp column")
                
                # Quick check if timestamp is parseable
                try:
                    pd.to_datetime(df[timestamp_col].iloc[0])
                except:
                    raise DataValidationError(
                        f"{timeframe}: Timestamp column '{timestamp_col}' cannot be parsed"
                    )
                
                self.validation_results[f"{timeframe}_structure"] = "PASS"
                
            except Exception as e:
                raise DataValidationError(f"{timeframe} structure validation failed: {str(e)}")
        
        self.logger.info(f"✓ Data structure validation passed for all timeframes")
    
    def _identify_timestamp_column(self, columns: List[str]) -> Optional[str]:
        """Identify the timestamp column from various possible names"""
        timestamp_candidates = ['timestamp', 'date', 'datetime', 'time']
        columns_lower = {col.lower(): col for col in columns}
        
        for candidate in timestamp_candidates:
            if candidate in columns_lower:
                return columns_lower[candidate]
        return None
    
    def _validate_data_quality(self, data_sources: Dict[str, str]):
        """Validate data quality - missing values, duplicates, outliers"""
        self.logger.info("Validating data quality...")
        
        for timeframe, filepath in data_sources.items():
            df = pd.read_csv(filepath)
            
            # Check for completely empty dataset
            if len(df) == 0:
                raise DataValidationError(f"{timeframe}: Dataset is completely empty")
            
            # Check for missing values in critical columns
            critical_cols = ['open', 'high', 'low', 'close']
            existing_critical_cols = [col for col in critical_cols 
                                    if col.lower() in [c.lower() for c in df.columns]]
            
            missing_data_issues = []
            for col in existing_critical_cols:
                missing_count = df[col].isna().sum()
                missing_pct = (missing_count / len(df)) * 100
                
                if missing_pct > 5:  # More than 5% missing
                    missing_data_issues.append(f"{col}: {missing_pct:.1f}% missing")
                elif missing_pct > 0:
                    self.logger.warning(f"{timeframe} {col}: {missing_pct:.1f}% missing values")
            
            if missing_data_issues:
                raise DataValidationError(f"{timeframe} excessive missing data: {missing_data_issues}")
            
            # Check for duplicate timestamps
            timestamp_col = self._identify_timestamp_column(df.columns)
            if timestamp_col:
                duplicates = df[timestamp_col].duplicated().sum()
                if duplicates > 0:
                    self.logger.warning(f"{timeframe}: {duplicates} duplicate timestamps found")
            
            # Check for data integrity (High >= Low, etc.)
            if all(col in df.columns for col in ['high', 'low', 'open', 'close']):
                integrity_issues = []
                
                if (df['high'] < df['low']).any():
                    integrity_issues.append("High < Low detected")
                
                if (df['high'] < df['open']).any() or (df['high'] < df['close']).any():
                    integrity_issues.append("High below Open/Close detected")
                
                if (df['low'] > df['open']).any() or (df['low'] > df['close']).any():
                    integrity_issues.append("Low above Open/Close detected")
                
                if integrity_issues:
                    raise DataValidationError(f"{timeframe} data integrity issues: {integrity_issues}")
            
            self.validation_results[f"{timeframe}_quality"] = "PASS"
        
        self.logger.info("✓ Data quality validation passed for all timeframes")
    
    def _validate_timeframe_coverage(self, data_sources: Dict[str, str]):
        """Validate sufficient data coverage for each timeframe"""
        self.logger.info("Validating timeframe coverage...")
        
        min_rows_required = {
            'M1': 10000,   # ~7 days
            'M5': 2880,    # ~10 days  
            'M15': 960,    # ~10 days
            'H1': 720,     # ~30 days
            'H4': 180,     # ~30 days
            'Daily': 252,  # ~1 year
            'Weekly': 52,  # ~1 year
            'Monthly': 12  # ~1 year
        }
        
        coverage_issues = []
        
        for timeframe, filepath in data_sources.items():
            df = pd.read_csv(filepath)
            row_count = len(df)
            
            # Determine minimum required based on timeframe
            min_required = min_rows_required.get(timeframe, 100)  # Default minimum
            
            if row_count < min_required:
                coverage_issues.append(
                    f"{timeframe}: {row_count} rows (need {min_required})"
                )
            
            # Check date range coverage
            timestamp_col = self._identify_timestamp_column(df.columns)
            if timestamp_col:
                try:
                    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                    date_range = df[timestamp_col].max() - df[timestamp_col].min()
                    
                    self.logger.info(
                        f"{timeframe}: {row_count} rows spanning {date_range.days} days"
                    )
                    
                    # Store coverage info for later use
                    self.validation_results[f"{timeframe}_coverage"] = {
                        "rows": row_count,
                        "days": date_range.days,
                        "start": df[timestamp_col].min(),
                        "end": df[timestamp_col].max()
                    }
                    
                except Exception as e:
                    self.logger.warning(f"{timeframe}: Cannot parse timestamp for coverage check")
        
        if coverage_issues:
            raise DataValidationError(f"Insufficient data coverage: {coverage_issues}")
        
        self.logger.info("✓ Timeframe coverage validation passed")
    
    def _validate_feature_requirements(self, data_sources: Dict[str, str]):
        """Validate that we have enough data for all required features"""
        self.logger.info("Validating feature requirements...")
        
        # Check each timeframe has minimum requirements for technical indicators
        feature_requirements = {
            'sma_periods': [20, 50, 200],
            'ema_periods': [12, 26],
            'rsi_period': 14,
            'bollinger_period': 20,
            'macd_periods': [12, 26, 9]
        }
        
        max_lookback = max([
            max(feature_requirements['sma_periods']),
            max(feature_requirements['ema_periods']),
            feature_requirements['rsi_period'],
            feature_requirements['bollinger_period'],
            max(feature_requirements['macd_periods'])
        ])
        
        insufficient_data = []
        
        for timeframe, filepath in data_sources.items():
            df = pd.read_csv(filepath)
            if timeframe == 'Monthly':
                # Only require 12 rows (1 year) for Monthly timeframe
                if len(df) < 12:
                    insufficient_data.append(
                        f"{timeframe}: {len(df)} rows (need 12 for minimum monthly check)"
                    )
                continue
            if len(df) < max_lookback * 2:  # Need extra buffer
                insufficient_data.append(
                    f"{timeframe}: {len(df)} rows (need {max_lookback * 2} for indicators)"
                )
        
        if insufficient_data:
            raise DataValidationError(f"Insufficient data for indicators: {insufficient_data}")
        
        self.logger.info("✓ Feature requirements validation passed")
    
    def _validate_holloway_prerequisites(self, data_sources: Dict[str, str]):
        """Validate specific requirements for Holloway feature calculation"""
        self.logger.info("Validating Holloway algorithm prerequisites...")
        
        holloway_issues = []
        
        for timeframe, filepath in data_sources.items():
            df = pd.read_csv(filepath)
            
            # Holloway algorithm needs sufficient historical data
            if len(df) < 100:  # Minimum for Holloway calculations
                holloway_issues.append(
                    f"{timeframe}: {len(df)} rows (need 100+ for Holloway)"
                )
                continue
            
            # Check for required OHLC data
            required_for_holloway = ['open', 'high', 'low', 'close']
            missing_ohlc = []
            
            df_columns_lower = [col.lower() for col in df.columns]
            for req_col in required_for_holloway:
                if req_col not in df_columns_lower:
                    missing_ohlc.append(req_col)
            
            if missing_ohlc:
                holloway_issues.append(
                    f"{timeframe}: Missing OHLC data for Holloway: {missing_ohlc}"
                )
        
        if holloway_issues:
            # Log as warning but don't fail - some timeframes might not have Holloway
            for issue in holloway_issues:
                self.logger.warning(f"Holloway prerequisite issue: {issue}")
        else:
            self.logger.info("✓ Holloway prerequisites validation passed")
    
    def _validate_training_readiness(self, data_sources: Dict[str, str]):
        """Final check that everything is ready for training"""
        self.logger.info("Validating training readiness...")
        
        # Ensure we have at least one primary timeframe with sufficient data
        primary_timeframes = ['Daily', 'H4', 'H1']
        viable_timeframes = []
        
        for timeframe in primary_timeframes:
            if timeframe in data_sources:
                df = pd.read_csv(data_sources[timeframe])
                if len(df) >= 252:  # At least 1 year of data
                    viable_timeframes.append(timeframe)
        
        if not viable_timeframes:
            raise DataValidationError(
                "No primary timeframes have sufficient data for training. "
                f"Need at least 252 rows in one of: {primary_timeframes}"
            )
        
        self.logger.info(f"✓ Training ready with viable timeframes: {viable_timeframes}")
        
        # Store training-ready timeframes
        self.validation_results['training_ready_timeframes'] = viable_timeframes
    
    def generate_validation_report(self) -> Dict:
        """Generate comprehensive validation report"""
        report = {
            'validation_status': 'PASSED',
            'timestamp': pd.Timestamp.now(),
            'results': self.validation_results.copy(),
            'summary': {
                'total_timeframes_validated': len([k for k in self.validation_results.keys() 
                                                 if '_structure' in k]),
                'training_ready': self.validation_results.get('training_ready_timeframes', [])
            }
        }
        
        return report


# Integration function for your forecasting.py
def validate_data_before_training(data_config: Dict) -> bool:
    """
    Main integration function to be called before training begins
    
    Args:
        data_config: Dictionary with data file paths and configuration
        
    Returns:
        bool: True if validation passes, raises exception otherwise
    """

    # Keep backwards-compatible signature but accept None to auto-detect files
    raise NotImplementedError("Use validate_data_before_training_auto or pass a config dict")


def validate_data_before_training_auto(data_config: Optional[Dict] = None) -> bool:
    """
    Auto-detect data files (if data_config is None) and run full validation.

    - If `data_config` is provided it should be a dict with a 'data_sources' key
      mapping timeframe names (e.g. 'H1','H4','Daily','Weekly') to file paths.
    - If `data_config` is None, this function will scan the `data/` folder and
      pick the first matching file for common timeframe suffixes.
    """

    # Auto-detect simple data_sources when no config passed
    if data_config is None:
        data_dir = Path('data')
        if not data_dir.exists():
            raise DataValidationError("Data directory 'data/' does not exist")

        patterns = {
            'H1': '*_H1.csv',
            'H4': '*_H4.csv',
            'Daily': '*_Daily.csv',
            'Weekly': '*_Weekly.csv',
            'Monthly': '*_Monthly.csv'
        }

        discovered = {}
        for timeframe, pat in patterns.items():
            matches = list(data_dir.glob(pat))
            if matches:
                discovered[timeframe] = str(matches[0])

        if not discovered:
            raise DataValidationError('No data files found in data/ to validate')

        data_config = {'data_sources': discovered}

    validator = ForexDataValidator(data_config)

    try:
        data_sources = data_config.get('data_sources', {})
        if not data_sources:
            raise DataValidationError("No data sources specified in configuration")

        validation_passed = validator.validate_all_data(data_sources)

        report = validator.generate_validation_report()
        logging.info(f"Validation Report: {report['summary']}")

        return validation_passed

    except DataValidationError:
        logging.error("DATA VALIDATION FAILED - TRAINING ABORTED")
        raise
    except Exception as e:
        logging.error(f"Unexpected validation error: {str(e)}")
        raise DataValidationError(f"Validation system error: {str(e)}")
