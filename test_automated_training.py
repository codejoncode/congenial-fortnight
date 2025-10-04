#!/usr/bin/env python3
"""
Unit tests for automated_training.py to ensure training stops on critical errors.
"""

import os
import sys
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add project root to path
BASE_APP_DIR = os.environ.get('APP_ROOT', os.getcwd())
sys.path.insert(0, BASE_APP_DIR)

from scripts.automated_training import AutomatedTrainer
from scripts.forecasting import HybridPriceForecastingEnsemble


class TestAutomatedTrainingErrors:
    """Test that automated training properly handles and stops on errors."""

    def test_training_stops_on_missing_data_files(self):
        """Test that training fails when essential data files are missing."""
        trainer = AutomatedTrainer()

        # Mock pre_training_data_fix to return False (data validation failed)
        with patch('scripts.automated_training.pre_training_data_fix', return_value=False):
            with pytest.raises(SystemExit) as exc_info:
                trainer.run_automated_training(pairs=['EURUSD'], dry_run=True, dry_iterations=1)

            assert exc_info.value.code == 1  # Should exit with error code

    def test_training_stops_on_empty_feature_dataframe(self):
        """Test that training fails when feature engineering produces empty dataframe."""
        trainer = AutomatedTrainer()

        # Mock ForecastingSystem to return empty features
        with patch('scripts.automated_training.ForecastingSystem') as mock_forecasting:
            mock_instance = MagicMock()
            mock_instance._prepare_features.return_value = pd.DataFrame()  # Empty dataframe
            mock_forecasting.return_value = mock_instance

            with pytest.raises(ValueError, match="Feature engineering produced empty dataframe"):
                trainer.run_automated_training(pairs=['EURUSD'], dry_run=True, dry_iterations=1)

    def test_training_stops_on_missing_target_column(self):
        """Test that training fails when target column is missing from features."""
        trainer = AutomatedTrainer()

        # Mock ForecastingSystem to return features without target
        with patch('scripts.automated_training.ForecastingSystem') as mock_forecasting:
            mock_instance = MagicMock()
            # Return dataframe with features but no target_1d
            mock_instance._prepare_features.return_value = pd.DataFrame({
                'feature1': [1, 2, 3],
                'feature2': [4, 5, 6]
            })
            mock_forecasting.return_value = mock_instance

            with pytest.raises(ValueError, match="No target column found in features"):
                trainer.run_automated_training(pairs=['EURUSD'], dry_run=True, dry_iterations=1)

    def test_training_stops_on_mismatched_dataset_lengths(self):
        """Test that training fails when X and y have different lengths."""
        trainer = AutomatedTrainer()

        # Mock ForecastingSystem to return mismatched lengths
        with patch('scripts.automated_training.ForecastingSystem') as mock_forecasting:
            mock_instance = MagicMock()
            # Mock load_and_prepare_datasets to return mismatched data
            mock_instance.load_and_prepare_datasets.return_value = (
                pd.DataFrame({'f1': [1, 2, 3]}),  # X_train: 3 rows
                pd.Series([1, 2]),  # y_train: 2 rows (mismatched)
                None, None
            )
            mock_forecasting.return_value = mock_instance

            with pytest.raises(ValueError, match="Mismatched lengths"):
                trainer.run_automated_training(pairs=['EURUSD'], dry_run=True, dry_iterations=1)

    def test_training_stops_on_insufficient_samples(self):
        """Test that training fails when there are too few samples."""
        trainer = AutomatedTrainer()

        # Mock ForecastingSystem to return very small dataset
        with patch('scripts.automated_training.ForecastingSystem') as mock_forecasting:
            mock_instance = MagicMock()
            mock_instance.load_and_prepare_datasets.return_value = (
                pd.DataFrame({'f1': [1]}),  # Only 1 sample
                pd.Series([1]),
                None, None
            )
            mock_forecasting.return_value = mock_instance

            with pytest.raises(ValueError, match="Empty training arrays"):
                trainer.run_automated_training(pairs=['EURUSD'], dry_run=True, dry_iterations=1)

    def test_fred_api_key_missing_stops_fundamentals(self):
        """Test that missing FRED API key prevents fundamentals loading."""
        # Temporarily remove FRED_API_KEY from environment
        original_key = os.environ.get('FRED_API_KEY')
        if 'FRED_API_KEY' in os.environ:
            del os.environ['FRED_API_KEY']

        try:
            fs = HybridPriceForecastingEnsemble('EURUSD')
            fundamentals = fs._load_fundamental_data()

            # Should return empty dataframe when key is missing
            assert fundamentals.empty, "Fundamentals should be empty when FRED API key is missing"

        finally:
            # Restore original key
            if original_key:
                os.environ['FRED_API_KEY'] = original_key

    def test_csv_parsing_errors_are_handled(self):
        """Test that CSV parsing errors are caught and handled."""
        trainer = AutomatedTrainer()

        # Mock ForecastingSystem to raise an exception during data loading
        with patch('scripts.automated_training.ForecastingSystem') as mock_forecasting:
            mock_instance = MagicMock()
            mock_instance._prepare_features.side_effect = FileNotFoundError("CSV file not found")
            mock_forecasting.return_value = mock_instance

            # Should catch the exception and continue to final results
            results = trainer.run_automated_training(pairs=['EURUSD'], dry_run=True, dry_iterations=1)

            # Should have error in results
            assert 'EURUSD' in results
            assert 'error' in results['EURUSD']
            assert 'CSV file not found' in str(results['EURUSD']['error'])


if __name__ == '__main__':
    pytest.main([__file__])