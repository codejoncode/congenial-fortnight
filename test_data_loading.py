import unittest
import pandas as pd
import os
import sys
from parameterized import parameterized
from pathlib import Path

# Add project root to path to allow imports from 'scripts'
BASE_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_APP_DIR not in sys.path:
    sys.path.insert(0, BASE_APP_DIR)

from scripts.forecasting import HybridPriceForecastingEnsemble

class TestDataLoading(unittest.TestCase):

    def setUp(self):
        """Set up test environment by defining absolute paths."""
        self.base_dir = Path(__file__).resolve().parent
        self.data_dir = self.base_dir / "data"
        self.models_dir = self.base_dir / "models"

    @parameterized.expand([
        ("EURUSD"),
        ("XAUUSD"),
    ])
    def test_all_data_sources_are_present_and_loadable(self, pair):
        """
        Verify that for each pair, all expected timeframes (H1, H4, Daily, Monthly)
        and fundamental data sources are present and can be loaded without errors.
        """
        try:
            forecasting_system = HybridPriceForecastingEnsemble(pair=pair, data_dir=self.data_dir)
        except Exception as e:
            self.fail(f"Failed to instantiate HybridPriceForecastingEnsemble for {pair}. Error: {e}")

        # Test if intraday data is loaded
        self.assertFalse(forecasting_system.intraday_data.empty, f"Intraday (H1) data for {pair} should not be empty.")
        self.assertIsInstance(forecasting_system.intraday_data.index, pd.DatetimeIndex, f"Intraday data index for {pair} should be a DatetimeIndex.")

        # Test Monthly data
        self.assertFalse(forecasting_system.monthly_data.empty, f"Monthly data for {pair} should not be empty.")
        self.assertIsInstance(forecasting_system.monthly_data.index, pd.DatetimeIndex, f"Monthly data index for {pair} should be a DatetimeIndex.")

        # Test the main consolidated price_data
        self.assertFalse(forecasting_system.price_data.empty, f"Consolidated price_data for {pair} should not be empty.")
        self.assertGreater(len(forecasting_system.price_data), 100, f"Consolidated price_data for {pair} should have a substantial number of rows.")

        # Test specific timeframe loading logic that was previously failing
        # Test H4 data loading
        h4_df = forecasting_system._load_daily_price_file(pair=pair, timeframe_hint='H4')
        self.assertIsNotNone(h4_df, f"H4 DataFrame for {pair} should not be None.")
        self.assertFalse(h4_df.empty, f"H4 data for {pair} failed to load or is empty. Check file and parsing logic.")
        self.assertIsInstance(h4_df.index, pd.DatetimeIndex, f"H4 data index for {pair} should be a DatetimeIndex.")

        # Test Weekly data loading - This is now removed as we don't have weekly files.
        # weekly_df = forecasting_system._load_daily_price_file(pair=pair, timeframe_hint='Weekly')
        # self.assertIsNotNone(weekly_df, f"Weekly DataFrame for {pair} should not be None.")
        # self.assertFalse(weekly_df.empty, f"Weekly data for {pair} failed to load or is empty. Check file and parsing logic.")
        # self.assertIsInstance(weekly_df.index, pd.DatetimeIndex, f"Weekly data index for {pair} should be a DatetimeIndex.")

    @unittest.skipIf(not os.environ.get('FRED_API_KEY'), "FRED_API_KEY not set, skipping fundamental data test.")
    def test_fundamental_data_loads(self):
        """
        Verify that fundamental economic data is loaded correctly.
        """
        try:
            # We only need to test this for one pair as it's pair-independent
            forecasting_system = HybridPriceForecastingEnsemble(pair='EURUSD', data_dir=self.data_dir)
        except Exception as e:
            self.fail(f"Failed to instantiate HybridPriceForecastingEnsemble for fundamental data test. Error: {e}")

        self.assertFalse(forecasting_system.fundamental_data.empty, "Fundamental data should not be empty.")

if __name__ == '__main__':
    unittest.main()
