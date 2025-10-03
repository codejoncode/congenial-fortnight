import unittest
import os
from pathlib import Path

# Auto-load .env for tests so FRED_API_KEY (and other keys) are available
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # dotenv not installed or load failed; tests will rely on environment
    pass

# Use the robust loader directly to test timeframes
from robust_data_loader import ForexDataLoader

class TestDataLoading(unittest.TestCase):

    def setUp(self):
        self.base_dir = Path(__file__).resolve().parent
        self.data_dir = self.base_dir / "data"

    def test_all_timeframes_for_pairs(self):
        """Verify both EURUSD and XAUUSD have five timeframes (H1, H4, Daily, Weekly, Monthly)."""

        pairs = ['EURUSD', 'XAUUSD']
        for pair in pairs:
            loader = ForexDataLoader()

            data_config = {
                'Daily': f'data/{pair}_Daily.csv',
                'H4': f'data/{pair}_H4.csv',
                'H1': f'data/{pair}_H1.csv',
                'Weekly': f'data/{pair}_Weekly.csv',
                'Monthly': f'data/{pair}_Monthly.csv'
            }

            try:
                loaded = loader.load_all_timeframes(data_config)
            except Exception as e:
                self.fail(f"Loader failed for {pair}: {e}")

            # Ensure all five expected timeframes loaded
            self.assertEqual(len(loaded), 5, f"Expected 5 timeframes for {pair}; loaded: {list(loaded.keys())}")

            # Verify each dataframe is non-empty and has a timestamp column
            for tf, df in loaded.items():
                self.assertIsNotNone(df, f"{pair} {tf} should not be None")
                self.assertFalse(df.empty, f"{pair} {tf} should not be empty")
                self.assertIn('timestamp', [c.lower() for c in df.columns], f"{pair} {tf} must have a timestamp column")

    @unittest.skipIf(not os.environ.get('FRED_API_KEY'), "FRED_API_KEY not set, skipping fundamental data test.")
    def test_fundamental_data_loads(self):
        """Verify that fundamental economic data is loaded correctly."""
        # We'll just check the loader can be used by the fundamental pipeline indirectly
        from scripts.fundamental_pipeline import FundamentalDataPipeline
        try:
            fd = FundamentalDataPipeline(self.data_dir)
            df = fd.load_all_series_as_df()
        except Exception as e:
            self.fail(f"Fundamental pipeline failed: {e}")

        self.assertFalse(df.empty, "Fundamental data should not be empty.")

if __name__ == '__main__':
    unittest.main()
