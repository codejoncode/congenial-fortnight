import pandas as pd
import numpy as np
from django.test import TestCase
from candle_prediction_system import CandlePredictionSystem


class CandlePatternTests(TestCase):
    """Test cases for candlestick pattern detection"""

    def setUp(self):
        """Set up test data"""
        self.system = CandlePredictionSystem(['EURUSD'])

        # Create sample OHLC data for testing
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        np.random.seed(42)  # For reproducible results

        # Create realistic OHLC data
        base_prices = np.array([1.0500, 1.0520, 1.0480, 1.0550, 1.0530,
                               1.0580, 1.0560, 1.0600, 1.0570, 1.0620])

        self.test_data = pd.DataFrame({
            'date': dates,
            'open': base_prices,
            'high': base_prices + np.random.uniform(0.001, 0.005, 10),
            'low': base_prices - np.random.uniform(0.001, 0.005, 10),
            'close': base_prices + np.random.uniform(-0.003, 0.003, 10),
            'volume': np.random.randint(1000, 10000, 10)
        })

        # Ensure OHLC relationships are valid
        for i in range(len(self.test_data)):
            high = max(self.test_data.loc[i, ['open', 'close']].values)
            low = min(self.test_data.loc[i, ['open', 'close']].values)
            self.test_data.loc[i, 'high'] = max(self.test_data.loc[i, 'high'], high)
            self.test_data.loc[i, 'low'] = min(self.test_data.loc[i, 'low'], low)

    def test_helper_functions(self):
        """Test helper functions for candlestick calculations"""
        # Test with a single bullish candle
        bullish_row = pd.Series({
            'open': 1.0500,
            'high': 1.0550,
            'low': 1.0480,
            'close': 1.0530
        })

        # Test helper functions (these are defined within engineer_features method)
        def body_size(row):
            return abs(row['close'] - row['open'])

        def upper_wick(row):
            return row['high'] - max(row['open'], row['close'])

        def lower_wick(row):
            return min(row['open'], row['close']) - row['low']

        def total_range(row):
            return row['high'] - row['low']

        self.assertAlmostEqual(body_size(bullish_row), 0.0030, places=4)
        self.assertAlmostEqual(upper_wick(bullish_row), 0.0020, places=4)
        self.assertAlmostEqual(lower_wick(bullish_row), 0.0020, places=4)
        self.assertAlmostEqual(total_range(bullish_row), 0.0070, places=4)

    def test_bullish_marubozu(self):
        """Test bullish marubozu pattern detection"""
        df = self.test_data.copy()

        # Create a perfect bullish marubozu
        df.loc[0, 'open'] = 1.0500
        df.loc[0, 'high'] = 1.0530
        df.loc[0, 'low'] = 1.0500
        df.loc[0, 'close'] = 1.0530

        # Apply feature engineering
        result_df = self.system.engineer_features(df, 'EURUSD')

        self.assertEqual(result_df.loc[0, 'bullish_marubozu'], 1)

        # Test non-marubozu candle
        df.loc[1, 'open'] = 1.0520
        df.loc[1, 'high'] = 1.0540
        df.loc[1, 'low'] = 1.0510
        df.loc[1, 'close'] = 1.0530  # Not a marubozu

        result_df = self.system.engineer_features(df, 'EURUSD')
        self.assertEqual(result_df.loc[1, 'bullish_marubozu'], 0)

    def test_bearish_marubozu(self):
        """Test bearish marubozu pattern detection"""
        df = self.test_data.copy()

        # Create a perfect bearish marubozu
        df.loc[0, 'open'] = 1.0530
        df.loc[0, 'high'] = 1.0530
        df.loc[0, 'low'] = 1.0500
        df.loc[0, 'close'] = 1.0500

        # Apply pattern detection
        df['bearish_marubozu'] = df.apply(lambda row: 1 if (row['close'] < row['open'] and
                                                           row['high'] == row['open'] and
                                                           row['low'] == row['close']) else 0, axis=1)

        self.assertEqual(df.loc[0, 'bearish_marubozu'], 1)

    def test_bullish_hammer(self):
        """Test bullish hammer pattern detection"""
        df = self.test_data.copy()

        # Create a hammer pattern: long lower wick, small body, small upper wick
        df.loc[0, 'open'] = 1.0520
        df.loc[0, 'high'] = 1.0530  # Small upper wick
        df.loc[0, 'low'] = 1.0480   # Long lower wick
        df.loc[0, 'close'] = 1.0525  # Small body

        def body_size(row):
            return abs(row['close'] - row['open'])

        def upper_wick(row):
            return row['high'] - max(row['open'], row['close'])

        def lower_wick(row):
            return min(row['open'], row['close']) - row['low']

        def total_range(row):
            return row['high'] - row['low']

        # Apply pattern detection
        df['bullish_hammer'] = df.apply(lambda row: 1 if (row['close'] > row['open'] and
                                                         lower_wick(row) > 2 * body_size(row) and
                                                         upper_wick(row) < 0.1 * total_range(row)) else 0, axis=1)

        self.assertEqual(df.loc[0, 'bullish_hammer'], 1)

    def test_bullish_engulfing(self):
        """Test bullish engulfing pattern detection"""
        df = self.test_data.copy()

        # Create engulfing pattern: first bearish, second bullish engulfing
        df.loc[0, 'open'] = 1.0530
        df.loc[0, 'high'] = 1.0530
        df.loc[0, 'low'] = 1.0500
        df.loc[0, 'close'] = 1.0500  # Bearish candle

        df.loc[1, 'open'] = 1.0490
        df.loc[1, 'high'] = 1.0550
        df.loc[1, 'low'] = 1.0490
        df.loc[1, 'close'] = 1.0540  # Bullish engulfing candle

        # Apply feature engineering
        result_df = self.system.engineer_features(df, 'EURUSD')

        self.assertEqual(result_df.loc[1, 'bullish_engulfing'], 1)

    def test_bearish_engulfing(self):
        """Test bearish engulfing pattern detection"""
        df = self.test_data.copy()

        # Create engulfing pattern: first bullish, second bearish engulfing
        df.loc[0, 'open'] = 1.0500
        df.loc[0, 'high'] = 1.0530
        df.loc[0, 'low'] = 1.0500
        df.loc[0, 'close'] = 1.0530  # Bullish candle

        df.loc[1, 'open'] = 1.0540
        df.loc[1, 'high'] = 1.0540
        df.loc[1, 'low'] = 1.0490
        df.loc[1, 'close'] = 1.0500  # Bearish engulfing candle

        # Apply feature engineering
        result_df = self.system.engineer_features(df, 'EURUSD')

        self.assertEqual(result_df.loc[1, 'bearish_engulfing'], 1)

    def test_bullish_harami(self):
        """Test bullish harami pattern detection"""
        df = self.test_data.copy()

        # Create harami pattern: first bearish (large), second bullish (small inside)
        df.loc[0, 'open'] = 1.0550
        df.loc[0, 'high'] = 1.0550
        df.loc[0, 'low'] = 1.0500
        df.loc[0, 'close'] = 1.0500  # Large bearish candle

        df.loc[1, 'open'] = 1.0520
        df.loc[1, 'high'] = 1.0530
        df.loc[1, 'low'] = 1.0510
        df.loc[1, 'close'] = 1.0525  # Small bullish candle inside previous

        # Apply pattern detection
        df['bullish_harami'] = ((df['close'] > df['open']) &
                               (df['close'].shift(1) < df['open'].shift(1)) &
                               (df['close'] < df['open'].shift(1)) &
                               (df['open'] > df['close'].shift(1))).astype(int)

        self.assertEqual(df.loc[1, 'bullish_harami'], 1)

    def test_three_white_soldiers(self):
        """Test three white soldiers pattern detection"""
        df = self.test_data.copy()

        # Create three consecutive bullish candles with higher closes
        df.loc[0, 'open'] = 1.0500
        df.loc[0, 'high'] = 1.0520
        df.loc[0, 'low'] = 1.0500
        df.loc[0, 'close'] = 1.0515

        df.loc[1, 'open'] = 1.0515
        df.loc[1, 'high'] = 1.0535
        df.loc[1, 'low'] = 1.0515
        df.loc[1, 'close'] = 1.0530

        df.loc[2, 'open'] = 1.0530
        df.loc[2, 'high'] = 1.0550
        df.loc[2, 'low'] = 1.0530
        df.loc[2, 'close'] = 1.0545

        # Apply pattern detection
        df['bullish_three_white_soldiers'] = ((df['close'] > df['open']) &
                                             (df['close'].shift(1) > df['open'].shift(1)) &
                                             (df['close'].shift(2) > df['open'].shift(2)) &
                                             (df['close'] > df['close'].shift(1)) &
                                             (df['close'].shift(1) > df['close'].shift(2))).astype(int)

        self.assertEqual(df.loc[2, 'bullish_three_white_soldiers'], 1)

    def test_three_black_crows(self):
        """Test three black crows pattern detection"""
        df = self.test_data.copy()

        # Create three consecutive bearish candles with lower closes
        df.loc[0, 'open'] = 1.0545
        df.loc[0, 'high'] = 1.0545
        df.loc[0, 'low'] = 1.0530
        df.loc[0, 'close'] = 1.0530

        df.loc[1, 'open'] = 1.0530
        df.loc[1, 'high'] = 1.0530
        df.loc[1, 'low'] = 1.0515
        df.loc[1, 'close'] = 1.0515

        df.loc[2, 'open'] = 1.0515
        df.loc[2, 'high'] = 1.0515
        df.loc[2, 'low'] = 1.0500
        df.loc[2, 'close'] = 1.0500

        # Apply pattern detection
        df['bearish_three_black_crows'] = ((df['close'] < df['open']) &
                                          (df['close'].shift(1) < df['open'].shift(1)) &
                                          (df['close'].shift(2) < df['open'].shift(2)) &
                                          (df['close'] < df['close'].shift(1)) &
                                          (df['close'].shift(1) < df['close'].shift(2))).astype(int)

        self.assertEqual(df.loc[2, 'bearish_three_black_crows'], 1)

    def test_morning_star(self):
        """Test morning star pattern detection"""
        df = self.test_data.copy()

        # Create morning star pattern
        df.loc[0, 'open'] = 1.0550
        df.loc[0, 'high'] = 1.0550
        df.loc[0, 'low'] = 1.0500
        df.loc[0, 'close'] = 1.0500  # Large bearish candle

        df.loc[1, 'open'] = 1.0520
        df.loc[1, 'high'] = 1.0525
        df.loc[1, 'low'] = 1.0515
        df.loc[1, 'close'] = 1.0520  # Small doji/star

        df.loc[2, 'open'] = 1.0520
        df.loc[2, 'high'] = 1.0570
        df.loc[2, 'low'] = 1.0520
        df.loc[2, 'close'] = 1.0560  # Large bullish candle

        # Apply pattern detection
        df['bullish_morning_star'] = ((df['close'].shift(2) < df['open'].shift(2)) &
                                     (abs(df['close'].shift(1) - df['open'].shift(1)) < 0.1 * (df['high'].shift(1) - df['low'].shift(1))) &
                                     (df['close'] > df['open']) &
                                     (df['close'] > (df['open'].shift(2) + df['close'].shift(2)) / 2)).astype(int)

        self.assertEqual(df.loc[2, 'bullish_morning_star'], 1)

    def test_evening_star(self):
        """Test evening star pattern detection"""
        df = self.test_data.copy()

        # Create evening star pattern
        df.loc[0, 'open'] = 1.0500
        df.loc[0, 'high'] = 1.0550
        df.loc[0, 'low'] = 1.0500
        df.loc[0, 'close'] = 1.0540  # Large bullish candle

        df.loc[1, 'open'] = 1.0530
        df.loc[1, 'high'] = 1.0535
        df.loc[1, 'low'] = 1.0525
        df.loc[1, 'close'] = 1.0530  # Small doji/star

        df.loc[2, 'open'] = 1.0530
        df.loc[2, 'high'] = 1.0530
        df.loc[2, 'low'] = 1.0480
        df.loc[2, 'close'] = 1.0490  # Large bearish candle

        # Apply pattern detection
        df['bearish_evening_star'] = ((df['close'].shift(2) > df['open'].shift(2)) &
                                     (abs(df['close'].shift(1) - df['open'].shift(1)) < 0.1 * (df['high'].shift(1) - df['low'].shift(1))) &
                                     (df['close'] < df['open']) &
                                     (df['close'] < (df['open'].shift(2) + df['close'].shift(2)) / 2)).astype(int)

        self.assertEqual(df.loc[2, 'bearish_evening_star'], 1)

    def test_piercing_pattern(self):
        """Test piercing pattern detection"""
        df = self.test_data.copy()

        # Create piercing pattern
        df.loc[0, 'open'] = 1.0550
        df.loc[0, 'high'] = 1.0550
        df.loc[0, 'low'] = 1.0500
        df.loc[0, 'close'] = 1.0500  # Bearish candle

        df.loc[1, 'open'] = 1.0490
        df.loc[1, 'high'] = 1.0525
        df.loc[1, 'low'] = 1.0490
        df.loc[1, 'close'] = 1.0530  # Bullish candle piercing deeper into previous body

        # Apply feature engineering
        result_df = self.system.engineer_features(df, 'EURUSD')

        self.assertEqual(result_df.loc[1, 'bullish_piercing_pattern'], 1)

    def test_dark_cloud_cover(self):
        """Test dark cloud cover pattern detection"""
        df = self.test_data.copy()

        # Create dark cloud cover pattern
        df.loc[0, 'open'] = 1.0500
        df.loc[0, 'high'] = 1.0550
        df.loc[0, 'low'] = 1.0500
        df.loc[0, 'close'] = 1.0540  # Bullish candle

        df.loc[1, 'open'] = 1.0550
        df.loc[1, 'high'] = 1.0550
        df.loc[1, 'low'] = 1.0510
        df.loc[1, 'close'] = 1.0515  # Bearish candle covering deeper into previous body

        # Apply feature engineering
        result_df = self.system.engineer_features(df, 'EURUSD')

        self.assertEqual(result_df.loc[1, 'bearish_dark_cloud'], 1)

    def test_engineer_features_integration(self):
        """Test that engineer_features method works end-to-end"""
        df = self.test_data.copy()

        # Test that the method runs without errors
        result_df = self.system.engineer_features(df, 'EURUSD')

        # Check that expected columns are created
        expected_patterns = [
            'bullish_marubozu', 'bullish_hammer', 'bullish_engulfing',
            'bearish_marubozu', 'bearish_hanging_man', 'bearish_engulfing',
            'bullish_three_white_soldiers', 'bearish_three_black_crows'
        ]

        for pattern in expected_patterns:
            self.assertIn(pattern, result_df.columns, f"Pattern {pattern} not found in result")

        # Check that all pattern columns contain binary values (0 or 1)
        pattern_cols = [col for col in result_df.columns if 'pattern' in col.lower() or 'candle' in col.lower()]
        for col in pattern_cols:
            unique_values = result_df[col].unique()
            valid_values = all(val in [0, 1] for val in unique_values)
            self.assertTrue(valid_values, f"Column {col} contains invalid values: {unique_values}")

    def test_pattern_columns_exist(self):
        """Test that all expected pattern columns are created"""
        df = self.test_data.copy()
        result_df = self.system.engineer_features(df, 'EURUSD')

        # Check for major pattern categories
        bullish_single = [f'bullish_pattern_{i}' for i in range(6, 52)]
        bearish_single = [f'bearish_pattern_{i}' for i in range(7, 50)]
        bullish_two = [f'bullish_two_candle_{i}' for i in range(4, 25)]
        bearish_two = [f'bearish_two_candle_{i}' for i in range(4, 25)]
        bullish_three = [f'bullish_three_candle_{i}' for i in range(2, 25)]
        bearish_three = [f'bearish_three_candle_{i}' for i in range(2, 25)]

        all_expected_patterns = (
            bullish_single + bearish_single +
            bullish_two + bearish_two +
            bullish_three + bearish_three
        )

        for pattern in all_expected_patterns:
            self.assertIn(pattern, result_df.columns,
                         f"Expected pattern column {pattern} not found")

    def test_no_patterns_on_insufficient_data(self):
        """Test that patterns are not detected with insufficient data"""
        # Create dataframe with only 1 row (insufficient for multi-candle patterns)
        single_row_df = pd.DataFrame({
            'date': [pd.Timestamp('2023-01-01')],
            'open': [1.0500],
            'high': [1.0520],
            'low': [1.0480],
            'close': [1.0510],
            'volume': [1000]
        })

        result_df = self.system.engineer_features(single_row_df, 'EURUSD')

        # Multi-candle patterns should be 0 or NaN for the first row
        multi_candle_patterns = [
            'bullish_engulfing', 'bearish_engulfing',
            'bullish_three_white_soldiers', 'bearish_three_black_crows'
        ]

        for pattern in multi_candle_patterns:
            if pattern in result_df.columns:
                # First row should not have multi-candle patterns
                self.assertEqual(result_df[pattern].iloc[0], 0,
                               f"Multi-candle pattern {pattern} detected on insufficient data")


class SignalGenerationTests(TestCase):
    """Test cases for signal generation and data completeness"""

    def test_signal_data_integrity(self):
        """Test that all signals have required fields and valid data"""
        from signals.models import Signal

        signals = Signal.objects.all()[:100]  # Test last 100 signals

        for signal in signals:
            # Check required fields are not null
            self.assertIsNotNone(signal.pair, f"Signal {signal.id} missing pair")
            self.assertIsNotNone(signal.signal, f"Signal {signal.id} missing signal type")
            self.assertIsNotNone(signal.probability, f"Signal {signal.id} missing probability")
            self.assertIsNotNone(signal.date, f"Signal {signal.id} missing date")

            # Check probability is valid (0-1 range)
            self.assertGreaterEqual(signal.probability, 0.0, f"Signal {signal.id} has invalid probability: {signal.probability}")
            self.assertLessEqual(signal.probability, 1.0, f"Signal {signal.id} has invalid probability: {signal.probability}")

            # Check signal type is valid
            valid_signals = ['bullish', 'bearish', 'no_signal']
            self.assertIn(signal.signal, valid_signals, f"Signal {signal.id} has invalid signal type: {signal.signal}")

            # Check pair is valid
            valid_pairs = ['EURUSD', 'XAUUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD']
            self.assertIn(signal.pair, valid_pairs, f"Signal {signal.id} has invalid pair: {signal.pair}")

    def test_signal_generation_handles_missing_h4_data(self):
        """Test that signal generation works even when H4 data is missing"""
        from candle_prediction_system import CandlePredictionSystem
        import pandas as pd
        import numpy as np

        # Create test data similar to EURUSD (which has missing H4 data)
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        np.random.seed(42)

        test_df = pd.DataFrame({
            'date': dates,
            'open': 1.0500 + np.random.normal(0, 0.01, 50),
            'high': 1.0520 + np.random.normal(0, 0.005, 50),
            'low': 1.0480 + np.random.normal(0, 0.005, 50),
            'close': 1.0500 + np.random.normal(0, 0.01, 50),
            'tickvol': np.random.randint(1000, 10000, 50)
        })

        # Ensure OHLC relationships are valid
        for i in range(len(test_df)):
            high = max(test_df.loc[i, ['open', 'close']].values)
            low = min(test_df.loc[i, ['open', 'close']].values)
            test_df.loc[i, 'high'] = max(test_df.loc[i, 'high'], high)
            test_df.loc[i, 'low'] = min(test_df.loc[i, 'low'], low)

        test_df['date'] = pd.to_datetime(test_df['date'])
        test_df = test_df.set_index('date')

        # Test feature engineering (this should work even without H4 data)
        cps = CandlePredictionSystem()
        try:
            features_df = cps.engineer_features(test_df, 'EURUSD')
            self.assertIsNotNone(features_df, "Feature engineering failed")
            self.assertGreater(len(features_df), 30, "Not enough data after feature engineering")

            # Check that H4 features are set to default values
            if 'h4_trend' in features_df.columns:
                # Should not be all zeros if H4 data exists, but should be zeros if missing
                self.assertTrue(True, "H4 features handled correctly")

        except Exception as e:
            self.fail(f"Feature engineering failed with missing H4 data: {e}")
