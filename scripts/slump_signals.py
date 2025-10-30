import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class SlumpSignalEngine:
    """
    Engine for generating slump signals - signals that identify market downturns,
    bearish reversals, and capitulation events.
    """

    def __init__(self):
        """Initialize the slump signal engine."""
        self.signal_names = [
            'bearish_engulfing_patterns',
            'shooting_star_rejections',
            'bearish_hammer_failures',
            'volume_climax_declines',
            'rsi_divergence_bearish',
            'macd_bearish_crossovers',
            'stochastic_bearish_signals',
            'bollinger_bearish_squeezes',
            'fibonacci_retracement_breaks',
            'momentum_divergence_bearish'
        ]

    def generate_all_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all slump signals for the given dataframe.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with slump signal columns added
        """
        df = df.copy()

        # Generate each signal
        df = self._bearish_engulfing_patterns(df)
        df = self._shooting_star_rejections(df)
        # DISABLED: bearish_hammer_failures - 47.57% accuracy, -13% correlation (worse than random)
        # df = self._bearish_hammer_failures(df)
        df = self._volume_climax_declines(df)
        # DISABLED: rsi_divergence_bearish - 48.68% accuracy, -2.9% correlation (worse than random)
        # df = self._rsi_divergence_bearish(df)
        # DISABLED: macd_bearish_crossovers - 49.19% accuracy, -1.3% correlation (worse than random)
        # df = self._macd_bearish_crossovers(df)
        df = self._stochastic_bearish_signals(df)
        df = self._bollinger_bearish_squeezes(df)
        df = self._fibonacci_retracement_breaks(df)
        df = self._momentum_divergence_bearish(df)

        return df

    def _bearish_engulfing_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bearish engulfing pattern - large bearish candle engulfs previous bullish candle."""
        signal_col = 'bearish_engulfing_patterns_signal'

        # Calculate candle body sizes
        body_current = abs(df['Close'] - df['Open'])
        body_prev = abs(df['Close'].shift(1) - df['Open'].shift(1))

        # Bearish engulfing conditions
        bearish_candle = df['Close'] < df['Open']
        prev_bullish = df['Close'].shift(1) > df['Open'].shift(1)
        engulfing = df['Open'] >= df['Close'].shift(1)
        engulfed = df['Close'] <= df['Open'].shift(1)

        df[signal_col] = (bearish_candle & prev_bullish & engulfing & engulfed).astype(int)
        return df

    def _shooting_star_rejections(self, df: pd.DataFrame) -> pd.DataFrame:
        """Shooting star pattern - small body with long upper wick, rejection at highs."""
        signal_col = 'shooting_star_rejections_signal'

        # Calculate candle components
        body = abs(df['Close'] - df['Open'])
        upper_wick = df['High'] - np.maximum(df['Open'], df['Close'])
        lower_wick = np.minimum(df['Open'], df['Close']) - df['Low']
        total_range = df['High'] - df['Low']

        # Shooting star conditions
        small_body = body <= (total_range * 0.3)
        long_upper_wick = upper_wick >= (total_range * 0.6)
        small_lower_wick = lower_wick <= (total_range * 0.2)

        df[signal_col] = (small_body & long_upper_wick & small_lower_wick).astype(int)
        return df

    def _bearish_hammer_failures(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bearish hammer - hammer pattern that fails to reverse bullish momentum."""
        signal_col = 'bearish_hammer_failures_signal'

        # Calculate hammer components
        body = abs(df['Close'] - df['Open'])
        upper_wick = df['High'] - np.maximum(df['Open'], df['Close'])
        lower_wick = np.minimum(df['Open'], df['Close']) - df['Low']
        total_range = df['High'] - df['Low']

        # Hammer conditions
        small_body = body <= (total_range * 0.3)
        long_lower_wick = lower_wick >= (total_range * 0.6)
        small_upper_wick = upper_wick <= (total_range * 0.2)

        # Bearish failure - hammer but price continues down
        hammer = small_body & long_lower_wick & small_upper_wick
        next_close_lower = df['Close'].shift(-1) < df['Close']

        df[signal_col] = (hammer & next_close_lower).astype(int)
        return df

    def _volume_climax_declines(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume climax - high volume accompanied by price decline."""
        signal_col = 'volume_climax_declines_signal'

        if 'Volume' not in df.columns:
            df[signal_col] = 0
            return df

        # Calculate volume moving averages
        vol_ma20 = df['Volume'].rolling(20).mean()
        vol_ma50 = df['Volume'].rolling(50).mean()

        # Volume climax conditions
        high_volume = df['Volume'] > vol_ma20 * 1.5
        very_high_volume = df['Volume'] > vol_ma50 * 2.0
        price_decline = df['Close'] < df['Open']

        df[signal_col] = ((high_volume | very_high_volume) & price_decline).astype(int)
        return df

    def _rsi_divergence_bearish(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bearish RSI divergence - price makes higher high, RSI makes lower high."""
        signal_col = 'rsi_divergence_bearish_signal'

        # Calculate RSI
        rsi = talib.RSI(df['Close'], timeperiod=14)

        # Find swing highs in price and RSI
        price_high1 = df['High'].rolling(5, center=True).max()
        price_high2 = df['High'].shift(5).rolling(5, center=True).max()

        rsi_high1 = rsi.rolling(5, center=True).max()
        rsi_high2 = rsi.shift(5).rolling(5, center=True).max()

        # Bearish divergence conditions
        price_higher_high = price_high1 > price_high2
        rsi_lower_high = rsi_high1 < rsi_high2

        df[signal_col] = (price_higher_high & rsi_lower_high).astype(int)
        return df

    def _macd_bearish_crossovers(self, df: pd.DataFrame) -> pd.DataFrame:
        """MACD bearish crossover - signal line crosses below MACD line."""
        signal_col = 'macd_bearish_crossovers_signal'

        # Calculate MACD
        macd, macdsignal, macdhist = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

        # Bearish crossover - signal crosses below MACD
        prev_signal_above = macdsignal.shift(1) > macd.shift(1)
        current_signal_below = macdsignal <= macd

        df[signal_col] = (prev_signal_above & current_signal_below).astype(int)
        return df

    def _stochastic_bearish_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stochastic bearish signals - overbought conditions and bearish crossovers."""
        signal_col = 'stochastic_bearish_signals_signal'

        # Calculate Stochastic
        slowk, slowd = talib.STOCH(df['High'], df['Low'], df['Close'],
                                  fastk_period=14, slowk_period=3, slowd_period=3)

        # Bearish conditions
        overbought = slowk > 80
        bearish_crossover = (slowk.shift(1) > slowd.shift(1)) & (slowk <= slowd)

        df[signal_col] = (overbought | bearish_crossover).astype(int)
        return df

    def _bollinger_bearish_squeezes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bollinger Band bearish squeeze - price touches upper band during squeeze."""
        signal_col = 'bollinger_bearish_squeezes_signal'

        # Calculate Bollinger Bands
        upper, middle, lower = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

        # Calculate band width (squeeze indicator)
        band_width = (upper - lower) / middle
        squeeze = band_width < band_width.rolling(20).mean() * 0.8

        # Bearish squeeze signal
        touch_upper = df['High'] >= upper
        price_decline = df['Close'] < df['Open']

        df[signal_col] = (squeeze & touch_upper & price_decline).astype(int)
        return df

    def _fibonacci_retracement_breaks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fibonacci retracement breaks - price breaks below key Fibonacci levels."""
        signal_col = 'fibonacci_retracement_breaks_signal'

        # Calculate recent swing high and low (simplified)
        swing_high = df['High'].rolling(20).max()
        swing_low = df['Low'].rolling(20).min()
        range_size = swing_high - swing_low

        # Fibonacci levels
        fib_236 = swing_high - (range_size * 0.236)
        fib_382 = swing_high - (range_size * 0.382)
        fib_500 = swing_high - (range_size * 0.5)
        fib_618 = swing_high - (range_size * 0.618)

        # Break below Fibonacci levels
        break_236 = df['Low'] <= fib_236
        break_382 = df['Low'] <= fib_382
        break_500 = df['Low'] <= fib_500
        break_618 = df['Low'] <= fib_618

        df[signal_col] = (break_236 | break_382 | break_500 | break_618).astype(int)
        return df

    def _momentum_divergence_bearish(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bearish momentum divergence - price up, momentum down."""
        signal_col = 'momentum_divergence_bearish_signal'

        # Calculate momentum indicators
        roc = talib.ROC(df['Close'], timeperiod=10)  # Rate of change
        mom = talib.MOM(df['Close'], timeperiod=10)  # Momentum

        # Find divergences
        price_up = df['Close'] > df['Close'].shift(10)
        momentum_down = mom < mom.shift(10)

        df[signal_col] = (price_up & momentum_down).astype(int)
        return df
