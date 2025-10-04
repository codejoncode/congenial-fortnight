"""
Day Trading Signal Generator
Implements 10 classic and modern day trading signals for intraday forex data.
"""
import pandas as pd
import numpy as np

class DayTradingSignalGenerator:
    def __init__(self, df):
        self.df = df.copy()

    def moving_average_crossover(self, fast=9, slow=21):
        self.df['ma_fast'] = self.df['close'].rolling(fast).mean()
        self.df['ma_slow'] = self.df['close'].rolling(slow).mean()
        self.df['ma_cross_signal'] = np.where(self.df['ma_fast'] > self.df['ma_slow'], 1, -1)
        return self.df['ma_cross_signal']

    def rsi_signal(self, period=14, overbought=70, oversold=30):
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        self.df['rsi'] = rsi
        self.df['rsi_signal'] = np.where(rsi > overbought, -1, np.where(rsi < oversold, 1, 0))
        return self.df['rsi_signal']

    def bollinger_band_signal(self, period=20, num_std=2):
        ma = self.df['close'].rolling(period).mean()
        std = self.df['close'].rolling(period).std()
        upper = ma + num_std * std
        lower = ma - num_std * std
        self.df['bb_signal'] = np.where(self.df['close'] > upper, -1, np.where(self.df['close'] < lower, 1, 0))
        return self.df['bb_signal']

    def macd_signal(self, fast=12, slow=26, signal=9):
        ema_fast = self.df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = self.df['close'].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        self.df['macd_signal'] = np.where(macd > macd_signal, 1, -1)
        return self.df['macd_signal']

    def stochastic_signal(self, k_period=14, d_period=3):
        low_min = self.df['low'].rolling(k_period).min()
        high_max = self.df['high'].rolling(k_period).max()
        k = 100 * (self.df['close'] - low_min) / (high_max - low_min + 1e-9)
        d = k.rolling(d_period).mean()
        self.df['stoch_signal'] = np.where(k > 80, -1, np.where(k < 20, 1, 0))
        return self.df['stoch_signal']

    def atr_breakout_signal(self, period=14, multiplier=1.5):
        high_low = self.df['high'] - self.df['low']
        high_close = np.abs(self.df['high'] - self.df['close'].shift())
        low_close = np.abs(self.df['low'] - self.df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        self.df['atr_upper'] = self.df['close'].shift() + multiplier * atr
        self.df['atr_lower'] = self.df['close'].shift() - multiplier * atr
        self.df['atr_breakout_signal'] = np.where(self.df['close'] > self.df['atr_upper'], 1, np.where(self.df['close'] < self.df['atr_lower'], -1, 0))
        return self.df['atr_breakout_signal']

    def volume_spike_signal(self, period=20, spike_mult=2):
        avg_vol = self.df['volume'].rolling(period).mean()
        self.df['vol_spike_signal'] = np.where(self.df['volume'] > spike_mult * avg_vol, 1, 0)
        return self.df['vol_spike_signal']

    def engulfing_candle_signal(self):
        prev_open = self.df['open'].shift()
        prev_close = self.df['close'].shift()
        bull = (self.df['close'] > self.df['open']) & (prev_close < prev_open) & (self.df['open'] < prev_close) & (self.df['close'] > prev_open)
        bear = (self.df['close'] < self.df['open']) & (prev_close > prev_open) & (self.df['open'] > prev_close) & (self.df['close'] < prev_open)
        self.df['engulf_signal'] = np.where(bull, 1, np.where(bear, -1, 0))
        return self.df['engulf_signal']

    def doji_signal(self, threshold=0.1):
        body = np.abs(self.df['close'] - self.df['open'])
        range_ = self.df['high'] - self.df['low']
        self.df['doji_signal'] = np.where((body / (range_ + 1e-9)) < threshold, 1, 0)
        return self.df['doji_signal']

    def session_open_breakout_signal(self, session='London', window=3):
        # Placeholder: session logic to be implemented
        self.df['session_breakout_signal'] = 0
        return self.df['session_breakout_signal']

    def generate_all_signals(self):
        self.moving_average_crossover()
        self.rsi_signal()
        self.bollinger_band_signal()
        self.macd_signal()
        self.stochastic_signal()
        self.atr_breakout_signal()
        self.volume_spike_signal()
        self.engulfing_candle_signal()
        self.doji_signal()
        self.session_open_breakout_signal()
        return self.df
