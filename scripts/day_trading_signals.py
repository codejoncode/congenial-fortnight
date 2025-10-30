#!/usr/bin/env python3
"""
Day Trading Signals Module

This module implements high-frequency day-trading signal generation for forex trading.
Provides multiple intraday entry opportunities while maintaining strong edge through
multi-timeframe confirmation.

Signals implemented:
- H1 Breakout Pullbacks
- VWAP Reversion Signals
- EMA Ribbon Compression Breakouts
- MACD Zero-Cross Scalping
- Volume Spike Reversal
- RSI Mean Reversion
- Inside-Outside Bar Patterns
- Time-of-Day Momentum
- Correlation Divergence (EURUSD vs XAUUSD)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import talib

class DayTradingSignalGenerator:
    def __init__(self, atr_multiplier: float = 0.5, volume_spike_threshold: float = 2.0):
        self.atr_multiplier = atr_multiplier
        self.volume_spike_threshold = volume_spike_threshold
    
    def h1_breakout_pullbacks(self, df: pd.DataFrame, buffer_pips: float = 3) -> pd.DataFrame:
        """H1 Breakout Pullback Signals"""
        # Calculate prior H1 high/low
        df['h1_high_prev'] = df['High'].shift(1)
        df['h1_low_prev'] = df['Low'].shift(1)
        
        # Breakout detection
        df['breakout_up'] = df['Close'] > (df['h1_high_prev'] + buffer_pips/10000)
        df['breakout_down'] = df['Close'] < (df['h1_low_prev'] - buffer_pips/10000)
        
        # Pullback detection (next bar pulls back toward breakout level)
        df['pullback_up'] = (df['breakout_up'].shift(1) & 
                            (df['Low'] <= df['h1_high_prev'].shift(1)) &
                            (df['Close'] > df['h1_high_prev'].shift(1)))
        df['pullback_down'] = (df['breakout_down'].shift(1) & 
                              (df['High'] >= df['h1_low_prev'].shift(1)) &
                              (df['Close'] < df['h1_low_prev'].shift(1)))
        
        df['h1_breakout_signal'] = df['pullback_up'] * 1 + df['pullback_down'] * -1
        return df
    
    def vwap_reversion_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """VWAP Touch and Reversion Signals"""
        if 'Volume' not in df.columns:
            df['vwap_signal'] = 0
            return df
            
        # Calculate VWAP
        df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['vwap'] = (df['typical_price'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        
        # ATR for distance measurement
        df['atr_14'] = talib.ATR(df['High'], df['Low'], df['Close'], 14)
        
        # H4 trend direction (simplified)
        df['ema_50'] = talib.EMA(df['Close'], 50)
        df['h4_trend_up'] = df['Close'] > df['ema_50']
        
        # VWAP reversion conditions
        vwap_distance = abs(df['Close'] - df['vwap']) / df['atr_14']
        df['vwap_long'] = (df['Close'] < df['vwap']) & (vwap_distance > self.atr_multiplier) & df['h4_trend_up']
        df['vwap_short'] = (df['Close'] > df['vwap']) & (vwap_distance > self.atr_multiplier) & ~df['h4_trend_up']
        
        df['vwap_signal'] = df['vwap_long'] * 1 + df['vwap_short'] * -1
        return df
    
    def ema_ribbon_compression(self, df: pd.DataFrame) -> pd.DataFrame:
        """EMA Ribbon Compression Breakout Signals"""
        # EMA Ribbon
        for period in [8, 13, 21, 34]:
            df[f'ema_{period}'] = talib.EMA(df['Close'], period)
        
        # Ribbon compression (standard deviation of EMAs)
        ema_values = df[['ema_8', 'ema_13', 'ema_21', 'ema_34']].values
        df['ribbon_compression'] = np.std(ema_values, axis=1) / df['Close']
        
        # Breakout beyond outer bands
        df['ribbon_upper'] = df['ema_8'].rolling(20).max()
        df['ribbon_lower'] = df['ema_8'].rolling(20).min()
        
        compression_threshold = df['ribbon_compression'].rolling(50).quantile(0.2)
        df['compressed'] = df['ribbon_compression'] < compression_threshold
        
        df['ribbon_breakout_up'] = df['compressed'].shift(1) & (df['Close'] > df['ribbon_upper'].shift(1))
        df['ribbon_breakout_down'] = df['compressed'].shift(1) & (df['Close'] < df['ribbon_lower'].shift(1))
        
        df['ribbon_signal'] = df['ribbon_breakout_up'] * 1 + df['ribbon_breakout_down'] * -1
        return df
    
    def macd_zero_cross_scalps(self, df: pd.DataFrame, df_daily: pd.DataFrame) -> pd.DataFrame:
        """MACD Zero-Cross Scalping Signals (M15 with H4/Daily confirmation)"""
        # MACD calculation
        macd, signal, hist = talib.MACD(df['Close'])
        df['macd_hist'] = hist
        
        # Zero cross detection
        df['macd_cross_up'] = (df['macd_hist'] > 0) & (df['macd_hist'].shift(1) <= 0)
        df['macd_cross_down'] = (df['macd_hist'] < 0) & (df['macd_hist'].shift(1) >= 0)
        
        # Daily/H4 MACD alignment (simplified - assumes daily data aligned)
        daily_macd, daily_signal, daily_hist = talib.MACD(df_daily['Close'])
        df['daily_macd_bullish'] = daily_hist.iloc[-1] > 0  # Use latest daily signal
        
        # Aligned signals only
        df['macd_scalp_long'] = df['macd_cross_up'] & df['daily_macd_bullish']
        df['macd_scalp_short'] = df['macd_cross_down'] & ~df['daily_macd_bullish']
        
        df['macd_scalp_signal'] = df['macd_scalp_long'] * 1 + df['macd_scalp_short'] * -1
        return df
    
    def volume_spike_reversal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume Spike Reversal Signals"""
        if 'Volume' not in df.columns:
            df['volume_reversal_signal'] = 0
            return df
            
        # Volume spike detection
        df['volume_ma_20'] = df['Volume'].rolling(20).mean()
        df['volume_spike'] = df['Volume'] > (df['volume_ma_20'] * self.volume_spike_threshold)
        
        # Inside bar detection (high < prev_high and low > prev_low)
        df['inside_bar'] = ((df['High'] < df['High'].shift(1)) & 
                           (df['Low'] > df['Low'].shift(1)))
        
        # Direction of volume spike
        df['spike_direction'] = np.where(df['Close'] > df['Open'], 1, -1)
        
        # Reversal signal: volume spike followed by inside bar, trade opposite to spike direction
        df['volume_reversal'] = (df['volume_spike'].shift(1) & df['inside_bar'] & 
                                ~df['volume_spike'])  # Current bar not also spiking
        
        df['volume_reversal_signal'] = np.where(df['volume_reversal'], 
                                               -df['spike_direction'].shift(1), 0)
        return df
    
    def rsi_mean_reversion(self, df: pd.DataFrame) -> pd.DataFrame:
        """Intraday RSI Mean-Reversion Signals"""
        # RSI calculation
        df['rsi_14'] = talib.RSI(df['Close'], 14)
        
        # ATR for regime filter
        df['atr_20'] = talib.ATR(df['High'], df['Low'], df['Close'], 20)
        df['atr_regime'] = df['atr_20'] <= df['atr_20'].rolling(50).quantile(0.7)  # Low-moderate ATR
        
        # Mean reversion signals
        df['rsi_oversold'] = (df['rsi_14'] < 20) & df['atr_regime']
        df['rsi_overbought'] = (df['rsi_14'] > 80) & df['atr_regime']
        
        df['rsi_mean_reversion_signal'] = df['rsi_oversold'] * 1 + df['rsi_overbought'] * -1
        return df
    
    def inside_outside_bar_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inside-Outside Bar Pattern Signals"""
        # Inside bar detection
        df['inside_bar'] = ((df['High'] < df['High'].shift(1)) & 
                           (df['Low'] > df['Low'].shift(1)))
        
        # Outside bar breakout
        df['outside_up'] = (df['inside_bar'].shift(1) & 
                           (df['High'] > df['High'].shift(1)))
        df['outside_down'] = (df['inside_bar'].shift(1) & 
                             (df['Low'] < df['Low'].shift(1)))
        
        df['inside_outside_signal'] = df['outside_up'] * 1 + df['outside_down'] * -1
        return df
    
    def time_of_day_momentum(self, df: pd.DataFrame, lookback_days: int = 100) -> pd.DataFrame:
        """Time-of-Day Momentum Cycle Signals"""
        # Extract hour from timestamp
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
        else:
            df['hour'] = 12  # Default if no datetime
        
        # Calculate hourly returns
        df['hourly_return'] = df['Close'].pct_change()
        
        # Historical average return by hour
        hourly_stats = df.groupby('hour')['hourly_return'].agg(['mean', 'std']).reset_index()
        hourly_stats = hourly_stats.sort_values('mean', ascending=False)
        
        # Top 2 momentum hours
        top_hours = hourly_stats.head(2)['hour'].values
        
        # Signal generation for top momentum hours
        df['momentum_hour'] = df['hour'].isin(top_hours)
        df['momentum_direction'] = df['hour'].map(
            dict(zip(hourly_stats['hour'], np.where(hourly_stats['mean'] > 0, 1, -1)))
        )
        
        df['time_momentum_signal'] = np.where(df['momentum_hour'], df['momentum_direction'], 0)
        return df
    
    def correlation_divergence(self, df_eur: pd.DataFrame, df_xau: pd.DataFrame) -> pd.DataFrame:
        """Correlation Divergence Signals between EURUSD & XAUUSD"""
        # Align dataframes on timestamp
        merged = pd.merge(df_eur, df_xau, on='DATE', suffixes=('_eur', '_xau'))
        
        # Rolling correlation
        merged['correlation'] = merged['CLOSE_eur'].rolling(12).corr(merged['CLOSE_xau'])
        
        # Trend detection
        merged['eur_trend'] = merged['CLOSE_eur'] > merged['CLOSE_eur'].rolling(20).mean()
        merged['xau_trend'] = merged['CLOSE_xau'] > merged['CLOSE_xau'].rolling(20).mean()
        
        # Divergence signal
        low_correlation = merged['correlation'] < 0.3
        merged['correlation_signal'] = np.where(
            low_correlation & merged['eur_trend'], 1,  # Trade EUR if trending
            np.where(low_correlation & merged['xau_trend'], -1, 0)  # Trade XAU if trending
        )
        
        return merged
    
    def range_expansion(self, df: pd.DataFrame) -> pd.DataFrame:
        """Signal 9: Range Expansion - Breakouts on expanding ranges."""
        df = df.copy()

        # True range
        df['tr'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift(1)),
                abs(df['Low'] - df['Close'].shift(1))
            )
        )

        # Range expansion (current TR > average of last 5)
        df['range_ma'] = df['tr'].rolling(5).mean()
        df['range_expansion'] = df['tr'] > df['range_ma'] * 1.5

        # Breakout signals on expansion
        df['range_expansion_buy'] = (
            df['range_expansion'] &
            (df['Close'] > df['Close'].shift(1)) &
            (df['High'] > df['High'].shift(1))
        )

        df['range_expansion_sell'] = (
            df['range_expansion'] &
            (df['Close'] < df['Close'].shift(1)) &
            (df['Low'] < df['Low'].shift(1))
        )

        df['range_expansion_signal'] = df['range_expansion_buy'] * 1 + df['range_expansion_sell'] * -1
        return df
    
    def order_flow_imbalance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Signal 10: Order Flow Imbalance - Imbalance between buy/sell volume."""
        df = df.copy()

        if 'Volume' not in df.columns:
            df['order_flow_signal'] = 0
            return df

        # Simplified order flow (using close vs open direction)
        df['buy_volume'] = np.where(df['Close'] > df['Open'], df['Volume'], 0)
        df['sell_volume'] = np.where(df['Close'] < df['Open'], df['Volume'], 0)

        # Rolling imbalance
        df['buy_volume_ma'] = df['buy_volume'].rolling(10).mean()
        df['sell_volume_ma'] = df['sell_volume'].rolling(10).mean()

        df['order_flow_buy'] = (
            (df['buy_volume'] > df['sell_volume'] * 1.5) &
            (df['buy_volume_ma'] > df['sell_volume_ma'])
        )

        df['order_flow_sell'] = (
            (df['sell_volume'] > df['buy_volume'] * 1.5) &
            (df['sell_volume_ma'] > df['buy_volume_ma'])
        )

        df['order_flow_signal'] = df['order_flow_buy'] * 1 + df['order_flow_sell'] * -1
        return df
    
    def generate_all_signals(self, df: pd.DataFrame, df_daily: pd.DataFrame = None, 
                           df_pair2: pd.DataFrame = None) -> pd.DataFrame:
        """Generate all day trading signals"""
        
        # Apply all signal generators
        df = self.h1_breakout_pullbacks(df)
        df = self.vwap_reversion_signals(df)
        df = self.ema_ribbon_compression(df)
        
        if df_daily is not None:
            df = self.macd_zero_cross_scalps(df, df_daily)
        
        df = self.volume_spike_reversal(df)
        df = self.rsi_mean_reversion(df)
        df = self.inside_outside_bar_patterns(df)
        df = self.time_of_day_momentum(df)
        df = self.range_expansion(df)
        df = self.order_flow_imbalance(df)
        
        # Composite signal strength
        signal_columns = [col for col in df.columns if col.endswith('_signal')]
        df['composite_signal_strength'] = df[signal_columns].sum(axis=1)
        df['signal_count'] = (df[signal_columns] != 0).sum(axis=1)
        
        return df
