"""
Candlestick Pattern Recognition using TA-Lib
"""
import pandas as pd
import talib

def add_candlestick_patterns(df):
    df = df.copy()
    df['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
    df['engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
    df['doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
    df['morning_star'] = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])
    df['evening_star'] = talib.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'])
    df['shooting_star'] = talib.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close'])
    df['harami'] = talib.CDLHARAMI(df['open'], df['high'], df['low'], df['close'])
    df['dark_cloud'] = talib.CDLDARKCLOUDCOVER(df['open'], df['high'], df['low'], df['close'])
    return df
