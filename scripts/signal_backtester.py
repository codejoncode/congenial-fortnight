"""
Intraday Signal Backtester
"""
import pandas as pd
import numpy as np

def backtest_signal(df, signal_col, take_profit=0.002, stop_loss=0.001):
    df = df.copy()
    df['trade_return'] = 0.0
    for i in range(1, len(df)):
        if df[signal_col].iloc[i-1] == 1:
            entry = df['close'].iloc[i-1]
            high = df['high'].iloc[i]
            low = df['low'].iloc[i]
            if high - entry >= take_profit:
                df['trade_return'].iloc[i] = take_profit
            elif entry - low >= stop_loss:
                df['trade_return'].iloc[i] = -stop_loss
            else:
                df['trade_return'].iloc[i] = df['close'].iloc[i] - entry
        elif df[signal_col].iloc[i-1] == -1:
            entry = df['close'].iloc[i-1]
            high = df['high'].iloc[i]
            low = df['low'].iloc[i]
            if entry - low >= take_profit:
                df['trade_return'].iloc[i] = take_profit
            elif high - entry >= stop_loss:
                df['trade_return'].iloc[i] = -stop_loss
            else:
                df['trade_return'].iloc[i] = entry - df['close'].iloc[i]
    df['cum_return'] = df['trade_return'].cumsum()
    return df
