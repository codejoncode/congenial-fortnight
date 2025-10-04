"""
Slump Signal Generator
Contrarian slump detection for forex time series.
"""
import pandas as pd
import numpy as np

def generate_slump_signals(df, window=10, threshold=-0.02):
    df = df.copy()
    df['slump_return'] = df['close'].pct_change(window)
    df['slump_signal'] = np.where(df['slump_return'] < threshold, 1, 0)
    return df['slump_signal']
