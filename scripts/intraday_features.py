"""
Intraday Feature Engineering for M15, M30, H1
"""
import pandas as pd
import numpy as np

def add_intraday_features(df):
    df = df.copy()
    # Example features
    df['return_1'] = df['close'].pct_change()
    df['range'] = df['high'] - df['low']
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour if 'timestamp' in df.columns else np.nan
    df['minute'] = pd.to_datetime(df['timestamp']).dt.minute if 'timestamp' in df.columns else np.nan
    # Add more engineered features as needed
    return df
