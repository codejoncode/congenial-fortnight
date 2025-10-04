#!/usr/bin/env python3
"""
Feature Engineering Script for Multi-Timeframe Forex Dataset
Implements CFT_007 and CFT_008 (except frontend/charting)
- Adds lagged features, rolling stats, momentum, trend, interaction terms
- Adds fundamental surprise features
- Exports enriched dataset for EURUSD (repeat for XAUUSD as needed)
"""
import pandas as pd
import numpy as np

# Load base multi-timeframe dataset (replace with your actual file)
base_file = 'data/EURUSD_Daily.csv'  # Use the correct aligned daily file
try:
    df = pd.read_csv(base_file, parse_dates=['date'])
except Exception:
    df = pd.read_csv(base_file)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

# --- Step 1: Yesterday’s Outcome and Return ---
if 'signal' in df.columns:
    df['prev_day_signal'] = df['signal'].shift(1).map({'bullish':1,'bearish':0})
if 'daily_close' in df.columns and 'daily_open' in df.columns:
    df['prev_day_return'] = (df['daily_close'] / df['daily_open']).shift(1) - 1

# --- Step 2: Last 10-Day Lag Features ---
core_indicators = [c for c in ['daily_RSI_14','H4_MACD_histogram','holloway_bull_count'] if c in df.columns]
for ind in core_indicators:
    for lag in range(1,11):
        df[f'{ind}_lag{lag}'] = df[ind].shift(lag)

# --- Step 3: Rolling-Window Highs, Lows, Mean, Std ---
windows = [30,100]
price_cols = [c for c in ['daily_high','daily_low'] if c in df.columns]
for w in windows:
    for col in price_cols:
        df[f'{col}_max_{w}d'] = df[col].rolling(w).max()
        df[f'{col}_min_{w}d'] = df[col].rolling(w).min()
    # indicator rolling stats
    for ind in ['daily_RSI_14','holloway_count_diff']:
        if ind in df.columns:
            df[f'{ind}_mean_{w}d'] = df[ind].rolling(w).mean()
            df[f'{ind}_std_{w}d'] = df[ind].rolling(w).std()

# --- Step 4: Momentum and Trend Slope ---
if 'daily_close' in df.columns:
    df['momentum_10d'] = df['daily_close'].pct_change(10)
    def slope(series):
        x = np.arange(len(series))
        return np.polyfit(x, series, 1)[0] if series.notnull().all() else np.nan
    df['trend_slope_30d'] = df['daily_close'].rolling(30).apply(slope, raw=False)

# --- Step 5: Interaction Terms ---
if 'daily_RSI_14' in df.columns and 'daily_RSI_14_mean_30d' in df.columns:
    df['RSI_ratio_30d'] = df['daily_RSI_14'] / df['daily_RSI_14_mean_30d']
if 'H4_RSI_14' in df.columns and 'daily_RSI_14' in df.columns:
    df['RSI_diff_H4_daily'] = df['H4_RSI_14'] - df['daily_RSI_14']

# --- Step 6: Fundamental Surprise Features ---
if 'fund_CPI_actual' in df.columns and 'fund_CPI_estimate' in df.columns:
    df['CPI_surprise'] = df['fund_CPI_actual'] - df['fund_CPI_estimate']
if 'fund_NFP_actual' in df.columns and 'fund_NFP_estimate' in df.columns:
    df['NFP_surprise'] = df['fund_NFP_actual'] - df['fund_NFP_estimate']
for w in windows:
    for col in ['CPI_surprise','NFP_surprise']:
        if col in df.columns:
            df[f'{col}_max_{w}d']  = df[col].rolling(w).max()
            df[f'{col}_min_{w}d']  = df[col].rolling(w).min()
            df[f'{col}_mean_{w}d'] = df[col].rolling(w).mean()
            df[f'{col}_std_{w}d']  = df[col].rolling(w).std()
for col in ['CPI_surprise','NFP_surprise']:
    if col in df.columns:
        df[f'{col}_positive_count_10d'] = (df[col] > 0).rolling(10).sum()
        df[f'{col}_negative_count_10d'] = (df[col] < 0).rolling(10).sum()
        df[f'{col}_events_count_10d']   = df[col].rolling(10).count()
        for lag in range(1,11):
            df[f'{col}_lag{lag}'] = df[col].shift(lag)

# --- Step 7: Finalize and Export ---
# Drop rows with NaNs in critical features (be conservative)
critical = ['prev_day_signal'] + [f'{ind}_lag10' for ind in core_indicators if ind in df.columns]
df_final = df.dropna(subset=critical)
# Save enriched dataset
out_file = 'EURUSD_multi_timeframe_enriched.csv'
df_final.to_csv(out_file, index=False)
print(f"✅ Enriched dataset saved: {out_file} ({len(df_final)} rows, {len(df_final.columns)} columns)")
