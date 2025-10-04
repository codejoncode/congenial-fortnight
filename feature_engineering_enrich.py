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
base_file = 'data/EURUSD_Daily.csv'
df = pd.read_csv(base_file, parse_dates=['timestamp'])

# Standardize column names for downstream logic
df = df.rename(columns={
    'open': 'daily_open',
    'high': 'daily_high',
    'low': 'daily_low',
    'close': 'daily_close',
    'volume': 'daily_volume',
    'timestamp': 'date'
})


# --- Step 1: Yesterday’s Outcome and Return ---
if 'signal' in df.columns:
    df['prev_day_signal'] = df['signal'].shift(1).map({'bullish':1,'bearish':0})
else:
    # If no signal column, skip prev_day_signal
    df['prev_day_signal'] = np.nan
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



# --- Step 7: Advanced Feature Engineering ---
def rolling_high(df, col, window):
    return df[col].rolling(window).max()
def rolling_low(df, col, window):
    return df[col].rolling(window).min()

# Rolling highs/lows and distances
for w in [10, 20, 30, 50, 100]:
    df[f'high_{w}d'] = rolling_high(df, 'daily_high', w) if 'daily_high' in df.columns else np.nan
    df[f'low_{w}d'] = rolling_low(df, 'daily_low', w) if 'daily_low' in df.columns else np.nan
    df[f'distance_from_high_{w}d'] = (df['daily_close'] - df[f'high_{w}d']) / df[f'high_{w}d'] if 'daily_close' in df.columns else np.nan
    df[f'distance_from_low_{w}d'] = (df['daily_close'] - df[f'low_{w}d']) / df[f'low_{w}d'] if 'daily_close' in df.columns else np.nan

# Classical pivot points and Fibonacci retracements
if all(col in df.columns for col in ['daily_high', 'daily_low', 'daily_close']):
    df['pivot_point'] = (df['daily_high'] + df['daily_low'] + df['daily_close']) / 3
    df['fib_23_6'] = df['daily_high'] - 0.236 * (df['daily_high'] - df['daily_low'])
else:
    df['pivot_point'] = np.nan
    df['fib_23_6'] = np.nan

# Consecutive bull/bear streaks
if all(col in df.columns for col in ['daily_close', 'daily_open']):
    bull = (df['daily_close'] > df['daily_open']).astype(int)
    bear = (df['daily_close'] < df['daily_open']).astype(int)
    df['consecutive_bull_days'] = bull.groupby((bull != bull.shift()).cumsum()).cumsum()
    df['consecutive_bear_days'] = bear.groupby((bear != bear.shift()).cumsum()).cumsum()
else:
    df['consecutive_bull_days'] = np.nan
    df['consecutive_bear_days'] = np.nan

# Pascal reversal logic (3-day streak reversal)
# Improved: Mark a reversal if there was a 3+ day bull streak followed by a bear day, or 3+ day bear streak followed by a bull day
if all(col in df.columns for col in ['consecutive_bull_days', 'consecutive_bear_days', 'daily_close', 'daily_open']):
    # Bull streak reversal: 3+ bull days, then a bear day
    bull_reversal = (df['consecutive_bull_days'].shift(1) >= 3) & (df['daily_close'] < df['daily_open'])
    # Bear streak reversal: 3+ bear days, then a bull day
    bear_reversal = (df['consecutive_bear_days'].shift(1) >= 3) & (df['daily_close'] > df['daily_open'])
    df['reversal_prob_3d'] = (bull_reversal | bear_reversal).astype(int)
    df['pascal_reversal_signal'] = df['reversal_prob_3d']
else:
    df['reversal_prob_3d'] = np.nan
    df['pascal_reversal_signal'] = np.nan

# Z-score 10d
if 'daily_close' in df.columns:
    df['z_score_10d'] = (df['daily_close'] - df['daily_close'].rolling(10).mean()) / (df['daily_close'].rolling(10).std() + 1e-9)
else:
    df['z_score_10d'] = np.nan

# Bollinger Band position
if 'daily_close' in df.columns:
    ma20 = df['daily_close'].rolling(20).mean()
    std20 = df['daily_close'].rolling(20).std()
    df['bb_upper'] = ma20 + 2 * std20
    df['bb_lower'] = ma20 - 2 * std20
    df['bb_position'] = (df['daily_close'] - ma20) / (2 * std20 + 1e-9)
else:
    df['bb_position'] = np.nan

# ATR-based volatility regime
if all(col in df.columns for col in ['daily_high', 'daily_low', 'daily_close']):
    tr = df[['daily_high', 'daily_low', 'daily_close']].copy()
    tr['prev_close'] = tr['daily_close'].shift(1)
    tr['tr1'] = tr['daily_high'] - tr['daily_low']
    tr['tr2'] = (tr['daily_high'] - tr['prev_close']).abs()
    tr['tr3'] = (tr['daily_low'] - tr['prev_close']).abs()
    tr['tr'] = tr[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr_14'] = tr['tr'].rolling(14).mean()
    q33 = df['atr_14'].quantile(0.33)
    q66 = df['atr_14'].quantile(0.66)
    df['volatility_regime'] = pd.cut(df['atr_14'], bins=[-float('inf'), q33, q66, float('inf')], labels=['low','normal','high'])
else:
    df['volatility_regime'] = np.nan

# Holloway bull count max 10d (simulate if missing)
if 'holloway_bull_count' in df.columns:
    df['holloway_bull_count_max_10d'] = df['holloway_bull_count'].rolling(10).max()
else:
    # Simulate with rolling max of consecutive_bull_days as a placeholder
    df['holloway_bull_count_max_10d'] = df['consecutive_bull_days'].rolling(10).max() if 'consecutive_bull_days' in df.columns else np.nan

# Lagged outcome columns (simulate daily_direction if missing)
if 'daily_direction' not in df.columns:
    # Simulate as sign of daily return (1 for up, -1 for down, 0 for no change)
    if 'daily_close' in df.columns:
        df['daily_direction'] = np.sign(df['daily_close'] - df['daily_close'].shift(1))
    else:
        df['daily_direction'] = np.nan
# For lag k, day_outcome_lagk = daily_direction shifted -k (so that df['day_outcome_lag1'].shift(-1) == df['daily_direction'])
for lag in range(1, 11):
    col = f'day_outcome_lag{lag}'
    df[col] = df['daily_direction'].shift(lag)

# --- Step 8: Finalize and Export ---
# Ensure all required diagnostic columns exist
required_columns = [
    'high_10d', 'low_10d', 'distance_from_high_10d', 'pivot_point', 'fib_23_6',
    'consecutive_bull_days', 'consecutive_bear_days', 'reversal_prob_3d',
    'z_score_10d', 'bb_position', 'volatility_regime', 'holloway_bull_count_max_10d',
    'pascal_reversal_signal'
]
for col in required_columns:
    if col not in df.columns:
        df[col] = np.nan

critical = [f'{ind}_lag10' for ind in core_indicators if ind in df.columns]
df_final = df.dropna(subset=critical) if critical else df.copy()
out_file = 'EURUSD_multi_timeframe_enriched.csv'
df_final.to_csv(out_file, index=False)
print(f"✅ Enriched dataset saved: {out_file} ({len(df_final)} rows, {len(df_final.columns)} columns)")
