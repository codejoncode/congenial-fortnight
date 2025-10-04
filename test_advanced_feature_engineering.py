# test_advanced_feature_engineering.py
"""
Advanced Feature Engineering Diagnostic Test
Validates presence, alignment, and quality of all engineered features before training.
"""
import pandas as pd

# Load your final engineered dataset
df = pd.read_csv('EURUSD_multi_timeframe_enriched.csv', parse_dates=['date'])

# 1. Column presence
required_columns = [
    'high_10d', 'low_10d', 'distance_from_high_10d', 'pivot_point', 'fib_23_6',
    'consecutive_bull_days', 'consecutive_bear_days', 'reversal_prob_3d',
    'z_score_10d', 'bb_position', 'volatility_regime', 'holloway_bull_count_max_10d',
    'pascal_reversal_signal', # ...add all other engineered features
]
missing = [col for col in required_columns if col not in df.columns]
assert not missing, f"Missing columns: {missing}"

# 2. No look-ahead bias (example for lagged features)
for lag in range(1, 11):
    col = f'day_outcome_lag{lag}'
    if col in df.columns:
        # Debug: print first 20 rows for lag 1
        if lag == 1:
            print('DEBUG: daily_direction vs day_outcome_lag1 (first 20 rows)')
            print(df[['daily_direction', 'day_outcome_lag1']].head(20))
            print('DEBUG: day_outcome_lag1.shift(-1) vs daily_direction (first 20 rows)')
            print(pd.DataFrame({
                'day_outcome_lag1.shift(-1)': df['day_outcome_lag1'].shift(-1).head(20),
                'daily_direction': df['daily_direction'].head(20)
            }))
        # Only check where both are not NaN
        mask = (~df[col].shift(-lag).isna()) & (~df['daily_direction'].isna())
        assert (df[col].shift(-lag)[mask] == df['daily_direction'][mask]).all(), f"Look-ahead bias in {col}"

# 3. No critical NaNs
for col in required_columns:
    if col in df.columns:
        null_pct = df[col].isnull().mean()
        assert null_pct < 0.05, f"Too many NaNs in {col}: {null_pct:.2%}"

# 4. Sufficient variance
for col in required_columns:
    if col in df.columns:
        assert df[col].nunique() > 1, f"Feature {col} is constant!"

# 5. Alignment
assert df['date'].is_monotonic_increasing and df['date'].is_unique, "Date index misaligned!"

# 6. Extreme value sanity
assert df['z_score_10d'].abs().max() < 10, "Z-score out of plausible range"
assert df['consecutive_bull_days'].min() >= 0, "Negative streaks?"
assert set(df['reversal_prob_3d'].dropna().unique()).issubset({0, 1, True, False}), "Non-binary Pascal flag"

print("All advanced feature diagnostics passed!")
