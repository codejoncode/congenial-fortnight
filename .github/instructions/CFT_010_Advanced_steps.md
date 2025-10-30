[//]: # (Append advanced diagnostics section at the end)

---

## Advanced Feature Engineering Diagnostic Checklist

- [x] Column Presence: Assert that every engineered feature exists in the final DataFrame.
- [x] No Look-Ahead Bias: Check that all lagged/rolling features use only past data.
- [x] No Critical NaNs: Assert that the percentage of missing values is below 5% for each critical feature.
- [x] Sufficient Variance: Assert that each feature has more than one unique value (not constant).
- [x] Alignment: Check that all features are aligned on the same date index, monotonic and unique.
- [x] Extreme Value Sanity: For features like Z-scores, streaks, and Pascal flags, check that their value ranges are plausible.

---

## Example Diagnostic Script (Python)

```python
import pandas as pd

# Load your final engineered dataset
df = pd.read_csv('EURUSD_multi_timeframe_enriched.csv', parse_dates=['Date'])

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
		assert (df[col].shift(-lag) == df['daily_direction']).all(skipna=True), f"Look-ahead bias in {col}"

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
assert df['Date'].is_monotonic_increasing and df['Date'].is_unique, "Date index misaligned!"

# 6. Extreme value sanity
assert df['z_score_10d'].abs().max() < 10, "Z-score out of plausible range"
assert df['consecutive_bull_days'].min() >= 0, "Negative streaks?"
assert set(df['reversal_prob_3d'].dropna().unique()).issubset({0, 1, True, False}), "Non-binary Pascal flag"

print("All advanced feature diagnostics passed!")
```
## What to Build (Summary of Features)

### 1. Directional Sequence Features
 - [x] Pascal reversal probability flags for 3, 5, 7+ day streaks.

- [x] 10, 20, 30, 50, 100-day highs/lows.
 [x] Add Pascal reversal probability flags for 3, 5, 7+ day streaks.
- [x] Distance from current price to those extremes.
- [x] Classical pivot points, support/resistance, Fibonacci retracements.

### 3. Holloway Count Extremes
- [x] Max/min bull and bear counts over 10, 20, 30, 50, 100 days.

- [x] Z-score calculations for 5, 10, 20, 50 days.
- [x] Bollinger Band position and extreme flags.

- [x] ATR-based volatility regime classification (low/normal/high).
- [x] Volatility expansion/contraction flags.

- [x] High-probability reversal signals when streaks and extremes align.

---

## Implementation Steps

 [x] Calculate rolling highs/lows for 10, 20, 30, 50, 100 days.
 [x] Calculate distance from current price to rolling highs/lows.
 [x] Compute classical pivot points, support/resistance, and Fibonacci retracements.
 [x] Compute consecutive bull/bear streaks.
- [x] Calculate rolling Z-scores for multiple periods.
- [x] Add Bollinger Band position and extreme flags.

- [x] Compute ATR-based volatility regime features.
- [x] Add Holloway count extremes.
- [x] Implement Pascal reversal confluence logic.

---

## Finalization
- [x] Validate all new features (no look-ahead bias, correct alignment).
- [x] Delete any old models.
- [x] Run training diagnostics and ensure all checks pass.
- [x] When all is complete, proceed to CFT_001 for training.

---

## Definition of Done
- [x] All features in the guide are implemented and validated.
- [x] Models are deleted and diagnostics pass.
- [x] Ready for CFT_001 training.