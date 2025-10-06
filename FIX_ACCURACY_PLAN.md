# IMMEDIATE FIXES FOR ACCURACY ISSUES

## 🎯 Goal: Get from 52% to 60%+ accuracy BEFORE adding more signals

## Fix #1: Remove Harmful Signals (IMMEDIATE)
These signals actively hurt performance - remove them now:
- `bearish_hammer_failures_signal` (47.57% accuracy, -13% correlation)
- `rsi_divergence_bearish_signal` (48.68% accuracy, -2.9% correlation)  
- `macd_bearish_crossovers_signal` (49.19% accuracy, -1.3% correlation)

**File:** `scripts/slump_signals.py`
**Action:** Comment out these 3 methods in `generate_all_signals()`

## Fix #2: Fix Day Trading Signals Column Overlap (IMMEDIATE)
**File:** `scripts/forecasting.py` line ~1155
**Change:**
```python
# BEFORE:
df = df.join(signals_only_df, how='left')

# AFTER:
df = df.join(signals_only_df, how='left', rsuffix='_day_trading')
```

## Fix #3: Add Feature Selection (HIGH PRIORITY)
**File:** `scripts/forecasting.py` in `_prepare_features()`
**Add after feature engineering:**
```python
# Feature selection - keep only features with variance
feature_cols = [c for c in df.columns if c not in ['target_1d', 'next_close_change']]
variance = df[feature_cols].var()
low_variance_cols = variance[variance < 0.0001].index.tolist()
df = df.drop(columns=low_variance_cols)
logger.info(f"Removed {len(low_variance_cols)} low-variance features")
```

## Fix #4: Check Target Variable Quality (HIGH PRIORITY)
**File:** `scripts/forecasting.py` in `_engineer_features()`
**Add target validation:**
```python
# After target creation, add:
logger.info(f"Target distribution: {df['target_1d'].value_counts().to_dict()}")
logger.info(f"Target balance: {df['target_1d'].mean():.3f}")
if df['target_1d'].isna().sum() > 0:
    logger.warning(f"Target has {df['target_1d'].isna().sum()} NaN values!")
```

## Fix #5: Handle Missing Fundamentals (MEDIUM PRIORITY)
**File:** `scripts/forecasting.py` in `_add_fundamental_features()`
**After merging fundamentals:**
```python
# Forward-fill fundamental data (release schedules are infrequent)
fund_cols = [c for c in df.columns if c.startswith('fund_')]
df[fund_cols] = df[fund_cols].fillna(method='ffill').fillna(0)
logger.info(f"Forward-filled {len(fund_cols)} fundamental features")
```

## Fix #6: Remove Duplicate Features (MEDIUM PRIORITY)
Many Holloway features appear multiple times with different prefixes.
**File:** `scripts/forecasting.py` in `_prepare_features()`
**Add deduplication:**
```python
# Remove duplicate columns
df = df.loc[:, ~df.columns.duplicated()]
logger.info(f"Removed duplicate columns, now have {len(df.columns)} features")
```

## Fix #7: Add Feature Importance Logging (MEDIUM PRIORITY)
**File:** `scripts/robust_lightgbm_config.py`
**After training, add:**
```python
# Log feature importance
importance_df = pd.DataFrame({
    'feature': model.feature_name_,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
importance_df.to_csv(f'{pair}_feature_importance.csv', index=False)
logger.info(f"Top 10 features: {importance_df.head(10)['feature'].tolist()}")
```

## Testing Checklist
After each fix:
1. [ ] Run `python test_automated_training.py`
2. [ ] Check `EURUSD_signal_evaluation.csv` for improvements
3. [ ] Verify no ERROR logs
4. [ ] Check accuracy increased
5. [ ] Check mean signal accuracy increased

## Expected Results After All Fixes
- Validation accuracy: 58-62%
- Test accuracy: 60-65%
- Mean signal accuracy: > 52%
- At least 20 signals with accuracy > 53%
- No column overlap errors
- No NaN-related errors

## Order of Implementation
1. Fix #2 (column overlap) - 2 minutes
2. Fix #1 (remove bad signals) - 3 minutes
3. Fix #6 (deduplicate) - 3 minutes
4. Fix #3 (feature selection) - 5 minutes
5. Fix #5 (missing values) - 5 minutes
6. Fix #4 (target validation) - 5 minutes
7. Fix #7 (importance logging) - 10 minutes

**Total time:** ~30-40 minutes to implement all fixes
**Expected improvement:** +8-10% accuracy
