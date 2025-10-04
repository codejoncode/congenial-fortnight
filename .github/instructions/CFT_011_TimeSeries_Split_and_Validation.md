# Time-Series Split and Validation/Test Reporting for Model Training

This document describes the best-practice approach for splitting time-series data for model training, validation, and testing, as implemented in this repo.

## 1. Time-Series Split (70/15/15)
- **Training (0–70%)**: Fit all parameters and perform Bayesian hyperparameter searches.
- **Validation (70–85%)**: Tune thresholds (e.g., Pascal reversal counts, drift triggers) and select the best regime-specific models.
- **Test (85–100%)**: Evaluate final performance (directional accuracy, drawdown, Sharpe) on data the model has never seen, including the most recent market conditions.

**Python Example:**
```python
n = len(df)
train_end = int(0.70 * n)
valid_end = int(0.85 * n)
df_train = df.iloc[:train_end]
df_valid = df.iloc[train_end:valid_end]
df_test  = df.iloc[valid_end:]
```

## 2. Hook Splits Into Your Pipeline
- In the data-prep step (before LightGBM training), compute these indices and assign X_train, y_train, X_val, y_val, X_test, y_test.
- Pass early_stopping_rounds using (X_val, y_val) so you never peek at df_test until after training is complete.

## 3. Report Validation and Test Metrics Separately
- After training completes, evaluate both validation accuracy and test accuracy:
```python
val_acc  = model.score(X_val, y_val)
test_acc = model.score(X_test, y_test)
print(f"\u25B6\uFE0F Validation Accuracy: {val_acc:.2%}")
print(f"\u25B6\uFE0F Test       Accuracy: {test_acc:.2%}")
```
- Log directional accuracy, drawdown, and Sharpe on both sets so you can see any over-fitting between validation and test.

## 4. (Optional) Walk-Forward Windows
- For robustness, wrap steps 1–3 in a loop that advances the split forward by a fixed window (e.g., 1 month).
- Aggregate performance across all folds to estimate true out-of-sample performance.

## 5. Measure Your Realistic Range
- Your final test accuracies across folds will empirically define your “Low,” “Mid,” and “High” bands.
- If your aggregated fold accuracies sit, for example, between 68% and 75%, that becomes your observed mid-range.

By explicitly coding these splits and reporting both validation and test results, you ensure a clean separation between model tuning and true performance measurement. Only then can you credibly claim a 60–65%, 68–75%, or 75–85% accuracy band based on the actual data your repo contains.
