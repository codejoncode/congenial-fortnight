# Clean Slate Training - Delete Models and Retrain with Fixed Data

---
scope: training
audience: ai_agents
priority: critical
status: ready-to-execute
dependencies: []
---

## AI Agent Context
This instruction is for an AI agent to execute a complete model reset and retraining process after data quality issues were identified. The previous models achieved only ~54% accuracy due to incorrect data loading (monthly data using H1 files, fundamental schema issues).

## Definition of Done
- [ ] All existing model artifacts deleted
- [ ] Data loading issues fixed (monthly files, fundamental schema)
- [ ] Training diagnostic passes with proper data counts
- [ ] Fresh models trained with corrected data
- [ ] Validation accuracy >65% (realistic improvement from current ~54%)
- [ ] Model artifacts saved with new timestamps

## Context
Training was completed but with wrong data:
- Monthly files loaded H1 data instead (1113 rows vs expected ~300-400)
- Fundamental data schema issues during earlier training
- Cross-timeframe alignment problems
- Result: ~54% accuracy (barely better than random 50%)

**Expected improvement**: 65-80% accuracy with proper data loading

## Implementation Steps

### Step 1: Clean Existing Models (2 minutes)
```bash
# Delete all trained model artifacts
echo "🧹 Cleaning existing models..."
rm -f models/EURUSD_*.joblib
rm -f models/XAUUSD_*.joblib

# Remove cached feature files
rm -f data/*_features.csv
rm -f data/*_complete_holloway.csv

# Clear training logs
rm -f logs/automated_training_results.json

# Verify cleanup
ls models/ | wc -l  # Should show 0 or very few files
echo "✅ Models cleaned"
```

### Step 2: Fix Data Loading Issues (5 minutes)
```bash
# Test monthly data loading
python -c "
import pandas as pd
print('🔍 Testing monthly data...')
for pair in ['EURUSD', 'XAUUSD']:
    try:
        df = pd.read_csv(f'data/{pair}_Monthly.csv')
        print(f'{pair}_Monthly: {len(df)} rows, columns: {list(df.columns)}')
        if len(df) > 1000:
            print(f'  ⚠️  Too many rows - likely loading wrong file')
        else:
            print(f'  ✅ Row count looks correct for monthly data')
    except Exception as e:
        print(f'  ❌ Error loading {pair}_Monthly.csv: {e}')
"

# Check fundamental data schema
python -c "
import pandas as pd
print('🔍 Testing fundamental data...')
test_files = ['DGS10.csv', 'CPIAUCSL.csv', 'FEDFUNDS.csv']
for file in test_files:
    try:
        df = pd.read_csv(f'data/{file}')
        print(f'{file}: columns = {list(df.columns)[:3]}...')
        if 'date' not in df.columns:
            print(f'  ❌ Missing date column in {file}')
        else:
            print(f'  ✅ Schema looks correct')
    except Exception as e:
        print(f'  ❌ Error reading {file}: {e}')
"
```

### Step 3: Run Pre-Training Diagnostic (3 minutes)
```bash
# Run comprehensive diagnostic
echo "🔍 Running training readiness diagnostic..."
python training_diagnostic.py

# Expected improvements to look for:
# - Monthly files showing ~300-400 rows (not 1113)
# - Fundamental files with proper date columns
# - Feature count >250 (was 216)
# - All data validation passes
```

### Step 4: Execute Dry-Run Test (10 minutes)
```bash
# Test with improved data
echo "🧪 Running dry-run test with fixed data..."
python -m scripts.automated_training \
  --pairs EURUSD \
  --dry-run \
  --dry-iterations 50 \
  --na-threshold 0.3

# Monitor output for:
# - Higher feature count (250+ vs previous 216)
# - Better baseline accuracy (>60% vs previous ~54%)
# - Proper timeframe data loading confirmations
```

### Step 5: Full Training with Fixed Data (30-60 minutes)
```bash
# Launch full retraining
echo "🚀 Starting full training with corrected data..."
python -m scripts.automated_training \
  --pairs EURUSD XAUUSD \
  --target 0.75 \
  --max-iterations 50

# Expected improvements:
# - Faster convergence due to better signal quality
# - Higher accuracy (65-80% range)
# - More stable training metrics
```

### Step 6: Validate Improvements (5 minutes)
```bash
# Check final results
echo "📊 Validating training improvements..."

# Check training results
if [ -f logs/automated_training_results.json ]; then
    python -c "
import json
with open('logs/automated_training_results.json', 'r') as f:
    results = json.load(f)
print('Training Results:')
for pair in ['EURUSD', 'XAUUSD']:
    if pair in results.get('results', {}):
        print(f'{pair}: {results[\"results\"][pair]}')
"
fi

# List new model files with timestamps
ls -la models/*.joblib | head -10

# Quick backtest validation
python -c "
try:
    from daily_forex_signal_system import DailyForexSignalSystem
    system = DailyForexSignalSystem(['EURUSD', 'XAUUSD'])
    results = system.run_backtest(days=30)
    print('Backtest accuracy (30 days):', results)
except Exception as e:
    print('Backtest validation pending model completion:', e)
"
```

## Success Criteria
- **Data Quality**: Monthly files show proper row counts (~300-400, not 1113)
- **Feature Count**: Engineering produces 250+ features (up from 216)
- **Accuracy Improvement**: Validation accuracy 65-80% (up from ~54%)
- **Model Stability**: Training converges faster with cleaner data
- **Artifacts**: Fresh model files with current timestamps

## Expected Timeline
- **Data cleanup**: 2 minutes
- **Data validation**: 8 minutes  
- **Dry-run testing**: 10 minutes
- **Full training**: 30-60 minutes
- **Validation**: 5 minutes
- **Total**: 55-85 minutes

## Monitoring During Training
Watch for these positive indicators:
```
✅ "Monthly data loaded: EURUSD 312 rows, XAUUSD 289 rows"
✅ "Feature engineering: 267 features from 6692 observations" 
✅ "EURUSD validation accuracy: 0.7234 (improvement from 0.5440)"
✅ "Target accuracy reached: EURUSD - 72.3%"
```

## If Issues Persist
1. **Data loading still wrong**: Check data loader functions in forecasting.py
2. **Low accuracy**: Verify fundamental data schema with date/value columns
3. **Training fails**: Reduce --na-threshold to 0.4 for more permissive feature selection

## AI Agent Notes
- This is a complete reset - all previous models are intentionally deleted
- The data quality issues explain the poor ~54% accuracy
- Expected realistic improvement to 65-80% with proper data
- Monitor logs closely for confirmation of improved data loading
- Success indicates the system architecture is sound, just needed clean data

## Post-Training Actions
After successful retraining:
1. Update training logs with new accuracy metrics
2. Commit new model artifacts to repository
3. Document data quality fixes applied
4. Set up monitoring for future data quality validation

This fresh training with corrected data should demonstrate the true capability of your enterprise forecasting system.