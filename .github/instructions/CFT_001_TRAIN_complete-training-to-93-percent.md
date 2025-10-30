# Complete Training to 93% Accuracy

---
scope: coding
audience: developers
priority: critical
status: todo
dependencies: []
---

## Definition of Done
- [ ] Achieve consistent 93% directional accuracy on EURUSD
- [ ] Achieve consistent 93% directional accuracy on XAUUSD  
- [ ] Complete automated training loop until target reached
- [ ] Generate final model artifacts (.joblib files)
- [ ] Validate accuracy with backtest results
- [ ] Document final model performance metrics

## Context
Based on TRAINING_READINESS_SUMMARY.md and current repo state, all infrastructure is ready for training. The system has enterprise-level regularization, Holloway Algorithm integration (400+ conditions), fundamental data integration, and robust training pipelines. We need to execute the final training runs to achieve the 93% accuracy target.

## Requirements
### MVP (Must Have)
- Environment loading patch applied to automated_training.py
- FRED API key properly loaded from .env
- Fundamental data schema correctly formatted (date, value columns)
- Training runs until 93% accuracy achieved
- Model artifacts saved to models/ directory

### Future Enhancements (Nice to Have)
- Cross-pair validation
- Multi-timeframe ensemble optimization
- Real-time monitoring dashboard

## Implementation Steps

### Step 1: Apply Environment Loading Fix
```bash
# Add environment loading to automated_training.py
# Insert after existing imports:
from pathlib import Path
from dotenv import load_dotenv

# Add before main() function:
def ensure_environment_loaded():
    env_paths = [
        Path(".env"),
        Path("../.env"), 
        Path(os.getcwd()) / ".env",
        Path(os.environ.get('APP_ROOT', os.getcwd())) / ".env"
    ]
    
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path, override=True)
            logger.info(f"Loaded environment from: {env_path}")
            break
    
    fred_key = os.getenv('FRED_API_KEY')
    if fred_key:
        logger.info(f"✅ FRED API key loaded: {fred_key[:8]}...")
        return True
    else:
        logger.warning("❌ FRED_API_KEY not found in environment")
        return False

# Add this call in main() after parser.parse_args():
ensure_environment_loaded()
```

### Step 2: Verify Data Schema
```bash
# Run fundamental data schema check
python restore_fundamental_backups.py

# Verify schema is correct
python -c "
import pandas as pd
df = pd.read_csv('data/DGS10.csv')
print('Columns:', list(df.columns))
print('First 3 rows:')
print(df.head(3))
assert 'date' in df.columns, 'Missing date column'
print('✅ Schema looks good')
"
```

### Step 3: Run Training Diagnostic
```bash
# Verify complete system readiness
python training_diagnostic.py

# Expected output should show:
# ✅ Environment variables loaded
# ✅ All required data files present
# ✅ Schema validation passed
# ✅ Feature engineering works
# ✅ Ready for training
```

### Step 4: Execute Training Loop
```bash
# Start with conservative dry-run
python -m scripts.automated_training --pairs EURUSD --dry-run --dry-iterations 50 --na-threshold 0.3

# If successful, run full training with loop until 93%
python -m scripts.automated_training --pairs EURUSD XAUUSD --target 0.93 --max-iterations 100
```

### Step 5: Monitor and Iterate
```bash
# Monitor logs for progress
tail -f logs/automated_training_results.json

# Check current models
ls -la models/

# Verify accuracy in logs
grep -i "accuracy" logs/automated_training_results.json
```

### Step 6: Validate Final Models
```bash
# Run backtest on final models
python -c "
from daily_forex_signal_system import DailyForexSignalSystem
system = DailyForexSignalSystem(['EURUSD', 'XAUUSD'])
results = system.run_backtest(days=30)
print('Final accuracy:', results)
"
```

## Success Criteria
- **Primary Target**: Both EURUSD and XAUUSD achieve ≥93% directional accuracy
- **Model Files**: All required .joblib artifacts generated and saved
- **Backtesting**: Historical validation confirms accuracy targets
- **Reproducibility**: Training process documented and repeatable
- **Performance**: Training completes within reasonable time (≤4 hours)

## Testing Requirements
- [ ] Unit tests for environment loading function
- [ ] Integration test for complete training pipeline  
- [ ] End-to-end test validating model artifacts work in prediction
- [ ] Backtest validation against historical data

## Known Issues & Mitigation
### Issue 1: FRED API Key Loading
- **Status**: Ready to fix
- **Solution**: Environment loading patch in Step 1

### Issue 2: Fundamental Data Schema  
- **Status**: Fixed in recent commits (8 minutes ago)
- **Validation**: restore_fundamental_backups.py script available

### Issue 3: Sample Size in Dry Runs
- **Status**: Known workaround
- **Solution**: Use --dry-iterations 50+ instead of default 10

## Loop Strategy for 93% Target
The training will run in iterations until the target is reached:

1. **Iteration 1-10**: Baseline establishment with enterprise regularization
2. **Iteration 11-25**: Holloway Algorithm features fully integrated
3. **Iteration 26-50**: Fundamental data alignment and cross-pair features  
4. **Iteration 51+**: Fine-tuning until 93% achieved

If accuracy plateaus below 93%, the system will:
- Adjust regularization parameters automatically
- Increase feature engineering depth
- Apply advanced ensemble weighting
- Continue until target reached or max iterations

## Dependencies Check
✅ All prerequisites completed:
- [x] Data schema standardized and fixed
- [x] Enterprise regularization system implemented  
- [x] Holloway Algorithm (400+ conditions) integrated
- [x] Fundamental data pipeline working
- [x] Robust training infrastructure ready
- [x] Environment loading solution identified