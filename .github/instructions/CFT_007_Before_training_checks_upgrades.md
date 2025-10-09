Please now consider what other fields can we add to the model?  yesterdays outcome bullish or bearish, yesterdays  features,  maybe we want to analyze the last 10 days features ,  or high's and lows of x periods  100 days 30 days with last 10 days of features on the board. Each time we our analyzing we the most data. What do you think of this? 


CFT_007_Before_training_checks_upgrades.md

Adding **lagged outcomes** and **rolling-window statistics** can enrich your model’s context and improve predictive power. Here are the top candidates:

1. Yesterday’s Directional Outcome  
   -  `prev_day_signal`: Boolean – whether yesterday’s model decision was bullish (1) or bearish (0).  
   -  `prev_day_return`: Yesterday’s daily return (Close / Open – 1).  
   Why? Captures autocorrelation in daily direction and momentum.

2. Lagged Features (Last 10 Days)  
   For each core indicator (e.g., `daily_RSI_14`, `H4_MACD_histogram`, `holloway_bull_count`), include the last 10 lagged values:  
   ```
   daily_RSI_14_lag1, daily_RSI_14_lag2, … daily_RSI_14_lag10  
   H4_MACD_hist_lag1, … H4_MACD_hist_lag10  
   holloway_bull_count_lag1, … holloway_bull_count_lag10  
   ```
   Why? Enables the model to detect short-term persistence or mean-reversion patterns in features.

3. Rolling-Window Highs & Lows  
   Compute rolling max and min of price and key indicators over longer windows:  
   -  Price extremes:  
     – `high_30d`: Maximum Daily High over past 30 days  
     – `low_30d`: Minimum Daily Low over past 30 days  
     – `high_100d`, `low_100d` similarly for 100-day window  
   -  Indicator extremes:  
     – `RSI_30d_max`, `RSI_30d_min`  
     – `holloway_bull_count_30d_max`, `holloway_bear_count_30d_min`  
   Why? Captures macro regime boundaries and breakout thresholds.

4. Rolling-Window Statistics  
   For each indicator, add the 30- and 100-day rolling mean and standard deviation:  
   ```
   daily_RSI_14_mean_30d, daily_RSI_14_std_30d  
   H4_ATR_14_mean_100d, H4_ATR_14_std_100d  
   holloway_count_diff_mean_30d, holloway_count_diff_std_30d  
   ```
   Why? Reveals shifts in volatility and central tendency.

5. Momentum & Trend Persistence  
   -  10-day momentum: `momentum_10d = Close / Close_lag10 - 1`  
   -  30-day trend slope: Slope of linear regression on the last 30 daily closes  
   Why? Quantifies sustained price movements.

6. Interaction Terms  
   -  Ratio of today’s RSI to its 30-day mean: `RSI_ratio_30d = daily_RSI_14 / daily_RSI_14_mean_30d`  
   -  Divergence indicator: `RSI_diff = daily_RSI_14 - H4_RSI_14`  
   Why? Highlights inter-timeframe divergences and regime shifts.

**Implementation Notes for AI Agent**  
- Generate all lagged and rolling features on the H4-anchored timeline, aligning indices carefully to avoid look-ahead bias.  
- Use pandas’ `shift()` for lag features and `rolling()` for window statistics.  
- After feature creation, prune highly correlated or low-variance columns before training.  

Incorporating yesterday’s outcomes plus rich lagged and rolling-window features ensures your model sees both very recent short-term patterns and longer-term regime context—maximizing the information available at each decision point.

Below is a detailed set of instructions designed for your AI agent to systematically build, align, analyze, and enhance your multi-timeframe forex data model, ensuring comprehensive feature inclusion, historical consistency, and operational guidelines. These instructions are tailored to your setup as of October 2025 and incorporate all the aspects we've discussed.
AI Agent Instructions for Multi-Timeframe Forex Data Model Construction and Analysis
Objective:
Construct, verify, and optimize a comprehensive, aligned dataset for EURUSD and XAUUSD that integrates multi-timeframe technical indicators, fundamental data, pattern recognition, and cross-pair features. Enable detailed historical analysis, feature tracking, and decision-making to enhance predictive accuracy for daily forex predictions.
1. Data Acquisition & Alignment
a. Load Existing CSV Data Files:
Target Files:
data/EURUSD_Daily.csv
data/EURUSD_WWeekly.csv
data/EURUSD_Monthly.csv
data/XAUUSD_Daily.csv
data/XAUUSD_WWeekly.csv
data/XAUUSD_Monthly.csv
Operational Steps:
Verify each file exists and contains valid rows (use len(pd.read_csv())).
Confirm date columns are correctly parsed as datetime objects (pd.to_datetime()).
b. Critical Alignment:
Construct aligned daily rows:
Use Date/timestamp as key.
For H4 data, generate daily features by aggregating or selecting the last bar of each day.
Handle missing data:
For days missing H1 or H4 data, interpolate or fill forward with care to avoid look-ahead bias.
Exclude or flag days where fundamental or macro data are unavailable.
c. Establish Main Timeline:
Set Primary Timeline:
Use H4 timeframe as the backbone.
Generate a master daily DataFrame (full_df) with columns:
date (date only)
H4 indicators: open, high, low, close, RSI, MACD, ATR, pattern features, Holloway counts, slowdown flags, fundamental data, macro estimates.
Cross-pair features: XAUUSD_Close, XAUUSD_RSI, correlation measures.
Lagged features (see section 4).
d. Maintain Maximum History:
Ensure that all data up to 2000 (for H4) is included, filling missing yearly segments with limitations noted.
For hourly data: Import only if deep backtest at hourly level or auxiliary entry/exit triggers are needed; otherwise, focus on H4.
2. Feature Inclusion & Generation
a. Technical & Pattern Indicators:
Generate for each timeframe:
OHLC (open, high, low, close)
RSI(14), MACD, Bollinger Bands, ADX
Moving Averages (SMA & EMA 20, 50, 100, 200)
Candlestick pattern scores (>200)
Harmonic patterns and chart patterns (if enabled)
Holloway algorithm features: bull_count, bear_count, count_diff, count_ratio, bully, beary, slowdown, streak metrics, cross signals.
b. Macro & Fundamental Data:
Fetch latest macro data via free-tier API:
Use environment variables for API keys.
Track quota limits and cache responses.
For each day:
Record macro releases, estimates, and actuals.
Log event impact, surprise metrics, and forecast errors.
Add features:
fund_macro_event_today (Boolean)
fund_estimate, fund_actual, fund_surprise
Use last available estimate if actual is missing.
c. Cross-Pair Features:
For both pairs:
Include other pair’s close, RSI, trend direction, and divergence measures.
Correlation with past 10/30/100 days computed daily.
Pattern and openness of both pairs for multivariate analysis.
d. Lagged & Rolling-Window Features:
Create features on aligned daily data:
Previous 1–10 days’ indicators (lag1 to lag10)
30-day and 100-day rolling max, min, mean, std of price, RSI, Holloway counts.
Momentum (e.g., 10-day return), trend slopes.
e. Finalize Data:
Drop any columns with near-zero variance or multicollinearity.
Validate that all features are correctly aligned and consistent.
3. Data Validation & Diagnostics
a. Check Feature Completeness:
Verify no missing values in critical features (isnull().sum())
Confirm date ranges match across pair datasets and macro data.
b. Verify Historical Depth:
Confirm daily dataset length (~6,697 rows for EURUSD, ~5,476 for XAUUSD).
Ensure monthly data extends to 2000, not truncated at 2010 or later.
c. Confirm Feature Set:
Cross-reference model report: ensure all listed features (technical indicators, patterns, Holloway, fundamentals, macro signals, cross-pair features) appear in data.
If some features are missing, regenerate from raw/derived data.
4. Model Versioning & Data Management
a. Save Processed Dataset:
Export aligned, feature-rich dataset:
data/EURUSD_final_train.csv
data/XAUUSD_final_train.csv
Version control these files for different retraining cycles.
b. Track Data Quality:
Log the number of rows, missing values, feature correlations, and distributions.
Maintain a changelog for schema updates or feature alterations.
5. API & Data Fetch Workflow
a. Free-tier API Management:
Use environment variables (API_KEYS) for credentials.
Limit calls via batching or caching.
When quota is exhausted:
Log the event.
Fall back on the last known forecast/estimate.
Gracefully exit or pause data fetching and training to prevent quota violations.
b. Fundamentals & Estimates:
Fetch scheduled macro releases (e.g., NFP, CPI).
Update features on the main aligned dataset with recent estimates/actuals.
Log whether values are estimated or actual.
6. Decision & Execution Logics
a. Holloway Algorithm & Trigger States:
Calculate slowdown[T/F], bull_count, bear_count, bully, beary.
Log thresholds and trigger points.
Use these signals for subsequent prediction or as model inputs.
b. Entry/Exit Strategy:
Use H1 data only for execution triggers:
Confirm signals generated by H4/ daily bias.
Enter/exit based on intra-day patterns (e.g., RSI cross, pattern completion).
Final Notes:
Full, aligned historical data is critical for model robustness.
Feature completeness and correctness validation ensures no missing critical signals.
Regularly update macro/fundamental features for real-time relevance.
Maintain full transparency of feature states, thresholds, and signals for auditability and model improvement.
End of Instructions
By following these steps meticulously, your AI agent will generate a high-quality, comprehensive, and fully aligned dataset optimized for predictive modeling in forex markets, taking full advantage of multi-timeframe insights, fundamental data, and cross-pair relationships.

# CFT_007_ENRICH_FeatureEnhancement_AgentGuidelines

---
scope: data
audience: ai_agents
author: system
priority: high
status: ready-to-implement
dependencies: [CFT_006_H4_Multi_Timeframe_Implementation_Plan]
---

## AI Agent Context
Enhance the existing multi-timeframe dataset by adding lagged outcome features, rolling-window statistics, and interaction terms. This will provide richer temporal context (recent history and regime extremes) for improved predictive performance.

## Definition of Done
- [ ] Yesterday’s outcome and return features added
- [ ] Last 10-day lag features for core indicators
- [ ] Rolling-window high/low & mean/std stats for 30- and 100-day windows
- [ ] Momentum and trend slope features computed
- [ ] Interaction terms (indicator ratios, divergences) generated
- [ ] New features validated and merged into final CSVs
- [ ] Documentation updated with feature list and code snippets

## Implementation Steps

### Step 1: Parse Existing Multi-Timeframe CSV
(2 mins)
```python
import pandas as pd
# Load existing H4-aligned dataset
df = pd.read_csv('EURUSD_vs_XAUUSD_multi_timeframe.csv', parse_dates=['Date'])
```

### Step 2: Add Yesterday’s Outcome and Return
(3 mins)
```python
# Previous day bullish/bearish label
df['prev_day_signal'] = df['signal'].shift(1).map({'bullish':1,'bearish':0})
# Previous day return
df['prev_day_return'] = (df['daily_close'] / df['daily_open']).shift(1) - 1
```

### Step 3: Create Last 10-Day Lag Features
(5 mins)
```python
core_indicators = ['daily_RSI_14','H4_MACD_histogram','holloway_bull_count']
for ind in core_indicators:
    for lag in range(1,11):
        df[f'{ind}_lag{lag}'] = df[ind].shift(lag)
```

### Step 4: Compute Rolling-Window Highs, Lows, Mean, Std
(5 mins)
```python
windows = [30,100]
price_cols = ['daily_high','daily_low']
for w in windows:
    for col in price_cols:
        df[f'{col}_max_{w}d'] = df[col].rolling(w).max()
        df[f'{col}_min_{w}d'] = df[col].rolling(w).min()
    # indicator rolling stats
def add_rolling_stats(col):
    for w in windows:
        df[f'{col}_mean_{w}d'] = df[col].rolling(w).mean()
        df[f'{col}_std_{w}d'] = df[col].rolling(w).std()
for ind in ['daily_RSI_14','holloway_count_diff']:
    add_rolling_stats(ind)
```

### Step 5: Compute Momentum and Trend Slope
(5 mins)
```python
# 10-day momentum
df['momentum_10d'] = df['daily_close'].pct_change(10)
# 30-day trend slope via linear regression
def slope(series):
    x = np.arange(len(series))
    return np.polyfit(x, series, 1)[0]
df['trend_slope_30d'] = df['daily_close'].rolling(30).apply(slope, raw=True)
```

### Step 6: Generate Interaction Terms
(3 mins)
```python
# RSI ratio to rolling mean
df['RSI_ratio_30d'] = df['daily_RSI_14'] / df['daily_RSI_14_mean_30d']
# Divergence between timeframes
df['RSI_diff_H4_daily'] = df['H4_RSI_14'] - df['daily_RSI_14']
```

### Step 7: Finalize and Export
(2 mins)
```python
# Drop rows with NaNs in critical features
df_final = df.dropna(subset=['prev_day_signal'] + [f'{ind}_lag10' for ind in core_indicators])
# Save enriched dataset
df_final.to_csv('EURUSD_multi_timeframe_enriched.csv', index=False)
```

## Success Criteria
- Enriched CSV contains >330 additional features
- No look-ahead bias (all features use past data only)
- Row count remains within 5% of original after pruning NaNs
- Agent documentation updated with code snippets and feature descriptions

## AI Agent Notes
- Ensure rolling computations align on the **current date** and use past data only.  
- Validate each new column’s non-null percentage; drop features with >50% NaNs.  
- Update `congenial_fortnight_features.txt` to include newly added fields.  
- Archive intermediate files to `archive/` upon successful run.

---
**End of Instructions for AI Agent**

Does this plan consider the last period fundamentals? high's lows over x periods  as well so we can forecast as we make decesions?

The current plan focuses on price and indicator extremes, but you’re right—you can gain extra edge by summarizing **fundamental surprises** and **estimate deviations** over look-back windows. To incorporate that:

1. Fundamental Surprise Highs/Lows  
   -  For each core macro series (CPI, NFP, Fed funds rate), compute:  
     – `fund_CPI_surprise_max_30d`: maximum (actual–forecast) over past 30 days  
     – `fund_CPI_surprise_min_30d`  
     – `fund_NFP_surprise_max_30d`, `fund_NFP_surprise_min_30d`  
     – Same for 100-day windows: `fund_*_surprise_max_100d`, etc.  
   -  These capture regime shifts in economic surprises that often precede market inflections.

2. Rolling Mean & Std of Surprise  
   -  `fund_CPI_surprise_mean_30d`, `fund_CPI_surprise_std_30d`  
   -  `fund_NFP_surprise_mean_30d`, `fund_NFP_surprise_std_30d`  
   -  And for 100-day windows.

3. Cumulative Surprise Count  
   -  `fund_high_impact_events_10d`: Count of high-impact releases in last 10 days.  
   -  `fund_positive_surprises_10d`: Count of surprise > 0.  
   -  `fund_negative_surprises_10d`: Count of surprise < 0.

4. Lagged Fundamental Features  
   -  `fund_CPI_surprise_lag1…lag10`  
   -  `fund_fed_rate_lag1…lag10`

Implement these the same as price indicators—using `shift()` and `rolling()`. This ensures your model not only sees raw surprises but also their **history, extremes, and volatility**, enriching its ability to forecast based on both technical and economic regime changes.

Based on the repository analysis, I can see this is a daily forex signal system with the following key components: Current Repository Structure:
Multiple data files for EURUSD and XAUUSD (H1, H4, Daily, Weekly, Monthly timeframes)
Fundamental data (FRED API integration with various economic indicators)
Training infrastructure with ensemble models
Holloway algorithm implementation
Feature engineering pipeline
Key Files Identified:
Price data: EURUSD_.csv, XAUUSD_.csv files for different timeframes
Fundamental data: Various economic indicators (DGS10.csv, VIXCLS.csv, etc.)
Core modules: forecasting.py, daily_forex_signal_system.py
Training scripts: automated_training.py
Configuration and environment files
Current Implementation Status:
Has multi-timeframe data structure
Includes fundamental data integration
Features Holloway algorithm with bull/bear counts
Has training pipeline with ensemble methods
Uses H1 data back to 2009, H4 data back to 2000
The repository shows a sophisticated forex prediction system that needs to be enhanced with the multi-timeframe cross-pair model we discussed, incorporating lagged features, rolling statistics, and fundamental analysis enhancements.
</tool_result> <tool_call>
<invoke name="create_text_file">
<parameter name="filename">CFT_008_AGENT_complete-multi-timeframe-implementation.md</parameter>
<parameter name="content"># CFT_008_AGENT Complete Multi-Timeframe Cross-Pair Implementation
scope: data-engineering
audience: ai-agents
priority: critical
dependencies: []
estimated_duration: 4-6 hours
target_accuracy: 75-85%
Mission Statement
Transform the current daily forex signal system into a comprehensive multi-timeframe cross-pair prediction engine with enhanced lagged features, rolling statistics, and fundamental analysis capabilities.
Pre-Implementation Checklist
 Verify H4 data availability (2000-2025) vs H1 data (2009-2025)
 Confirm fundamental data schema (date, value columns)
 Backup existing models before deletion
 Ensure FRED API key is loaded in environment
Architecture Overview
Core Decision: H4-Primary Timeline
Primary Timeframe: H4 (4-hour bars) - extends back to 2000
Secondary Timeframes: Daily, Weekly, Monthly
Execution Timeframe: H1 (for entry/exit timing only)
Training Data: 6,697 aligned daily rows for EURUSD, 5,476 for XAUUSD
Complete Feature Implementation
1. Core Multi-Timeframe Features (Per Pair)
H4 Features (Primary Intraday)

# Price Features
h4_open, h4_high, h4_low, h4_close
h4_range = h4_high - h4_low
h4_body_size = abs(h4_close - h4_open)
h4_upper_shadow = h4_high - max(h4_open, h4_close)
h4_lower_shadow = min(h4_open, h4_close) - h4_low

# Technical Indicators
h4_rsi_14, h4_rsi_21
h4_macd, h4_macd_signal, h4_macd_histogram
h4_atr_14
h4_sma_20, h4_sma_50, h4_ema_20, h4_ema_50
h4_bb_upper, h4_bb_middle, h4_bb_lower
h4_adx_14, h4_momentum_10

# Threshold Flags
h4_rsi_oversold = h4_rsi_14 < 30
h4_rsi_overbought = h4_rsi_14 > 70
h4_rsi_bull_cross = h4_rsi_14 > 50


# CFT_008_FundamentalSurprise_FeatureGuidelines

---
scope: data
audience: ai_agents
author: system
priority: medium
status: ready-to-implement
dependencies: [CFT_007_ENRICH_FeatureEnhancement_AgentGuidelines]
---

## AI Agent Context
Incorporate **fundamental surprise** features to enable the model to learn how sequences of estimate vs actual deviations influence future directional outcomes. These augment technical and price-based features with economic regime context.

## Definition of Done
- [ ] Fundamental surprise max/min over 30d & 100d windows added
- [ ] Rolling mean & std of surprise over 30d & 100d added
- [ ] Cumulative surprise counts over last 10 days added
- [ ] Lagged fundamental surprise features (lag1–lag10) added
- [ ] All features validated and no lookahead bias present

## Feature Engineering Steps

### Step 1: Load Enriched Dataset
```python
import pandas as pd
# Load existing enriched dataset
df = pd.read_csv('EURUSD_multi_timeframe_enriched.csv', parse_dates=['Date'])
```

### Step 2: Compute Daily Surprise Values
```python
# Surprise = actual - estimate for key events
df['CPI_surprise'] = df['fund_CPI_actual'] - df['fund_CPI_estimate']
df['NFP_surprise'] = df['fund_NFP_actual'] - df['fund_NFP_estimate']
# Additional fundamental series as needed
```

### Step 3: Rolling-Window Extremes & Stats (30d & 100d)
```python
windows = [30,100]
for w in windows:
    for col in ['CPI_surprise','NFP_surprise']:
        df[f'{col}_max_{w}d']  = df[col].rolling(w).max()
        df[f'{col}_min_{w}d']  = df[col].rolling(w).min()
        df[f'{col}_mean_{w}d'] = df[col].rolling(w).mean()
        df[f'{col}_std_{w}d']  = df[col].rolling(w).std()
```

### Step 4: Cumulative Surprise Counts (Last 10 Days)
```python
# Count positive/negative surprises in last 10 days
for col in ['CPI_surprise','NFP_surprise']:
    df[f'{col}_positive_count_10d'] = (df[col] > 0).rolling(10).sum()
    df[f'{col}_negative_count_10d'] = (df[col] < 0).rolling(10).sum()
    df[f'{col}_events_count_10d']   = df[col].rolling(10).count()
```

### Step 5: Lagged Surprise Features (lag1–lag10)
```python
fund_cols = ['CPI_surprise','NFP_surprise']
for col in fund_cols:
    for lag in range(1,11):
        df[f'{col}_lag{lag}'] = df[col].shift(lag)
```

### Step 6: Finalize & Export
```python
# Drop rows with NaNs beyond acceptable threshold
df_final = df.dropna(thresh=int(0.8*len(df.columns)))
df_final.to_csv('EURUSD_multi_timeframe_enriched_v2.csv', index=False)
```

## AI Agent Notes
- Validate that rolling windows do not introduce forward-looking data.  
- Use `shift(1)` appropriately for lag features.  
- Document any dropped rows and final row count.  
- Update feature registry list accordingly.

---
**End of Fundamental Surprise Feature Guidelines**

