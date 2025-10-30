# Complete Signal Inventory - Pre-Training Documentation

**Generated**: October 7, 2025  
**Status**: Ready for Training  
**Purpose**: Complete catalog of all signal systems before training

---

## 🎯 Signal Systems Overview

### Total Signal Systems: 8
1. **Holloway Algorithm** (Original) - 49 features × 4 timeframes = 196 features
2. **Day Trading Signals** - 9 active signals
3. **Slump Signals** - 29 active signals (3 disabled)
4. **Harmonic Patterns** - Pattern detection signals
5. **Chart Patterns** - Classical chart pattern recognition
6. **Elliott Wave** - Wave pattern analysis
7. **Ultimate Signal Repository** - SMC + Order Flow signals
8. **Enhanced Holloway Next-Candle** - SEPARATE system (not in main training)

---

## 📊 System 1: Holloway Algorithm (Original)

**File**: `scripts/holloway_algorithm.py`  
**Class**: `CompleteHollowayAlgorithm`  
**Integration**: Core feature in main training pipeline  
**Status**: ✅ Active

### Features Generated: 49 per timeframe

#### Multi-Timeframe Analysis (4 timeframes):
1. **H4** (4-hour)
2. **Daily**
3. **Weekly**
4. **Monthly**

**Total Features**: 49 × 4 = **196 Holloway features**

### Feature Breakdown (per timeframe):

#### Bull/Bear Count Features (8):
1. `bull_count` - Sum of all bullish signals (1000+ possible)
2. `bear_count` - Sum of all bearish signals (1000+ possible)
3. `bull_count_avg` - Double EMA smoothed (period: 27)
4. `bear_count_avg` - Double EMA smoothed (period: 27)
5. `bull_minus_bear` - Net count difference
6. `bull_above_avg` - Boolean flag
7. `bear_above_avg` - Boolean flag
8. `dominant_signal` - Bull=1, Bear=0

#### Historical Analysis (9 weighted periods):
9-17. Price oscillation scores (weights: 3.0, 2.7, 2.4, 2.1, 1.8, 1.5, 1.2, 0.9, 0.6)

#### Resistance/Support Levels (2):
18. `resistance_level` - 95th percentile
19. `support_level` - 12th percentile

#### Trend Indicators (16 boolean flags):
20. `is_rising_trend`
21. `is_falling_trend`
22. `is_neutral_trend`
23. `is_extreme_high`
24. `is_extreme_low`
25. `bull_momentum`
26. `bear_momentum`
27. `trend_strength`
28. `volatility_high`
29. `consolidation`
30. `breakout_up`
31. `breakout_down`
32. `reversal_up`
33. `reversal_down`
34. `acceleration_bull`
35. `acceleration_bear`

#### Pattern Counts (14):
36. `rise_pattern_count`
37. `fall_pattern_count`
38. `neutral_pattern_count`
39. `extreme_high_count`
40. `extreme_low_count`
41. `consecutive_bull`
42. `consecutive_bear`
43. `alternating_count`
44. `momentum_change`
45. `trend_change_frequency`
46. `volatility_score`
47. `consolidation_periods`
48. `breakout_frequency`
49. `reversal_frequency`

### Signal Calculation Logic:

**1000+ Holloway Signal Combinations**:
- Price vs 24 Moving Averages (12 EMA + 12 SMA)
- EMA alignment (132 combinations)
- SMA alignment (132 combinations)
- EMA vs SMA crosses (288 combinations)
- Fresh crossovers (48 + 288)

**Periods Used**: 5, 7, 10, 14, 20, 28, 50, 56, 100, 112, 200, 225

---

## 📊 System 2: Day Trading Signals

**File**: `scripts/day_trading_signals.py`  
**Class**: `DayTradingSignalGenerator`  
**Integration**: Core feature in main training pipeline  
**Status**: ✅ Active  
**Suffix**: `_day_trading` (to prevent column conflicts)

### Active Signals: 9

#### Signal 1: H1 Breakout Pullbacks
**Name**: `h1_breakout_pullbacks_signal_day_trading`  
**Logic**: 
- Detects H1 breakout followed by pullback to support
- Measures volume confirmation
- Validates breakout strength
**Output**: 0 (bearish), 1 (bullish)

#### Signal 2: VWAP Reversion
**Name**: `vwap_reversion_signal_day_trading`  
**Logic**:
- Calculates Volume Weighted Average Price
- Detects price deviation from VWAP
- Mean reversion when price > 2 std devs from VWAP
**Output**: 0 (bearish), 1 (bullish)

#### Signal 3: EMA Ribbon Compression
**Name**: `ema_ribbon_compression_signal_day_trading`  
**Logic**:
- Monitors 8/13/21 EMA spacing
- Detects compression (tight spacing)
- Predicts explosive move when compressed
**Output**: 0 (bearish), 1 (bullish)

#### Signal 4: MACD Zero Cross Scalps
**Name**: `macd_zero_cross_scalps_signal_day_trading`  
**Logic**:
- MACD line crossing zero line
- Histogram confirmation
- Signal line alignment
**Output**: 0 (bearish), 1 (bullish)

#### Signal 5: Volume Spike Reversal
**Name**: `volume_spike_reversal_signal_day_trading`  
**Logic**:
- Detects volume spikes (>2× avg)
- Checks for reversal candle patterns
- Confirms with momentum divergence
**Output**: 0 (bearish), 1 (bullish)

#### Signal 6: RSI Mean Reversion
**Name**: `rsi_mean_reversion_signal_day_trading`  
**Logic**:
- RSI extremes (>70 or <30)
- Divergence with price
- Reversal confirmation
**Output**: 0 (bearish), 1 (bullish)

#### Signal 7: Inside/Outside Bar Patterns
**Name**: `inside_outside_bar_patterns_signal_day_trading`  
**Logic**:
- Inside bar: High/low within previous bar
- Outside bar: High/low outside previous bar
- Breakout direction after inside bar
**Output**: 0 (bearish), 1 (bullish)

#### Signal 8: Time of Day Momentum
**Name**: `time_of_day_momentum_signal_day_trading`  
**Logic**:
- London open (08:00 GMT) momentum
- New York open (13:00 GMT) momentum
- Asian session fade patterns
**Output**: 0 (bearish), 1 (bullish)

#### Signal 9: Range Expansion
**Name**: `range_expansion_signal_day_trading`  
**Logic**:
- Detects narrowing range (consolidation)
- Measures ATR compression
- Predicts expansion direction
**Output**: 0 (bearish), 1 (bullish)

---

## 📊 System 3: Slump Signals

**File**: `scripts/slump_signals.py`  
**Class**: `SlumpSignalEngine`  
**Integration**: Core feature in main training pipeline  
**Status**: ✅ 29 Active, ❌ 3 Disabled

### Active Signals: 29

#### Signal 1: Bearish Engulfing Patterns
**Name**: `bearish_engulfing_signal`  
**Logic**: Current candle body completely engulfs previous bullish candle  
**Accuracy**: 52.4%  
**Status**: ✅ Active

#### Signal 2: Shooting Star Rejections
**Name**: `shooting_star_signal`  
**Logic**: Small body at bottom, long upper wick (>2× body), at resistance  
**Accuracy**: 53.1%  
**Status**: ✅ Active

#### Signal 3: Volume Climax Declines
**Name**: `volume_climax_signal`  
**Logic**: Volume spike (>3× avg) + bearish candle + momentum exhaustion  
**Accuracy**: 54.2%  
**Status**: ✅ Active

#### Signal 4: Stochastic Bearish Signals
**Name**: `stochastic_bearish_signal`  
**Logic**: Stochastic %K crosses below %D in overbought zone (>80)  
**Accuracy**: 51.8%  
**Status**: ✅ Active

#### Signal 5: Bollinger Bearish Squeeze
**Name**: `bollinger_squeeze_bearish_signal`  
**Logic**: Price at upper band + squeeze (narrow bands) + breakdown  
**Accuracy**: 52.9%  
**Status**: ✅ Active

#### Signal 6: Fibonacci Retracement Breaks
**Name**: `fibonacci_break_signal`  
**Logic**: Price breaks key Fib level (0.618, 0.5, 0.382) with volume  
**Accuracy**: 53.7%  
**Status**: ✅ Active

#### Signal 7: Momentum Divergence Bearish
**Name**: `momentum_divergence_signal`  
**Logic**: Price makes higher high, momentum indicator makes lower high  
**Accuracy**: 54.5%  
**Status**: ✅ Active

#### Signal 8: Three Black Crows
**Name**: `three_black_crows_signal`  
**Logic**: Three consecutive bearish candles with lower closes  
**Accuracy**: 52.1%  
**Status**: ✅ Active

#### Signal 9: Dark Cloud Cover
**Name**: `dark_cloud_cover_signal`  
**Logic**: Bearish candle opens above previous close, closes below midpoint  
**Accuracy**: 51.6%  
**Status**: ✅ Active

#### Signal 10: Evening Star
**Name**: `evening_star_signal`  
**Logic**: Bullish + doji/small + bearish (reversal pattern)  
**Accuracy**: 53.3%  
**Status**: ✅ Active

#### Signal 11: Bearish Harami
**Name**: `bearish_harami_signal`  
**Logic**: Small bearish candle contained within previous large bullish  
**Accuracy**: 50.8%  
**Status**: ✅ Active

#### Signal 12: Hanging Man
**Name**: `hanging_man_signal`  
**Logic**: Small body, long lower wick at top of uptrend  
**Accuracy**: 52.4%  
**Status**: ✅ Active

#### Signal 13: Bearish Belt Hold
**Name**: `bearish_belt_hold_signal`  
**Logic**: Opens at high, closes near low (strong bearish momentum)  
**Accuracy**: 51.9%  
**Status**: ✅ Active

#### Signal 14: Tweezer Top
**Name**: `tweezer_top_signal`  
**Logic**: Two candles with matching highs (resistance)  
**Accuracy**: 50.5%  
**Status**: ✅ Active

#### Signal 15: Bearish Kicker
**Name**: `bearish_kicker_signal`  
**Logic**: Gap down opening after bullish candle (strong reversal)  
**Accuracy**: 54.8%  
**Status**: ✅ Active

#### Signal 16: Triple Top
**Name**: `triple_top_signal`  
**Logic**: Three peaks at similar resistance level  
**Accuracy**: 53.6%  
**Status**: ✅ Active

#### Signal 17: Head and Shoulders
**Name**: `head_shoulders_signal`  
**Logic**: Left shoulder + head + right shoulder formation  
**Accuracy**: 55.2%  
**Status**: ✅ Active

#### Signal 18: Descending Triangle Break
**Name**: `descending_triangle_signal`  
**Logic**: Flat support + lower highs + breakdown  
**Accuracy**: 52.7%  
**Status**: ✅ Active

#### Signal 19: Death Cross
**Name**: `death_cross_signal`  
**Logic**: 50 EMA crosses below 200 EMA  
**Accuracy**: 53.9%  
**Status**: ✅ Active

#### Signal 20: Bearish Divergence RSI
**Name**: `rsi_bearish_divergence_signal`  
**Logic**: Price higher high, RSI lower high  
**Accuracy**: 54.1%  
**Status**: ✅ Active

#### Signal 21: Bearish ADX Trend
**Name**: `adx_bearish_signal`  
**Logic**: ADX > 25 + -DI > +DI (strong downtrend)  
**Accuracy**: 52.3%  
**Status**: ✅ Active

#### Signal 22: Bearish CCI Signal
**Name**: `cci_bearish_signal`  
**Logic**: CCI drops below 100 from overbought  
**Accuracy**: 51.4%  
**Status**: ✅ Active

#### Signal 23: Parabolic SAR Reversal
**Name**: `psar_bearish_signal`  
**Logic**: SAR flips above price (trend reversal)  
**Accuracy**: 52.8%  
**Status**: ✅ Active

#### Signal 24: On-Balance Volume Decline
**Name**: `obv_decline_signal`  
**Logic**: OBV declining while price rising (divergence)  
**Accuracy**: 53.2%  
**Status**: ✅ Active

#### Signal 25: Money Flow Weakness
**Name**: `mfi_weakness_signal`  
**Logic**: MFI < 20 or declining from overbought  
**Accuracy**: 50.9%  
**Status**: ✅ Active

#### Signal 26: Ichimoku Bearish Cloud
**Name**: `ichimoku_bearish_signal`  
**Logic**: Price below cloud + lagging span below price  
**Accuracy**: 54.3%  
**Status**: ✅ Active

#### Signal 27: Elder Ray Bear Power
**Name**: `elder_ray_bear_signal`  
**Logic**: Bear power negative and increasing  
**Accuracy**: 51.7%  
**Status**: ✅ Active

#### Signal 28: Williams %R Overbought
**Name**: `williams_r_signal`  
**Logic**: %R in overbought (-20 to 0) then drops  
**Accuracy**: 52.5%  
**Status**: ✅ Active

#### Signal 29: Average True Range Expansion
**Name**: `atr_expansion_signal`  
**Logic**: ATR expanding + bearish candle (volatility increase)  
**Accuracy**: 51.2%  
**Status**: ✅ Active

### Disabled Signals: 3

#### ❌ Signal 30: Bearish Hammer Failures
**Name**: `bearish_hammer_failures_signal`  
**Logic**: Hammer pattern fails (doesn't reverse)  
**Accuracy**: 47.57% ⚠️ Below 49% threshold  
**Status**: ❌ DISABLED (worse than random)

#### ❌ Signal 31: RSI Divergence Bearish
**Name**: `rsi_divergence_bearish_signal`  
**Logic**: RSI bearish divergence  
**Accuracy**: 48.68% ⚠️ Below 49% threshold  
**Status**: ❌ DISABLED (worse than random)

#### ❌ Signal 32: MACD Bearish Crossovers
**Name**: `macd_bearish_crossovers_signal`  
**Logic**: MACD line crosses below signal line  
**Accuracy**: 49.19% ⚠️ Below 49% threshold  
**Status**: ❌ DISABLED (worse than random)

**Note**: These 3 signals are commented out in the `generate_all_signals()` method

---

## 📊 System 4: Harmonic Patterns

**File**: `scripts/harmonic_patterns.py`  
**Class**: Not specified (pattern detection functions)  
**Integration**: Core feature in main training pipeline  
**Status**: ✅ Active

### Patterns Detected:

#### Pattern 1: Gartley Pattern
**Logic**: XABCD with Fibonacci ratios (0.618, 0.786)  
**Signals**: Bullish/Bearish Gartley completion

#### Pattern 2: Butterfly Pattern
**Logic**: XABCD with extended D point (1.272, 1.618)  
**Signals**: Bullish/Bearish Butterfly completion

#### Pattern 3: Bat Pattern
**Logic**: XABCD with 0.886 B retracement  
**Signals**: Bullish/Bearish Bat completion

#### Pattern 4: Crab Pattern
**Logic**: XABCD with extreme D extension (1.618)  
**Signals**: Bullish/Bearish Crab completion

#### Pattern 5: Shark Pattern
**Logic**: AB=CD with 0.886 to 1.13 ratios  
**Signals**: Bullish/Bearish Shark completion

**Output**: Boolean flags for each pattern completion + confidence score

---

## 📊 System 5: Chart Patterns

**File**: `scripts/chart_patterns.py`  
**Class**: Not specified (classical pattern detection)  
**Integration**: Core feature in main training pipeline  
**Status**: ✅ Active

### Classical Patterns:

1. **Double Top/Bottom** - Two peaks/troughs at resistance/support
2. **Triple Top/Bottom** - Three peaks/troughs at same level
3. **Head and Shoulders** - Peak between two lower peaks
4. **Inverse H&S** - Trough between two higher troughs
5. **Rising Wedge** - Converging upward trend lines (bearish)
6. **Falling Wedge** - Converging downward trend lines (bullish)
7. **Ascending Triangle** - Flat resistance + rising support
8. **Descending Triangle** - Flat support + falling resistance
9. **Symmetrical Triangle** - Converging support/resistance
10. **Flag Pattern** - Rectangular consolidation after strong move
11. **Pennant** - Small symmetrical triangle after strong move
12. **Cup and Handle** - U-shape + small consolidation

**Output**: Pattern detected (boolean) + breakout direction + target level

---

## 📊 System 6: Elliott Wave

**File**: `scripts/elliott_wave.py`  
**Class**: Not specified (wave analysis)  
**Integration**: Core feature in main training pipeline  
**Status**: ✅ Active

### Wave Analysis:

#### Impulse Waves (5 waves):
- Wave 1: Initial move
- Wave 2: Retracement (50-61.8%)
- Wave 3: Strongest move (1.618× wave 1)
- Wave 4: Retracement (38.2-50%)
- Wave 5: Final move

#### Corrective Waves (3 waves):
- Wave A: Initial correction
- Wave B: Counter-trend bounce
- Wave C: Final correction

**Output**: 
- Current wave position (1-5, A-C)
- Wave completion confidence
- Next wave prediction
- Fibonacci extension targets

---

## 📊 System 7: Ultimate Signal Repository

**File**: `scripts/ultimate_signal_repository.py`  
**Class**: `UltimateSignalRepository`  
**Integration**: Core feature in main training pipeline  
**Status**: ✅ Active

### Smart Money Concepts (SMC):

#### Signal 1: Order Blocks
**Logic**: Institutional accumulation/distribution zones  
**Detection**: Last bullish candle before drop (or vice versa)

#### Signal 2: Fair Value Gaps (FVG)
**Logic**: Imbalance areas (gaps) that price revisits  
**Detection**: Three-candle pattern with gap

#### Signal 3: Break of Structure (BOS)
**Logic**: Market structure shift (higher high/lower low break)  
**Detection**: Price breaks previous swing high/low

#### Signal 4: Change of Character (ChoCh)
**Logic**: Trend change signal  
**Detection**: Failed to make new high/low + reversal

#### Signal 5: Liquidity Sweeps
**Logic**: Stop hunt before reversal  
**Detection**: Spike above/below key level + quick reversal

### Order Flow Signals:

#### Signal 6: Volume Profile
**Logic**: Volume at price levels (VPOC, VAH, VAL)  
**Detection**: High volume nodes = support/resistance

#### Signal 7: Cumulative Delta
**Logic**: Buying vs selling pressure  
**Detection**: Delta divergence with price

#### Signal 8: Footprint Charts
**Logic**: Bid/ask volume at each price level  
**Detection**: Absorption, exhaustion patterns

#### Signal 9: Time and Sales
**Logic**: Large trades (whales)  
**Detection**: Unusual size trades

**Output**: Boolean flags + strength indicators for each signal

---

## 📊 System 8: Enhanced Holloway Next-Candle Predictor

**File**: `scripts/holloway_algorithm_next_candle.py`  
**Class**: `EnhancedHollowayPredictor`  
**Integration**: ❌ SEPARATE system (not in main training pipeline)  
**Status**: ✅ Production ready (independent use)

### Purpose:
Predict complete next candle (direction + OHLC) using advanced Holloway analysis

### Features: 115+
- Count vs average crossovers (8 signals) - FASTEST
- Historical levels (12 features)
- Explosion detection (8 features)
- Mirroring behavior (4 features)
- W/M patterns (12 features)
- Core Holloway (6 features)
- Momentum (4 features)
- RSI (3 features)
- Price action (5+ features)

### Models:
1. Direction Model: GradientBoost (85%+ target)
2. Open Model: RandomForest
3. High Model: RandomForest
4. Low Model: RandomForest
5. Close Model: RandomForest

**Status**: Independent system, runs separately from main training

---

## 📈 Technical Indicators (Base Features)

**File**: `scripts/forecasting.py::_engineer_features()`  
**Generated**: ~100 features

### Moving Averages (12):
- SMA: 5, 10, 20, 50, 100, 200
- EMA: 5, 10, 20, 50, 100, 200

### Momentum Indicators (8):
- RSI(14)
- MACD (12, 26, 9): line, signal, histogram
- Stochastic (14, 3): %K, %D
- ROC (Rate of Change)
- Williams %R

### Volatility Indicators (4):
- ATR (14)
- Bollinger Bands (20, 2): upper, middle, lower
- Standard deviation (20, 50)

### Statistical Features (6):
- Skewness (rolling 20)
- Kurtosis (rolling 20)
- Z-score
- Percentile rank

### Time-based Features (8):
- day_of_week (0-6)
- month (1-12)
- week_of_year (1-52)
- quarter (1-4)
- is_month_start
- is_month_end
- is_quarter_start
- is_quarter_end

### Lagged Features (15):
- close lag 1, 2, 3, 5, 10
- returns lag 1, 2, 3, 5, 10
- volume lag 1, 2, 3, 5, 10

---

## 📊 Fundamental Data

**File**: `scripts/fundamental_pipeline.py`  
**Series**: 23 FRED indicators  
**Prefix**: `fund_`

### Economic Indicators:

#### Exchange Rates (3):
1. DEXUSEU - USD/EUR
2. DEXJPUS - USD/JPY
3. DEXCHUS - USD/CHF

#### Interest Rates (4):
4. FEDFUNDS - Federal Funds Rate
5. DFF - Effective Federal Funds Rate
6. DGS10 - 10-Year Treasury
7. DGS2 - 2-Year Treasury

#### Inflation (2):
8. CPIAUCSL - US CPI
9. CPALTT01USM661S - OECD CPI

#### Employment (2):
10. UNRATE - Unemployment Rate
11. PAYEMS - Nonfarm Payrolls

#### Economic Activity (2):
12. INDPRO - Industrial Production
13. DGORDER - Durable Goods Orders

#### ECB (1):
14. ECBDFR - ECB Deposit Facility Rate

#### Euro Area (2):
15. CP0000EZ19M086NEST - Euro CPI
16. LRHUTTTTDEM156S - Germany Unemployment

#### Market Indicators (3):
17. VIXCLS - VIX Volatility Index
18. DCOILWTICO - WTI Crude Oil
19. DCOILBRENTEU - Brent Crude Oil

#### Trade (1):
20. BOPGSTB - Trade Balance

#### Additional (3):
21-23. Other macro indicators

**Processing**:
- Resampled to daily frequency
- Forward-filled (fundamentals release infrequently)
- Merged with price data on date

---

## 🎯 Total Feature Count Summary

### Before Cleaning: ~1400 features
- Technical Indicators: ~100
- Holloway Algorithm: 196 (49 × 4 timeframes)
- Day Trading Signals: 9
- Slump Signals: 29
- Harmonic Patterns: ~20
- Chart Patterns: ~15
- Elliott Wave: ~10
- Ultimate Signals: ~30
- Fundamental Data: 23
- Derived/Interaction: ~968

### After Cleaning: 574 features
- Deduplication: 1400 → 844
- Low-variance filter: 844 → 574
- Final features used in training

---

## ✅ Pre-Training Checklist

### Data Validation:
- [x] Price data loaded (EURUSD_H4.csv, XAUUSD_H4.csv)
- [x] Fundamental data loaded (23 FRED series)
- [x] Column names standardized (lowercase)
- [x] Date columns correct format
- [x] No missing OHLCV values

### Signal Systems:
- [x] Holloway Algorithm (196 features)
- [x] Day Trading Signals (9 signals)
- [x] Slump Signals (29 active, 3 disabled)
- [x] Harmonic Patterns (active)
- [x] Chart Patterns (active)
- [x] Elliott Wave (active)
- [x] Ultimate Signals (active)
- [x] Technical Indicators (100 features)
- [x] Fundamental Data (23 series)

### Model Configuration:
- [x] Old models deleted (fresh start)
- [x] LightGBM parameters configured
- [x] Training script ready (ultra_simple_train.py)
- [x] Logging enabled
- [x] Output directories exist

### Documentation:
- [x] All signals named and documented
- [x] Signal details provided
- [x] Architecture documented
- [x] Flow diagrams created

---

## 🚀 Ready to Train!

**Command**:
```bash
cd /workspaces/congenial-fortnight
source .venv/bin/activate
python ultra_simple_train.py 2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log
```

**Expected Output**:
- EURUSD: 65-70% validation accuracy
- XAUUSD: 75-80% validation accuracy
- Training time: 2-3 minutes per pair
- Features used: 570-580 (after cleaning)

**Models Saved**:
- `models/EURUSD_lightgbm_simple.joblib`
- `models/XAUUSD_lightgbm_simple.joblib`

---

**Total Signal Count**: 29 (slump) + 9 (day trading) + 196 (Holloway) + ~75 (other) = **309+ named signals**  
**Total Features After Cleaning**: **574 features**  
**Status**: ✅ **READY FOR TRAINING**

---

*Generated automatically before training - October 7, 2025*
