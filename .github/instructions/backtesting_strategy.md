# Backtesting & Trade Management Strategy

**Version**: 1.0  
**Last Updated**: October 6, 2025  
**Status**: Planning Document

---

## Table of Contents

1. [Signal Generation Targets](#signal-generation-targets)
2. [Trade Management Rules](#trade-management-rules)
3. [Backtesting Framework](#backtesting-framework)
4. [Risk Management](#risk-management)
5. [Performance Metrics](#performance-metrics)

---

## Signal Generation Targets

### Daily Signal Goals
Based on current system architecture with 574 features

**EURUSD (65.8% accuracy)**:
- **Target**: 1 high-confidence signal per day
- **Threshold**: Prediction probability ≥ 0.70 (70% model confidence)
- **Expected Win Rate**: ~65-70% (at 0.70 threshold)
- **Monthly Target**: 20-22 signals (trading days)

**XAUUSD (77.3% accuracy)**:
- **Target**: 1 high-confidence signal per day
- **Threshold**: Prediction probability ≥ 0.65 (65% model confidence)
- **Expected Win Rate**: ~70-75% (at 0.65 threshold)
- **Monthly Target**: 20-22 signals (trading days)

### Weekly Signal Aggregation
- **Both Pairs Combined**: 40-44 signals/month
- **Quality over Quantity**: Only trade when model confidence exceeds threshold
- **Skip Days**: If no signal meets confidence threshold, DO NOT TRADE

### Monthly Signal Distribution
```
Week 1: 10-11 signals (5-6 per pair)
Week 2: 10-11 signals (5-6 per pair)
Week 3: 10-11 signals (5-6 per pair)
Week 4: 10-11 signals (5-6 per pair)
-------------------------------------------
Total:  40-44 signals across both pairs
```

### Signal Quality Filters

**Minimum Requirements for Valid Signal**:
1. **Model Confidence**: ≥ 0.65 (XAUUSD) or ≥ 0.70 (EURUSD)
2. **Feature Completeness**: ≥ 95% non-null features
3. **Volatility Check**: Not during extreme volatility (VIX > 40)
4. **Fundamental Alignment**: No conflicting fundamental signals

**Signal Strength Categories**:
- **Strong (0.80+)**: Full position size
- **Medium (0.70-0.79)**: 75% position size
- **Weak (0.65-0.69)**: 50% position size (XAUUSD only)

---

## Trade Management Rules

### Entry Rules

#### Buy Signal (Bullish Prediction = 1)
**Conditions**:
- Model predicts next day close > today close
- Probability ≥ confidence threshold
- No conflicting fundamental data

**Entry**:
- **Time**: Market open next day (00:00 GMT for forex, 00:00 for gold)
- **Price**: Market order at open OR limit order at yesterday's close
- **Position Size**: Based on signal strength category

#### Sell Signal (Bearish Prediction = 0)
**Conditions**:
- Model predicts next day close < today close
- Probability ≥ confidence threshold
- No conflicting fundamental data

**Entry**:
- **Time**: Market open next day
- **Price**: Market order at open OR limit order at yesterday's close
- **Position Size**: Based on signal strength category

### Exit Rules

#### Profit Target
**EURUSD**:
- **Target**: +50 pips (0.0050)
- **Reasoning**: Daily ATR ~70-100 pips, target 50-70% of ATR
- **Take Profit**: Automatic at +50 pips

**XAUUSD**:
- **Target**: +500 pips ($5.00)
- **Reasoning**: Daily ATR ~$15-25, target 20-30% of ATR
- **Take Profit**: Automatic at +$5.00

#### Stop Loss
**EURUSD**:
- **Stop**: -30 pips (0.0030)
- **Risk:Reward**: 1:1.67 (excellent)
- **Stop Loss**: Automatic at -30 pips

**XAUUSD**:
- **Stop**: -300 pips ($3.00)
- **Risk:Reward**: 1:1.67
- **Stop Loss**: Automatic at -$3.00

#### Time-Based Exit
**End of Day Exit**:
- **Time**: 23:50 GMT (10 minutes before daily candle close)
- **Reasoning**: Model predicts next DAY direction, not intraday
- **Action**: Close position at market price regardless of P&L
- **Exception**: If position within 5 pips of TP, hold for TP

### Position Sizing

**Base Position Size** (per $10,000 account):
```
Signal Strength    | EURUSD Lots | XAUUSD Lots | Risk %
-------------------|-------------|-------------|--------
Strong (0.80+)     | 0.10        | 0.10        | 1.0%
Medium (0.70-0.79) | 0.075       | 0.075       | 0.75%
Weak (0.65-0.69)   | 0.05        | 0.05        | 0.5%
```

**Risk Per Trade**: 0.5-1.0% of account equity

**Max Concurrent Positions**: 2 (one per pair)

---

## Backtesting Framework

### Historical Data Requirements

**Time Period**:
- **Training Data**: 2000-2024 (for model training)
- **Backtest Period**: 2023-2024 (out-of-sample)
- **Walk-Forward**: Retrain every 6 months

**Data Granularity**:
- **Daily OHLCV**: Required for entry/exit prices
- **4-Hour OHLCV**: Required for intraday feature engineering
- **Monthly**: Required for Holloway monthly features
- **Fundamental**: Daily frequency (forward-filled)

### Backtesting Process

**Step 1: Data Preparation**
```python
# Load validation data (2023-2024)
validation_period = df[(df['timestamp'] >= '2023-01-01') & 
                       (df['timestamp'] <= '2024-12-31')]

# Ensure all features present
assert validation_period.shape[1] == 574 + 2  # 574 features + timestamp + target
```

**Step 2: Generate Predictions**
```python
# Load trained model
model = joblib.load('models/EURUSD_lightgbm_simple.joblib')

# Get predictions with probabilities
predictions = model.predict(X_validation)
probabilities = model.predict_proba(X_validation)

# Filter by confidence threshold
high_confidence_signals = probabilities[:, 1] >= 0.70  # Bull probability
```

**Step 3: Simulate Trades**
```python
for date, prediction, probability in signals:
    # Determine signal strength
    strength = get_signal_strength(probability)
    
    # Calculate position size
    lot_size = get_position_size(strength, account_balance)
    
    # Simulate entry
    entry_price = next_day_open_price
    
    # Simulate exit (TP/SL/Time)
    exit_price, exit_reason = simulate_exit(
        entry_price=entry_price,
        direction=prediction,  # 1=buy, 0=sell
        stop_loss_pips=30,
        take_profit_pips=50,
        daily_ohlc=next_day_data
    )
    
    # Calculate P&L
    trade_pnl = calculate_pnl(entry_price, exit_price, lot_size, direction)
    
    # Record trade
    trades.append({
        'date': date,
        'pair': 'EURUSD',
        'direction': 'BUY' if prediction == 1 else 'SELL',
        'entry': entry_price,
        'exit': exit_price,
        'pips': (exit_price - entry_price) * 10000,
        'pnl': trade_pnl,
        'exit_reason': exit_reason
    })
```

**Step 4: Calculate Metrics**
```python
# Win rate
win_rate = len([t for t in trades if t['pnl'] > 0]) / len(trades)

# Average win/loss
avg_win = mean([t['pnl'] for t in trades if t['pnl'] > 0])
avg_loss = mean([t['pnl'] for t in trades if t['pnl'] < 0])

# Profit factor
profit_factor = sum([t['pnl'] for t in trades if t['pnl'] > 0]) / \
                abs(sum([t['pnl'] for t in trades if t['pnl'] < 0]))

# Max drawdown
equity_curve = calculate_equity_curve(trades)
max_drawdown = calculate_max_drawdown(equity_curve)

# Sharpe ratio
sharpe = calculate_sharpe_ratio(equity_curve)
```

### Backtesting Script Template

Create file: `scripts/backtest_trading_system.py`

```python
#!/usr/bin/env python3
"""
Comprehensive Backtesting Script
Tests trained models on out-of-sample data (2023-2024)
"""
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingSystemBacktest:
    """Backtest trading system with realistic execution"""
    
    def __init__(self, pair: str, model_path: str, confidence_threshold: float):
        self.pair = pair
        self.model = joblib.load(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Trade parameters
        if pair == 'EURUSD':
            self.take_profit_pips = 50
            self.stop_loss_pips = 30
            self.pip_value = 0.0001
        elif pair == 'XAUUSD':
            self.take_profit_pips = 500
            self.stop_loss_pips = 300
            self.pip_value = 0.01
        
        self.trades = []
        self.account_balance = 10000  # Starting balance
        
    def simulate_trade(self, entry_date, prediction, probability, daily_ohlc):
        """Simulate a single trade with realistic execution"""
        
        # Determine position size based on signal strength
        if probability >= 0.80:
            lot_size = 0.10
        elif probability >= 0.70:
            lot_size = 0.075
        else:
            lot_size = 0.05
        
        # Entry at next day open
        entry_price = daily_ohlc['open']
        direction = 'BUY' if prediction == 1 else 'SELL'
        
        # Calculate TP/SL levels
        if direction == 'BUY':
            take_profit = entry_price + (self.take_profit_pips * self.pip_value)
            stop_loss = entry_price - (self.stop_loss_pips * self.pip_value)
        else:
            take_profit = entry_price - (self.take_profit_pips * self.pip_value)
            stop_loss = entry_price + (self.stop_loss_pips * self.pip_value)
        
        # Check if TP or SL hit during the day
        if direction == 'BUY':
            if daily_ohlc['high'] >= take_profit:
                exit_price = take_profit
                exit_reason = 'TP'
            elif daily_ohlc['low'] <= stop_loss:
                exit_price = stop_loss
                exit_reason = 'SL'
            else:
                exit_price = daily_ohlc['close']
                exit_reason = 'EOD'
        else:  # SELL
            if daily_ohlc['low'] <= take_profit:
                exit_price = take_profit
                exit_reason = 'TP'
            elif daily_ohlc['high'] >= stop_loss:
                exit_price = stop_loss
                exit_reason = 'SL'
            else:
                exit_price = daily_ohlc['close']
                exit_reason = 'EOD'
        
        # Calculate P&L
        if direction == 'BUY':
            pips = (exit_price - entry_price) / self.pip_value
        else:
            pips = (entry_price - exit_price) / self.pip_value
        
        pnl = pips * lot_size * (10 if self.pair == 'EURUSD' else 1)
        
        # Update account balance
        self.account_balance += pnl
        
        # Record trade
        trade = {
            'date': entry_date,
            'pair': self.pair,
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pips': pips,
            'pnl': pnl,
            'balance': self.account_balance,
            'exit_reason': exit_reason,
            'probability': probability
        }
        
        self.trades.append(trade)
        return trade
    
    def run_backtest(self, validation_data):
        """Run full backtest on validation data"""
        logger.info(f"Starting backtest for {self.pair}")
        logger.info(f"Confidence threshold: {self.confidence_threshold}")
        
        # Get predictions
        X = validation_data.drop(['target', 'timestamp'], axis=1, errors='ignore')
        y_true = validation_data['target']
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # Filter high-confidence signals
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            bull_prob = prob[1]
            
            # Only trade if confidence exceeds threshold
            if pred == 1 and bull_prob >= self.confidence_threshold:
                # Buy signal
                if i + 1 < len(validation_data):
                    next_day = validation_data.iloc[i + 1]
                    trade = self.simulate_trade(
                        entry_date=validation_data.iloc[i]['timestamp'],
                        prediction=1,
                        probability=bull_prob,
                        daily_ohlc=next_day
                    )
                    
            elif pred == 0 and (1 - bull_prob) >= self.confidence_threshold:
                # Sell signal
                if i + 1 < len(validation_data):
                    next_day = validation_data.iloc[i + 1]
                    trade = self.simulate_trade(
                        entry_date=validation_data.iloc[i]['timestamp'],
                        prediction=0,
                        probability=1 - bull_prob,
                        daily_ohlc=next_day
                    )
        
        # Calculate performance metrics
        self.calculate_metrics()
        
        return pd.DataFrame(self.trades)
    
    def calculate_metrics(self):
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            logger.warning("No trades to analyze")
            return
        
        df_trades = pd.DataFrame(self.trades)
        
        # Win rate
        wins = len(df_trades[df_trades['pnl'] > 0])
        losses = len(df_trades[df_trades['pnl'] < 0])
        win_rate = wins / len(df_trades) if len(df_trades) > 0 else 0
        
        # Average win/loss
        avg_win = df_trades[df_trades['pnl'] > 0]['pnl'].mean() if wins > 0 else 0
        avg_loss = df_trades[df_trades['pnl'] < 0]['pnl'].mean() if losses > 0 else 0
        
        # Profit factor
        total_profit = df_trades[df_trades['pnl'] > 0]['pnl'].sum()
        total_loss = abs(df_trades[df_trades['pnl'] < 0]['pnl'].sum())
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Net profit
        net_profit = df_trades['pnl'].sum()
        
        # Max drawdown
        equity_curve = df_trades['balance'].values
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Print results
        logger.info(f"\n{'='*60}")
        logger.info(f"BACKTEST RESULTS: {self.pair}")
        logger.info(f"{'='*60}")
        logger.info(f"Total Trades:     {len(df_trades)}")
        logger.info(f"Win Rate:         {win_rate*100:.2f}% ({wins}W / {losses}L)")
        logger.info(f"Average Win:      ${avg_win:.2f}")
        logger.info(f"Average Loss:     ${avg_loss:.2f}")
        logger.info(f"Profit Factor:    {profit_factor:.2f}")
        logger.info(f"Net Profit:       ${net_profit:.2f}")
        logger.info(f"Max Drawdown:     {max_drawdown*100:.2f}%")
        logger.info(f"Final Balance:    ${self.account_balance:.2f}")
        logger.info(f"ROI:              {((self.account_balance/10000 - 1)*100):.2f}%")
        logger.info(f"{'='*60}\n")
        
        # Exit reason breakdown
        logger.info("Exit Reason Breakdown:")
        exit_counts = df_trades['exit_reason'].value_counts()
        for reason, count in exit_counts.items():
            logger.info(f"  {reason}: {count} ({count/len(df_trades)*100:.1f}%)")


if __name__ == "__main__":
    # Example usage
    from scripts.forecasting import HybridPriceForecastingEnsemble
    
    # Load data
    ensemble = HybridPriceForecastingEnsemble('EURUSD')
    X_train, y_train, X_val, y_val = ensemble.load_and_prepare_datasets()
    
    # Reconstruct validation dataframe
    validation_df = X_val.copy()
    validation_df['target'] = y_val
    
    # Run backtest
    backtest = TradingSystemBacktest(
        pair='EURUSD',
        model_path='models/EURUSD_lightgbm_simple.joblib',
        confidence_threshold=0.70
    )
    
    trades_df = backtest.run_backtest(validation_df)
    
    # Save results
    trades_df.to_csv('backtest_results_EURUSD.csv', index=False)
    logger.info("Backtest complete. Results saved to backtest_results_EURUSD.csv")
```

---

## Risk Management

### Account Rules
- **Max Risk Per Trade**: 1% of account
- **Max Daily Loss**: 3% of account
- **Max Weekly Loss**: 5% of account
- **Max Monthly Loss**: 10% of account

### Position Limits
- **Max Concurrent Positions**: 2 (one per pair)
- **Max Lot Size**: 0.10 per $10,000 account
- **Min Account Balance**: $1,000 per 0.01 lot

### Circuit Breakers
**Daily Circuit Breaker**:
- If 3 consecutive losses, stop trading for the day
- If daily loss hits -3%, stop all trading for the day

**Weekly Circuit Breaker**:
- If weekly loss hits -5%, stop all trading for the week
- Review system performance before resuming

**Monthly Circuit Breaker**:
- If monthly loss hits -10%, stop all trading
- Retrain models before resuming

---

## Performance Metrics

### Key Metrics to Track

#### Win Rate
**Formula**: `Wins / Total Trades`  
**Target**: ≥ 60% (matching model accuracy)  
**Threshold**: Stop trading if drops below 55% for 20 trades

#### Profit Factor
**Formula**: `Gross Profit / Gross Loss`  
**Target**: ≥ 1.5  
**Threshold**: Review strategy if drops below 1.2

#### Average Win:Loss Ratio
**Formula**: `Average Win / Average Loss`  
**Target**: ≥ 1.5 (with 1:1.67 R:R)  
**Current**: ~1.67 (50 pips TP / 30 pips SL)

#### Max Drawdown
**Formula**: `(Trough - Peak) / Peak`  
**Target**: ≤ 20%  
**Threshold**: Stop trading if exceeds 25%

#### Sharpe Ratio
**Formula**: `(Return - Risk-Free Rate) / Std Dev of Returns`  
**Target**: ≥ 1.0  
**Good**: ≥ 1.5

#### Monthly ROI
**Target**: 5-10% per month  
**Conservative**: 3-5% per month  
**Aggressive**: 10-15% per month (higher risk)

### Trade Journal Template

Record every trade in CSV:

```csv
Date,Pair,Direction,Entry,Exit,Pips,PnL,Balance,ExitReason,Probability,Notes
2025-10-07,EURUSD,BUY,1.0850,1.0900,50,500,10500,TP,0.85,Strong signal
2025-10-08,XAUUSD,SELL,1950.50,1955.50,-500,-250,10250,SL,0.72,Stopped out
```

---

## Expected Performance (Projected)

Based on current model accuracy (65.8% EURUSD, 77.3% XAUUSD):

### EURUSD (20 trades/month, 0.70 threshold)
```
Win Rate:           65%
Average Win:        +50 pips = $500
Average Loss:       -30 pips = -$300
Wins:               13 trades × $500 = $6,500
Losses:             7 trades × -$300 = -$2,100
---------------------------------------------------
Monthly Net:        $4,400
Monthly ROI:        44% (on $10k account)
```

### XAUUSD (20 trades/month, 0.65 threshold)
```
Win Rate:           70%
Average Win:        +$5.00 = $500
Average Loss:       -$3.00 = -$300
Wins:               14 trades × $500 = $7,000
Losses:             6 trades × -$300 = -$1,800
---------------------------------------------------
Monthly Net:        $5,200
Monthly ROI:        52% (on $10k account)
```

### Combined Performance (40 trades/month)
```
Total Monthly Net:  $9,600
Combined ROI:       96% per month
Annual ROI:         ~1,150% (compounded)
```

**Note**: These are PROJECTIONS based on perfect execution. Real results will vary due to:
- Slippage (2-5 pips)
- Spread costs
- Execution delays
- Market conditions
- Model drift

**Realistic Expectation**: 30-50% monthly ROI after costs

---

## Next Steps

1. **Implement Backtesting Script**: Create `scripts/backtest_trading_system.py`
2. **Run Historical Backtest**: Test on 2023-2024 data
3. **Paper Trading**: 1 month forward testing with virtual money
4. **Micro Account**: Start with $100-500 real money
5. **Scale Up**: Gradually increase account size as confidence grows

---

*This strategy is based on current model performance. Adjust parameters based on backtest and live trading results.*
