# Forex Signal Service

A daily ML-driven trading signal service for EURUSD and XAUUSD.
Follow the signal, enter the trade manually in MT5, let SL/TP do the rest.

## Signal Performance (Honest Walk-Forward CV)

| Pair   | RR    | Win Rate | Breakeven | Expectancy | Threshold |
|--------|-------|----------|-----------|------------|-----------|
| EURUSD | 1.5:1 | 63.4%    | 40.0%     | +0.585R    | 0.71      |
| XAUUSD | 2.0:1 | 41.4%    | 33.3%     | +0.241R    | 0.50      |

All metrics from TimeSeriesSplit(n_splits=5) walk-forward cross-validation.
No lookahead bias. Target: did trade hit TP before SL within N daily bars?
Features: 37 TA indicators + 7 FRED macro fundamentals + 12 Fibonacci harmonic patterns.

## Quick Start

```bash
git clone https://github.com/codejoncode/congenial-fortnight.git
cd congenial-fortnight
pip install -r requirements.txt
python manage.py migrate
```

## Daily Workflow

```bash
# Full daily run (data + signals)
python manage.py daily_workflow

# Or step by step:
python manage.py fetch_price_data          # refresh H1, H4, Daily, Weekly
python manage.py run_daily_signal          # generate signals (auto-fetches if stale)
python manage.py run_daily_signal --force  # always regenerate
```

## Initial Training

```bash
# Fetch maximum history and train both pairs
python manage.py train_models --fetch-data --full

# Retrain one pair
python manage.py train_models --pair XAUUSD
```

## Auto-Scheduler

The scheduler runs daily at 22:00 UTC and retrains on the 1st and 15th:

```bash
# In one terminal: Django server
python manage.py runserver

# In another terminal: scheduler
python scheduler.py
```

Or use the provided startup scripts: `start.bat` (Windows) / `start.sh` (Linux/Mac).

## Email Notifications

Add to your `.env` file:

```env
ALERT_EMAIL_HOST_USER=your@gmail.com
ALERT_EMAIL_HOST_PASSWORD=xxxx xxxx xxxx xxxx   # Gmail App Password (16 chars)
ALERT_EMAIL_TO=your@gmail.com
SIGNAL_NOTIFY_THRESHOLD=0.60
```

Gmail setup: Account > Security > 2-Step Verification > App Passwords > Create.

## Architecture

```
signals/signal_engine.py        — Core ML engine (train + predict)
  Target:  TP/SL outcome (1=TP hit first, 0=SL hit first, NaN=unresolved)
  Models:  RF(class_weight=balanced) + XGB(scale_pos_weight) ensemble
  CV:      TimeSeriesSplit(n_splits=5), zero lookahead bias
  Features: 37 TA (trading/features.py) + 7 FRED macro fundamentals

trading/features.py             — Canonical feature engineering (single source of truth)
  37 features: log returns, SMAs, RSI, MACD, BB, ATR, candlestick geometry/patterns

signals/management/commands/
  fetch_price_data.py           — Fetch H1, H4, Daily, Weekly (yfinance + resample)
  train_models.py               — Train RF+XGB with per-pair RR/lookahead config
  run_daily_signal.py           — Generate signals, save to DB, send email alert
  daily_workflow.py             — Orchestrator: fetch -> train (optional) -> signal

data/                           — OHLCV CSVs + FRED fundamentals (DGS10, DGS2, VIX...)
models/                         — Trained model artifacts (gitignored)
```

## Trading Rules

- Signal appears: enter at market open next session
- Set SL and TP at the exact levels shown (ATR-based, already calculated)
- Let the trade run — do not move stops unless the signal explicitly updates
- Size per trade: 1-3% of account (adjustable by confidence in signal)
- No signal = no trade that day

## Risk Context

Positive expectancy means long-run profitability, not every trade wins.
At 36-43% win rate and 1.5-2:1 RR, expect ~3-4 losses for every 2 wins on average.
Proper position sizing (never more than 3% risk per trade) is what lets the account survive and compound.
