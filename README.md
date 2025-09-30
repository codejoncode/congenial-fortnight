# Forex Signal Service

## Getting Started

To get all project files, clone the repository:

```bash
git clone https://github.com/codejoncode/congenial-fortnight.git
cd congenial-fortnight
```

This will pull all files: Django backend, React frontend, ML script (`daily_forex_signal_system.py`), README, requirements.txt, etc.

## Project Overview

This project is a signal service for forex trading, specifically targeting EURUSD and XAUUSD pairs. It uses machine learning models to predict the direction of the next trading period (bullish or bearish) and provides stop-loss recommendations to minimize losses.

## Primary Goal: 90%+ Accurate Candle Prediction

Our ultimate objective is to achieve 90% or better accuracy in predicting and displaying the next candle's characteristics. This includes:

- **Direction Prediction**: Accurately forecasting bullish (green) or bearish (red) trends.
- **Price Levels**: Predicting high, low, open, close within tight tolerances.
- **Visual Representation**: Displaying golden candles with red/green fills, wicks, and hover tooltips showing OHLC.
- **Advanced Features**:
  - Trend direction analysis using moving averages, RSI, MACD.
  - ATR (Average True Range) for volatility and stop-loss calculation.
  - Variants from previous highs/lows: e.g., breakout levels, support/resistance.
  - Creative indicators: Fibonacci retracements, Bollinger Bands, PnF charts, momentum oscillators.
  - Price action patterns: Engulfing, Doji, Hammer, etc.
  - Machine learning ensemble: RF + XGB with isotonic calibration for confidence scoring.
- **No-Signal Logic**: Emit 'no signal' for low-confidence predictions to avoid false positives.
- **Backtesting**: Rigorous historical testing with slippage, commissions, and realistic P&L.

By integrating these elements creatively, we aim to provide traders with highly reliable signals for daily close-to-close trades.

### Key Features
- **Daily Signals**: Generates bullish/bearish signals for the next trading day based on historical data.
- **Stop Loss Optimization**: Provides ATR-based stop losses to minimize losses.
- **Historical Backtesting**: Exports historical data and trade details for analysis, including wins/losses, pips earned/lost, and average pips per trade.
- **Accuracy Improvement**: Strategies to enhance model confidence and accuracy.
- **Web Application**: Django backend serving data to a React frontend displaying candlestick charts, signals, and predictions.
- **Alerts**: Daily email/text notifications (using free services).
- **Data Updates**: Fetches daily OHLC data using free APIs (e.g., yfinance).

### Architecture
- **Backend**: Python Django API to run ML models and serve data.
- **Frontend**: React single-page app with candlestick charts showing historical trades, wins/losses, and current signals.
- **Models**: Pre-trained ML models (uploaded by user) for predictions.
- **Data**: Free daily OHLC data from yfinance or similar.

### How It Works
1. Models predict the direction of the next candle (bullish/green or bearish/red).
2. Signals are taken at close, with stop losses calculated.
3. Frontend displays golden candles with red/green fills, wicks, and hover details (open, high, low, close).
4. Runs ~1 hour before US close to provide signals after previous close.
5. Exports trade logs for lifetime pips analysis.

### Optimizing Stop Losses
- Use ATR (Average True Range) for dynamic stop losses.
- Backtest different multipliers (e.g., 0.5x for EURUSD, 0.8x for XAUUSD).
- Analyze historical losses to adjust thresholds.

### Improving Accuracy and Confidence
- Tune ML models (hyperparameters, features).
- Implement no-signal logic for low-confidence predictions.
- Add more technical indicators or price action features.
- Backtest extensively and refine based on results.

### Trade Export and Analysis
- Log each trade: entry/exit, SL, P&L, pips.
- Calculate lifetime pips: total earned, bullish/bearish breakdowns, average per trade.
- Export to CSV/JSON for further analysis.

### Running the Models
- Models are run in Google Colab (free).
- Backend serves predictions without running models locally.
- Update models in Colab, then upload to backend.

### Setup
1. Clone the repo.
2. Set up Django backend: `pip install -r requirements.txt`, `python manage.py migrate`, etc.
3. Set up React frontend: `npm install`, `npm start`.
4. Configure free data source (yfinance).
5. Upload ML models to backend.

### Free Services
- Data: yfinance (Yahoo Finance API, free).
- Alerts: Gmail for email (free), free SMS APIs if available (e.g., Twilio trial, but monitor limits).

### Navigation
- Easily switch between EURUSD and XAUUSD views in the frontend.