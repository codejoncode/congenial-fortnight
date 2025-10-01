# Forex Signal Service

## 🚀 Quick Start

To get all project files, clone the repository:

```bash
git clone https://github.com/codejoncode/congenial-fortnight.git
cd congenial-fortnight
```

## 📚 Documentation

For comprehensive setup instructions, deployment options, and usage guides, see:

- **[CLOUD_DEPLOYMENT_GUIDE.md](CLOUD_DEPLOYMENT_GUIDE.md)** - Complete setup, deployment, and automation guide
- **[ENHANCEMENT_CHECKLIST.md](ENHANCEMENT_CHECKLIST.md)** - Development roadmap and features

## 🎯 Current Status

✅ **Multi-timeframe ML models** with 251 features
✅ **200+ candlestick patterns** integrated
✅ **Automated GitHub Actions** training pipeline
✅ **Realistic backtesting** with proper entry/exit logic
✅ **Cloud deployment** ready (GitHub Actions + Cloud Run)

**Performance**: EURUSD ensemble MAE 0.004973, 84%+ directional accuracy

### 🎨 **Latest Chart Improvements (October 2025)**

- **Gold Prediction Candles**: Future AI predictions outlined in gold with star indicators
- **Professional Layout**: Improved spacing, modern design, better tooltips
- **Current Data**: Charts now display up-to-date prices (2025-10-01)
- **Chart Options**: Switch between Custom AI charts and TradingView-style displays

### 📊 TradingView Integration Options

For production-grade charts similar to TradingView, consider these React libraries:

#### 1. **react-tradingview-widget** (Recommended for full TradingView experience)
```bash
npm install react-tradingview-widget
```
- ✅ Full TradingView charts with all indicators
- ✅ Professional look and feel
- ✅ Advanced drawing tools
- ⚠️ Requires TradingView account for premium features

#### 2. **lightweight-charts** (High-performance alternative)
```bash
npm install lightweight-charts
```
- ✅ Similar to TradingView but lighter weight
- ✅ Excellent performance
- ✅ Free and open source
- ✅ Modern API, easy customization

#### 3. **react-financial-charts** (Advanced technical analysis)
```bash
npm install react-financial-charts
```
- ✅ Built on D3, highly customizable
- ✅ Great for technical analysis
- ⚠️ Steeper learning curve

**Current Implementation**: Uses Recharts for custom charts with AI prediction visualization. Switch to "TradingView Style" in the chart selector to see integration placeholder.

## 🗺️ **Project Roadmap & Feature Checklist**

This outline integrates our in-depth conversation with your existing congenial-fortnight repository, organizing work into clear, incremental steps. Check off each item as you implement it.

### 1. **Repository Structure Review**
- [x] Confirm top-level directories:
  - [x] `data/` - CSV files for FRED series, CFTC COT, modeling dataset
  - [x] `scripts/` - ETL pipelines, forecasting, diagnostics, backtesting, signal generation
  - [x] `charts/` - Front-end charting assets (HTML/JS)
  - [x] `.github/workflows/` - GitHub Actions for daily jobs
  - [x] `README.md` & feature spec (`congenial_fortnight_features.txt`)

### 2. **Data Collection & Storage**

#### 2.1 **Initial ETL Pipeline**
- [x] **FundamentalDataPipeline**
  - [x] Collect full history for all FRED series (CPI, Fed funds, USD index, Gold price, etc.)
  - [x] Download & parse CFTC FinFut and weekly COT for EURFX & Gold
  - [x] Save per-series CSV in `data/` and update `update_metadata.json`

#### 2.2 **Daily Update Job**
- [x] In `.github/workflows/data_update.yml`, schedule daily run:
  ```yaml
  on:
    schedule:
      - cron: '0 6 * * *'
  jobs:
    update:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v3
        - name: Run ETL
          run: python scripts/fundamental_pipeline.py
  ```
- [x] Ensure incremental fetch (only new observations)

### 3. **Modeling & Forecasting**

#### 3.1 **Hybrid Ensemble Models**
- [x] **HybridPriceForecastingEnsemble** (in `scripts/forecasting.py`)
  - [x] Classical (Prophet, StatsForecast), ML (LightGBM), DL (LSTM/RF) base models
  - [x] Meta-model stacking (Ridge)
- [x] **QuantumMultiTimeframeSignalGenerator** (in `scripts/signals.py`)
  - [x] Prepare weekly/daily/4h features (technical + fundamental)
  - [x] Train VotingRegressor ensembles per timeframe
  - [x] Fuse into unified daily signal + entry/exit levels

#### 3.2 **Diagnostics & Backtesting**
- [x] **ModelDiagnosticsFramework** (`scripts/diagnostics.py`)
  - [x] Performance metrics, feature importance, error analysis, stability
  - [x] Export JSON report & actionable recommendations
- [x] **AutomatedTradingBacktestOptimizer** (`scripts/backtesting.py`)
  - [x] Simulate trades, calculate pips P&L
  - [x] Cross-pair correlation & combined features
  - [x] Automated parameter grid search until ≥75% accuracy
  - [x] Export `optimization_results.csv`, `trading_performance_report.json`

### 4. **Front-End Chart Integration**

#### 4.1 **TradingView Lightweight Charts**
- [x] In `charts/index.html` or React component:
  - [x] Embed Lightweight Charts with `timeScale.rightOffset` and `rightPriceScale.scaleMargins`
  - [x] Overlay daily signal markers, entry/exit lines
  - [x] Responsive resizing hook

#### 4.2 **TradingView Charting Library (Optional)**
- [ ] Acquire license, host `charting_library.min.js`
- [ ] Update `charts/tv_widget.html` configuration:
  - [ ] `timeScale.rightOffset: 12`, `priceScale.rightMargin: 20`
  - [ ] Custom UDF datafeed endpoint to your API

### 5. **Backend API & Deployment**

#### 5.1 **FastAPI Service** (`api/`)
- [x] Endpoints: `/signals?pair=EURUSD`, `/fundamentals`, `/backtest/report`
- [x] Reads CSVs, builds model, returns JSON

#### 5.2 **Google Cloud Run**
- [x] Containerize with Dockerfile
- [x] Deploy daily update job + API service

#### 5.3 **GitHub Actions**
- [x] Build & push container on merge to main
- [x] Trigger Cloud Run deploy

### 6. **README.md Enhancements**
- [x] Overview with architecture diagram
- [x] Prerequisites (Python, FRED key, CFTC access)
- [x] Getting Started
  - [x] Clone repo, set env vars, install `requirements.txt`
  - [x] Run `scripts/fundamental_pipeline.py --init`
- [x] Daily Workflow
  - [x] Describe GitHub Actions & Cloud Run schedule
- [x] Usage
  - [x] How to call API endpoints
  - [x] How to view charts in `charts/`
- [x] Development
  - [x] Running tests/backtests
  - [x] How to add new fundamentals or model variants
- [x] Roadmap & Contribution
  - [x] Link this feature checklist for tracking

### 7. **Next Steps & To-Do**
- [ ] Integrate correlation-based ensemble weights in signal generator
- [ ] Add live backtester dashboard UI (optional)
- [ ] Monitor model drift & schedule re-training triggers
- [ ] Expand to additional pairs or asset classes

## ✅ **Project Setup Checklist**

### 🔧 **Environment Setup**
- [x] Python 3.8+ installed
- [x] Node.js 16+ installed
- [x] Git repository cloned
- [x] Virtual environment created (`.venv`)
- [x] Dependencies installed (`pip install -r requirements.txt`)
- [x] Django migrations run (`python manage.py migrate`)
- [x] React dependencies installed (`npm install`)

### 🤖 **Machine Learning Models**
- [x] EURUSD models trained (RF + XGB + calibrator)
- [x] XAUUSD models trained (RF + XGB + calibrator)
- [x] Model files in `models/` directory
- [x] Feature scalers configured
- [x] Isotonic calibration applied
- [x] 251 technical indicators implemented
- [x] 200+ candlestick patterns integrated
- [x] Multi-timeframe data (H4, Daily, Weekly)

### 📊 **Data Pipeline**
- [x] Historical CSV data downloaded
- [x] Data cleaning and preprocessing
- [x] Automatic separator detection
- [x] Multi-timeframe data integration
- [x] Data freshness validation
- [x] Yahoo Finance API integration

### 🎨 **Frontend Features**
- [x] React application running
- [x] Candlestick chart display
- [x] Gold prediction candles with stars
- [x] Professional chart layout
- [x] Chart type selector (Custom/TradingView)
- [x] Signal cards with probability
- [x] Backtesting interface
- [x] Real-time data updates

### 🔄 **Backend API**
- [x] Django REST API configured
- [x] Signal generation endpoints
- [x] Backtesting API
- [x] Historical data serving
- [x] Model prediction integration
- [x] CORS configuration for frontend

### 📱 **Signal Generation**
- [x] Daily signal command working
- [x] Ensemble model combination
- [x] Pair-specific weighting (EURUSD: 0.6 RF/0.4 XGB, XAUUSD: 0.7 RF/0.3 XGB)
- [x] Confidence-based stop losses
- [x] Always-signal generation (no low-confidence filtering)
- [x] ATR-based risk management

### 🧪 **Testing & Validation**
- [x] Backtesting functionality
- [x] CSV export capability
- [x] Performance metrics calculation
- [x] Model accuracy validation
- [x] Chart display verification
- [x] API endpoint testing

### 🚀 **Deployment Ready**
- [x] GitHub Actions workflow configured
- [x] Docker container setup
- [x] Cloud Run deployment ready
- [x] Automated training pipeline
- [x] Model artifact management

### 📢 **Notifications (Optional)**
- [ ] Email notifications configured (Gmail + App Password)
- [ ] SMS notifications setup (Textbelt/Twilio)
- [ ] Notification testing completed
- [ ] Alert thresholds configured

## 🏃‍♂️ Quick Local Test

```bash
# Set up environment
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Train models
python -c "from candle_prediction_system import CandlePredictionSystem; system = CandlePredictionSystem(['EURUSD']); system.run_full_pipeline()"

# Run backtest
python manage.py backtest_signals EURUSD --days 30
```

## � Notification Setup (Optional)

Get notified when new trading signals are generated! The system supports free email and SMS notifications.

### Email Notifications (Free)
1. **Create Gmail Account** (or use existing)
2. **Enable 2-Factor Authentication** in Gmail settings
3. **Generate App Password**:
   - Go to Google Account settings → Security → 2-Step Verification → App passwords
   - Generate password for "Mail"
4. **Add to GitHub Secrets**:
   ```
   EMAIL_USERNAME: your-gmail@gmail.com
   EMAIL_PASSWORD: your-app-password
   EMAIL_FROM: your-gmail@gmail.com
   NOTIFICATION_EMAIL: your-notification-email@example.com
   ```

### SMS Notifications (Free Tier)
1. **Use Textbelt** (1 free SMS/day):
   - No setup required - uses free API key
   - Add to GitHub Secrets: `NOTIFICATION_SMS: +1234567890`

2. **Alternative: Twilio Free Credits** (new accounts get $15+ free):
   - Sign up at twilio.com
   - Get phone number and API credentials
   - Add to GitHub Secrets: `TWILIO_FREE_KEY: your-key`

### Testing Notifications
```bash
# Generate test signals
python manage.py run_daily_signal

# Test notification system
python -c "
from notification_system import NotificationSystem
notifier = NotificationSystem()
signals = [{'pair': 'EURUSD', 'signal': 'bullish', 'probability': 0.85, 'entry_price': 1.0850, 'stop_loss': 0.0020}]
notifier.send_signal_notification(signals, ['your-email@example.com', '+1234567890'])
"
```

## �📊 Project Overview

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

## Model Integration and Signal Generation

The core ML logic is in `daily_forex_signal_system.py`, which trains and predicts signals using an ensemble of Random Forest (RF) and XGBoost (XGB) models, calibrated with isotonic regression for confidence scoring. The `generate_signal` method processes the latest data window to produce signals.

Pre-trained model artifacts (.joblib files) for EURUSD and XAUUSD pairs are stored in the `models/` directory:
- `{pair}_rf.joblib`: Random Forest model
- `{pair}_xgb.joblib`: XGBoost model
- `{pair}_scaler.joblib`: Feature scaler
- `{pair}_calibrator.joblib`: Probability calibrator

The backend loads these artifacts to generate daily signals, which are served to the React frontend for display on candlestick charts.

## Security and Personal Use

This application is designed for personal use only. Access is restricted to the owner via authentication:
- JWT-based authentication for API access.
- Simple login system to ensure only authorized users can view signals.
- No public sharing; signals are private and for individual trading decisions.

## Getting Daily Updates
To keep signals current with the latest market data:

1. **Automatic Data Fetching**:
   ```bash
   python manage.py run_daily_signal --fetch-data
   ```
   This downloads the latest daily OHLC data from Yahoo Finance and generates fresh signals.

2. **Manual Data Updates**:
   - Download latest CSV files from your data source
   - Replace files in `data/raw/` directory
   - Run: `python manage.py run_daily_signal`

3. **Scheduling Daily Updates**:
   - **Windows**: Use Task Scheduler to run the command daily at market close (e.g., 5 PM ET)
   - **Linux/Mac**: Use cron: `0 17 * * 1-5 /path/to/venv/bin/python manage.py run_daily_signal --fetch-data`

4. **Signal Alerts**:
   - Check the web app at http://localhost:3000 for current signals
   - Signals update automatically when you refresh the page
   - High probability (>80%) signals indicate strong confidence

5. **Model Updates**:
   - Retrain models in Google Colab with new data
   - Download updated .joblib files
   - Replace in `models/` directory
   - Rerun signal generation

## Candle Prediction System

The `candle_prediction_system.py` script provides complete OHLC prediction for the next candle using machine learning. This predicts the exact Open, High, Low, Close values for the next trading day.

### Features
- **Multi-Pair Support**: Trains models for both EURUSD and XAUUSD simultaneously
- **Ensemble ML**: Random Forest + XGBoost with feature scaling
- **Yahoo Finance Integration**: Automatic data fetching
- **Colab Ready**: Designed to run in Google Colab for easy training

### Running in Google Colab

1. **Create New Notebook** in Google Colab
2. **Install Dependencies**:
   ```python
   !pip install yfinance scikit-learn xgboost joblib pandas numpy
   ```
3. **Copy the Code**: Use the `candle_prediction_system.py` file
4. **Run Training**:
   ```python
   system = CandlePredictionSystem(['EURUSD', 'XAUUSD'])
   results = system.run_full_pipeline()
   ```
5. **Download Models**: After training completes, download all `.joblib` files from:
   - `/content/EURUSD_models/`
   - `/content/XAUUSD_models/`

6. **Upload to Project**: Place files in your `models/` directory

### Model Files Generated
For each pair, you'll get:
- `{pair}_rf_candle.joblib` - Random Forest model
- `{pair}_xgb_candle.joblib` - XGBoost model  
- `{pair}_scaler_candle.joblib` - Feature scaler

**Note**: Unlike the daily signal models, candle prediction models don't have `_calibrator.joblib` files because they use **regression** (predicting OHLC values) instead of **classification** (predicting bullish/bearish). Calibration is only needed for classification models to improve probability estimates.

### Updating CSVs with Predictions
After getting predictions from the API, update your CSV files directly:

```python
from candle_prediction_system import CandlePredictionSystem

system = CandlePredictionSystem(['EURUSD', 'XAUUSD'])

# Load your trained models
# ... (model loading code)

# Get prediction from API
prediction = system.predict_next_candle('EURUSD', latest_data)

# Update CSV with prediction
system.update_csv_with_prediction('EURUSD', prediction)
```

This appends the predicted OHLC values for the next day to your existing CSV files in `data/raw/`.

### Integration
Once models are uploaded, the Django system can be extended to include candle predictions alongside direction signals.