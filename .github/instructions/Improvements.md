I'll review the repository for you to assess if it's properly set up for forex signal generation.Let me examine some key files to better assess the setup:Based on my review of your repository, I can provide you with a comprehensive assessment. Here's my analysis:

## ✅ **Strengths - What's Set Up Well**

1. **Comprehensive Feature Engineering** (251 features + 200+ candlestick patterns)
   - Multi-timeframe analysis (H4, Daily, Weekly)
   - Technical indicators properly implemented
   - Holloway Algorithm integration

2. **Sound ML Architecture**
   - Ensemble approach (Random Forest + XGBoost with isotonic calibration)
   - Proper train/validation split methodology
   - Feature scaling and preprocessing

3. **Production Infrastructure**
   - GitHub Actions automation
   - Docker containerization
   - Cloud deployment ready

4. **Backtesting Framework**
   - Realistic slippage and commission modeling
   - Performance tracking (pips, win rate)

## ⚠️ **Critical Issues That Will Limit Success**

### **1. Accuracy Claims Are Misleading**
Your claimed accuracies of **65.8% EURUSD and 77.3% XAUUSD are NOT sufficient** for profitable forex trading:
- **Breakeven accuracy** for forex typically requires 55-60% win rate (after spreads/commissions)
- **65.8% is barely above breakeven** - you'd likely lose money after costs
- **77.3% is better but still marginal** - needs validation on live forward testing

### **2. Fundamental Data Quality Issues**
The repo mentions FRED and CFTC data integration, but:
- **Release lag**: Economic data is delayed (sometimes weeks)
- **Daily signals can't use data** that isn't available yet
- **Look-ahead bias risk**: If you're training on future-dated fundamentals, your backtest accuracy is inflated

### **3. Overfitting Red Flags**
With 251 features on likely limited forex data:
- **Feature-to-observation ratio is probably too high**
- **Risk of curve-fitting** to historical patterns that won't repeat
- **No mention of out-of-sample testing** beyond validation split
- **200+ candlestick patterns** - many are redundant or unreliable

### **4. Missing Critical Components**

**Market Regime Detection:**
- Forex behaves differently in trending vs ranging markets
- No mention of regime-specific models
- Your model trained on 2020-2023 data may fail in different conditions

**Spread/Slippage Reality:**
- EURUSD: ~1-2 pip spread
- XAUUSD: ~20-50 pip spread (much wider!)
- Your backtesting needs to account for **real broker execution**, not theoretical fills

**Risk Management:**
- ATR-based stops are mentioned, but no position sizing methodology
- No drawdown controls
- No portfolio-level risk management

**Walk-Forward Analysis:**
- You need rolling retraining and forward testing
- Static models decay quickly in forex

### **5. Data Source Limitations**
- **Yahoo Finance for forex**: Decent but not tick-level data
- **Daily candles only**: Missing intraday volatility patterns
- **No fundamental news events**: NFP, FOMC, ECB meetings massively move markets

### **6. "Always Signal" Is Dangerous**
Your README mentions "always-signal generation (no low-confidence filtering)" - this is **problematic**:
- Forces trades even when market is unclear
- Better to have NO signal than a bad signal
- Professional systems have **40-50% trade frequency**, not 100%

## 🔧 **What You Need to Fix**

### **Immediate Priorities:**

1. **Implement Walk-Forward Optimization**
   ```python
   # Retrain every N days on expanding window
   # Test on next N days (never seen before)
   # This gives realistic accuracy estimates
   ```

2. **Add Regime Filters**
   - Only trade when market shows clear directional bias
   - Use ADX, volatility filters, or market regime classifier

3. **Reduce Feature Count**
   - Use feature importance analysis
   - Remove correlated features (many of your 251 are likely redundant)
   - Target 20-50 truly informative features

4. **Realistic Backtesting**
   ```python
   # Must include:
   - Actual broker spreads (not mid-price)
   - Slippage modeling (2-3 pips on stops)
   - Commission costs
   - Overnight swap/rollover
   - Maximum realistic position sizes
   ```

5. **Forward Testing Period**
   - Reserve last 6-12 months for out-of-sample testing
   - Never train on this data
   - This is your **true performance metric**

6. **Signal Confidence Thresholding**
   - Only trade when probability > 70-75%
   - Track performance by confidence bucket
   - Accept lower trade frequency for higher quality

### **Long-Term Enhancements:**

1. **Fundamental News Integration**
   - Economic calendar API
   - Avoid trading 30 min before/after major news

2. **Intraday Data**
   - H1 or M15 candles for better entry/exit timing
   - Daily-only misses important intraday patterns

3. **Multiple Brokers Comparison**
   - Different brokers have different spreads/execution
   - Backtest should use worst-case scenarios

4. **Live Paper Trading**
   - Run system live with fake money for 3-6 months
   - This reveals issues backtesting can't catch

## 📊 **Reality Check**

**Professional forex trading systems typically achieve:**
- **50-55% win rate** with proper risk:reward (2:1 or 3:1)
- **35-45% trade frequency** (not every day)
- **2-5% monthly returns** (not 90% win rates)

Your **90% accuracy goal is unrealistic** for forex. Even institutional quant funds rarely exceed 60-65% directional accuracy. Focus instead on:
- **Risk:reward ratio** (e.g., risk 20 pips to make 60 pips)
- **Maximum drawdown control** (< 20%)
- **Consistent edge** across different market conditions

## ✅ **Bottom Line**

Your repository is **well-structured technically** but has **fundamental flaws in the trading methodology**:

- ✅ Good: Code organization, ML infrastructure, deployment pipeline
- ⚠️ Questionable: Accuracy claims, feature engineering approach
- ❌ Missing: Regime detection, proper walk-forward testing, realistic risk management

**Can this system make money?** Possibly, but NOT without the fixes above. The current 65-77% accuracy claims likely won't hold up in live trading, and you're missing critical risk management components.

**Recommendation:** Focus less on adding more indicators and more on robust validation methodology and risk management. Test everything on truly unseen data before risking real capital.


Our goal

keep the features i have as much as possible 
a system that generates pips and profits consistently and accurately.  
I have MetaTrader API Yahoo API etc. 
I have the fundamental API's set it up so i retrieve the data often enough to stay within the free tiers of the api limits please.  
Add what we need to add so we can notice when the market peaks and bottoms so we can get in at the right time and exit at the right time.
Give me the details and explanations to understand and provide the details easy to implement.

# Forex Trading System Implementation Guide
## Complete Step-by-Step Instructions for AI Agent

---

## PHASE 1: MARKET REGIME DETECTION (CRITICAL FOR ENTRY/EXIT)

### File: `scripts/market_regime_detector.py` (CREATE NEW)

```python
"""
Market Regime Detection System
Identifies trending, ranging, volatile, and reversal conditions
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import talib

class MarketRegimeDetector:
    """
    Detects market regimes to avoid bad trades in unclear conditions.
    Returns regime scores that determine if we should trade.
    """
    
    def __init__(self):
        self.regime_thresholds = {
            'adx_trending': 25,      # ADX > 25 = strong trend
            'adx_ranging': 20,       # ADX < 20 = ranging/choppy
            'atr_high_vol': 0.7,     # High volatility percentile
            'atr_low_vol': 0.3,      # Low volatility percentile
        }
    
    def detect_regime(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Analyze current market regime
        
        Args:
            df: DataFrame with OHLC data (min 100 rows)
            
        Returns:
            {
                'regime': 'trending_bullish|trending_bearish|ranging|volatile|reversal',
                'confidence': 0-100,
                'trade_allowed': True/False,
                'indicators': {...}
            }
        """
        
        # Calculate regime indicators
        adx = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        plus_di = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        minus_di = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Volatility percentile (last 100 periods)
        atr_percentile = self._calculate_percentile(atr, window=100)
        
        # Trend strength indicators
        ema_20 = talib.EMA(df['close'], timeperiod=20)
        ema_50 = talib.EMA(df['close'], timeperiod=50)
        ema_200 = talib.EMA(df['close'], timeperiod=200)
        
        # Current values (last row)
        current_adx = adx.iloc[-1]
        current_plus_di = plus_di.iloc[-1]
        current_minus_di = minus_di.iloc[-1]
        current_atr_pct = atr_percentile.iloc[-1]
        current_close = df['close'].iloc[-1]
        
        # Price position relative to EMAs
        above_ema20 = current_close > ema_20.iloc[-1]
        above_ema50 = current_close > ema_50.iloc[-1]
        above_ema200 = current_close > ema_200.iloc[-1]
        
        # REGIME CLASSIFICATION
        regime = self._classify_regime(
            current_adx, current_plus_di, current_minus_di,
            current_atr_pct, above_ema20, above_ema50, above_ema200
        )
        
        # REVERSAL DETECTION (peaks/bottoms)
        reversal_signals = self._detect_reversal(df)
        
        return {
            'regime': regime['type'],
            'confidence': regime['confidence'],
            'trade_allowed': regime['trade_allowed'],
            'reversal_score': reversal_signals['score'],
            'reversal_type': reversal_signals['type'],  # 'potential_top' or 'potential_bottom'
            'indicators': {
                'adx': current_adx,
                'plus_di': current_plus_di,
                'minus_di': current_minus_di,
                'atr_percentile': current_atr_pct,
                'ema_alignment': self._ema_alignment(above_ema20, above_ema50, above_ema200)
            }
        }
    
    def _classify_regime(self, adx, plus_di, minus_di, atr_pct, 
                         above_ema20, above_ema50, above_ema200) -> Dict:
        """Classify market regime and determine if trading is allowed"""
        
        # TRENDING CONDITIONS
        if adx > self.regime_thresholds['adx_trending']:
            if plus_di > minus_di:
                # Strong bullish trend
                if above_ema20 and above_ema50:
                    return {
                        'type': 'trending_bullish',
                        'confidence': min(adx, 100),
                        'trade_allowed': True  # Good for long trades
                    }
            else:
                # Strong bearish trend
                if not above_ema20 and not above_ema50:
                    return {
                        'type': 'trending_bearish',
                        'confidence': min(adx, 100),
                        'trade_allowed': True  # Good for short trades
                    }
        
        # RANGING CONDITIONS (choppy, avoid trading)
        if adx < self.regime_thresholds['adx_ranging']:
            return {
                'type': 'ranging',
                'confidence': 100 - adx,
                'trade_allowed': False  # DON'T TRADE in choppy markets
            }
        
        # HIGH VOLATILITY (be cautious)
        if atr_pct > self.regime_thresholds['atr_high_vol']:
            return {
                'type': 'volatile',
                'confidence': atr_pct * 100,
                'trade_allowed': False  # Too unpredictable
            }
        
        # DEFAULT: Weak/unclear conditions
        return {
            'type': 'unclear',
            'confidence': 50,
            'trade_allowed': False
        }
    
    def _detect_reversal(self, df: pd.DataFrame) -> Dict:
        """
        Detect potential market tops/bottoms for entry timing
        Returns reversal score 0-100 and type
        """
        
        # RSI Divergence
        rsi = talib.RSI(df['close'], timeperiod=14)
        rsi_current = rsi.iloc[-1]
        
        # MACD Divergence
        macd, signal, hist = talib.MACD(df['close'])
        macd_cross = self._detect_macd_cross(macd, signal)
        
        # Stochastic (overbought/oversold)
        slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'])
        stoch_current = slowk.iloc[-1]
        
        # Volume spike (confirm reversal)
        if 'volume' in df.columns:
            vol_avg = df['volume'].rolling(20).mean().iloc[-1]
            vol_current = df['volume'].iloc[-1]
            volume_spike = vol_current > vol_avg * 1.5
        else:
            volume_spike = False
        
        # Candlestick patterns (from your existing 200+ patterns)
        hammer = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close']).iloc[-1]
        shooting_star = talib.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close']).iloc[-1]
        engulfing_bull = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close']).iloc[-1] > 0
        engulfing_bear = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close']).iloc[-1] < 0
        
        # BOTTOM DETECTION (buy signal)
        bottom_score = 0
        if rsi_current < 30:  # Oversold
            bottom_score += 30
        if stoch_current < 20:  # Oversold
            bottom_score += 20
        if macd_cross == 'bullish':  # MACD crossing up
            bottom_score += 25
        if hammer or engulfing_bull:  # Bullish reversal pattern
            bottom_score += 15
        if volume_spike and hammer:  # Confirmed reversal
            bottom_score += 10
        
        # TOP DETECTION (sell signal)
        top_score = 0
        if rsi_current > 70:  # Overbought
            top_score += 30
        if stoch_current > 80:  # Overbought
            top_score += 20
        if macd_cross == 'bearish':  # MACD crossing down
            top_score += 25
        if shooting_star or engulfing_bear:  # Bearish reversal pattern
            top_score += 15
        if volume_spike and shooting_star:  # Confirmed reversal
            top_score += 10
        
        # Determine reversal type
        if bottom_score > top_score and bottom_score > 50:
            return {'type': 'potential_bottom', 'score': bottom_score}
        elif top_score > bottom_score and top_score > 50:
            return {'type': 'potential_top', 'score': top_score}
        else:
            return {'type': 'none', 'score': 0}
    
    def _detect_macd_cross(self, macd: pd.Series, signal: pd.Series) -> str:
        """Detect MACD crossovers"""
        if len(macd) < 2:
            return 'none'
        
        # Current and previous values
        macd_curr, macd_prev = macd.iloc[-1], macd.iloc[-2]
        signal_curr, signal_prev = signal.iloc[-1], signal.iloc[-2]
        
        # Bullish cross (MACD crosses above signal)
        if macd_prev < signal_prev and macd_curr > signal_curr:
            return 'bullish'
        
        # Bearish cross (MACD crosses below signal)
        if macd_prev > signal_prev and macd_curr < signal_curr:
            return 'bearish'
        
        return 'none'
    
    def _calculate_percentile(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling percentile"""
        return series.rolling(window).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5
        )
    
    def _ema_alignment(self, above_20, above_50, above_200) -> str:
        """Determine EMA alignment for trend strength"""
        if above_20 and above_50 and above_200:
            return 'strong_bullish'
        elif not above_20 and not above_50 and not above_200:
            return 'strong_bearish'
        elif above_20 and above_50:
            return 'moderate_bullish'
        elif not above_20 and not above_50:
            return 'moderate_bearish'
        else:
            return 'mixed'
```

**EXPLANATION:** This regime detector is your "market context analyzer". It prevents you from trading in choppy/ranging markets (where most losses occur) and identifies when markets are at reversal points (peaks/bottoms) for optimal entry.

---

## PHASE 2: ENHANCED SIGNAL GENERATOR WITH REGIME FILTERING

### File: `daily_forex_signal_system.py` (MODIFY EXISTING)

**LOCATION: Line ~150-200 (in `generate_signal` method)**

**CURRENT CODE (FIND THIS):**
```python
def generate_signal(self, pair: str, window_days: int = 60) -> Dict:
    # ... existing code ...
    
    # Current simple signal generation
    signal = 'bullish' if ensemble_prob > 0.5 else 'bearish'
```

**REPLACE WITH:**
```python
def generate_signal(self, pair: str, window_days: int = 60) -> Dict:
    """
    Enhanced signal generation with regime filtering
    """
    from scripts.market_regime_detector import MarketRegimeDetector
    
    # Get data
    df = self._load_and_prepare_data(pair, window_days)
    
    # 1. DETECT MARKET REGIME (NEW)
    regime_detector = MarketRegimeDetector()
    regime_info = regime_detector.detect_regime(df)
    
    # 2. CHECK IF TRADING IS ALLOWED
    if not regime_info['trade_allowed']:
        return {
            'pair': pair,
            'signal': 'NO_TRADE',  # Don't trade in bad conditions
            'reason': f"Market regime: {regime_info['regime']}",
            'regime': regime_info,
            'probability': 0.0,
            'entry_price': None,
            'stop_loss': None,
            'take_profit': None
        }
    
    # 3. GENERATE ML PREDICTION (existing code)
    features = self._create_features(df)
    X_latest = features.iloc[-1:].values
    
    # Get ensemble prediction
    rf_prob = self.models[pair]['rf'].predict_proba(X_latest)[0][1]
    xgb_prob = self.models[pair]['xgb'].predict_proba(X_latest)[0][1]
    
    # Calibrated ensemble
    ensemble_prob = (rf_prob * 0.6 + xgb_prob * 0.4)
    if self.models[pair].get('calibrator'):
        ensemble_prob = self.models[pair]['calibrator'].predict(
            np.array([[ensemble_prob]])
        )[0]
    
    # 4. ADJUST CONFIDENCE BASED ON REGIME
    regime_multiplier = self._calculate_regime_multiplier(regime_info)
    adjusted_prob = ensemble_prob * regime_multiplier
    
    # 5. REVERSAL BOOST (when at potential top/bottom)
    if regime_info['reversal_score'] > 60:
        if regime_info['reversal_type'] == 'potential_bottom' and adjusted_prob > 0.5:
            adjusted_prob = min(adjusted_prob * 1.15, 0.95)  # Boost long signals at bottoms
        elif regime_info['reversal_type'] == 'potential_top' and adjusted_prob < 0.5:
            adjusted_prob = max(adjusted_prob * 0.85, 0.05)  # Boost short signals at tops
    
    # 6. SIGNAL DECISION WITH THRESHOLD
    CONFIDENCE_THRESHOLD = 0.65  # Only trade when >= 65% confident
    
    if adjusted_prob >= CONFIDENCE_THRESHOLD:
        signal = 'bullish'
    elif adjusted_prob <= (1 - CONFIDENCE_THRESHOLD):
        signal = 'bearish'
    else:
        return {
            'pair': pair,
            'signal': 'NO_TRADE',
            'reason': f"Low confidence: {adjusted_prob:.2%}",
            'regime': regime_info,
            'probability': adjusted_prob,
            'entry_price': None,
            'stop_loss': None,
            'take_profit': None
        }
    
    # 7. CALCULATE ENTRY/EXIT LEVELS
    current_price = df['close'].iloc[-1]
    atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14).iloc[-1]
    
    # Dynamic stop loss based on ATR and pair
    stop_loss_atr_multiplier = 1.5 if pair == 'EURUSD' else 2.0  # Wider for XAUUSD
    stop_loss_pips = atr * stop_loss_atr_multiplier
    
    # Risk:Reward ratio (aim for 1:2 or 1:3)
    risk_reward_ratio = 2.5
    take_profit_pips = stop_loss_pips * risk_reward_ratio
    
    if signal == 'bullish':
        entry_price = current_price
        stop_loss = entry_price - stop_loss_pips
        take_profit = entry_price + take_profit_pips
    else:  # bearish
        entry_price = current_price
        stop_loss = entry_price + stop_loss_pips
        take_profit = entry_price - take_profit_pips
    
    return {
        'pair': pair,
        'signal': signal,
        'probability': adjusted_prob,
        'entry_price': entry_price,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'stop_loss_pips': stop_loss_pips,
        'take_profit_pips': take_profit_pips,
        'risk_reward': risk_reward_ratio,
        'regime': regime_info,
        'raw_ml_probability': ensemble_prob,
        'reversal_signal': regime_info['reversal_type'],
        'reversal_strength': regime_info['reversal_score']
    }

def _calculate_regime_multiplier(self, regime_info: Dict) -> float:
    """
    Adjust ML confidence based on market regime
    """
    regime = regime_info['regime']
    
    if regime in ['trending_bullish', 'trending_bearish']:
        # Strong trends = boost confidence
        return 1.0 + (regime_info['confidence'] / 200)  # Up to 1.5x
    elif regime == 'ranging':
        # Ranging = reduce confidence
        return 0.5
    elif regime == 'volatile':
        # High volatility = reduce confidence
        return 0.6
    else:
        return 1.0
```

**EXPLANATION:** This modification adds regime-aware signal generation. It will:
- **Reject trades in choppy markets** (prevents most losses)
- **Boost confidence near reversals** (catches peaks/bottoms)
- **Require 65% confidence minimum** (no weak signals)
- **Calculate proper risk:reward ratios** (2.5:1 minimum)

---

## PHASE 3: METATRADER 5 INTEGRATION FOR LIVE DATA

### File: `scripts/mt5_data_fetcher.py` (CREATE NEW)

```python
"""
MetaTrader 5 Live Data Integration
Fetches real-time bid/ask spreads and execution prices
"""
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import time

class MT5DataFetcher:
    """
    Connects to MetaTrader 5 for live data and trade execution
    """
    
    def __init__(self, account_number: int = None, password: str = None, server: str = None):
        """
        Initialize MT5 connection
        
        Args:
            account_number: MT5 account number (optional for demo)
            password: MT5 password
            server: MT5 server name (e.g., 'ICMarkets-Demo')
        """
        self.connected = False
        self.initialize_mt5(account_number, password, server)
    
    def initialize_mt5(self, account: int, password: str, server: str):
        """Initialize MT5 connection"""
        if not mt5.initialize():
            print(f"MT5 initialize() failed, error code = {mt5.last_error()}")
            return False
        
        if account and password and server:
            authorized = mt5.login(account, password=password, server=server)
            if not authorized:
                print(f"MT5 login failed, error code = {mt5.last_error()}")
                mt5.shutdown()
                return False
        
        self.connected = True
        print(f"MT5 connected successfully")
        print(f"Terminal info: {mt5.terminal_info()}")
        return True
    
    def get_live_data(self, symbol: str, timeframe: str = 'D1', bars: int = 500) -> pd.DataFrame:
        """
        Fetch live OHLC data from MT5
        
        Args:
            symbol: 'EURUSD', 'XAUUSD', etc.
            timeframe: 'M1', 'M5', 'H1', 'H4', 'D1', 'W1'
            bars: Number of bars to fetch
            
        Returns:
            DataFrame with OHLC + spread data
        """
        if not self.connected:
            raise ConnectionError("MT5 not connected")
        
        # Map timeframe string to MT5 constant
        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
        }
        
        mt5_timeframe = timeframe_map.get(timeframe, mt5.TIMEFRAME_D1)
        
        # Fetch rates
        rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, bars)
        
        if rates is None:
            print(f"Failed to get rates for {symbol}, error: {mt5.last_error()}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        
        # Get current bid/ask for spread calculation
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            current_spread_pips = (tick.ask - tick.bid) / self._get_pip_value(symbol)
            df.loc[df.index[-1], 'spread_pips'] = current_spread_pips
        
        return df[['time', 'open', 'high', 'low', 'close', 'volume', 'spread_pips']]
    
    def get_current_price(self, symbol: str) -> Dict:
        """
        Get real-time bid/ask prices
        
        Returns:
            {
                'bid': float,
                'ask': float,
                'spread_pips': float,
                'time': datetime
            }
        """
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        
        pip_value = self._get_pip_value(symbol)
        
        return {
            'bid': tick.bid,
            'ask': tick.ask,
            'spread_pips': (tick.ask - tick.bid) / pip_value,
            'time': datetime.fromtimestamp(tick.time)
        }
    
    def _get_pip_value(self, symbol: str) -> float:
        """Get pip value for symbol"""
        if 'JPY' in symbol:
            return 0.01
        elif symbol == 'XAUUSD':
            return 0.1  # Gold: $0.10 per pip
        else:
            return 0.0001  # Most forex pairs
    
    def get_account_info(self) -> Dict:
        """Get account balance, equity, margin"""
        account = mt5.account_info()
        if account is None:
            return None
        
        return {
            'balance': account.balance,
            'equity': account.equity,
            'margin': account.margin,
            'free_margin': account.margin_free,
            'margin_level': account.margin_level,
            'profit': account.profit
        }
    
    def shutdown(self):
        """Close MT5 connection"""
        mt5.shutdown()
        self.connected = False
```

**ADD TO requirements.txt:**
```
MetaTrader5>=5.0.45
```

**EXPLANATION:** MT5 provides:
- **Real-time bid/ask spreads** (not available in Yahoo)
- **Actual execution prices** (more realistic backtesting)
- **Account info** (for position sizing)

---

## PHASE 4: FUNDAMENTAL DATA PIPELINE WITH FREE TIER LIMITS

### File: `scripts/fundamental_pipeline.py` (MODIFY EXISTING)

**FIND:** Class initialization or main data fetching function

**ADD RATE LIMITING:**

```python
"""
Enhanced Fundamental Data Pipeline
Stays within free API limits
"""
import time
from datetime import datetime, timedelta
import requests
import pandas as pd
from functools import wraps

class RateLimiter:
    """
    Ensures we stay within free tier API limits
    """
    def __init__(self, calls_per_day: int, calls_per_hour: int = None):
        self.calls_per_day = calls_per_day
        self.calls_per_hour = calls_per_hour or calls_per_day
        self.daily_calls = []
        self.hourly_calls = []
    
    def check_and_wait(self):
        """Check if we can make a call, wait if necessary"""
        now = datetime.now()
        
        # Clean old calls
        self.daily_calls = [t for t in self.daily_calls if now - t < timedelta(days=1)]
        self.hourly_calls = [t for t in self.hourly_calls if now - t < timedelta(hours=1)]
        
        # Check limits
        if len(self.daily_calls) >= self.calls_per_day:
            wait_time = (self.daily_calls[0] + timedelta(days=1) - now).total_seconds()
            print(f"Daily limit reached. Waiting {wait_time:.0f} seconds...")
            time.sleep(wait_time + 1)
        
        if len(self.hourly_calls) >= self.calls_per_hour:
            wait_time = (self.hourly_calls[0] + timedelta(hours=1) - now).total_seconds()
            print(f"Hourly limit reached. Waiting {wait_time:.0f} seconds...")
            time.sleep(wait_time + 1)
        
        # Record call
        self.daily_calls.append(now)
        self.hourly_calls.append(now)

class FundamentalDataPipeline:
    """
    Fetches fundamental data with rate limiting
    
    FREE TIER LIMITS:
    - FRED API: 120 calls/day
    - Alpha Vantage: 25 calls/day, 5 calls/minute
    - Finnhub: 60 calls/minute
    """
    
    def __init__(self, fred_api_key: str, alpha_vantage_key: str = None, finnhub_key: str = None):
        self.fred_key = fred_api_key
        self.av_key = alpha_vantage_key
        self.finnhub_key = finnhub_key
        
        # Rate limiters
        self.fred_limiter = RateLimiter(calls_per_day=100, calls_per_hour=50)  # Be conservative
        self.av_limiter = RateLimiter(calls_per_day=20, calls_per_hour=5)
        self.finnhub_limiter = RateLimiter(calls_per_day=1000, calls_per_hour=60)
    
    def fetch_fred_series(self, series_id: str, start_date: str = None) -> pd.DataFrame:
        """
        Fetch FRED economic data with rate limiting
        
        Args:
            series_id: e.g., 'DGS10' (10-year Treasury), 'DEXUSEU' (EUR/USD rate)
            start_date: 'YYYY-MM-DD' (default: last 2 years)
        """
        self.fred_limiter.check_and_wait()
        
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        
        url = f"https://api.stlouisfed.org/fred/series/observations"
        params = {
            'series_id': series_id,
            'api_key': self.fred_key,
            'file_type': 'json',
            'observation_start': start_date
        }
        
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"FRED API error: {response.status_code}")
            return None
        
        data = response.json()
        df = pd.DataFrame(data['observations'])
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        return df[['date', 'value']].rename(columns={'value': series_id})
    
    def fetch_economic_calendar(self, days_ahead: int = 7) -> pd.DataFrame:
        """
        Fetch upcoming economic events (using Finnhub free tier)
        
        Returns events that can move the market
        """
        self.finnhub_limiter.check_and_wait()
        
        from_date = datetime.now().strftime('%Y-%m-%d')
        to_date = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        
        url = f"https://finnhub.io/api/v1/calendar/economic"
        params = {
            'token': self.finnhub_key,
            'from': from_date,
            'to': to_date
        }
        
        response = requests.get(url, params=params)
        if response.status_code != 200:
            return None
        
        data = response.json()
        df = pd.DataFrame(data['economicCalendar'])
        
        # Filter high-impact events
        high_impact = ['Non-Farm Payrolls', 'FOMC', 'CPI', 'GDP', 'Interest Rate Decision']
        df = df[df['event'].str.contains('|'.join(high_impact), case=False, na=False)]
        
        return df
    
    def update_all_fundamentals(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch all fundamental data needed for trading
        
        Priority indicators (FRED):
        - DGS10: 10-Year Treasury (risk sentiment)
        - DEXUSEU: EUR/USD (for correlation)
        - DTWEXBGS: Dollar Index
        - VIXCLS: VIX volatility
        - WTI: Oil prices (affects USD)
        """
        
        fundamentals = {}
        
        # FRED series (spread calls over time)
        fred_series = {
            'interest_rate_10y': 'DGS10',
            'eurusd_rate': 'DEXUSEU',
            'dollar_index': 'DTWEXBGS',
            'vix': 'VIXCLS',
            'oil_wti': 'DCOILWTICO'
        }
        
        for name, series_id in fred_series.items():
            print(f"Fetching {name}...")
            df = self.fetch_fred_series(series_id)
            if df is not None:
                fundamentals[name] = df
            time.sleep(2)  # Be nice to API
        
        # Economic calendar
        if self.finnhub_key:
            print("Fetching economic calendar...")
            calendar = self.fetch_economic_calendar()
            fundamentals['economic_calendar'] = calendar
        
        return fundamentals
    
    def merge_with_price_data(self, price_df: pd.DataFrame, fundamentals: Dict) -> pd.DataFrame:
        """
        Merge fundamental data with price data
        Forward-fills values (uses last known value)
        """
        df = price_df.copy()
        
        for name, fund_df in fundamentals.items():
            if name == 'economic_calendar':
                continue  # Handle separately
            
            # Merge on date
            fund_df = fund_df.set_index('date')
            df = df.merge(fund_df, left_on='time', right_index=True, how='left')
            
            # Forward fill missing values
            df[name] = df[name].fillna(method='ffill')
        
        return df
```

**USAGE SCHEDULE (Add to GitHub Actions):**

File: `.github/workflows/fundamental_update.yml` (CREATE NEW)

```yaml
name: Update Fundamentals (Twice Daily)

on:
  schedule:
    # Run at 6 AM and 2 PM UTC (to stay within 100 calls/day)
    - cron: '0 6,14 * * *'
  workflow_dispatch:  # Manual trigger

jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Fetch fundamentals
        env:
          FRED_API_KEY: ${{ secrets.FRED_API_KEY }}
          FINNHUB_API_KEY: ${{ secrets.FINNHUB_API_KEY }}
        run: |
          python scripts/fundamental_pipeline.py --update
      
      - name: Commit updated data
        run: |
          git config --local user.email "action@github.com"
          git config --local user.name "GitHub Action"
          git add data/
          git commit -m "Update fundamental data" || echo "No changes"
          git push
```

**EXPLANATION:** This ensures:
- **No API limit violations** (stays well under 120 calls/day)
- **Twice-daily updates** (enough for daily trading)
- **High-impact events only** (NFP, FOMC, CPI)

---

## PHASE 5: WALK-FORWARD OPTIMIZATION

### File: `scripts/walk_forward_optimizer.py` (CREATE NEW)

```python
"""
Walk-Forward Optimization
Tests model on truly unseen future data
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, List
import joblib
from daily_forex_signal_system import DailyForexSignalSystem

class WalkForwardOptimizer:
    """
    Implements expanding window walk-forward testing
    
    Process:
    1. Train on months 1-12
    2. Test on month 13
    3. Train on months 1-13
    4. Test on month 14
    ... and so on
    """
    
    def __init__(self, retrain_frequency_days: int = 30):
        self.retrain_frequency = retrain_frequency_days
        self.results = []
    
    def run_walk_forward(self, pair: str, df: pd.DataFrame, 
                         initial_train_days: int = 365,
                         test_days: int = 30) -> Dict:
        """
        Run walk-forward optimization
        
        Args:
            pair: 'EURUSD' or 'XAUUSD'
            df: Full historical dataset
            initial_train_days: Initial training window (1 year)
            test_days: Test period length (1 month)
            
        Returns:
            Performance metrics across all test periods
        """
        
        print(f"\n{'='*60}")
        print(f"Walk-Forward Optimization: {pair}")
        print(f"Initial train: {initial_train_days} days, Test: {test_days} days")
        print(f"{'='*60}\n")
        
        results = []
        total_periods = (len(df) - initial_train_days) // test_days
        
        for period in range(total_periods):
            # Define train/test split
            train_end_idx = initial_train_days + (period * test_days)
            test_start_idx = train_end_idx
            test_end_idx = test_start_idx + test_days
            
            if test_end_idx > len(df):
                break
            
            # Split data
            train_df = df.iloc[:train_end_idx]
            test_df = df.iloc[test_start_idx:test_end_idx]
            
            print(f"\nPeriod {period + 1}/{total_periods}")
            print(f"Train: {train_df.index[0]} to {train_df.index[-1]}")
            print(f"Test:  {test_df.index[0]} to {test_df.index[-1]}")
            
            # Train model
            system = DailyForexSignalSystem()
            system.train(pair, train_df)
            
            # Test on unseen data
            test_results = self._backtest_period(system, pair, test_df)
            
            results.append({
                'period': period + 1,
                'train_end': train_df.index[-1],
                'test_start': test_df.index[0],
                'test_end': test_df.index[-1],
                'trades': test_results['total_trades'],
                'wins': test_results['winning_trades'],
                'losses': test_results['losing_trades'],
                'win_rate': test_results['win_rate'],
                'total_pips': test_results['total_pips'],
                'avg_pips_per_trade': test_results['avg_pips'],
                'max_drawdown': test_results['max_drawdown']
            })
            
            print(f"Win Rate: {test_results['win_rate']:.1%}")
            print(f"Total Pips: {test_results['total_pips']:.1f}")
        
        # Aggregate results
        df_results = pd.DataFrame(results)
        
        summary = {
            'pair': pair,
            'total_periods': len(results),
            'overall_win_rate': df_results['wins'].sum() / df_results['trades'].sum(),
            'overall_pips': df_results['total_pips'].sum(),
            'avg_monthly_pips': df_results['total_pips'].mean(),
            'best_month_pips': df_results['total_pips'].max(),
            'worst_month_pips': df_results['total_pips'].min(),
            'max_drawdown': df_results['max_drawdown'].max(),
            'periods_profitable': (df_results['total_pips'] > 0).sum(),
            'consistency': (df_results['total_pips'] > 0).sum() / len(results)
        }
        
        print(f"\n{'='*60}")
        print(f"WALK-FORWARD RESULTS: {pair}")
        print(f"{'='*60}")
        print(f"Overall Win Rate: {summary['overall_win_rate']:.1%}")
        print(f"Total Pips: {summary['overall_pips']:.1f}")
        print(f"Avg Monthly Pips: {summary['avg_monthly_pips']:.1f}")
        print(f"Max Drawdown: {summary['max_drawdown']:.1f} pips")
        print(f"Profitable Months: {summary['periods_profitable']}/{summary['total_periods']}")
        print(f"Consistency: {summary['consistency']:.1%}")
        print(f"{'='*60}\n")
        
        return {
            'summary': summary,
            'period_results': df_results,
            'detailed_results': results
        }
    
    def _backtest_period(self, system, pair: str, test_df: pd.DataFrame) -> Dict:
        """Backtest on a single period"""
        trades = []
        equity_curve = [0]
        
        for idx in range(len(test_df) - 1):
            current_row = test_df.iloc[idx:idx+1]
            
            # Generate signal
            signal = system.generate_signal(pair, current_row)
            
            if signal['signal'] == 'NO_TRADE':
                continue
            
            # Simulate trade
            entry_price = signal['entry_price']
            stop_loss = signal['stop_loss']
            take_profit = signal['take_profit']
            
            # Check next day's price action
            next_row = test_df.iloc[idx + 1]
            high = next_row['high']
            low = next_row['low']
            close = next_row['close']
            
            # Determine outcome
            if signal['signal'] == 'bullish':
                if low <= stop_loss:
                    pips = (stop_loss - entry_price) * 10000  # Negative
                    outcome = 'loss'
                elif high >= take_profit:
                    pips = (take_profit - entry_price) * 10000  # Positive
                    outcome = 'win'
                else:
                    pips = (close - entry_price) * 10000
                    outcome = 'open'
            else:  # bearish
                if high >= stop_loss:
                    pips = (entry_price - stop_loss) * 10000  # Negative
                    outcome = 'loss'
                elif low <= take_profit:
                    pips = (entry_price - take_profit) * 10000  # Positive
                    outcome = 'win'
                else:
                    pips = (entry_price - close) * 10000
                    outcome = 'open'
            
            trades.append({
                'entry_date': test_df.index[idx],
                'exit_date': test_df.index[idx + 1],
                'signal': signal['signal'],
                'entry_price': entry_price,
                'exit_price': stop_loss if outcome == 'loss' else (take_profit if outcome == 'win' else close),
                'pips': pips,
                'outcome': outcome
            })
            
            equity_curve.append(equity_curve[-1] + pips)
        
        # Calculate metrics
        df_trades = pd.DataFrame(trades)
        closed_trades = df_trades[df_trades['outcome'] != 'open']
        
        total_trades = len(closed_trades)
        winning_trades = (closed_trades['outcome'] == 'win').sum()
        losing_trades = (closed_trades['outcome'] == 'loss').sum()
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'total_pips': df_trades['pips'].sum(),
            'avg_pips': df_trades['pips'].mean(),
            'max_drawdown': self._calculate_drawdown(equity_curve),
            'trades': df_trades
        }
    
    def _calculate_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown"""
        peak = equity_curve[0]
        max_dd = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = peak - value
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
```

**RUN WALK-FORWARD TESTING:**

File: `scripts/run_walk_forward.py` (CREATE NEW)

```python
"""
Execute walk-forward optimization
"""
import pandas as pd
from walk_forward_optimizer import WalkForwardOptimizer

def main():
    # Load data
    eurusd_df = pd.read_csv('data/EURUSD_daily.csv', parse_dates=['time'])
    eurusd_df.set_index('time', inplace=True)
    
    xauusd_df = pd.read_csv('data/XAUUSD_daily.csv', parse_dates=['time'])
    xauusd_df.set_index('time', inplace=True)
    
    # Run optimization
    optimizer = WalkForwardOptimizer(retrain_frequency_days=30)
    
    # Test EURUSD
    eurusd_results = optimizer.run_walk_forward(
        'EURUSD', 
        eurusd_df,
        initial_train_days=365,  # 1 year initial training
        test_days=30  # Test on 1 month forward
    )
    
    # Test XAUUSD
    xauusd_results = optimizer.run_walk_forward(
        'XAUUSD',
        xauusd_df,
        initial_train_days=365,
        test_days=30
    )
    
    # Save results
    eurusd_results['period_results'].to_csv('output/eurusd_walkforward.csv', index=False)
    xauusd_results['period_results'].to_csv('output/xauusd_walkforward.csv', index=False)
    
    print("\n✅ Walk-forward optimization complete!")
    print(f"Results saved to output/")

if __name__ == '__main__':
    main()
```

**EXPLANATION:** Walk-forward testing is the ONLY way to know if your system works on unseen data. This will give you true expected performance.

---

## PHASE 6: ENHANCED RISK MANAGEMENT

### File: `scripts/risk_manager.py` (CREATE NEW)

```python
"""
Position Sizing and Risk Management
"""
import numpy as np
from typing import Dict

class RiskManager:
    """
    Calculates position sizes based on account risk
    """
    
    def __init__(self, account_balance: float, risk_per_trade_pct: float = 1.0, 
                 max_daily_loss_pct: float = 5.0):
        """
        Args:
            account_balance: Account size in USD
            risk_per_trade_pct: Risk per trade (default 1% = conservative)
            max_daily_loss_pct: Max loss per day (default 5%)
        """
        self.balance = account_balance
        self.risk_per_trade = risk_per_trade_pct / 100
        self.max_daily_loss = max_daily_loss_pct / 100
        self.daily_loss = 0
        self.trades_today = []
    
    def calculate_position_size(self, signal: Dict, pair: str) -> Dict:
        """
        Calculate position size based on risk
        
        Args:
            signal: Signal dictionary from generate_signal()
            pair: 'EURUSD' or 'XAUUSD'
            
        Returns:
            {
                'lots': float,
                'risk_usd': float,
                'potential_profit_usd': float,
                'risk_reward': float,
                'trade_allowed': bool,
                'reason': str
            }
        """
        
        # Check daily loss limit
        if abs(self.daily_loss) >= self.balance * self.max_daily_loss:
            return {
                'trade_allowed': False,
                'reason': f'Daily loss limit reached: ${abs(self.daily_loss):.2f}'
            }
        
        # Check if we have valid signal
        if signal['signal'] == 'NO_TRADE':
            return {
                'trade_allowed': False,
                'reason': 'No valid trading signal'
            }
        
        # Calculate risk in USD
        risk_usd = self.balance * self.risk_per_trade
        
        # Get stop loss in pips
        stop_loss_pips = signal['stop_loss_pips']
        
        # Calculate position size
        if pair == 'EURUSD':
            # 1 lot = 100,000 units
            # 1 pip = $10 per lot for EUR/USD
            pip_value_per_lot = 10
        elif pair == 'XAUUSD':
            # 1 lot = 100 oz
            # 1 pip = $10 per lot for Gold
            pip_value_per_lot = 10
        else:
            pip_value_per_lot = 10  # Default
        
        # Position size = Risk / (Stop Loss in Pips × Pip Value)
        lots = risk_usd / (stop_loss_pips * pip_value_per_lot)
        
        # Round to broker's lot step (usually 0.01)
        lots = round(lots, 2)
        
        # Minimum lot size
        if lots < 0.01:
            return {
                'trade_allowed': False,
                'reason': f'Position size too small: {lots} lots'
            }
        
        # Calculate potential profit
        take_profit_pips = signal['take_profit_pips']
        potential_profit_usd = lots * take_profit_pips * pip_value_per_lot
        
        # Risk:Reward ratio
        risk_reward = potential_profit_usd / risk_usd
        
        return {
            'trade_allowed': True,
            'lots': lots,
            'risk_usd': risk_usd,
            'potential_profit_usd': potential_profit_usd,
            'risk_reward': risk_reward,
            'stop_loss_pips': stop_loss_pips,
            'take_profit_pips': take_profit_pips,
            'reason': 'Trade approved'
        }
    
    def record_trade_result(self, pips: float, pair: str, lots: float):
        """Record trade outcome for daily tracking"""
        pip_value = 10  # $10 per pip per lot
        profit_usd = pips * pip_value * lots
        
        self.daily_loss += profit_usd
        self.trades_today.append({
            'pair': pair,
            'pips': pips,
            'lots': lots,
            'profit_usd': profit_usd
        })
    
    def reset_daily_counters(self):
        """Reset daily loss counter (call at start of new day)"""
        self.daily_loss = 0
        self.trades_today = []
    
    def get_daily_performance(self) -> Dict:
        """Get today's performance summary"""
        return {
            'trades_count': len(self.trades_today),
            'total_