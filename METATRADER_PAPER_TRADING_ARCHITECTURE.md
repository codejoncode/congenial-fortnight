# 📊 MetaTrader Paper Trading & TradingView Integration - Enterprise Architecture

## 🎯 Project Overview

**Objective:** Build an enterprise-level paper trading system that:
- Connects to MetaTrader 4/5 for order execution simulation
- Displays live charts via TradingView
- Shows all signals, patterns, and trade levels in real-time
- Maintains free-tier API usage while maximizing data quality
- Scales to pro account when needed

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          FRONTEND (React)                            │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐ │
│  │  TradingView     │  │  Signal Display  │  │  Order Manager   │ │
│  │  Chart Widget    │  │  & Patterns      │  │  & Positions     │ │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                            WebSocket / REST API
                                    │
┌─────────────────────────────────────────────────────────────────────┐
│                      DJANGO BACKEND (API Layer)                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐ │
│  │  Signal Service  │  │  Paper Trading   │  │  Data Aggregator │ │
│  │  (ML + Harmonic) │  │  Engine          │  │  (Multi-Source)  │ │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
┌───────────────────▼──┐  ┌────────▼────────┐  ┌──▼──────────────────┐
│  MetaTrader Bridge   │  │  Free Data APIs │  │  WebSocket Streams  │
│  (MT4/MT5 Python)    │  │  • Alpha Vantage│  │  • Binance (Crypto) │
│  • ZeroMQ or REST    │  │  • Twelve Data  │  │  • Finnhub          │
│  • Order Simulation  │  │  • Yahoo Finance│  │  • Polygon.io       │
│  • Position Tracking │  │  • FRED (Macro) │  │  (Free tier)        │
└──────────────────────┘  └─────────────────┘  └─────────────────────┘
```

---

## 📋 Component Breakdown

### **1. Frontend Components**

#### **A. TradingView Chart Integration**
```javascript
// components/TradingViewChart.js
- Lightweight Charts or TradingView Widget
- Real-time price updates
- Signal overlays (arrows, boxes)
- Support/resistance levels
- Pattern drawings (harmonic, chart patterns)
- Trade entry/SL/TP visualization
```

#### **B. Signal Display Panel**
```javascript
// components/SignalPanel.js
- Live signal feed (ML, Harmonic, Quantum)
- Confidence scores
- R:R ratios
- Entry price, SL, TP levels
- Pattern details (for harmonic)
- One-click paper trade execution
```

#### **C. Order Management**
```javascript
// components/OrderManager.js
- Open positions display
- Pending orders
- Trade history
- P&L tracking (pips and $)
- Position sizing calculator
- Risk management tools
```

#### **D. Performance Dashboard**
```javascript
// components/PerformanceDashboard.js
- Win rate statistics
- Average R:R achieved
- Total pips gained/lost
- Drawdown tracking
- Equity curve
- Trade distribution charts
```

---

### **2. Backend Services**

#### **A. Paper Trading Engine** (`paper_trading_engine.py`)
```python
class PaperTradingEngine:
    """
    Simulates MetaTrader order execution without real money
    Tracks positions, calculates P&L, manages risk
    """
    - execute_order(signal, lot_size)
    - close_position(position_id, price)
    - update_positions(current_prices)
    - calculate_pnl(position)
    - check_stop_loss_take_profit()
    - get_open_positions()
    - get_trade_history()
```

#### **B. MetaTrader Bridge** (`mt_bridge.py`)
```python
# Options for MT4/MT5 connection:
# 1. MetaTrader5 Python package (official, MT5 only)
# 2. ZeroMQ bridge (MT4/MT5, more flexible)
# 3. REST API wrapper (custom solution)

class MetaTraderBridge:
    - connect()
    - get_current_price(symbol)
    - get_historical_data(symbol, timeframe, bars)
    - simulate_order(order_type, symbol, volume, sl, tp)
    - get_account_info()
    - get_positions()
```

#### **C. Data Aggregation Service** (`data_aggregator.py`)
```python
class DataAggregator:
    """
    Manages multiple free-tier API connections
    Rotates sources to stay within limits
    Caches data to minimize API calls
    """
    sources = {
        'alpha_vantage': AlphaVantageAPI,
        'twelve_data': TwelveDataAPI,
        'yahoo_finance': YahooFinanceAPI,
        'finnhub': FinnhubAPI,
        'polygon': PolygonAPI,
        'binance': BinanceWebSocket  # For crypto
    }
    
    - get_realtime_price(symbol)
    - get_historical_ohlc(symbol, interval, limit)
    - aggregate_from_multiple_sources()
    - cache_data(symbol, data, ttl)
    - rotate_api_source()
```

#### **D. WebSocket Server** (`websocket_server.py`)
```python
# Django Channels for WebSocket support
class TradingWebSocket:
    - connect()
    - receive(message)
    - send_price_update(symbol, price)
    - send_signal_alert(signal)
    - send_trade_execution(order)
    - broadcast_to_clients()
```

---

### **3. Free-Tier API Strategy**

#### **Price Data Sources** (Rotating for Max Coverage)

| API | Free Tier Limit | Data Type | Use Case |
|-----|----------------|-----------|----------|
| **Alpha Vantage** | 25 req/day | Forex, Stocks | Daily data, fundamentals |
| **Twelve Data** | 800 req/day | Forex, Stocks, Crypto | Primary real-time source |
| **Yahoo Finance** | Unlimited* | Stocks, Forex, Indices | Backup + historical |
| **Finnhub** | 60 req/min | Stocks, Forex | Real-time quotes |
| **Polygon.io** | 5 req/min | Stocks, Options | Aggregated data |
| **Binance WS** | Unlimited | Crypto | Crypto real-time (free) |
| **FRED** | Unlimited | Economic data | Fundamentals |

**Strategy:**
- Rotate APIs to spread load
- Cache aggressively (Redis)
- Use WebSockets where available
- Fallback chain for redundancy

#### **API Rotation Logic**
```python
# Prioritize by rate limits
1. Twelve Data (800/day) - Primary for forex
2. Yahoo Finance - Backup
3. Finnhub (60/min) - Real-time fallback
4. Alpha Vantage (25/day) - Special requests only

# Auto-rotate when limit reached
if api_limit_reached(current_api):
    switch_to_next_api()
```

---

### **4. Database Schema**

#### **Paper Trades Table**
```sql
CREATE TABLE paper_trades (
    id SERIAL PRIMARY KEY,
    signal_id VARCHAR(100),
    pair VARCHAR(20),
    order_type VARCHAR(10),  -- 'buy' or 'sell'
    entry_price DECIMAL(10,5),
    stop_loss DECIMAL(10,5),
    take_profit_1 DECIMAL(10,5),
    take_profit_2 DECIMAL(10,5),
    take_profit_3 DECIMAL(10,5),
    lot_size DECIMAL(10,2),
    entry_time TIMESTAMP,
    exit_time TIMESTAMP,
    exit_price DECIMAL(10,5),
    pips_gained DECIMAL(10,2),
    profit_loss DECIMAL(10,2),
    status VARCHAR(20),  -- 'open', 'closed', 'pending'
    signal_type VARCHAR(50),
    signal_source VARCHAR(50),
    risk_reward_ratio DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_paper_trades_status ON paper_trades(status);
CREATE INDEX idx_paper_trades_pair ON paper_trades(pair);
CREATE INDEX idx_paper_trades_entry_time ON paper_trades(entry_time);
```

#### **Price Cache Table**
```sql
CREATE TABLE price_cache (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20),
    timestamp TIMESTAMP,
    open DECIMAL(10,5),
    high DECIMAL(10,5),
    low DECIMAL(10,5),
    close DECIMAL(10,5),
    volume BIGINT,
    source VARCHAR(50),
    timeframe VARCHAR(10),  -- '1m', '5m', '1h', '1d'
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_price_cache_symbol_time ON price_cache(symbol, timestamp DESC);
```

#### **Performance Metrics Table**
```sql
CREATE TABLE performance_metrics (
    id SERIAL PRIMARY KEY,
    date DATE,
    pair VARCHAR(20),
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    win_rate DECIMAL(5,2),
    total_pips DECIMAL(10,2),
    total_pnl DECIMAL(10,2),
    avg_risk_reward DECIMAL(5,2),
    max_drawdown DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

### **5. TradingView Integration Options**

#### **Option A: TradingView Widget (Easiest)**
```javascript
// Free, limited customization
<script src="https://s3.tradingview.com/tv.js"></script>
<div id="tradingview_chart"></div>
<script>
new TradingView.widget({
  container_id: "tradingview_chart",
  symbol: "FX:EURUSD",
  interval: "60",
  theme: "dark",
  style: "1",
  locale: "en",
  toolbar_bg: "#f1f3f6",
  enable_publishing: false,
  withdateranges: true,
  hide_side_toolbar: false,
  allow_symbol_change: true,
  save_image: false,
  studies: ["RSI@tv-basicstudies", "MASimple@tv-basicstudies"]
});
</script>
```

#### **Option B: Lightweight Charts (Full Control)**
```javascript
// Free, full customization, self-hosted
import { createChart } from 'lightweight-charts';

const chart = createChart(document.getElementById('chart'), {
    width: 800,
    height: 400,
    layout: {
        background: { color: '#222' },
        textColor: '#DDD',
    },
    grid: {
        vertLines: { color: '#444' },
        horzLines: { color: '#444' },
    },
});

const candlestickSeries = chart.addCandlestickSeries();
candlestickSeries.setData([
    { time: '2025-01-01', open: 1.0850, high: 1.0880, low: 1.0820, close: 1.0860 },
    // ...more data
]);

// Add signal markers
const markers = [
    {
        time: '2025-01-01',
        position: 'belowBar',
        color: '#2196F3',
        shape: 'arrowUp',
        text: 'ML Signal: LONG @ 1.0850'
    }
];
candlestickSeries.setMarkers(markers);

// Add lines for SL/TP
const lineSeries = chart.addLineSeries({
    color: 'red',
    lineWidth: 2,
});
lineSeries.setData([
    { time: '2025-01-01', value: 1.0820 },  // Stop Loss
]);
```

#### **Option C: Chart.js + Candlestick Plugin (Alternative)**
```javascript
// Good for simpler charts, less forex-specific
import Chart from 'chart.js/auto';
import { CandlestickController, CandlestickElement } from 'chartjs-chart-financial';

Chart.register(CandlestickController, CandlestickElement);
```

**Recommendation: Lightweight Charts** - Best balance of features, performance, and customization

---

### **6. Real-Time Data Flow**

```
1. Data Collection (Every 1-5 seconds)
   └─> Data Aggregator checks cache
       └─> If expired:
           └─> Request from API (rotated source)
           └─> Cache result (Redis, TTL: 5-60s)
       └─> Return cached data

2. Price Update (Every 1-5 seconds)
   └─> WebSocket broadcasts to all connected clients
       └─> Frontend updates TradingView chart
       └─> Paper Trading Engine checks SL/TP

3. Signal Generation (Every 1-5 minutes or on new candle)
   └─> Multi-Model Signal Aggregator runs
       └─> Generate signals
       └─> WebSocket broadcasts new signals
       └─> Frontend displays alerts

4. Trade Execution (On user action or auto)
   └─> User clicks "Execute Trade" or auto-trade enabled
       └─> Paper Trading Engine simulates order
       └─> Store in database
       └─> WebSocket updates position list
       └─> Frontend shows confirmation
```

---

### **7. MetaTrader Bridge Implementation**

#### **Method 1: MT5 Python Package (Recommended for MT5)**
```python
import MetaTrader5 as mt5

class MT5Bridge:
    def __init__(self):
        if not mt5.initialize():
            raise Exception("MT5 initialization failed")
    
    def get_price(self, symbol):
        tick = mt5.symbol_info_tick(symbol)
        return {
            'bid': tick.bid,
            'ask': tick.ask,
            'time': tick.time
        }
    
    def get_bars(self, symbol, timeframe, count):
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        return pd.DataFrame(rates)
    
    def simulate_order(self, symbol, order_type, volume, sl, tp):
        # For paper trading, we DON'T actually place orders
        # Just simulate the execution
        price = self.get_price(symbol)
        return {
            'order_id': self._generate_order_id(),
            'entry_price': price['ask'] if order_type == 'buy' else price['bid'],
            'status': 'filled',
            'time': datetime.now()
        }
```

#### **Method 2: ZeroMQ Bridge (MT4/MT5, More Flexible)**
```python
import zmq

class ZMQBridge:
    def __init__(self, host='localhost', port=5555):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{host}:{port}")
    
    def send_command(self, command):
        self.socket.send_json(command)
        return self.socket.recv_json()
    
    def get_price(self, symbol):
        return self.send_command({
            'action': 'GET_PRICE',
            'symbol': symbol
        })
    
    def get_bars(self, symbol, timeframe, count):
        return self.send_command({
            'action': 'GET_BARS',
            'symbol': symbol,
            'timeframe': timeframe,
            'count': count
        })
```

**MQL4/MQL5 EA for ZeroMQ:**
```mql5
// EA that runs in MT4/MT5 and communicates via ZeroMQ
#property strict
#include <Zmq/Zmq.mqh>

Context context;
Socket socket(context, ZMQ_REP);

int OnInit() {
    socket.bind("tcp://*:5555");
    return(INIT_SUCCEEDED);
}

void OnTick() {
    ZmqMsg request;
    if(socket.recv(request, true)) {
        string command = request.getData();
        string response = ProcessCommand(command);
        ZmqMsg reply(response);
        socket.send(reply);
    }
}

string ProcessCommand(string cmd) {
    // Parse JSON command
    // Execute action (get price, get bars, etc.)
    // Return JSON response
}
```

---

### **8. Frontend Technology Stack**

```javascript
// package.json
{
  "dependencies": {
    "react": "^18.2.0",
    "lightweight-charts": "^4.1.0",
    "axios": "^1.6.0",
    "socket.io-client": "^4.6.0",
    "recharts": "^2.10.0",  // For performance charts
    "zustand": "^4.4.0",    // State management
    "react-query": "^3.39.0",  // Data fetching
    "tailwindcss": "^3.4.0",   // Styling
    "date-fns": "^3.0.0"       // Date utilities
  }
}
```

---

### **9. Key Features Checklist**

#### **Phase 1: Core Infrastructure** (Week 1-2)
- [ ] Backend API endpoints for paper trading
- [ ] Database schema setup
- [ ] Data aggregator with API rotation
- [ ] WebSocket server setup
- [ ] Basic frontend with TradingView chart

#### **Phase 2: Trading Features** (Week 3-4)
- [ ] Paper trading engine
- [ ] Order execution simulation
- [ ] Position tracking
- [ ] P&L calculations
- [ ] Signal display on chart
- [ ] SL/TP visualization

#### **Phase 3: Advanced Features** (Week 5-6)
- [ ] MetaTrader bridge (MT5 or ZeroMQ)
- [ ] Auto-trading (optional)
- [ ] Performance analytics dashboard
- [ ] Trade history with filtering
- [ ] Export reports (CSV, PDF)
- [ ] Risk management tools

#### **Phase 4: Enterprise Features** (Week 7-8)
- [ ] Multi-account support
- [ ] Pro account migration path
- [ ] Advanced charting (patterns, indicators)
- [ ] Alert system (email, SMS, push)
- [ ] Backtesting integration
- [ ] API rate limit monitoring
- [ ] Data quality checks
- [ ] Fallback mechanisms

---

## 🎯 Pro Account Migration Strategy

### **Free Tier → Pro Tier**

| Feature | Free Tier | Pro Tier |
|---------|-----------|----------|
| **Data Updates** | 1-5 min | Real-time (<1s) |
| **API Limits** | Shared, rotated | Dedicated, higher |
| **Concurrent Users** | 10-50 | Unlimited |
| **Historical Data** | 1 year | 10+ years |
| **Pairs** | EURUSD, XAUUSD | All major + exotics |
| **Storage** | Basic | Enhanced + backups |
| **Support** | Community | Priority |

### **Migration Path:**
1. Start with free tier for validation
2. Monitor API usage and data quality
3. Identify bottlenecks
4. Upgrade specific APIs (e.g., Twelve Data Pro)
5. Add premium data sources (e.g., IEX Cloud, Polygon paid)
6. Implement dedicated WebSocket connections

---

## 📊 Expected Performance

### **Free Tier**
- **Latency:** 1-5 seconds (cached)
- **Update Frequency:** Every 1-5 minutes
- **Concurrent Users:** 10-50
- **Cost:** $0/month
- **Data Coverage:** EURUSD, XAUUSD, major pairs

### **Pro Tier** (Future)
- **Latency:** <1 second
- **Update Frequency:** Real-time
- **Concurrent Users:** Unlimited
- **Cost:** $50-200/month (depending on APIs)
- **Data Coverage:** All pairs + exotics

---

## 🚀 Implementation Priority

### **Must-Have (MVP)**
1. ✅ Paper trading engine
2. ✅ TradingView chart with Lightweight Charts
3. ✅ Signal display and execution
4. ✅ Basic P&L tracking
5. ✅ Data aggregation from free APIs

### **Should-Have (V1.0)**
6. MetaTrader bridge (MT5 Python package)
7. WebSocket real-time updates
8. Performance dashboard
9. Trade history with filters

### **Nice-to-Have (V1.1+)**
10. Auto-trading
11. Advanced pattern visualization
12. Multiple timeframe charts
13. Custom indicators
14. Export/reporting

---

## 📝 Next Steps

1. **Review and approve architecture**
2. **Set up development environment**
3. **Implement core backend services**
4. **Build frontend components**
5. **Integrate data sources**
6. **Test paper trading flow**
7. **Deploy and monitor**

---

**Ready to proceed with implementation?** 🚀

I can start building:
- Backend paper trading engine
- Data aggregation service
- Frontend TradingView integration
- WebSocket server
- MetaTrader bridge

Let me know which component to prioritize!
