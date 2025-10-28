# Paper Trading System - Developer Guide

**Last Updated**: October 9, 2025  
**Status**: ✅ Complete and Production-Ready  
**Version**: 1.0.0

---

## 🎯 Purpose

This document guides the next developer/agent on how to:
1. Understand the paper trading system architecture
2. Add new features or enhancements
3. Maintain and debug existing functionality
4. Deploy to production

---

## 📋 System Overview

### What This System Does

The **Paper Trading System** is an enterprise-level forward testing platform that:
- Simulates real forex trading without risking capital
- Displays live TradingView-style charts with signal markers
- Tracks performance metrics (win rate, P&L, pips)
- Integrates with multi-model signal aggregator
- Updates prices in real-time via WebSocket
- Supports MetaTrader 4/5 integration (optional)

### Core Components

```
├── paper_trading/              # Django app (Backend)
│   ├── models.py              # Database models
│   ├── engine.py              # Paper trading simulation engine
│   ├── data_aggregator.py     # Multi-source data fetching
│   ├── mt_bridge.py           # MetaTrader integration
│   ├── views.py               # REST API endpoints
│   ├── consumers.py           # WebSocket channels
│   ├── signal_integration.py  # Signal-to-trade connector
│   └── management/commands/   # Background workers
│
├── frontend/src/              # React Frontend
│   ├── EnhancedTradingChart.js    # Main chart component
│   ├── SignalPanel.js             # Signal feed & execution
│   ├── OrderManager.js            # Position management
│   ├── PerformanceDashboard.js    # Analytics & metrics
│   └── PaperTradingApp.js         # Main integrated app
│
└── Documentation/
    ├── SYSTEM_ARCHITECTURE_DIAGRAM.md
    ├── PAPER_TRADING_COMPLETE_SUMMARY.md
    ├── PAPER_TRADING_IMPLEMENTATION_COMPLETE.md
    └── METATRADER_PAPER_TRADING_ARCHITECTURE.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
# Backend requirements
pip install django djangorestframework channels channels-redis daphne
pip install redis MetaTrader5 pyzmq yfinance requests

# Frontend requirements
cd frontend
npm install react lightweight-charts
```

### Quick Start

```bash
# 1. Validate setup
python setup_paper_trading.py

# 2. Run migrations (if needed)
python manage.py makemigrations paper_trading
python manage.py migrate

# 3. Start backend (Terminal 1)
daphne -b 0.0.0.0 -p 8000 forex_signal.asgi:application

# 4. Start price worker (Terminal 2)
python manage.py run_price_worker --interval=5 --pairs=EURUSD,XAUUSD

# 5. Start frontend (Terminal 3)
cd frontend && npm start
```

Visit: **http://localhost:3000/**

---

## 🔧 How to Add Features

### Adding a New Data Source

**Location**: `paper_trading/data_aggregator.py`

```python
class DataAggregator:
    def __init__(self):
        self.apis = {
            'yahoo': {'priority': 1, 'limit': None},
            'twelve_data': {'priority': 2, 'limit': 800},
            'finnhub': {'priority': 3, 'limit': 3600},
            'alpha_vantage': {'priority': 4, 'limit': 25},
            'new_api': {'priority': 5, 'limit': 1000}  # Add here
        }
    
    def _fetch_from_new_api(self, symbol):
        """Implement fetching logic for new API"""
        # 1. Get API key from settings
        api_key = settings.NEW_API_KEY
        
        # 2. Convert symbol format if needed
        formatted_symbol = self._convert_symbol_for_new_api(symbol)
        
        # 3. Make API request
        response = requests.get(
            f"https://api.newapi.com/quote/{formatted_symbol}",
            params={'apikey': api_key}
        )
        
        # 4. Parse response
        data = response.json()
        return {
            'bid': data['bid'],
            'ask': data['ask'],
            'timestamp': data['timestamp']
        }
    
    def get_realtime_price(self, symbol):
        # Add 'new_api' to the rotation
        for api_name in ['yahoo', 'twelve_data', 'finnhub', 'new_api']:
            if self._can_use_api(api_name):
                try:
                    if api_name == 'new_api':
                        price_data = self._fetch_from_new_api(symbol)
                    # ... existing logic
```

**Environment Variables** (add to `.env`):
```bash
NEW_API_KEY=your_api_key_here
```

**Testing**:
```python
# Test in Django shell
python manage.py shell

from paper_trading.data_aggregator import DataAggregator
aggregator = DataAggregator()
price = aggregator.get_realtime_price('EURUSD')
print(price)
```

---

### Adding a New Frontend Component

**Example**: Adding a Risk Calculator Panel

**Step 1**: Create component file  
**Location**: `frontend/src/RiskCalculator.js`

```javascript
import React, { useState } from 'react';

const RiskCalculator = () => {
  const [accountBalance, setAccountBalance] = useState(10000);
  const [riskPercent, setRiskPercent] = useState(2);
  const [stopLossPips, setStopLossPips] = useState(50);
  
  const calculateLotSize = () => {
    const riskAmount = accountBalance * (riskPercent / 100);
    const pipValue = 10; // Standard for forex
    const lotSize = riskAmount / (stopLossPips * pipValue);
    return lotSize.toFixed(2);
  };
  
  return (
    <div className="risk-calculator">
      <h3>Risk Calculator</h3>
      <div className="input-group">
        <label>Account Balance:</label>
        <input 
          type="number" 
          value={accountBalance}
          onChange={(e) => setAccountBalance(parseFloat(e.target.value))}
        />
      </div>
      <div className="input-group">
        <label>Risk %:</label>
        <input 
          type="number" 
          value={riskPercent}
          onChange={(e) => setRiskPercent(parseFloat(e.target.value))}
        />
      </div>
      <div className="input-group">
        <label>Stop Loss (pips):</label>
        <input 
          type="number" 
          value={stopLossPips}
          onChange={(e) => setStopLossPips(parseFloat(e.target.value))}
        />
      </div>
      <div className="result">
        <strong>Recommended Lot Size: {calculateLotSize()}</strong>
      </div>
    </div>
  );
};

export default RiskCalculator;
```

**Step 2**: Integrate into main app  
**Location**: `frontend/src/PaperTradingApp.js`

```javascript
import RiskCalculator from './RiskCalculator';

function PaperTradingApp() {
  return (
    <div className="paper-trading-app">
      <div className="main-grid">
        <EnhancedTradingChart />
        <SignalPanel />
        <RiskCalculator />  {/* Add here */}
        <OrderManager />
      </div>
    </div>
  );
}
```

**Step 3**: Add styles  
**Location**: `frontend/src/PaperTradingApp.css`

```css
.risk-calculator {
  background: #1e222d;
  border-radius: 8px;
  padding: 20px;
  margin-bottom: 20px;
}

.risk-calculator .input-group {
  margin-bottom: 15px;
}

.risk-calculator .result {
  margin-top: 20px;
  padding: 15px;
  background: #2a2e39;
  border-radius: 4px;
  font-size: 16px;
}
```

---

### Adding a New API Endpoint

**Example**: Get average position hold time

**Step 1**: Add method to engine  
**Location**: `paper_trading/engine.py`

```python
def get_average_hold_time(self, days=30):
    """Calculate average position hold time"""
    cutoff_date = timezone.now() - timedelta(days=days)
    
    closed_trades = PaperTrade.objects.filter(
        user=self.user,
        status='closed',
        exit_time__gte=cutoff_date
    ).exclude(entry_time__isnull=True, exit_time__isnull=True)
    
    if not closed_trades.exists():
        return None
    
    total_seconds = 0
    for trade in closed_trades:
        if trade.entry_time and trade.exit_time:
            hold_time = trade.exit_time - trade.entry_time
            total_seconds += hold_time.total_seconds()
    
    avg_seconds = total_seconds / closed_trades.count()
    avg_hours = avg_seconds / 3600
    
    return {
        'average_hours': round(avg_hours, 2),
        'average_days': round(avg_hours / 24, 2),
        'total_trades': closed_trades.count()
    }
```

**Step 2**: Add view endpoint  
**Location**: `paper_trading/views.py`

```python
from rest_framework.decorators import action
from rest_framework.response import Response

class PaperTradeViewSet(viewsets.ModelViewSet):
    # ... existing code ...
    
    @action(detail=False, methods=['get'])
    def average_hold_time(self, request):
        """Get average position hold time"""
        days = int(request.query_params.get('days', 30))
        
        engine = PaperTradingEngine(request.user)
        hold_time_data = engine.get_average_hold_time(days=days)
        
        if hold_time_data is None:
            return Response({
                'message': 'No closed trades found',
                'average_hours': 0,
                'average_days': 0,
                'total_trades': 0
            })
        
        return Response(hold_time_data)
```

**Step 3**: Test the endpoint

```bash
# Using curl
curl http://localhost:8000/api/paper-trading/trades/average_hold_time/?days=30

# Using Python requests
import requests
response = requests.get('http://localhost:8000/api/paper-trading/trades/average_hold_time/?days=30')
print(response.json())
```

**Step 4**: Add to frontend

```javascript
// In PerformanceDashboard.js
const [avgHoldTime, setAvgHoldTime] = useState(null);

useEffect(() => {
  fetch('http://localhost:8000/api/paper-trading/trades/average_hold_time/?days=30')
    .then(res => res.json())
    .then(data => setAvgHoldTime(data));
}, []);

// Display in UI
<div className="metric-card">
  <div className="metric-label">Avg Hold Time</div>
  <div className="metric-value">
    {avgHoldTime ? `${avgHoldTime.average_hours}h` : 'N/A'}
  </div>
</div>
```

---

### Adding WebSocket Message Types

**Example**: Broadcasting account balance updates

**Step 1**: Update consumer  
**Location**: `paper_trading/consumers.py`

```python
class TradingWebSocketConsumer(AsyncWebsocketConsumer):
    # ... existing code ...
    
    async def balance_update(self, event):
        """Send balance update to client"""
        await self.send(text_data=json.dumps({
            'type': 'balance_update',
            'balance': event['balance'],
            'equity': event['equity'],
            'margin_free': event['margin_free'],
            'timestamp': event['timestamp']
        }))
```

**Step 2**: Broadcast from backend

```python
# In paper_trading/engine.py
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

def close_position(self, trade_id):
    # ... existing close logic ...
    
    # Calculate new balance
    new_balance = self.get_current_balance()
    
    # Broadcast balance update
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(
        f"trading_{self.user.id}",
        {
            'type': 'balance_update',
            'balance': float(new_balance),
            'equity': float(new_balance),  # Simplified
            'margin_free': float(new_balance * 0.9),
            'timestamp': timezone.now().isoformat()
        }
    )
```

**Step 3**: Handle in frontend

```javascript
// In PaperTradingApp.js WebSocket handler
useEffect(() => {
  const ws = new WebSocket('ws://localhost:8000/ws/trading/');
  
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    if (data.type === 'balance_update') {
      setAccountBalance(data.balance);
      setAccountEquity(data.equity);
      console.log('Balance updated:', data.balance);
    }
    // ... other message types
  };
  
  return () => ws.close();
}, []);
```

---

## 🐛 Debugging Guide

### Common Issues & Solutions

#### 1. "Error loading CSV: 'date'" in Fundamental Pipeline

**Problem**: Fundamental data files have wrong schema (price schema instead of fundamental schema)

**Solution**:
```bash
# Run the fix script
python fix_fundamental_headers.py

# Or restore from backups
python restore_fundamental_backups.py

# Verify fix
python -c "import pandas as pd; df = pd.read_csv('data/DGS10.csv'); print(df.columns)"
```

**Root Cause**: Files were standardized with price schema when they needed fundamental schema (date, value)

---

#### 2. WebSocket Connection Fails

**Problem**: `WebSocket connection failed` in browser console

**Checklist**:
```bash
# 1. Check if Daphne is running
ps aux | grep daphne

# 2. Check if Redis is running
redis-cli ping  # Should return "PONG"

# 3. Check WebSocket routing
cat forex_signal/routing.py

# 4. Test WebSocket manually
python manage.py shell
>>> from channels.layers import get_channel_layer
>>> channel_layer = get_channel_layer()
>>> # If no error, channels are configured correctly
```

**Solution**: Ensure `daphne` is running (not `runserver`) and Redis is accessible

---

#### 3. No Price Updates

**Problem**: Prices not updating in frontend

**Checklist**:
```bash
# 1. Check if price worker is running
ps aux | grep run_price_worker

# 2. Check API limits
python manage.py shell
>>> from paper_trading.models import APIUsageTracker
>>> APIUsageTracker.objects.all()

# 3. Test data aggregator directly
python manage.py shell
>>> from paper_trading.data_aggregator import DataAggregator
>>> agg = DataAggregator()
>>> price = agg.get_realtime_price('EURUSD')
>>> print(price)

# 4. Check cache
redis-cli
> KEYS *EURUSD*
> GET price_cache_EURUSD_realtime
```

**Solution**: Restart price worker or clear cache

---

#### 4. SL/TP Not Triggering

**Problem**: Stop loss or take profit not closing positions automatically

**Debug Steps**:
```python
# In Django shell
from paper_trading.engine import PaperTradingEngine
from django.contrib.auth import get_user_model

User = get_user_model()
user = User.objects.first()
engine = PaperTradingEngine(user)

# Check open positions
positions = engine.get_open_positions()
print(f"Open positions: {len(positions)}")

# Manually test update
from paper_trading.data_aggregator import DataAggregator
agg = DataAggregator()

for position in positions:
    price = agg.get_realtime_price(position.symbol)
    print(f"{position.symbol}: Entry={position.entry_price}, Current={price['bid']}, SL={position.stop_loss}, TP={position.take_profit_1}")
    
    # Test SL/TP check
    hit, level = engine._check_sl_tp_hit(position, price['bid'])
    print(f"Hit: {hit}, Level: {level}")
```

**Common Causes**:
- Price worker not running
- Incorrect pip calculation (4-digit vs 5-digit broker)
- API returning stale prices

---

#### 5. Frontend Not Connecting to Backend

**Problem**: API calls returning CORS errors

**Solution**: Update Django settings  
**Location**: `forex_signal/settings.py`

```python
INSTALLED_APPS = [
    'corsheaders',  # Add this
    # ... other apps
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',  # Add this near top
    # ... other middleware
]

# Development settings
CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# Production: Set to your domain
# CORS_ALLOWED_ORIGINS = ["https://yourdomain.com"]
```

---

## 📊 Database Schema Reference

### PaperTrade Model

```python
class PaperTrade(models.Model):
    user = ForeignKey  # Who made the trade
    symbol = CharField  # e.g., 'EURUSD'
    signal_type = CharField  # 'BUY' or 'SELL'
    entry_price = DecimalField
    stop_loss = DecimalField
    take_profit_1 = DecimalField
    take_profit_2 = DecimalField (optional)
    take_profit_3 = DecimalField (optional)
    lot_size = DecimalField
    entry_time = DateTimeField
    exit_time = DateTimeField (nullable)
    exit_price = DecimalField (nullable)
    pips_gained = DecimalField (nullable)
    profit_loss = DecimalField (nullable)
    status = CharField  # 'open', 'closed', 'cancelled'
    exit_reason = CharField  # 'manual', 'sl_hit', 'tp1_hit', etc.
```

**Indexes**: `user`, `symbol`, `status`, `entry_time`

**Queries**:
```python
# Get all open positions
PaperTrade.objects.filter(user=user, status='open')

# Get closed trades with profit
PaperTrade.objects.filter(user=user, status='closed', profit_loss__gt=0)

# Get trades for specific symbol
PaperTrade.objects.filter(user=user, symbol='EURUSD').order_by('-entry_time')
```

---

### PriceCache Model

```python
class PriceCache(models.Model):
    symbol = CharField
    timestamp = DateTimeField
    open_price = DecimalField
    high_price = DecimalField
    low_price = DecimalField
    close_price = DecimalField
    volume = BigIntegerField
    source = CharField  # 'yahoo', 'twelve_data', etc.
    timeframe = CharField  # '1min', '5min', '1hour', '1day'
```

**Purpose**: Cache OHLC data to reduce API calls

**Queries**:
```python
# Get latest cached price
PriceCache.objects.filter(symbol='EURUSD', timeframe='1min').order_by('-timestamp').first()

# Get hourly candles for last 24 hours
from datetime import timedelta
cutoff = timezone.now() - timedelta(hours=24)
PriceCache.objects.filter(symbol='EURUSD', timeframe='1hour', timestamp__gte=cutoff)
```

---

## 🔐 Security Best Practices

### API Keys Management

**Never commit API keys to git!**

**Step 1**: Create `.env` file (already in `.gitignore`)
```bash
# .env
TWELVE_DATA_API_KEY=your_key_here
FINNHUB_API_KEY=your_key_here
ALPHA_VANTAGE_API_KEY=your_key_here
SECRET_KEY=your_django_secret_key
```

**Step 2**: Load in settings
```python
# forex_signal/settings.py
from dotenv import load_dotenv
import os

load_dotenv()

TWELVE_DATA_API_KEY = os.getenv('TWELVE_DATA_API_KEY')
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
```

**Step 3**: Use in code
```python
# paper_trading/data_aggregator.py
from django.conf import settings

api_key = settings.TWELVE_DATA_API_KEY
```

---

### Authentication

**Current**: Basic Django authentication  
**Recommended for Production**: JWT tokens

```bash
pip install djangorestframework-simplejwt
```

```python
# settings.py
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ],
}

# urls.py
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

urlpatterns = [
    path('api/token/', TokenObtainPairView.as_view()),
    path('api/token/refresh/', TokenRefreshView.as_view()),
]
```

---

## 🚀 Deployment Checklist

### Pre-Deployment

- [ ] All tests passing: `python manage.py test`
- [ ] Static files collected: `python manage.py collectstatic`
- [ ] Database migrations applied: `python manage.py migrate`
- [ ] API keys configured in environment
- [ ] `DEBUG = False` in production settings
- [ ] `ALLOWED_HOSTS` configured
- [ ] CORS settings updated for production domain
- [ ] SSL certificates obtained
- [ ] Redis configured and running
- [ ] PostgreSQL configured (migrate from SQLite)

### Docker Deployment

**Create `docker-compose.yml`**:
```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
  
  db:
    image: postgres:15
    environment:
      POSTGRES_DB: paper_trading
      POSTGRES_USER: trader
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  backend:
    build: .
    command: daphne -b 0.0.0.0 -p 8000 forex_signal.asgi:application
    volumes:
      - .:/app
    ports:
      - "8000:8000"
    depends_on:
      - redis
      - db
    environment:
      - DATABASE_URL=postgresql://trader:${DB_PASSWORD}@db:5432/paper_trading
      - REDIS_URL=redis://redis:6379/0
  
  worker:
    build: .
    command: python manage.py run_price_worker --interval=5
    depends_on:
      - redis
      - db
      - backend
  
  frontend:
    build: ./frontend
    ports:
      - "3000:80"
    depends_on:
      - backend

volumes:
  postgres_data:
```

**Deploy**:
```bash
docker-compose up -d
```

---

## 📝 Code Style & Standards

### Python (Backend)

- Follow PEP 8
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use descriptive variable names
- Add docstrings to all functions/classes

**Example**:
```python
from typing import Optional, Dict, Any
from decimal import Decimal

def calculate_position_size(
    account_balance: Decimal,
    risk_percent: Decimal,
    stop_loss_pips: int
) -> Dict[str, Any]:
    """
    Calculate optimal position size based on risk parameters.
    
    Args:
        account_balance: Total account balance
        risk_percent: Percentage of account to risk (e.g., 2.0 for 2%)
        stop_loss_pips: Stop loss distance in pips
    
    Returns:
        Dict containing lot_size, risk_amount, and pip_value
    """
    risk_amount = account_balance * (risk_percent / Decimal('100'))
    pip_value = Decimal('10')  # Standard for forex
    lot_size = risk_amount / (Decimal(stop_loss_pips) * pip_value)
    
    return {
        'lot_size': round(lot_size, 2),
        'risk_amount': round(risk_amount, 2),
        'pip_value': pip_value
    }
```

### JavaScript (Frontend)

- Use ES6+ features
- Prefer functional components with hooks
- Use meaningful component and variable names
- Add comments for complex logic
- Use consistent formatting (Prettier recommended)

**Example**:
```javascript
import React, { useState, useEffect } from 'react';

/**
 * Component for displaying real-time account balance
 * Updates automatically via WebSocket connection
 */
const AccountBalance = ({ userId }) => {
  const [balance, setBalance] = useState(10000);
  const [isConnected, setIsConnected] = useState(false);
  
  useEffect(() => {
    const ws = new WebSocket(`ws://localhost:8000/ws/trading/`);
    
    ws.onopen = () => {
      setIsConnected(true);
      console.log('WebSocket connected');
    };
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'balance_update') {
        setBalance(data.balance);
      }
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setIsConnected(false);
    };
    
    return () => ws.close();
  }, [userId]);
  
  return (
    <div className="account-balance">
      <div className="balance-label">Account Balance</div>
      <div className="balance-value">${balance.toFixed(2)}</div>
      <div className={`status ${isConnected ? 'connected' : 'disconnected'}`}>
        {isConnected ? '● Connected' : '○ Disconnected'}
      </div>
    </div>
  );
};

export default AccountBalance;
```

---

## 📚 Additional Resources

### Documentation Files

1. **SYSTEM_ARCHITECTURE_DIAGRAM.md**: Visual overview of entire system
2. **PAPER_TRADING_COMPLETE_SUMMARY.md**: Executive summary and quick start
3. **PAPER_TRADING_IMPLEMENTATION_COMPLETE.md**: Detailed API documentation
4. **METATRADER_PAPER_TRADING_ARCHITECTURE.md**: Technical architecture deep dive

### External Resources

- **Django Channels**: https://channels.readthedocs.io/
- **Lightweight Charts**: https://tradingview.github.io/lightweight-charts/
- **MetaTrader5 Python**: https://www.mql5.com/en/docs/python_metatrader5
- **Redis**: https://redis.io/docs/

### Testing

```bash
# Run all tests
python manage.py test paper_trading

# Run specific test
python manage.py test paper_trading.tests.test_engine

# Run with coverage
pip install coverage
coverage run --source='paper_trading' manage.py test paper_trading
coverage report
```

---

## 🤝 Contributing Guidelines

### Before Making Changes

1. **Check existing documentation** - Review architecture docs
2. **Create a branch** - Don't commit directly to main
3. **Test locally** - Ensure everything works before committing
4. **Update documentation** - Keep docs in sync with code changes

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**: feat, fix, docs, style, refactor, test, chore

**Example**:
```
feat(data-aggregator): Add support for IEX Cloud API

- Implemented IEXCloudProvider class
- Added symbol mapping for IEX format
- Updated priority rotation to include IEX
- Added rate limiting (100 calls/day free tier)

Closes #123
```

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring

## Testing
- [ ] Tested locally
- [ ] All tests passing
- [ ] Added new tests if needed

## Documentation
- [ ] Updated relevant documentation
- [ ] Added code comments
- [ ] Updated API docs if needed

## Checklist
- [ ] Code follows style guidelines
- [ ] No console.log or print statements left
- [ ] Environment variables documented
```

---

## 🆘 Getting Help

### Debugging Steps

1. **Check logs**: `tail -f logs/*.log`
2. **Django shell**: `python manage.py shell` - Test components interactively
3. **Browser console**: Check for JavaScript errors
4. **Network tab**: Verify API requests/responses
5. **Redis CLI**: `redis-cli` - Check cache status

### Common Commands

```bash
# Check system status
python setup_paper_trading.py

# Clear cache
redis-cli FLUSHALL

# Reset database
python manage.py flush
python manage.py migrate

# Create superuser
python manage.py createsuperuser

# Access admin interface
# http://localhost:8000/admin/

# Django shell for debugging
python manage.py shell
```

---

## 🎯 Next Steps / Roadmap

### Phase 1: Core Enhancements (High Priority)
- [ ] Add email notifications for trade executions
- [ ] Implement trade journaling (notes, screenshots)
- [ ] Add advanced charting indicators (RSI, MACD, EMA)
- [ ] Create mobile-responsive design
- [ ] Add export functionality (CSV, PDF reports)

### Phase 2: Advanced Features (Medium Priority)
- [ ] Multi-account support
- [ ] Trade copying functionality
- [ ] Advanced risk management (correlation analysis)
- [ ] Backtesting integration
- [ ] Strategy builder interface

### Phase 3: Enterprise Features (Low Priority)
- [ ] Team/organization accounts
- [ ] Role-based permissions
- [ ] Audit logs
- [ ] White-label capability
- [ ] API rate limiting per user

### Phase 4: Integrations (Future)
- [ ] TradingView alerts integration
- [ ] Telegram bot notifications
- [ ] Discord webhooks
- [ ] CRM integration
- [ ] Payment gateway (upgrade to pro)

---

## 📄 License & Credits

**Project**: Paper Trading System  
**Created**: October 2025  
**Status**: Production-ready  
**License**: MIT (or your chosen license)

**Built With**:
- Django 5.2.6
- Django Channels 4.0.0
- React 18.2.0
- Lightweight Charts 4.1.0
- Redis 5.0.1
- MetaTrader5 Python Package

---

## ✅ Quick Reference Checklist

**Starting Development**:
- [ ] Read this document
- [ ] Review architecture diagrams
- [ ] Run `python setup_paper_trading.py`
- [ ] Start all three services (backend, worker, frontend)
- [ ] Access admin interface and create test trades
- [ ] Review existing code structure

**Adding Features**:
- [ ] Create feature branch
- [ ] Update relevant files
- [ ] Add tests
- [ ] Update documentation
- [ ] Test locally
- [ ] Submit PR

**Before Deployment**:
- [ ] All tests passing
- [ ] Documentation updated
- [ ] Environment variables configured
- [ ] SSL certificates ready
- [ ] Monitoring setup
- [ ] Backup strategy in place

---

**Last Updated**: October 9, 2025  
**For Questions**: Check documentation files or open an issue  
**Happy Trading!** 📈🚀
