import React, { useState, useEffect } from 'react';
import axios from 'axios';
import CandlestickChart from './CandlestickChart';
import TradingViewChart from './TradingViewChart';
import './App.css';

// API configuration
const API_BASE_URL = process.env.NODE_ENV === 'production'
  ? 'https://congenial-fortnight-1034520618737.europe-west1.run.app'
  : 'http://localhost:8000';

/*
TRADINGVIEW INTEGRATION OPTIONS FOR REACT:
...
*/

function App() {
  const [signals, setSignals] = useState([]);
  const [backtestResults, setBacktestResults] = useState(null);
  const [showBacktest, setShowBacktest] = useState(false);
  const [showChart, setShowChart] = useState(false);
  const [backtestPair, setBacktestPair] = useState('EURUSD');
  const [chartPair, setChartPair] = useState('EURUSD');
  const [backtestDays, setBacktestDays] = useState(30);
  const [chartType, setChartType] = useState('custom');
  const [darkMode, setDarkMode] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [portfolio, setPortfolio] = useState({
    balance: 10000,
    positions: [],
    history: []
  });
  const [notifications, setNotifications] = useState([]);
  const [showPortfolio, setShowPortfolio] = useState(false);

  useEffect(() => {
    fetchSignals();
    const savedDarkMode = localStorage.getItem('darkMode') === 'true';
    setDarkMode(savedDarkMode);
  }, []);

  useEffect(() => {
    let interval;
    if (autoRefresh) {
      interval = setInterval(fetchSignals, 30000);
    }
    return () => clearInterval(interval);
  }, [autoRefresh]);

  useEffect(() => {
    if (signals.length > 0) {
      const lastSignal = signals[0];
      const lastNotification = notifications[notifications.length - 1];
      if (!lastNotification || lastNotification.id !== lastSignal.id) {
        const newNotification = {
          id: lastSignal.id,
          message: `New ${lastSignal.signal} signal for ${lastSignal.pair} (${(lastSignal.probability * 100).toFixed(1)}% confidence)`,
          timestamp: new Date(),
          type: 'signal'
        };
        setNotifications(prev => [newNotification, ...prev.slice(0, 9)]);
      }
    }
  }, [signals, notifications]);

  const fetchSignals = () => {
    axios.get(`${API_BASE_URL}/api/signals/`)
      .then(response => setSignals(response.data))
      .catch(error => console.error('Error fetching signals:', error));
  };

  const toggleDarkMode = () => {
    const newDark = !darkMode;
    setDarkMode(newDark);
    localStorage.setItem('darkMode', newDark.toString());
  };

  const clearNotifications = () => setNotifications([]);

  const simulateTrade = (signal) => {
    const positionSize = 1000;
    const currentPrice = 1.0850;
    const stopLoss = signal.stop_loss || (signal.signal === 'bullish' ? currentPrice * 0.98 : currentPrice * 1.02);
    const newPosition = {
      id: Date.now(),
      pair: signal.pair,
      type: signal.signal,
      entryPrice: currentPrice,
      stopLoss,
      size: positionSize,
      timestamp: new Date(),
      status: 'open'
    };
    setPortfolio(prev => ({
      ...prev,
      positions: [...prev.positions, newPosition]
    }));
    setPortfolio(prev => ({
      ...prev,
      history: [...prev.history, { ...newPosition, exitPrice: null, pnl: 0, status: 'open' }]
    }));
  };

  const downloadCSV = () => {
    const url = `${API_BASE_URL}/api/backtest/csv/?pair=${backtestPair}&days=${backtestDays}`;
    window.open(url, '_blank');
  };

  const runBacktest = () => {
    setBacktestResults({ status: 'running', message: `Starting backtest for ${backtestPair} over ${backtestDays} days...` });
    axios.get(`${API_BASE_URL}/api/backtest/?pair=${backtestPair}&days=${backtestDays}`)
      .then(response => setBacktestResults(response.data))
      .catch(error => {
        console.error('Error running backtest:', error);
        setBacktestResults({
          status: 'error',
          message: `Backtest failed: ${error.response?.data?.error || error.message}`,
          error: error.response?.data?.error || error.message
        });
      });
  };

  return (
    <div className="App" style={{
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      backgroundColor: darkMode ? '#1a1a1a' : '#f5f5f5',
      color: darkMode ? '#ffffff' : '#333333',
      minHeight: '100vh',
      padding: '20px',
      transition: 'all 0.3s ease'
    }}>
      {/* Notification Panel */}
      {notifications.length > 0 && (
        <div style={{ position: 'fixed', top: '20px', right: '20px', zIndex: 1000, maxWidth: '400px' }}>
          {notifications.slice(0, 3).map((note, i) => (
            <div key={note.id} style={{
              backgroundColor: darkMode ? '#333' : 'white',
              color: darkMode ? '#fff' : '#333',
              padding: '15px',
              marginBottom: '10px',
              borderRadius: '8px',
              boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
              border: '1px solid #ddd',
              animation: 'slideIn 0.5s ease-out',
              transform: `translateY(${i * -10}px)`
            }}>
              <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>🔔 New Signal</div>
              <div style={{ fontSize: '14px' }}>{note.message}</div>
              <div style={{ fontSize: '12px', color: '#666', marginTop: '5px' }}>
                {note.timestamp.toLocaleTimeString()}
              </div>
            </div>
          ))}
          {notifications.length > 3 && (
            <div style={{ textAlign: 'center', marginTop: '10px' }}>
              <button onClick={clearNotifications} style={{
                padding: '5px 10px', backgroundColor: '#6c757d', color: 'white',
                border: 'none', borderRadius: '4px', cursor: 'pointer'
              }}>
                Clear All ({notifications.length})
              </button>
            </div>
          )}
        </div>
      )}

      <header className="App-header" style={{
        backgroundColor: '#2c3e50', color: 'white', padding: '20px',
        borderRadius: '10px', marginBottom: '30px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
      }}>
        {/* header content omitted for brevity */}
      </header>

      {/* Main Content Area */}
      <div style={{ marginTop: '20px' }}>
        {!showBacktest ? (
          <>
            {/* Portfolio & Signals Sections */}
          </>
        ) : (
          <div>
            {/* Backtest Section */}
          </div>
        )}

        {showChart && (
          <div>
            {/* Chart Section */}
          </div>
        )}

        <div style={{
          textAlign: 'center', marginTop: '30px', padding: '20px',
          backgroundColor: darkMode ? '#333' : '#ecf0f1', borderRadius: '8px'
        }}>
          <button onClick={fetchSignals} style={{
            padding: '12px 25px', backgroundColor: '#3498db', color: 'white',
            border: 'none', borderRadius: '6px', cursor: 'pointer',
            fontSize: '16px', fontWeight: '500', transition: 'all 0.3s ease',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
          }}
            onMouseOver={e => e.currentTarget.style.backgroundColor = '#2980b9'}
            onMouseOut={e => e.currentTarget.style.backgroundColor = '#3498db'}>
            🔄 Refresh Signals
          </button>
        </div>
      </div>
    </div>  // ← This closes the <div className="App">
  );
}

export default App;
