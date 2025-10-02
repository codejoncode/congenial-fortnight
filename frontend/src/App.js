import React, { useState, useEffect } from 'react';
import axios from 'axios';
import CandlestickChart from './CandlestickChart';
import TradingViewChart from './TradingViewChart2';
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
    <div className={`App ${darkMode ? 'dark-mode' : ''}`} style={{
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      backgroundColor: darkMode ? '#0f0f23' : '#f5f5f5',
      color: darkMode ? '#ffffff' : '#333333',
      minHeight: '100vh',
      padding: '20px',
      transition: 'all 0.3s ease'
    }}>
      {/* Enhanced Notification Panel */}
      {notifications.length > 0 && (
        <div style={{ 
          position: 'fixed', 
          top: '20px', 
          right: '20px', 
          zIndex: 1000, 
          maxWidth: '400px',
          maxHeight: '400px',
          overflowY: 'auto'
        }}>
          {notifications.slice(0, 5).map((note, i) => (
            <div key={note.id} className="glass-card" style={{
              background: note.signal === 'bullish' 
                ? 'linear-gradient(45deg, rgba(39, 174, 96, 0.9), rgba(46, 204, 113, 0.9))' 
                : 'linear-gradient(45deg, rgba(231, 76, 60, 0.9), rgba(192, 57, 43, 0.9))',
              color: 'white', 
              padding: '15px', 
              margin: '8px 0',
              borderRadius: '15px', 
              boxShadow: '0 8px 32px rgba(31, 38, 135, 0.37)',
              backdropFilter: 'blur(10px)',
              border: '1px solid rgba(255, 255, 255, 0.18)',
              transform: `translateX(${i * -5}px)`,
              opacity: 1 - (i * 0.1),
              transition: 'all 0.3s ease'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span style={{ fontSize: '20px' }}>
                    {note.signal === 'bullish' ? '🚀' : '�'}
                  </span>
                  <strong style={{ fontSize: '16px' }}>{note.pair}</strong>
                </div>
                <div style={{ fontSize: '12px', opacity: 0.8 }}>
                  {(note.probability * 100).toFixed(1)}%
                </div>
              </div>
              
              <div style={{ 
                fontSize: '14px', 
                marginTop: '8px',
                textTransform: 'uppercase',
                fontWeight: 'bold',
                letterSpacing: '1px'
              }}>
                {note.signal} Signal Detected!
              </div>
              
              <div style={{ 
                fontSize: '11px', 
                marginTop: '8px', 
                opacity: 0.7,
                display: 'flex',
                justifyContent: 'space-between'
              }}>
                <span>AI Confidence: {(note.probability * 100).toFixed(1)}%</span>
                <span>{new Date(note.timestamp).toLocaleTimeString()}</span>
              </div>
            </div>
          ))}
          
          {notifications.length > 5 && (
            <div style={{
              textAlign: 'center',
              color: darkMode ? '#aaa' : '#666',
              fontSize: '12px',
              marginTop: '10px'
            }}>
              +{notifications.length - 5} more notifications
            </div>
          )}
        </div>
      )}

      <header className="App-header" style={{
        backgroundColor: darkMode ? '#2c3e50' : '#34495e',
        color: 'white',
        padding: '20px',
        borderRadius: '10px',
        marginBottom: '30px',
        boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div>
          <h1 style={{ margin: 0, fontSize: '28px', fontWeight: 'bold' }}>
            🚀 AI Forex Signal System
          </h1>
          <p style={{ margin: '5px 0 0 0', fontSize: '14px', opacity: 0.8 }}>
            Advanced ML predictions with {signals.length} active signals
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
          {/* Auto Refresh Toggle */}
          <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              style={{ marginRight: '5px' }}
            />
            <span style={{ fontSize: '12px' }}>Auto Refresh</span>
          </label>

          {/* Dark Mode Toggle */}
          <button
            onClick={toggleDarkMode}
            style={{
              padding: '8px 12px',
              backgroundColor: darkMode ? '#f39c12' : '#e67e22',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '12px',
              fontWeight: '500',
              transition: 'all 0.2s ease'
            }}
          >
            {darkMode ? '☀️ Light' : '🌙 Dark'}
          </button>

          {/* Notification Bell */}
          {notifications.length > 0 && (
            <div style={{ position: 'relative' }}>
              <button
                onClick={() => setNotifications([])}
                style={{
                  padding: '8px 12px',
                  backgroundColor: '#e74c3c',
                  color: 'white',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontSize: '12px',
                  fontWeight: '500'
                }}
              >
                🔔 Clear All ({notifications.length})
              </button>
            </div>
          )}
        </div>
      </header>

      {/* Main Content Area */}
      <div style={{ marginTop: '20px' }}>
        {!showBacktest && !showChart ? (
          <>
            {/* Portfolio Section */}
            {showPortfolio && (
              <div style={{
                backgroundColor: darkMode ? '#333' : 'white',
                padding: '25px',
                borderRadius: '12px',
                marginBottom: '25px',
                boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
              }}>
                <h2 style={{ marginBottom: '20px', color: darkMode ? '#fff' : '#2c3e50' }}>
                  💼 Portfolio Overview
                </h2>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '20px', marginBottom: '20px' }}>
                  <div style={{
                    backgroundColor: darkMode ? '#2c3e50' : '#ecf0f1',
                    padding: '15px',
                    borderRadius: '8px',
                    textAlign: 'center'
                  }}>
                    <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#27ae60' }}>
                      ${portfolio.balance.toLocaleString()}
                    </div>
                    <div style={{ fontSize: '14px', color: '#7f8c8d' }}>Account Balance</div>
                  </div>
                  <div style={{
                    backgroundColor: darkMode ? '#2c3e50' : '#ecf0f1',
                    padding: '15px',
                    borderRadius: '8px',
                    textAlign: 'center'
                  }}>
                    <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#3498db' }}>
                      {portfolio.positions.length}
                    </div>
                    <div style={{ fontSize: '14px', color: '#7f8c8d' }}>Open Positions</div>
                  </div>
                  <div style={{
                    backgroundColor: darkMode ? '#2c3e50' : '#ecf0f1',
                    padding: '15px',
                    borderRadius: '8px',
                    textAlign: 'center'
                  }}>
                    <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#e74c3c' }}>
                      {portfolio.history.length}
                    </div>
                    <div style={{ fontSize: '14px', color: '#7f8c8d' }}>Total Trades</div>
                  </div>
                </div>
                
                {portfolio.positions.length > 0 && (
                  <div>
                    <h3 style={{ marginBottom: '15px' }}>Active Positions</h3>
                    {portfolio.positions.map(position => (
                      <div key={position.id} style={{
                        backgroundColor: darkMode ? '#2c3e50' : '#f8f9fa',
                        padding: '15px',
                        borderRadius: '8px',
                        marginBottom: '10px',
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center'
                      }}>
                        <div>
                          <strong>{position.pair}</strong> - {position.type}
                          <div style={{ fontSize: '12px', color: '#7f8c8d' }}>
                            Entry: {position.entryPrice} | Stop Loss: {position.stopLoss}
                          </div>
                        </div>
                        <div style={{ textAlign: 'right' }}>
                          <div style={{ fontWeight: 'bold' }}>${position.size}</div>
                          <div style={{ fontSize: '12px', color: '#7f8c8d' }}>
                            {new Date(position.timestamp).toLocaleDateString()}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Trading Signals Section */}
            <div style={{
              backgroundColor: darkMode ? '#333' : 'white',
              padding: '25px',
              borderRadius: '12px',
              marginBottom: '25px',
              boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
            }}>
              <h2 style={{ marginBottom: '20px', color: darkMode ? '#fff' : '#2c3e50' }}>
                🎯 Current Trading Signals
              </h2>
              
              {signals.length === 0 ? (
                <div style={{
                  textAlign: 'center',
                  padding: '40px',
                  color: '#7f8c8d'
                }}>
                  <div style={{ fontSize: '48px', marginBottom: '10px' }}>📊</div>
                  <div>No signals available. Click refresh to fetch latest data.</div>
                </div>
              ) : (
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
                  gap: '20px'
                }}>
                  {signals.map((signal, index) => (
                    <div key={index} style={{
                      backgroundColor: darkMode ? '#2c3e50' : '#f8f9fa',
                      border: `3px solid ${signal.signal === 'bullish' ? '#27ae60' : '#e74c3c'}`,
                      borderRadius: '12px',
                      padding: '20px',
                      position: 'relative',
                      transition: 'transform 0.2s ease',
                      cursor: 'pointer'
                    }}
                    onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-2px)'}
                    onMouseLeave={e => e.currentTarget.style.transform = 'translateY(0)'}
                    >
                      <div style={{
                        position: 'absolute',
                        top: '10px',
                        right: '10px',
                        backgroundColor: signal.signal === 'bullish' ? '#27ae60' : '#e74c3c',
                        color: 'white',
                        padding: '4px 8px',
                        borderRadius: '12px',
                        fontSize: '12px',
                        fontWeight: 'bold'
                      }}>
                        {(signal.probability * 100).toFixed(1)}%
                      </div>
                      
                      <div style={{ fontSize: '24px', marginBottom: '10px' }}>
                        {signal.signal === 'bullish' ? '🚀' : '📉'} {signal.pair}
                      </div>
                      
                      <div style={{
                        fontSize: '18px',
                        fontWeight: 'bold',
                        color: signal.signal === 'bullish' ? '#27ae60' : '#e74c3c',
                        marginBottom: '10px',
                        textTransform: 'uppercase'
                      }}>
                        {signal.signal}
                      </div>
                      
                      <div style={{ fontSize: '14px', color: '#7f8c8d', marginBottom: '15px' }}>
                        Generated: {new Date(signal.date || Date.now()).toLocaleString()}
                      </div>
                      
                      {signal.stop_loss && (
                        <div style={{ fontSize: '12px', marginBottom: '10px' }}>
                          Stop Loss: {signal.stop_loss.toFixed(4)}
                        </div>
                      )}
                      
                      <div style={{ display: 'flex', gap: '10px', marginTop: '15px' }}>
                        <button
                          onClick={() => simulateTrade(signal)}
                          style={{
                            flex: 1,
                            padding: '8px 12px',
                            backgroundColor: '#3498db',
                            color: 'white',
                            border: 'none',
                            borderRadius: '6px',
                            cursor: 'pointer',
                            fontSize: '12px',
                            fontWeight: '500'
                          }}
                        >
                          📈 Simulate Trade
                        </button>
                        <button
                          onClick={() => {setChartPair(signal.pair); setShowChart(true);}}
                          style={{
                            flex: 1,
                            padding: '8px 12px',
                            backgroundColor: '#9b59b6',
                            color: 'white',
                            border: 'none',
                            borderRadius: '6px',
                            cursor: 'pointer',
                            fontSize: '12px',
                            fontWeight: '500'
                          }}
                        >
                          📊 View Chart
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Quick Actions */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
              gap: '20px',
              marginBottom: '25px'
            }}>
              <button
                onClick={() => setShowBacktest(true)}
                style={{
                  padding: '20px',
                  backgroundColor: '#9b59b6',
                  color: 'white',
                  border: 'none',
                  borderRadius: '12px',
                  cursor: 'pointer',
                  fontSize: '16px',
                  fontWeight: '500',
                  transition: 'all 0.3s ease',
                  boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
                }}
                onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-2px)'}
                onMouseLeave={e => e.currentTarget.style.transform = 'translateY(0)'}
              >
                📈 Run Backtest
              </button>
              
              <button
                onClick={() => setShowChart(true)}
                style={{
                  padding: '20px',
                  backgroundColor: '#e67e22',
                  color: 'white',
                  border: 'none',
                  borderRadius: '12px',
                  cursor: 'pointer',
                  fontSize: '16px',
                  fontWeight: '500',
                  transition: 'all 0.3s ease',
                  boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
                }}
                onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-2px)'}
                onMouseLeave={e => e.currentTarget.style.transform = 'translateY(0)'}
              >
                📊 View Charts
              </button>
              
              <button
                onClick={() => setShowPortfolio(!showPortfolio)}
                style={{
                  padding: '20px',
                  backgroundColor: '#27ae60',
                  color: 'white',
                  border: 'none',
                  borderRadius: '12px',
                  cursor: 'pointer',
                  fontSize: '16px',
                  fontWeight: '500',
                  transition: 'all 0.3s ease',
                  boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
                }}
                onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-2px)'}
                onMouseLeave={e => e.currentTarget.style.transform = 'translateY(0)'}
              >
                💼 Portfolio
              </button>
            </div>
          </>
        ) : showBacktest ? (
          // Backtest Section
          <div style={{
            backgroundColor: darkMode ? '#333' : 'white',
            padding: '25px',
            borderRadius: '12px',
            marginBottom: '25px',
            boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <h2 style={{ color: darkMode ? '#fff' : '#2c3e50' }}>📈 Backtesting Results</h2>
              <button
                onClick={() => setShowBacktest(false)}
                style={{
                  padding: '8px 16px',
                  backgroundColor: '#95a5a6',
                  color: 'white',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: 'pointer'
                }}
              >
                ← Back to Signals
              </button>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '20px', marginBottom: '25px' }}>
              <div>
                <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>Currency Pair:</label>
                <select
                  value={backtestPair}
                  onChange={(e) => setBacktestPair(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '10px',
                    borderRadius: '6px',
                    border: '1px solid #ddd',
                    backgroundColor: darkMode ? '#2c3e50' : 'white',
                    color: darkMode ? '#fff' : '#333'
                  }}
                >
                  <option value="EURUSD">EUR/USD</option>
                  <option value="XAUUSD">XAU/USD (Gold)</option>
                </select>
              </div>
              
              <div>
                <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>Days to Test:</label>
                <input
                  type="number"
                  value={backtestDays}
                  onChange={(e) => setBacktestDays(parseInt(e.target.value))}
                  min="7"
                  max="365"
                  style={{
                    width: '100%',
                    padding: '10px',
                    borderRadius: '6px',
                    border: '1px solid #ddd',
                    backgroundColor: darkMode ? '#2c3e50' : 'white',
                    color: darkMode ? '#fff' : '#333'
                  }}
                />
              </div>
              
              <div style={{ display: 'flex', alignItems: 'end', gap: '10px' }}>
                <button
                  onClick={runBacktest}
                  disabled={backtestResults?.status === 'running'}
                  style={{
                    flex: 1,
                    padding: '10px',
                    backgroundColor: backtestResults?.status === 'running' ? '#95a5a6' : '#3498db',
                    color: 'white',
                    border: 'none',
                    borderRadius: '6px',
                    cursor: backtestResults?.status === 'running' ? 'not-allowed' : 'pointer',
                    fontWeight: '500'
                  }}
                >
                  {backtestResults?.status === 'running' ? '⏳ Running...' : '🚀 Run Test'}
                </button>
                <button
                  onClick={downloadCSV}
                  style={{
                    padding: '10px 15px',
                    backgroundColor: '#27ae60',
                    color: 'white',
                    border: 'none',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontWeight: '500'
                  }}
                >
                  📥 CSV
                </button>
              </div>
            </div>

            {backtestResults && (
              <div style={{
                backgroundColor: darkMode ? '#2c3e50' : '#f8f9fa',
                padding: '20px',
                borderRadius: '8px',
                border: `2px solid ${
                  backtestResults.status === 'running' ? '#f39c12' :
                  backtestResults.status === 'error' ? '#e74c3c' :
                  backtestResults.total_return > 0 ? '#27ae60' : '#e74c3c'
                }`
              }}>
                {backtestResults.status === 'running' ? (
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '48px', marginBottom: '10px' }}>⏳</div>
                    <div style={{ fontSize: '18px', fontWeight: 'bold' }}>Running Backtest...</div>
                    <div style={{ color: '#7f8c8d' }}>{backtestResults.message}</div>
                  </div>
                ) : backtestResults.status === 'error' ? (
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '48px', marginBottom: '10px' }}>❌</div>
                    <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#e74c3c' }}>Error</div>
                    <div style={{ color: '#7f8c8d' }}>{backtestResults.message}</div>
                  </div>
                ) : (
                  <div>
                    <h3 style={{ marginBottom: '15px' }}>Results for {backtestPair} ({backtestDays} days)</h3>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '15px' }}>
                      <div style={{ textAlign: 'center' }}>
                        <div style={{ 
                          fontSize: '24px', 
                          fontWeight: 'bold',
                          color: backtestResults.total_return > 0 ? '#27ae60' : '#e74c3c'
                        }}>
                          {backtestResults.total_return > 0 ? '+' : ''}{backtestResults.total_return?.toFixed(2)}%
                        </div>
                        <div style={{ fontSize: '14px', color: '#7f8c8d' }}>Total Return</div>
                      </div>
                      <div style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#3498db' }}>
                          {backtestResults.total_trades || 0}
                        </div>
                        <div style={{ fontSize: '14px', color: '#7f8c8d' }}>Total Trades</div>
                      </div>
                      <div style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#9b59b6' }}>
                          {((backtestResults.win_rate || 0) * 100).toFixed(1)}%
                        </div>
                        <div style={{ fontSize: '14px', color: '#7f8c8d' }}>Win Rate</div>
                      </div>
                      <div style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#e67e22' }}>
                          {backtestResults.sharpe_ratio?.toFixed(2) || 'N/A'}
                        </div>
                        <div style={{ fontSize: '14px', color: '#7f8c8d' }}>Sharpe Ratio</div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        ) : (
          // Chart Section
          <div style={{
            backgroundColor: darkMode ? '#333' : 'white',
            padding: '25px',
            borderRadius: '12px',
            marginBottom: '25px',
            boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <h2 style={{ color: darkMode ? '#fff' : '#2c3e50' }}>📊 Price Chart - {chartPair}</h2>
              <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                <select
                  value={chartPair}
                  onChange={(e) => setChartPair(e.target.value)}
                  style={{
                    padding: '8px 12px',
                    borderRadius: '6px',
                    border: '1px solid #ddd',
                    backgroundColor: darkMode ? '#2c3e50' : 'white',
                    color: darkMode ? '#fff' : '#333'
                  }}
                >
                  <option value="EURUSD">EUR/USD</option>
                  <option value="XAUUSD">XAU/USD</option>
                </select>
                <select
                  value={chartType}
                  onChange={(e) => setChartType(e.target.value)}
                  style={{
                    padding: '8px 12px',
                    borderRadius: '6px',
                    border: '1px solid #ddd',
                    backgroundColor: darkMode ? '#2c3e50' : 'white',
                    color: darkMode ? '#fff' : '#333'
                  }}
                >
                  <option value="custom">Custom AI Chart</option>
                  <option value="tradingview">TradingView Style</option>
                </select>
                <button
                  onClick={() => setShowChart(false)}
                  style={{
                    padding: '8px 16px',
                    backgroundColor: '#95a5a6',
                    color: 'white',
                    border: 'none',
                    borderRadius: '6px',
                    cursor: 'pointer'
                  }}
                >
                  ← Back
                </button>
              </div>
            </div>

            {chartType === 'custom' ? (
              <CandlestickChart pair={chartPair} />
            ) : (
              <TradingViewChart pair={chartPair} darkMode={darkMode} />
            )}
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
