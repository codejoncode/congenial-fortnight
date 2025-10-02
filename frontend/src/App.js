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

1. react-tradingview-widget - Direct TradingView integration
   npm install react-tradingview-widget
   - Full TradingView charts with indicators
   - Professional look and feel
   - Requires TradingView account for advanced features

2. lightweight-charts - Lightweight charting library
   npm install lightweight-charts
   - Similar to TradingView but lighter
   - Good performance, modern API
   - Free and open source

3. react-financial-charts - Advanced financial charting
   npm install react-financial-charts
   - Built on D3, highly customizable
   - Great for technical analysis
   - Steeper learning curve

Current implementation uses Recharts for simplicity and customization.
For production TradingView-like experience, consider lightweight-charts.
*/

function App() {
  const [signals, setSignals] = useState([]);
  const [backtestResults, setBacktestResults] = useState(null);
  const [showBacktest, setShowBacktest] = useState(false);
  const [showChart, setShowChart] = useState(false);
  const [backtestPair, setBacktestPair] = useState('EURUSD');
  const [chartPair, setChartPair] = useState('EURUSD');
  const [backtestDays, setBacktestDays] = useState(30);
  const [chartType, setChartType] = useState('custom'); // 'custom' or 'tradingview'
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
    
    // Load dark mode preference
    const savedDarkMode = localStorage.getItem('darkMode') === 'true';
    setDarkMode(savedDarkMode);
  }, []);

  // Auto-refresh signals
  useEffect(() => {
    let interval;
    if (autoRefresh) {
      interval = setInterval(fetchSignals, 30000); // Refresh every 30 seconds
    }
    return () => clearInterval(interval);
  }, [autoRefresh]);

  // Check for new signals and show notifications
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
        setNotifications(prev => [newNotification, ...prev.slice(0, 9)]); // Keep last 10 notifications
      }
    }
  }, [signals, notifications]);

  const fetchSignals = () => {
    axios.get(`${API_BASE_URL}/api/signals/`)
      .then(response => {
        setSignals(response.data);
      })
      .catch(error => {
        console.error('Error fetching signals:', error);
      });
  };

  const toggleDarkMode = () => {
    const newDarkMode = !darkMode;
    setDarkMode(newDarkMode);
    localStorage.setItem('darkMode', newDarkMode.toString());
  };

  const clearNotifications = () => {
    setNotifications([]);
  };

  const simulateTrade = (signal) => {
    // Simple portfolio simulation
    const positionSize = 1000; // $1000 per trade
    const currentPrice = 1.0850; // Mock current price
    const stopLoss = signal.stop_loss || (signal.signal === 'bullish' ? currentPrice * 0.98 : currentPrice * 1.02);
    
    const newPosition = {
      id: Date.now(),
      pair: signal.pair,
      type: signal.signal,
      entryPrice: currentPrice,
      stopLoss: stopLoss,
      size: positionSize,
      timestamp: new Date(),
      status: 'open'
    };
    
    setPortfolio(prev => ({
      ...prev,
      positions: [...prev.positions, newPosition]
    }));
    
    // Add to history
    setPortfolio(prev => ({
      ...prev,
      history: [...prev.history, {
        ...newPosition,
        exitPrice: null,
        pnl: 0,
        status: 'open'
      }]
    }));
  };

  const downloadCSV = () => {
    const url = `${API_BASE_URL}/api/backtest/csv/?pair=${backtestPair}&days=${backtestDays}`;
    window.open(url, '_blank');
  };

  const runBacktest = () => {
    setBacktestResults({ status: 'running', message: `Starting backtest for ${backtestPair} over ${backtestDays} days...` });
    
    axios.get(`${API_BASE_URL}/api/backtest/?pair=${backtestPair}&days=${backtestDays}`)
      .then(response => {
        setBacktestResults(response.data);
      })
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
        <div style={{
          position: 'fixed',
          top: '20px',
          right: '20px',
          zIndex: 1000,
          maxWidth: '400px'
        }}>
          {notifications.slice(0, 3).map((notification, index) => (
            <div key={notification.id} style={{
              backgroundColor: darkMode ? '#333' : 'white',
              color: darkMode ? '#fff' : '#333',
              padding: '15px',
              marginBottom: '10px',
              borderRadius: '8px',
              boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
              border: '1px solid #ddd',
              animation: 'slideIn 0.5s ease-out',
              transform: `translateY(${index * -10}px)`
            }}>
              <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>🔔 New Signal</div>
              <div style={{ fontSize: '14px' }}>{notification.message}</div>
              <div style={{ fontSize: '12px', color: '#666', marginTop: '5px' }}>
                {notification.timestamp.toLocaleTimeString()}
              </div>
            </div>
          ))}
          {notifications.length > 3 && (
            <div style={{ textAlign: 'center', marginTop: '10px' }}>
              <button onClick={clearNotifications} style={{
                padding: '5px 10px',
                backgroundColor: '#6c757d',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer'
              }}>
                Clear All ({notifications.length})
              </button>
            </div>
          )}
        </div>
      )}

      <header className="App-header" style={{
        backgroundColor: darkMode ? '#2c3e50' : '#2c3e50',
        color: 'white',
        padding: '20px',
        borderRadius: '10px',
        marginBottom: '30px',
        boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
          <h1 style={{ margin: '0', fontSize: '2.5em', fontWeight: '300' }}>
            📈 Forex Signals Dashboard
          </h1>
          
          <div style={{ display: 'flex', gap: '15px', alignItems: 'center' }}>
            {/* Dark Mode Toggle */}
            <button 
              onClick={toggleDarkMode}
              style={{
                padding: '8px 16px',
                backgroundColor: darkMode ? '#f39c12' : '#34495e',
                color: 'white',
                border: 'none',
                borderRadius: '20px',
                cursor: 'pointer',
                fontSize: '14px',
                transition: 'all 0.3s ease'
              }}
            >
              {darkMode ? '☀️ Light' : '🌙 Dark'}
            </button>
            
            {/* Auto Refresh Toggle */}
            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
                style={{ margin: 0 }}
              />
              <span style={{ fontSize: '14px' }}>Auto Refresh (30s)</span>
            </label>
            
            {/* Portfolio Toggle */}
            <button 
              onClick={() => setShowPortfolio(!showPortfolio)}
              style={{
                padding: '8px 16px',
                backgroundColor: showPortfolio ? '#e74c3c' : '#27ae60',
                color: 'white',
                border: 'none',
                borderRadius: '6px',
                cursor: 'pointer',
                fontSize: '14px'
              }}
            >
              {showPortfolio ? '📊 Hide Portfolio' : '💼 Portfolio'}
            </button>
          </div>
        </div>
        
        <div style={{ display: 'flex', gap: '15px', alignItems: 'center', flexWrap: 'wrap' }}>
          <button 
            onClick={() => setShowBacktest(!showBacktest)}
            style={{
              padding: '12px 20px',
              backgroundColor: showBacktest ? '#e74c3c' : '#3498db',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '14px',
              fontWeight: '500',
              transition: 'all 0.3s ease'
            }}
          >
            {showBacktest ? '❌ Hide Backtest' : '📊 Show Backtest Results'}
          </button>
          <button 
            onClick={() => setShowChart(!showChart)} 
            style={{
              padding: '12px 20px',
              backgroundColor: showChart ? '#e74c3c' : '#27ae60',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '14px',
              fontWeight: '500',
              transition: 'all 0.3s ease'
            }}
          >
            {showChart ? '📉 Hide Chart' : '📈 Show Chart'}
          </button>
        </div>
        
        {/* Main Content Area */}
        <div style={{ marginTop: '20px' }}>
          {!showBacktest ? (
          <>
            {/* Portfolio Section */}
            {showPortfolio && (
              <div style={{
                backgroundColor: darkMode ? '#333' : 'white',
                padding: '20px',
                borderRadius: '8px',
                marginBottom: '20px',
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
              }}>
                <h2 style={{ margin: '0 0 15px 0', color: darkMode ? '#fff' : '#333' }}>💼 Portfolio Overview</h2>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '15px' }}>
                  <div style={{ textAlign: 'center', padding: '15px', backgroundColor: darkMode ? '#444' : '#f8f9fa', borderRadius: '6px' }}>
                    <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#27ae60' }}>
                      ${portfolio.balance.toFixed(2)}
                    </div>
                    <div style={{ fontSize: '14px', color: darkMode ? '#ccc' : '#666' }}>Account Balance</div>
                  </div>
                  <div style={{ textAlign: 'center', padding: '15px', backgroundColor: darkMode ? '#444' : '#f8f9fa', borderRadius: '6px' }}>
                    <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#3498db' }}>
                      {portfolio.positions.length}
                    </div>
                    <div style={{ fontSize: '14px', color: darkMode ? '#ccc' : '#666' }}>Open Positions</div>
                  </div>
                  <div style={{ textAlign: 'center', padding: '15px', backgroundColor: darkMode ? '#444' : '#f8f9fa', borderRadius: '6px' }}>
                    <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#f39c12' }}>
                      {portfolio.history.length}
                    </div>
                    <div style={{ fontSize: '14px', color: darkMode ? '#ccc' : '#666' }}>Total Trades</div>
                  </div>
                </div>
                
                {portfolio.positions.length > 0 && (
                  <div style={{ marginTop: '20px' }}>
                    <h3 style={{ color: darkMode ? '#fff' : '#333' }}>Open Positions</h3>
                    <div style={{ display: 'grid', gap: '10px' }}>
                      {portfolio.positions.map(position => (
                        <div key={position.id} style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                          padding: '10px',
                          backgroundColor: darkMode ? '#444' : '#f8f9fa',
                          borderRadius: '6px',
                          border: `1px solid ${position.type === 'bullish' ? '#27ae60' : '#e74c3c'}`
                        }}>
                          <div>
                            <strong>{position.pair}</strong> - {position.type.toUpperCase()}
                            <br />
                            <small style={{ color: darkMode ? '#ccc' : '#666' }}>
                              Entry: ${position.entryPrice.toFixed(4)} | Size: ${position.size}
                            </small>
                          </div>
                          <div style={{ textAlign: 'right' }}>
                            <div style={{ fontWeight: 'bold', color: position.type === 'bullish' ? '#27ae60' : '#e74c3c' }}>
                              P/L: ${(Math.random() * 200 - 100).toFixed(2)}
                            </div>
                            <small style={{ color: darkMode ? '#ccc' : '#666' }}>
                              {position.timestamp.toLocaleDateString()}
                            </small>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            <h2 style={{ margin: '30px 0 20px 0', fontSize: '1.8em', fontWeight: '300', color: darkMode ? '#fff' : '#ecf0f1' }}>
              🎯 Current Signals
            </h2>
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', 
              gap: '15px',
              marginBottom: '20px'
            }}>
              {(() => {
                // Group signals by pair and get the most recent signal for each pair
                const signalsByPair = {};
                signals.forEach(signal => {
                  if (!signalsByPair[signal.pair] || new Date(signal.date) > new Date(signalsByPair[signal.pair].date)) {
                    signalsByPair[signal.pair] = signal;
                  }
                });
                
                // Convert to array and sort by pair name for consistent display
                const recentSignals = Object.values(signalsByPair).sort((a, b) => a.pair.localeCompare(b.pair));
                
                return recentSignals.map(signal => (
                  <div key={signal.id} style={{
                    backgroundColor: darkMode ? '#333' : 'white',
                    padding: '20px',
                    borderRadius: '8px',
                    boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                    border: `2px solid ${signal.signal === 'bullish' ? '#27ae60' : '#e74c3c'}`,
                    transition: 'transform 0.2s ease'
                  }}>
                    <div style={{ 
                      fontSize: '1.2em', 
                      fontWeight: 'bold', 
                      color: signal.signal === 'bullish' ? '#27ae60' : '#e74c3c',
                      marginBottom: '10px'
                    }}>
                      {signal.pair} - {signal.signal.toUpperCase()}
                    </div>
                    <div style={{ fontSize: '0.9em', color: darkMode ? '#ccc' : '#666', marginBottom: '5px' }}>
                      Confidence: {(signal.probability * 100).toFixed(1)}%
                    </div>
                    <div style={{ fontSize: '0.9em', color: darkMode ? '#ccc' : '#666', marginBottom: '5px' }}>
                      Stop Loss: {signal.stop_loss?.toFixed(4)}
                    </div>
                    <div style={{ fontSize: '0.8em', color: darkMode ? '#aaa' : '#999' }}>
                      Date: {signal.date}
                    </div>
                    <button 
                      onClick={() => simulateTrade(signal)}
                      style={{
                        marginTop: '10px',
                        padding: '8px 16px',
                        backgroundColor: signal.signal === 'bullish' ? '#27ae60' : '#e74c3c',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        fontSize: '12px',
                        fontWeight: 'bold'
                      }}
                    >
                      💰 Simulate Trade
                    </button>
                  </div>
                ));
              })()}
            </div>
          </>
        ) : (
          <div>
            <h2 style={{ color: darkMode ? '#fff' : '#333' }}>Backtest Analysis</h2>
            <div style={{ backgroundColor: darkMode ? '#333' : '#f8f9fa', padding: '20px', borderRadius: '8px', marginBottom: '20px' }}>
              <div style={{ display: 'flex', gap: '20px', alignItems: 'center', flexWrap: 'wrap', marginBottom: '15px' }}>
                <div>
                  <label style={{ 
                    fontWeight: '600', 
                    color: darkMode ? '#fff' : '#495057',
                    marginRight: '10px',
                    fontSize: '1.1em'
                  }}>
                    Pair:
                  </label>
                  <select 
                    value={backtestPair} 
                    onChange={(e) => setBacktestPair(e.target.value)}
                    style={{
                      padding: '8px 12px',
                      borderRadius: '4px',
                      border: '1px solid #ced4da',
                      fontSize: '1em',
                      backgroundColor: darkMode ? '#444' : 'white',
                      color: darkMode ? '#fff' : '#333',
                      cursor: 'pointer'
                    }}
                  >
                    <option value="EURUSD">EURUSD</option>
                    <option value="XAUUSD">XAUUSD</option>
                    <option value="GBPUSD">GBPUSD</option>
                    <option value="USDJPY">USDJPY</option>
                    <option value="AUDUSD">AUDUSD</option>
                    <option value="USDCAD">USDCAD</option>
                    <option value="USDCHF">USDCHF</option>
                    <option value="NZDUSD">NZDUSD</option>
                  </select>
                </div>
                <div>
                  <label style={{ 
                    fontWeight: '600', 
                    color: darkMode ? '#fff' : '#495057',
                    marginRight: '10px',
                    fontSize: '1.1em'
                  }}>
                    Days:
                  </label>
                  <input 
                    type="number" 
                    value={backtestDays} 
                    onChange={(e) => setBacktestDays(e.target.value)}
                    min="10" 
                    max="365"
                    style={{
                      padding: '8px 12px',
                      borderRadius: '4px',
                      border: '1px solid #ced4da',
                      fontSize: '1em',
                      backgroundColor: darkMode ? '#444' : 'white',
                      color: darkMode ? '#fff' : '#333',
                      width: '80px'
                    }}
                  />
                </div>
                <button 
                  onClick={runBacktest}
                  style={{
                    padding: '10px 20px',
                    backgroundColor: '#007bff',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontSize: '1em',
                    fontWeight: '500'
                  }}
                >
                  Run Backtest
                </button>
                {backtestResults && (
                  <button 
                    onClick={downloadCSV} 
                    style={{ 
                      padding: '10px 20px',
                      backgroundColor: '#28a745', 
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: 'pointer',
                      fontSize: '1em',
                      fontWeight: '500'
                    }}
                  >
                    📥 Download CSV
                  </button>
                )}
              </div>
            
            {backtestResults && (
              <div>
                {backtestResults.message && (
                  <div style={{
                    backgroundColor: backtestResults.status === 'completed' ? '#d4edda' : (backtestResults.status === 'error' ? '#f8d7da' : '#fff3cd'),
                    color: backtestResults.status === 'completed' ? '#155724' : (backtestResults.status === 'error' ? '#721c24' : '#856404'),
                    padding: '10px', 
                    margin: '10px 0', 
                    borderRadius: '5px', 
                    border: '1px solid ' + (backtestResults.status === 'completed' ? '#c3e6cb' : backtestResults.status === 'error' ? '#f5c6cb' : '#ffeaa7')
                  }}>
                    <strong>
                      {backtestResults.status === 'completed' ? '✅' : backtestResults.status === 'error' ? '❌' : '⏳'} {backtestResults.message}
                    </strong>
                  </div>
                )}
                
                {!backtestResults.error && backtestResults.pair && (
                  <div>
                    <h3>{backtestResults.pair} - {backtestResults.days} Days Enhanced Backtest</h3>

                    {/* Overall Statistics */}
                    <div style={{backgroundColor: '#f0f0f0', padding: '10px', margin: '10px 0', borderRadius: '5px'}}>
                      <h4>Overall Performance</h4>
                      <p>Accuracy: {(backtestResults.overall_accuracy * 100).toFixed(1)}%</p>
                      <p>Total Signals: {backtestResults.total_signals}</p>
                      <p>Wins: {backtestResults.wins} | Losses: {backtestResults.losses}</p>
                    </div>

                    {/* Pips Analysis */}
                    <div style={{backgroundColor: '#e8f4f8', padding: '10px', margin: '10px 0', borderRadius: '5px'}}>
                      <h4>💰 Pips Analysis</h4>
                      <p>Total Pips Won: <span style={{color: 'green'}}>{backtestResults.total_pips_won?.toFixed(1)}</span></p>
                      <p>Total Pips Lost: <span style={{color: 'red'}}>{Math.abs(backtestResults.total_pips_lost)?.toFixed(1)}</span></p>
                      <p><strong>Net Pips: <span style={{color: backtestResults.net_pips >= 0 ? 'green' : 'red'}}>{backtestResults.net_pips?.toFixed(1)}</span></strong></p>
                      <p>Average Win: {backtestResults.avg_win_pips?.toFixed(2)} pips</p>
                      <p>Average Loss: {Math.abs(backtestResults.avg_loss_pips)?.toFixed(2)} pips</p>
                      <p>Profit Factor: {backtestResults.profit_factor?.toFixed(2)}</p>
                      <p>Largest Win: {backtestResults.largest_win?.toFixed(1)} pips</p>
                      <p>Largest Loss: {Math.abs(backtestResults.largest_loss)?.toFixed(1)} pips</p>
                    </div>

                    {/* Probability Analysis */}
                    <h4>🎯 Accuracy by Probability Range</h4>
                    <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '10px'}}>
                      {Object.entries(backtestResults.probability_bins || {}).map(([range, data]) => (
                        <div key={range} style={{
                          border: '1px solid #ddd',
                          padding: '8px',
                          borderRadius: '4px',
                          backgroundColor: data.accuracy > 0.6 ? '#d4edda' : (data.accuracy > 0.5 ? '#fff3cd' : '#f8d7da')
                        }}>
                          <strong>{range}:</strong><br/>
                          {data.total > 0 ? (data.accuracy * 100).toFixed(1) : 0}% accuracy<br/>
                          ({data.correct}/{data.total} correct)
                        </div>
                      ))}
                    </div>

                    {/* Recent Trades */}
                    <h4>📊 Recent Trade Details</h4>
                    <div style={{maxHeight: '300px', overflowY: 'auto', border: '1px solid #ddd', borderRadius: '4px'}}>
                      <table style={{width: '100%', borderCollapse: 'collapse'}}>
                        <thead>
                          <tr style={{backgroundColor: '#f8f9fa'}}>
                            <th style={{padding: '8px', border: '1px solid #ddd'}}>Date</th>
                            <th style={{padding: '8px', border: '1px solid #ddd'}}>Signal</th>
                            <th style={{padding: '8px', border: '1px solid #ddd'}}>Probability</th>
                            <th style={{padding: '8px', border: '1px solid #ddd'}}>Result</th>
                            <th style={{padding: '8px', border: '1px solid #ddd'}}>Pips</th>
                          </tr>
                        </thead>
                        <tbody>
                          {(backtestResults.recent_results || []).map((result, index) => (
                            <tr key={index} style={{
                              backgroundColor: result.correct ? '#d4edda' : '#f8d7da'
                            }}>
                              <td style={{padding: '8px', border: '1px solid #ddd'}}>{result.date}</td>
                              <td style={{padding: '8px', border: '1px solid #ddd'}}>{result.signal}</td>
                              <td style={{padding: '8px', border: '1px solid #ddd'}}>{(result.probability * 100).toFixed(1)}%</td>
                              <td style={{padding: '8px', border: '1px solid #ddd'}}>
                                {result.correct ? '✅ Win' : '❌ Loss'}
                              </td>
                              <td style={{
                                padding: '8px',
                                border: '1px solid #ddd',
                                color: result.pips >= 0 ? 'green' : 'red',
                                fontWeight: 'bold'
                              }}>
                                {result.pips?.toFixed(1)}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
                
              </div>
            )}

            {backtestResults && backtestResults.error && (
              <div style={{backgroundColor: '#f8d7da', color: '#721c24', padding: '10px', margin: '10px 0', borderRadius: '5px', border: '1px solid #f5c6cb'}}>
                <strong>❌ Error: {backtestResults.error}</strong>
              </div>
            )}

          </div>

        {showChart && (
        <div style={{
          marginTop: '30px',
          padding: '25px',
          backgroundColor: darkMode ? '#333' : 'white',
          borderRadius: '10px',
          boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
          border: '1px solid #e0e0e0'
        }}>
          <h2 style={{ 
            margin: '0 0 20px 0', 
            color: darkMode ? '#fff' : '#2c3e50', 
            fontSize: '1.8em', 
            fontWeight: '300',
            borderBottom: '2px solid #3498db',
            paddingBottom: '10px'
          }}>
            📊 Price Chart - {chartPair}
          </h2>
          <div style={{
            marginBottom: '20px',
            padding: '15px',
            backgroundColor: darkMode ? '#444' : '#f8f9fa',
            borderRadius: '6px',
            border: '1px solid #dee2e6'
          }}>
            <div style={{ display: 'flex', gap: '20px', alignItems: 'center', flexWrap: 'wrap' }}>
              <div>
                <label style={{ 
                  fontWeight: '600', 
                  color: darkMode ? '#fff' : '#495057',
                  marginRight: '10px',
                  fontSize: '1.1em'
                }}>
                  Select Pair:
                </label>
                <select 
                  value={chartPair} 
                  onChange={(e) => setChartPair(e.target.value)}
                  style={{
                    padding: '8px 12px',
                    borderRadius: '4px',
                    border: '1px solid #ced4da',
                    fontSize: '1em',
                    backgroundColor: darkMode ? '#555' : 'white',
                    color: darkMode ? '#fff' : '#333',
                    cursor: 'pointer'
                  }}
                >
                  <option value="EURUSD">EURUSD</option>
                  <option value="XAUUSD">XAUUSD</option>
                  <option value="GBPUSD">GBPUSD</option>
                  <option value="USDJPY">USDJPY</option>
                  <option value="AUDUSD">AUDUSD</option>
                  <option value="USDCAD">USDCAD</option>
                  <option value="USDCHF">USDCHF</option>
                  <option value="NZDUSD">NZDUSD</option>
                </select>
              </div>
              
              <div>
                <label style={{ 
                  fontWeight: '600', 
                  color: darkMode ? '#fff' : '#495057',
                  marginRight: '10px',
                  fontSize: '1.1em'
                }}>
                  Chart Type:
                </label>
                <select 
                  value={chartType} 
                  onChange={(e) => setChartType(e.target.value)}
                  style={{
                    padding: '8px 12px',
                    borderRadius: '4px',
                    border: '1px solid #ced4da',
                    fontSize: '1em',
                    backgroundColor: darkMode ? '#555' : 'white',
                    color: darkMode ? '#fff' : '#333',
                    cursor: 'pointer'
                  }}
                >
                  <option value="custom">Custom Chart (with AI Predictions)</option>
                  <option value="tradingview">TradingView Style</option>
                </select>
              </div>
            </div>
            
            <div style={{ 
              marginTop: '10px', 
              fontSize: '0.9em', 
              color: darkMode ? '#ccc' : '#6c757d',
              fontStyle: 'italic'
            }}>
              💡 <strong>Custom Chart:</strong> Gold star (★) indicates AI prediction candle • Gold outline shows predicted price movement<br/>
              💡 <strong>TradingView:</strong> Professional charting experience (requires additional setup)
            </div>
          </div>
          
          {chartType === 'custom' ? (
            <CandlestickChart pair={chartPair} />
          ) : (
            <TradingViewChart pair={chartPair} />
          )}
        </div>
      )}
      
      <div style={{ 
        textAlign: 'center', 
        marginTop: '30px',
        padding: '20px',
        backgroundColor: darkMode ? '#333' : '#ecf0f1',
        borderRadius: '8px'
      }}>
        <button 
          onClick={fetchSignals} 
          style={{
            padding: '12px 25px',
            backgroundColor: '#3498db',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            cursor: 'pointer',
            fontSize: '16px',
            fontWeight: '500',
            transition: 'all 0.3s ease',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
          }}
          onMouseOver={(e) => e.target.style.backgroundColor = '#2980b9'}
          onMouseOut={(e) => e.target.style.backgroundColor = '#3498db'}
        >
          🔄 Refresh Signals
        </button>
      </div>
    </div>
  );
}

export default App;
