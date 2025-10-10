/**
 * Complete Paper Trading App
 * Integrates all components into one dashboard
 */
import React, { useState } from 'react';
import EnhancedTradingChart from './components/EnhancedTradingChart';
import SignalPanel from './components/SignalPanel';
import OrderManager from './components/OrderManager';
import PerformanceDashboard from './components/PerformanceDashboard';
import './PaperTradingApp.css';

const PaperTradingApp = () => {
  const [selectedPair, setSelectedPair] = useState('EURUSD');
  const [selectedInterval, setSelectedInterval] = useState('1h');
  const [activeView, setActiveView] = useState('trading'); // 'trading' or 'performance'

  const pairs = ['EURUSD', 'XAUUSD', 'GBPUSD', 'USDJPY'];
  const intervals = ['5m', '15m', '30m', '1h', '4h', '1d'];

  return (
    <div className="paper-trading-app">
      {/* Header */}
      <header className="app-header">
        <div className="header-left">
          <h1>📊 Paper Trading System</h1>
          <p className="subtitle">Forward testing with live signals & MetaTrader integration</p>
        </div>
        <div className="header-right">
          <button
            className={`view-btn ${activeView === 'trading' ? 'active' : ''}`}
            onClick={() => setActiveView('trading')}
          >
            📈 Trading
          </button>
          <button
            className={`view-btn ${activeView === 'performance' ? 'active' : ''}`}
            onClick={() => setActiveView('performance')}
          >
            📊 Performance
          </button>
        </div>
      </header>

      {/* Symbol & Interval Selector */}
      <div className="controls-bar">
        <div className="control-group">
          <label>Pair:</label>
          <select value={selectedPair} onChange={(e) => setSelectedPair(e.target.value)}>
            {pairs.map(pair => (
              <option key={pair} value={pair}>{pair}</option>
            ))}
          </select>
        </div>
        <div className="control-group">
          <label>Interval:</label>
          <select value={selectedInterval} onChange={(e) => setSelectedInterval(e.target.value)}>
            {intervals.map(interval => (
              <option key={interval} value={interval}>{interval}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Main Content */}
      {activeView === 'trading' ? (
        <div className="trading-view">
          {/* Top Row: Chart + Signals */}
          <div className="top-row">
            <div className="chart-container">
              <EnhancedTradingChart symbol={selectedPair} interval={selectedInterval} />
            </div>
            <div className="signals-container">
              <SignalPanel pair={selectedPair} />
            </div>
          </div>

          {/* Bottom Row: Order Manager */}
          <div className="bottom-row">
            <OrderManager />
          </div>
        </div>
      ) : (
        <div className="performance-view">
          <PerformanceDashboard days={30} />
        </div>
      )}

      {/* Footer */}
      <footer className="app-footer">
        <div className="footer-left">
          <span className="status-dot"></span>
          Connected to backend
        </div>
        <div className="footer-right">
          <span>Multi-Model Signal Aggregator</span>
          <span>|</span>
          <span>Free Tier APIs</span>
          <span>|</span>
          <span>v1.0.0</span>
        </div>
      </footer>
    </div>
  );
};

export default PaperTradingApp;
