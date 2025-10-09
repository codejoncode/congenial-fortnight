import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './UnifiedSignals.css';

const API_BASE_URL = process.env.NODE_ENV === 'production'
  ? 'https://congenial-fortnight-1034520618737.europe-west1.run.app'
  : 'http://localhost:8000';

const UnifiedSignals = ({ pair = 'EURUSD', mode = 'parallel', onSignalUpdate }) => {
  const [signals, setSignals] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [selectedPair, setSelectedPair] = useState(pair);
  const [selectedMode, setSelectedMode] = useState(mode);

  const pairs = ['EURUSD', 'XAUUSD', 'GBPUSD', 'USDJPY'];
  const modes = [
    { value: 'parallel', label: 'All Signals', description: 'Show all opportunities' },
    { value: 'confluence', label: 'Confluence Only', description: 'Both systems agree' },
    { value: 'weighted', label: 'Weighted', description: 'Quality-based combination' }
  ];

  useEffect(() => {
    fetchUnifiedSignals();
    // Auto-refresh every 60 seconds
    const interval = setInterval(fetchUnifiedSignals, 60000);
    return () => clearInterval(interval);
  }, [selectedPair, selectedMode]);

  const fetchUnifiedSignals = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.get(`${API_BASE_URL}/api/signals/unified/`, {
        params: { pair: selectedPair, mode: selectedMode }
      });
      
      setSignals(response.data);
      setLastUpdate(new Date());
      
      // Notify parent component of new signals
      if (onSignalUpdate) {
        onSignalUpdate(response.data);
      }
      
      setLoading(false);
    } catch (err) {
      console.error('Error fetching unified signals:', err);
      setError(err.message || 'Failed to fetch signals');
      setLoading(false);
    }
  };

  const formatPrice = (price) => {
    if (!price) return 'N/A';
    return typeof price === 'number' ? price.toFixed(5) : price;
  };

  const formatPercentage = (value) => {
    if (!value) return 'N/A';
    return `${(value * 100).toFixed(1)}%`;
  };

  const getSignalColor = (type) => {
    if (!type) return 'gray';
    const lowerType = type.toLowerCase();
    if (lowerType === 'long' || lowerType === 'bullish' || lowerType === 'buy') return 'green';
    if (lowerType === 'short' || lowerType === 'bearish' || lowerType === 'sell') return 'red';
    return 'gray';
  };

  const getQualityBadge = (signal) => {
    if (signal.quality) {
      const quality = typeof signal.quality === 'number' ? signal.quality : parseFloat(signal.quality);
      if (quality >= 0.8) return { text: 'EXCELLENT', color: 'gold' };
      if (quality >= 0.6) return { text: 'GOOD', color: 'green' };
      return { text: 'FAIR', color: 'orange' };
    }
    return null;
  };

  const renderMLSignal = (signal, index) => {
    const quality = getQualityBadge(signal);
    
    return (
      <div key={`ml-${index}`} className="signal-card ml-signal">
        <div className="signal-header">
          <div className="signal-source">
            <span className="source-badge ml-badge">🤖 ML Pip-Based</span>
            {quality && (
              <span className={`quality-badge quality-${quality.color}`}>
                {quality.text}
              </span>
            )}
          </div>
          <div className={`signal-type ${getSignalColor(signal.type)}`}>
            {signal.type?.toUpperCase() || 'N/A'}
          </div>
        </div>

        <div className="signal-body">
          <div className="signal-metrics">
            <div className="metric">
              <span className="metric-label">Entry:</span>
              <span className="metric-value">{formatPrice(signal.entry)}</span>
            </div>
            <div className="metric">
              <span className="metric-label">Stop Loss:</span>
              <span className="metric-value">{formatPrice(signal.stop_loss)}</span>
            </div>
            <div className="metric">
              <span className="metric-label">Take Profit:</span>
              <span className="metric-value">{formatPrice(signal.take_profit)}</span>
            </div>
          </div>

          <div className="signal-stats">
            <div className="stat">
              <span className="stat-label">Risk Pips:</span>
              <span className="stat-value">{signal.risk_pips || 'N/A'}</span>
            </div>
            <div className="stat">
              <span className="stat-label">Reward Pips:</span>
              <span className="stat-value">{signal.reward_pips || 'N/A'}</span>
            </div>
            <div className="stat">
              <span className="stat-label">R:R Ratio:</span>
              <span className="stat-value highlight">{signal.risk_reward_ratio?.toFixed(2) || 'N/A'}</span>
            </div>
            <div className="stat">
              <span className="stat-label">Confidence:</span>
              <span className="stat-value">{formatPercentage(signal.confidence)}</span>
            </div>
          </div>

          {signal.reasoning && (
            <div className="signal-reasoning">
              <strong>Analysis:</strong> {signal.reasoning}
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderHarmonicSignal = (signal, index) => {
    const quality = getQualityBadge(signal);
    
    return (
      <div key={`harmonic-${index}`} className="signal-card harmonic-signal">
        <div className="signal-header">
          <div className="signal-source">
            <span className="source-badge harmonic-badge">📐 Harmonic Pattern</span>
            {quality && (
              <span className={`quality-badge quality-${quality.color}`}>
                {quality.text}
              </span>
            )}
          </div>
          <div className={`signal-type ${getSignalColor(signal.type)}`}>
            {signal.type?.toUpperCase() || 'N/A'}
          </div>
        </div>

        <div className="signal-body">
          <div className="pattern-info">
            <strong>Pattern:</strong> {signal.pattern?.replace(/_/g, ' ').toUpperCase() || 'N/A'}
          </div>

          <div className="signal-metrics">
            <div className="metric">
              <span className="metric-label">Entry (D):</span>
              <span className="metric-value">{formatPrice(signal.entry)}</span>
            </div>
            <div className="metric">
              <span className="metric-label">Stop Loss:</span>
              <span className="metric-value">{formatPrice(signal.stop_loss)}</span>
            </div>
          </div>

          <div className="targets">
            <div className="target-header">Fibonacci Targets:</div>
            <div className="target-list">
              {signal.target_1 && (
                <div className="target">
                  <span className="target-label">T1 (38.2%):</span>
                  <span className="target-value">{formatPrice(signal.target_1)}</span>
                  <span className="target-rr">R:R {signal.risk_reward_t1?.toFixed(1) || 'N/A'}</span>
                </div>
              )}
              {signal.target_2 && (
                <div className="target">
                  <span className="target-label">T2 (61.8%):</span>
                  <span className="target-value">{formatPrice(signal.target_2)}</span>
                  <span className="target-rr">R:R {signal.risk_reward_t2?.toFixed(1) || 'N/A'}</span>
                </div>
              )}
              {signal.target_3 && (
                <div className="target">
                  <span className="target-label">T3 (100%):</span>
                  <span className="target-value">{formatPrice(signal.target_3)}</span>
                  <span className="target-rr">R:R {signal.risk_reward_t3?.toFixed(1) || 'N/A'}</span>
                </div>
              )}
            </div>
          </div>

          {signal.reasoning && (
            <div className="signal-reasoning">
              <strong>Analysis:</strong> {signal.reasoning}
            </div>
          )}

          {/* Pattern Points for Reference */}
          {signal.X && (
            <div className="pattern-points">
              <div className="points-header">Pattern Points:</div>
              <div className="points-grid">
                <span>X: {formatPrice(signal.X)}</span>
                <span>A: {formatPrice(signal.A)}</span>
                <span>B: {formatPrice(signal.B)}</span>
                <span>C: {formatPrice(signal.C)}</span>
                <span>D: {formatPrice(signal.D)}</span>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderRecommendation = () => {
    if (!signals?.recommendation) return null;

    const rec = signals.recommendation;
    const actionColor = getSignalColor(rec.action);
    const isConfluence = rec.confluence;

    return (
      <div className={`recommendation ${isConfluence ? 'confluence-recommendation' : ''}`}>
        {isConfluence && (
          <div className="confluence-badge">
            ⭐ CONFLUENCE SIGNAL
          </div>
        )}
        
        <div className="recommendation-header">
          <h3>System Recommendation</h3>
          <div className={`recommendation-action ${actionColor}`}>
            {rec.action}
          </div>
        </div>

        <div className="recommendation-body">
          <div className="recommendation-metric">
            <span className="label">Confidence:</span>
            <span className="value">{formatPercentage(rec.confidence)}</span>
          </div>
          <div className="recommendation-reason">
            {rec.reason}
          </div>
          <div className="recommendation-flags">
            {rec.has_ml && <span className="flag ml-flag">✓ ML Signal</span>}
            {rec.has_harmonic && <span className="flag harmonic-flag">✓ Harmonic Pattern</span>}
          </div>
        </div>
      </div>
    );
  };

  if (loading && !signals) {
    return (
      <div className="unified-signals-container">
        <div className="loading">
          <div className="spinner"></div>
          <p>Loading signals...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="unified-signals-container">
        <div className="error">
          <p>Error: {error}</p>
          <button onClick={fetchUnifiedSignals} className="retry-btn">
            Retry
          </button>
        </div>
      </div>
    );
  }

  const hasSignals = signals && (
    (signals.ml_signals && signals.ml_signals.length > 0) ||
    (signals.harmonic_signals && signals.harmonic_signals.length > 0)
  );

  return (
    <div className="unified-signals-container">
      <div className="signals-header">
        <h2>📊 Unified Trading Signals</h2>
        
        <div className="controls">
          <div className="control-group">
            <label>Pair:</label>
            <select 
              value={selectedPair} 
              onChange={(e) => setSelectedPair(e.target.value)}
              className="pair-select"
            >
              {pairs.map(p => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
          </div>

          <div className="control-group">
            <label>Mode:</label>
            <select 
              value={selectedMode} 
              onChange={(e) => setSelectedMode(e.target.value)}
              className="mode-select"
            >
              {modes.map(m => (
                <option key={m.value} value={m.value}>
                  {m.label} - {m.description}
                </option>
              ))}
            </select>
          </div>

          <button 
            onClick={fetchUnifiedSignals} 
            className="refresh-btn"
            disabled={loading}
          >
            🔄 Refresh
          </button>
        </div>

        {lastUpdate && (
          <div className="last-update">
            Last updated: {lastUpdate.toLocaleTimeString()}
          </div>
        )}
      </div>

      {!hasSignals ? (
        <div className="no-signals">
          <p>No signals available for {selectedPair} in {selectedMode} mode</p>
          <p className="hint">Try a different pair or mode</p>
        </div>
      ) : (
        <>
          {/* Recommendation Section */}
          {renderRecommendation()}

          {/* ML Signals Section */}
          {signals.ml_signals && signals.ml_signals.length > 0 && (
            <div className="signals-section">
              <h3 className="section-title">🤖 Machine Learning Signals</h3>
              <div className="signals-grid">
                {signals.ml_signals.map((signal, idx) => renderMLSignal(signal, idx))}
              </div>
            </div>
          )}

          {/* Harmonic Signals Section */}
          {signals.harmonic_signals && signals.harmonic_signals.length > 0 && (
            <div className="signals-section">
              <h3 className="section-title">📐 Harmonic Pattern Signals</h3>
              <div className="signals-grid">
                {signals.harmonic_signals.map((signal, idx) => renderHarmonicSignal(signal, idx))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default UnifiedSignals;
