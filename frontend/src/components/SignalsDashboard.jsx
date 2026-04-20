import React, { useState, useEffect } from 'react';
import axios from 'axios';

const SIGNAL_LABEL = { bullish: 'BUY', bearish: 'SELL', no_signal: 'WAIT' };

const SignalsDashboard = ({ apiBaseUrl, signals: propSignals, darkMode = false, onRequestGenerate }) => {
  const [signals, setSignals] = useState(propSignals || []);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [executingTrade, setExecutingTrade] = useState(null);
  const [tradeMessage, setTradeMessage] = useState(null);
  const [generating, setGenerating] = useState(false);

  // Fetch signals on mount if not provided as props
  useEffect(() => {
    if (!propSignals || propSignals.length === 0) {
      fetchSignals();
    } else {
      setSignals(propSignals);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [propSignals]);

  const fetchSignals = async () => {
    setLoading(true);
    setError('');
    try {
      const response = await axios.get(`${apiBaseUrl}/api/signals/`);
      setSignals(response.data);
    } catch (err) {
      console.error('Error fetching signals:', err);
      setError('Failed to load signals');
    } finally {
      setLoading(false);
    }
  };

  const executeTrade = async (signal) => {
    setExecutingTrade(signal.id);
    setTradeMessage(null);
    
    try {
      const response = await axios.post(`${apiBaseUrl}/api/paper-trades/execute/`, {
        pair: signal.pair,
        signal: signal.signal,
        stop_loss: signal.stop_loss,
        take_profit: signal.take_profit,
        entry_price: signal.entry_price,
        probability: signal.probability,
        lot_size: 0.1,
      });
      
      setTradeMessage({
        type: 'success',
        text: `✅ Paper trade executed! ${signal.pair} ${signal.signal.toUpperCase()} @ ${response.data.entry_price.toFixed(4)}`
      });
      
      // Auto-dismiss after 5 seconds
      setTimeout(() => setTradeMessage(null), 5000);
    } catch (error) {
      console.error('Trade execution failed:', error);
      setTradeMessage({
        type: 'error',
        text: `❌ Failed to execute trade: ${error.response?.data?.error || error.message}`
      });
      
      setTimeout(() => setTradeMessage(null), 5000);
    } finally {
      setExecutingTrade(null);
    }
  };

  const getSignalColor = (signal) => {
    return signal === 'bullish' ? '#3fb950' : '#f85149';
  };

  const getSignalIcon = (signal) => {
    return signal === 'bullish' ? '📈' : '📉';
  };

  const getConfidenceLevel = (probability) => {
    if (probability >= 0.8) return { text: 'Very High', color: '#00ff87', glow: 'rgba(0,255,135,0.3)' };
    if (probability >= 0.7) return { text: 'High', color: '#60efff', glow: 'rgba(96,239,255,0.3)' };
    if (probability >= 0.6) return { text: 'Medium', color: '#ffa500', glow: 'rgba(255,165,0,0.3)' };
    return { text: 'Low', color: '#ff6b6b', glow: 'rgba(255,107,107,0.3)' };
  };

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '40px' }}>
        <div style={{
          width: '50px',
          height: '50px',
          border: '5px solid #f3f3f3',
          borderTop: '5px solid #00ff87',
          borderRadius: '50%',
          animation: 'spin 1s linear infinite',
          margin: '0 auto'
        }} />
        <p style={{ marginTop: '20px', color: darkMode ? '#8b949e' : '#6c757d' }}>Loading signals...</p>
        <style>
          {`
            @keyframes spin {
              0% { transform: rotate(0deg); }
              100% { transform: rotate(360deg); }
            }
          `}
        </style>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{
        padding: '20px',
        backgroundColor: darkMode ? 'rgba(248,81,73,0.1)' : '#f8d7da',
        color: '#f85149',
        borderRadius: '12px',
        border: `1px solid ${darkMode ? 'rgba(248,81,73,0.3)' : '#f5c6cb'}`,
        textAlign: 'center'
      }}>
        <strong>⚠️ {error}</strong>
        <button
          onClick={fetchSignals}
          style={{
            marginTop: '10px',
            padding: '8px 16px',
            backgroundColor: '#f85149',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            cursor: 'pointer',
            fontWeight: '600'
          }}
        >
          Retry
        </button>
      </div>
    );
  }

  const handleGenerate = async () => {
    setGenerating(true);
    setError('');
    try {
      const res = await axios.post(`${apiBaseUrl}/api/signals/generate/`);
      const newSignals = res.data?.signals || [];
      setSignals(newSignals);
      if (newSignals.length === 0) setError('No signals generated yet — market data may still be loading.');
    } catch (err) {
      setError('Could not generate signals. Make sure the server is running.');
    } finally {
      setGenerating(false);
    }
  };

  if (!signals || signals.length === 0) {
    return (
      <div style={{
        padding: '48px 32px',
        textAlign: 'center',
        backgroundColor: darkMode ? 'rgba(22,27,34,0.5)' : '#f8f9fa',
        borderRadius: '16px',
        border: `2px dashed ${darkMode ? '#30363d' : '#dee2e6'}`
      }}>
        <div style={{ fontSize: '64px', marginBottom: '20px' }}>📊</div>
        <h2 style={{ color: darkMode ? '#c9d1d9' : '#212529', marginBottom: '12px', fontSize: '24px' }}>
          No signals yet
        </h2>
        <p style={{ color: darkMode ? '#6e7681' : '#6c757d', marginBottom: '28px', fontSize: '16px' }}>
          Tap the button below to get your first trading signal.
        </p>
        {error && (
          <p style={{ color: '#f85149', marginBottom: '16px', fontSize: '14px' }}>{error}</p>
        )}
        <button
          onClick={handleGenerate}
          disabled={generating}
          style={{
            padding: '16px 40px',
            fontSize: '18px',
            fontWeight: '700',
            color: 'white',
            background: generating ? '#6c757d' : 'linear-gradient(135deg, #667eea, #764ba2)',
            border: 'none',
            borderRadius: '12px',
            cursor: generating ? 'not-allowed' : 'pointer',
            boxShadow: generating ? 'none' : '0 6px 20px rgba(102,126,234,0.5)',
            transition: 'all 0.3s ease'
          }}
        >
          {generating ? 'Getting signals...' : 'Get Trading Signals'}
        </button>
      </div>
    );
  }

  return (
    <div style={{
      padding: '20px',
      backgroundColor: darkMode ? 'rgba(13,17,23,0.8)' : '#ffffff',
      borderRadius: '16px',
      boxShadow: darkMode ? '0 8px 32px rgba(0,0,0,0.4)' : '0 4px 6px rgba(0,0,0,0.1)'
    }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '28px',
        paddingBottom: '20px',
        borderBottom: `2px solid ${darkMode ? '#21262d' : '#e9ecef'}`
      }}>
        <h2 style={{
          margin: 0,
          fontSize: '32px',
          fontWeight: '700',
          background: darkMode ? 'linear-gradient(135deg, #00ff87, #60efff)' : 'linear-gradient(135deg, #667eea, #764ba2)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          backgroundClip: 'text',
          display: 'flex',
          alignItems: 'center',
          gap: '12px'
        }}>
          🎯 Active Trading Signals
        </h2>
        <button
          onClick={fetchSignals}
          style={{
            padding: '10px 20px',
            background: darkMode ? 'rgba(48,54,61,0.8)' : '#6c757d',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            cursor: 'pointer',
            fontSize: '14px',
            fontWeight: '600',
            transition: 'all 0.3s ease',
            boxShadow: '0 2px 8px rgba(0,0,0,0.2)'
          }}
          onMouseOver={(e) => e.currentTarget.style.background = darkMode ? 'rgba(56,139,253,0.8)' : '#5a6268'}
          onMouseOut={(e) => e.currentTarget.style.background = darkMode ? 'rgba(48,54,61,0.8)' : '#6c757d'}
        >
          🔄 Refresh
        </button>
      </div>

      {/* Trade Message */}
      {tradeMessage && (
        <div style={{
          padding: '16px 20px',
          backgroundColor: tradeMessage.type === 'success' 
            ? darkMode ? 'rgba(63,185,80,0.1)' : '#d4edda'
            : darkMode ? 'rgba(248,81,73,0.1)' : '#f8d7da',
          color: tradeMessage.type === 'success' ? '#3fb950' : '#f85149',
          borderRadius: '8px',
          marginBottom: '20px',
          border: `1px solid ${tradeMessage.type === 'success' ? '#3fb950' : '#f85149'}`,
          animation: 'slideIn 0.3s ease',
          fontWeight: '600'
        }}>
          {tradeMessage.text}
        </div>
      )}

      {/* Signals Grid */}
      <div className="signals-grid" style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))',
        gap: '24px'
      }}>
        {signals.map((signal, index) => {
          const confidence = getConfidenceLevel(signal.probability);
          const signalColor = getSignalColor(signal.signal);
          const icon = getSignalIcon(signal.signal);
          const isExecuting = executingTrade === signal.id;

          return (
            <div
              key={signal.id || index}
              className="signal-card"
              style={{
                position: 'relative',
                background: darkMode 
                  ? 'linear-gradient(145deg, rgba(30,30,46,0.95) 0%, rgba(20,20,36,0.95) 100%)'
                  : 'linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%)',
                backdropFilter: 'blur(15px)',
                borderRadius: '20px',
                padding: '28px',
                border: `2px solid ${signalColor}40`,
                boxShadow: `0 10px 40px ${darkMode ? 'rgba(0,0,0,0.5)' : 'rgba(0,0,0,0.1)'}, 0 0 0 1px ${darkMode ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)'}`,
                transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
                overflow: 'hidden',
                cursor: 'pointer'
              }}
              onMouseOver={(e) => {
                e.currentTarget.style.transform = 'translateY(-8px) scale(1.02)';
                e.currentTarget.style.boxShadow = `0 20px 60px ${signalColor}40, 0 0 40px ${signalColor}30`;
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.transform = 'translateY(0) scale(1)';
                e.currentTarget.style.boxShadow = `0 10px 40px ${darkMode ? 'rgba(0,0,0,0.5)' : 'rgba(0,0,0,0.1)'}, 0 0 0 1px ${darkMode ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)'}`;
              }}
            >
              {/* Animated Gradient Top Bar */}
              <div style={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                height: '4px',
                background: `linear-gradient(90deg, ${signalColor}, ${signalColor}88, ${signalColor})`,
                animation: 'borderRotate 3s linear infinite'
              }} />

              {/* Pair Header with Icon */}
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '20px'
              }}>
                <h3 style={{
                  margin: 0,
                  fontSize: '28px',
                  fontWeight: '700',
                  color: darkMode ? '#c9d1d9' : '#212529',
                  letterSpacing: '1px'
                }}>
                  {signal.pair}
                </h3>
                <div style={{
                  fontSize: '40px',
                  animation: 'float 3s ease-in-out infinite'
                }}>
                  {icon}
                </div>
              </div>

              {/* Signal Direction Badge */}
              <div style={{
                background: `linear-gradient(135deg, ${signalColor}, ${signalColor}cc)`,
                color: 'white',
                padding: '14px 24px',
                borderRadius: '12px',
                marginBottom: '20px',
                textAlign: 'center',
                fontWeight: '700',
                fontSize: '28px',
                letterSpacing: '4px',
                boxShadow: `0 4px 12px ${signalColor}50, inset 0 1px 0 rgba(255,255,255,0.2)`
              }}>
                {SIGNAL_LABEL[signal.signal] || signal.signal.toUpperCase()}
              </div>

              {/* Confidence Badge with Glow */}
              <div style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '8px',
                padding: '10px 18px',
                borderRadius: '12px',
                background: darkMode ? `rgba(0,0,0,0.3)` : 'rgba(255,255,255,0.5)',
                border: `1px solid ${confidence.color}`,
                fontSize: '14px',
                fontWeight: '600',
                color: confidence.color,
                marginBottom: '24px',
                animation: 'pulse 2s ease-in-out infinite',
                boxShadow: `0 0 20px ${confidence.glow}`
              }}>
                <span>⭐</span>
                <span>Confidence: {confidence.text} ({(signal.probability * 100).toFixed(1)}%)</span>
              </div>

              {/* Signal Details Grid — Entry / SL / TP / R:R */}
              <div style={{
                background: darkMode ? 'rgba(0,0,0,0.3)' : 'rgba(0,0,0,0.03)',
                padding: '20px',
                borderRadius: '12px',
                marginTop: '20px',
                border: `1px solid ${darkMode ? '#30363d' : '#e9ecef'}`
              }}>
                {(() => {
                  const priceCell = (label, value, color) => (
                    <div>
                      <div style={{
                        color: darkMode ? '#8b949e' : '#6c757d',
                        fontWeight: '500', marginBottom: '6px',
                        fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.5px'
                      }}>{label}</div>
                      <div style={{
                        color: color || (darkMode ? '#c9d1d9' : '#212529'),
                        fontWeight: '700', fontSize: '16px', fontFamily: '"Roboto Mono", monospace'
                      }}>
                        {typeof value === 'number' ? value.toFixed(value > 100 ? 2 : 5) : (value || '—')}
                      </div>
                    </div>
                  );
                  const isXAU = signal.pair && signal.pair.includes('XAU');
                  const fmt = (v) => typeof v === 'number' ? v.toFixed(isXAU ? 2 : 5) : '—';
                  const slColor = signal.signal === 'bullish' ? '#f85149' : '#3fb950';
                  const tpColor = signal.signal === 'bullish' ? '#3fb950' : '#f85149';
                  return (
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', fontSize: '14px' }}>
                      <div>
                        <div style={{ color: darkMode ? '#8b949e' : '#6c757d', fontWeight: '500', marginBottom: '6px', fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Entry</div>
                        <div style={{ color: signalColor, fontWeight: '700', fontSize: '16px', fontFamily: '"Roboto Mono", monospace' }}>
                          {signal.entry_price ? fmt(signal.entry_price) : '—'}
                        </div>
                      </div>
                      <div>
                        <div style={{ color: darkMode ? '#8b949e' : '#6c757d', fontWeight: '500', marginBottom: '6px', fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Stop Loss</div>
                        <div style={{ color: slColor, fontWeight: '700', fontSize: '16px', fontFamily: '"Roboto Mono", monospace' }}>
                          {signal.stop_loss ? fmt(signal.stop_loss) : '—'}
                        </div>
                      </div>
                      <div>
                        <div style={{ color: darkMode ? '#8b949e' : '#6c757d', fontWeight: '500', marginBottom: '6px', fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Take Profit</div>
                        <div style={{ color: tpColor, fontWeight: '700', fontSize: '16px', fontFamily: '"Roboto Mono", monospace' }}>
                          {signal.take_profit ? fmt(signal.take_profit) : '—'}
                        </div>
                      </div>
                      <div>
                        <div style={{ color: darkMode ? '#8b949e' : '#6c757d', fontWeight: '500', marginBottom: '6px', fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>R:R Ratio</div>
                        <div style={{ color: signal.risk_reward >= 2 ? '#00ff87' : signal.risk_reward >= 1.5 ? '#ffd700' : (darkMode ? '#c9d1d9' : '#212529'), fontWeight: '700', fontSize: '16px', fontFamily: '"Roboto Mono", monospace' }}>
                          {signal.risk_reward ? `1 : ${signal.risk_reward.toFixed(2)}` : '—'}
                        </div>
                      </div>
                    </div>
                  );
                })()}
              </div>

              {/* Date and source */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '12px', fontSize: '11px', color: darkMode ? '#6e7681' : '#adb5bd' }}>
                <span>{new Date(signal.date).toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' })}</span>
                {signal.source && <span style={{ textTransform: 'uppercase', letterSpacing: '0.5px' }}>{signal.source}</span>}
              </div>

              {/* Animated Probability Bar */}
              <div style={{ marginTop: '20px' }}>
                <div style={{
                  height: '10px',
                  backgroundColor: darkMode ? 'rgba(255,255,255,0.1)' : '#e9ecef',
                  borderRadius: '10px',
                  overflow: 'hidden',
                  position: 'relative'
                }}>
                  <div style={{
                    height: '100%',
                    width: `${signal.probability * 100}%`,
                    background: `linear-gradient(90deg, ${signalColor}, ${confidence.color})`,
                    borderRadius: '10px',
                    transition: 'width 1s ease',
                    position: 'relative',
                    boxShadow: `0 0 10px ${signalColor}80`
                  }}>
                    <div style={{
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      right: 0,
                      bottom: 0,
                      background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent)',
                      animation: 'shimmer 2s infinite'
                    }} />
                  </div>
                </div>
              </div>

              {/* Execute Trade Button */}
              <button
                onClick={() => executeTrade(signal)}
                disabled={isExecuting}
                style={{
                  marginTop: '24px',
                  width: '100%',
                  padding: '14px 24px',
                  background: isExecuting 
                    ? darkMode ? 'rgba(108,117,125,0.5)' : '#6c757d'
                    : signal.signal === 'bullish'
                      ? 'linear-gradient(135deg, #3fb950 0%, #2ea043 100%)'
                      : 'linear-gradient(135deg, #f85149 0%, #da3633 100%)',
                  color: 'white',
                  border: 'none',
                  borderRadius: '10px',
                  cursor: isExecuting ? 'not-allowed' : 'pointer',
                  fontSize: '16px',
                  fontWeight: '700',
                  textTransform: 'uppercase',
                  letterSpacing: '1px',
                  transition: 'all 0.3s ease',
                  boxShadow: isExecuting ? 'none' : `0 4px 12px ${signalColor}50`,
                  opacity: isExecuting ? 0.6 : 1
                }}
                onMouseOver={(e) => {
                  if (!isExecuting) {
                    e.currentTarget.style.transform = 'translateY(-2px)';
                    e.currentTarget.style.boxShadow = `0 6px 20px ${signalColor}70`;
                  }
                }}
                onMouseOut={(e) => {
                  if (!isExecuting) {
                    e.currentTarget.style.transform = 'translateY(0)';
                    e.currentTarget.style.boxShadow = `0 4px 12px ${signalColor}50`;
                  }
                }}
              >
                {isExecuting ? '⏳ Executing...' : '📊 Execute Paper Trade'}
              </button>
            </div>
          );
        })}
      </div>

      {/* Animations */}
      <style>
        {`
          @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
          }

          @keyframes borderRotate {
            0% { filter: hue-rotate(0deg); }
            100% { filter: hue-rotate(360deg); }
          }

          @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.05); opacity: 0.9; }
          }

          @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
          }

          @keyframes slideIn {
            from {
              opacity: 0;
              transform: translateY(-20px);
            }
            to {
              opacity: 1;
              transform: translateY(0);
            }
          }

          /* Responsive Design */
          @media (max-width: 768px) {
            .signals-grid {
              grid-template-columns: 1fr !important;
              gap: 16px !important;
            }
            
            .signal-card {
              padding: 20px !important;
              border-radius: 16px !important;
            }
          }

          @media (min-width: 769px) and (max-width: 1024px) {
            .signals-grid {
              grid-template-columns: repeat(2, 1fr) !important;
              gap: 20px !important;
            }
          }

          @media (min-width: 1440px) {
            .signals-grid {
              grid-template-columns: repeat(4, 1fr) !important;
              gap: 28px !important;
            }
          }

          /* Touch optimizations */
          @media (hover: none) and (pointer: coarse) {
            button {
              min-height: 48px !important;
            }
          }

          /* Reduced motion */
          @media (prefers-reduced-motion: reduce) {
            * {
              animation-duration: 0.01ms !important;
              transition-duration: 0.01ms !important;
            }
          }
        `}
      </style>
    </div>
  );
};

export default SignalsDashboard;
