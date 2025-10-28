import React, { useState, useEffect } from 'react';
import axios from 'axios';

const SignalsDashboard = ({ apiBaseUrl, signals: propSignals }) => {
  const [signals, setSignals] = useState(propSignals || []);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Fetch signals on mount if not provided as props
  useEffect(() => {
    if (!propSignals || propSignals.length === 0) {
      fetchSignals();
    } else {
      setSignals(propSignals);
    }
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

  const getSignalColor = (signal) => {
    return signal === 'bullish' ? '#28a745' : '#dc3545';
  };

  const getSignalIcon = (signal) => {
    return signal === 'bullish' ? '📈' : '📉';
  };

  const getConfidenceLevel = (probability) => {
    if (probability >= 0.8) return { text: 'Very High', color: '#155724', bg: '#d4edda' };
    if (probability >= 0.7) return { text: 'High', color: '#0c5460', bg: '#d1ecf1' };
    if (probability >= 0.6) return { text: 'Medium', color: '#856404', bg: '#fff3cd' };
    return { text: 'Low', color: '#721c24', bg: '#f8d7da' };
  };

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '40px' }}>
        <div style={{
          width: '50px',
          height: '50px',
          border: '5px solid #f3f3f3',
          borderTop: '5px solid #007bff',
          borderRadius: '50%',
          animation: 'spin 1s linear infinite',
          margin: '0 auto'
        }} />
        <p style={{ marginTop: '20px', color: '#6c757d' }}>Loading signals...</p>
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
        backgroundColor: '#f8d7da',
        color: '#721c24',
        borderRadius: '8px',
        border: '1px solid #f5c6cb',
        textAlign: 'center'
      }}>
        <strong>⚠️ {error}</strong>
        <button
          onClick={fetchSignals}
          style={{
            marginTop: '10px',
            padding: '8px 16px',
            backgroundColor: '#dc3545',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer'
          }}
        >
          Retry
        </button>
      </div>
    );
  }

  if (!signals || signals.length === 0) {
    return (
      <div style={{
        padding: '40px',
        textAlign: 'center',
        backgroundColor: '#f8f9fa',
        borderRadius: '8px',
        border: '2px dashed #dee2e6'
      }}>
        <div style={{ fontSize: '48px', marginBottom: '20px' }}>📊</div>
        <h3 style={{ color: '#6c757d', marginBottom: '10px' }}>No Signals Available</h3>
        <p style={{ color: '#adb5bd' }}>Generate signals to see trading recommendations</p>
      </div>
    );
  }

  return (
    <div style={{
      padding: '20px',
      backgroundColor: '#ffffff',
      borderRadius: '12px',
      boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
    }}>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '24px',
        paddingBottom: '16px',
        borderBottom: '2px solid #e9ecef'
      }}>
        <h2 style={{
          margin: 0,
          fontSize: '28px',
          fontWeight: '700',
          color: '#212529',
          display: 'flex',
          alignItems: 'center',
          gap: '10px'
        }}>
          🎯 Active Trading Signals
        </h2>
        <button
          onClick={fetchSignals}
          style={{
            padding: '8px 16px',
            backgroundColor: '#6c757d',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            cursor: 'pointer',
            fontSize: '14px',
            fontWeight: '500',
            transition: 'all 0.3s ease'
          }}
          onMouseOver={(e) => e.currentTarget.style.backgroundColor = '#5a6268'}
          onMouseOut={(e) => e.currentTarget.style.backgroundColor = '#6c757d'}
        >
          🔄 Refresh
        </button>
      </div>

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))',
        gap: '20px'
      }}>
        {signals.map((signal, index) => {
          const confidence = getConfidenceLevel(signal.probability);
          const signalColor = getSignalColor(signal.signal);
          const icon = getSignalIcon(signal.signal);

          return (
            <div
              key={signal.id || index}
              style={{
                backgroundColor: '#ffffff',
                borderRadius: '12px',
                padding: '24px',
                border: `3px solid ${signalColor}`,
                boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                transition: 'all 0.3s ease',
                position: 'relative',
                overflow: 'hidden'
              }}
              onMouseOver={(e) => {
                e.currentTarget.style.transform = 'translateY(-4px)';
                e.currentTarget.style.boxShadow = '0 8px 20px rgba(0,0,0,0.15)';
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = '0 4px 12px rgba(0,0,0,0.1)';
              }}
            >
              {/* Decorative gradient bar */}
              <div style={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                height: '6px',
                background: `linear-gradient(90deg, ${signalColor}, ${signalColor}88)`
              }} />

              {/* Pair Header */}
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '16px'
              }}>
                <h3 style={{
                  margin: 0,
                  fontSize: '24px',
                  fontWeight: '700',
                  color: '#212529',
                  letterSpacing: '1px'
                }}>
                  {signal.pair}
                </h3>
                <div style={{
                  fontSize: '32px'
                }}>
                  {icon}
                </div>
              </div>

              {/* Signal Direction */}
              <div style={{
                backgroundColor: signalColor,
                color: 'white',
                padding: '12px 20px',
                borderRadius: '8px',
                marginBottom: '16px',
                textAlign: 'center',
                fontWeight: '700',
                fontSize: '20px',
                textTransform: 'uppercase',
                letterSpacing: '1.5px',
                boxShadow: '0 2px 4px rgba(0,0,0,0.2)'
              }}>
                {signal.signal}
              </div>

              {/* Confidence Badge */}
              <div style={{
                display: 'inline-block',
                backgroundColor: confidence.bg,
                color: confidence.color,
                padding: '8px 16px',
                borderRadius: '20px',
                fontSize: '14px',
                fontWeight: '600',
                marginBottom: '20px',
                border: `1px solid ${confidence.color}`
              }}>
                Confidence: {confidence.text} ({(signal.probability * 100).toFixed(1)}%)
              </div>

              {/* Signal Details */}
              <div style={{
                backgroundColor: '#f8f9fa',
                padding: '16px',
                borderRadius: '8px',
                marginTop: '16px'
              }}>
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: '1fr 1fr',
                  gap: '12px',
                  fontSize: '14px'
                }}>
                  <div>
                    <div style={{ color: '#6c757d', fontWeight: '500', marginBottom: '4px' }}>
                      Stop Loss
                    </div>
                    <div style={{ color: '#212529', fontWeight: '700', fontSize: '16px' }}>
                      {signal.stop_loss ? signal.stop_loss.toFixed(4) : 'N/A'}
                    </div>
                  </div>
                  <div>
                    <div style={{ color: '#6c757d', fontWeight: '500', marginBottom: '4px' }}>
                      Date
                    </div>
                    <div style={{ color: '#212529', fontWeight: '700', fontSize: '16px' }}>
                      {new Date(signal.date).toLocaleDateString()}
                    </div>
                  </div>
                </div>
              </div>

              {/* Probability Bar */}
              <div style={{ marginTop: '16px' }}>
                <div style={{
                  height: '8px',
                  backgroundColor: '#e9ecef',
                  borderRadius: '10px',
                  overflow: 'hidden',
                  position: 'relative'
                }}>
                  <div style={{
                    height: '100%',
                    width: `${signal.probability * 100}%`,
                    backgroundColor: signalColor,
                    borderRadius: '10px',
                    transition: 'width 0.5s ease',
                    position: 'relative'
                  }}>
                    <div style={{
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      right: 0,
                      bottom: 0,
                      background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent)',
                      animation: 'shimmer 2s infinite'
                    }} />
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <style>
        {`
          @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
          }
        `}
      </style>
    </div>
  );
};

export default SignalsDashboard;
