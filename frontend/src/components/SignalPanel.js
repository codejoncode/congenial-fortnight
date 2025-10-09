/**
 * Signal Panel Component
 * Displays live signal feed with execution buttons
 */
import React, { useState, useEffect } from 'react';

const SignalPanel = ({ pair }) => {
  const [signals, setSignals] = useState([]);
  const [loading, setLoading] = useState(false);
  const [executing, setExecuting] = useState(null);

  useEffect(() => {
    loadSignals();

    // Poll for new signals every 10 seconds
    const interval = setInterval(loadSignals, 10000);
    return () => clearInterval(interval);
  }, [pair]);

  const loadSignals = async () => {
    try {
      setLoading(true);
      const queryParam = pair ? `?pair=${pair}` : '';
      const response = await fetch(`/api/signals/latest/${queryParam}`);
      const data = await response.json();
      
      if (data.signals) {
        setSignals(data.signals);
      }
    } catch (error) {
      console.error('Error loading signals:', error);
    } finally {
      setLoading(false);
    }
  };

  const executeSignal = async (signal) => {
    try {
      setExecuting(signal.id);

      // Prepare trade execution payload
      const tradeData = {
        pair: signal.pair,
        order_type: signal.direction.toLowerCase() === 'long' || signal.direction === 'buy' ? 'buy' : 'sell',
        entry_price: signal.entry_price,
        stop_loss: signal.stop_loss,
        take_profit_1: signal.take_profit_1,
        take_profit_2: signal.take_profit_2 || null,
        take_profit_3: signal.take_profit_3 || null,
        lot_size: calculateLotSize(signal),
        signal_id: signal.id,
        signal_type: signal.type,
        signal_source: 'multi_model_aggregator',
        notes: `Confidence: ${signal.confidence}%, R:R: ${signal.risk_reward_ratio}`,
      };

      const response = await fetch('/api/paper-trading/trades/execute/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(tradeData),
      });

      if (response.ok) {
        const result = await response.json();
        alert(`✅ Trade executed successfully! Trade ID: ${result.id}`);
        
        // Remove executed signal from list
        setSignals(signals.filter(s => s.id !== signal.id));
      } else {
        const error = await response.json();
        alert(`❌ Error executing trade: ${error.error || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Error executing signal:', error);
      alert(`❌ Error executing trade: ${error.message}`);
    } finally {
      setExecuting(null);
    }
  };

  const calculateLotSize = (signal) => {
    // Base lot size
    let lotSize = 0.01;

    // Increase based on confidence
    if (signal.confidence >= 95) {
      lotSize = 0.03;
    } else if (signal.confidence >= 85) {
      lotSize = 0.02;
    } else if (signal.confidence >= 75) {
      lotSize = 0.015;
    }

    // Boost for confluence signals
    if (signal.type && (signal.type.includes('CONFLUENCE') || signal.type.includes('ULTRA'))) {
      lotSize *= 1.5;
    }

    return Math.min(lotSize, 0.10); // Cap at 0.1 lots
  };

  const getSignalColor = (signal) => {
    const direction = signal.direction.toLowerCase();
    return direction === 'buy' || direction === 'long' ? '#2196F3' : '#e91e63';
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 90) return '#4caf50';
    if (confidence >= 75) return '#ff9800';
    return '#f44336';
  };

  return (
    <div
      style={{
        padding: '20px',
        background: '#2B2B2B',
        borderRadius: '8px',
        maxHeight: '600px',
        overflowY: 'auto',
      }}
    >
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '15px',
        }}
      >
        <h2 style={{ margin: 0, color: '#D9D9D9' }}>
          🎯 Signal Feed
          {pair && ` - ${pair}`}
        </h2>
        <button
          onClick={loadSignals}
          disabled={loading}
          style={{
            padding: '8px 16px',
            background: '#1976d2',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: loading ? 'not-allowed' : 'pointer',
            opacity: loading ? 0.6 : 1,
          }}
        >
          {loading ? '🔄 Loading...' : '🔄 Refresh'}
        </button>
      </div>

      {signals.length === 0 ? (
        <div style={{ textAlign: 'center', padding: '40px', color: '#999' }}>
          {loading ? 'Loading signals...' : 'No active signals'}
        </div>
      ) : (
        signals.map((signal) => (
          <div
            key={signal.id}
            style={{
              padding: '15px',
              margin: '10px 0',
              background: '#1E1E1E',
              borderRadius: '8px',
              borderLeft: `5px solid ${getSignalColor(signal)}`,
              boxShadow: '0 2px 4px rgba(0, 0, 0, 0.3)',
            }}
          >
            {/* Signal Header */}
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '10px',
              }}
            >
              <div>
                <span
                  style={{
                    fontSize: '18px',
                    fontWeight: 'bold',
                    color: '#D9D9D9',
                  }}
                >
                  {signal.pair}
                </span>
                <span
                  style={{
                    marginLeft: '10px',
                    padding: '4px 8px',
                    background: getSignalColor(signal),
                    color: 'white',
                    borderRadius: '4px',
                    fontSize: '14px',
                    fontWeight: 'bold',
                  }}
                >
                  {signal.direction.toUpperCase()}
                </span>
              </div>
              <div
                style={{
                  padding: '4px 12px',
                  background: getConfidenceColor(signal.confidence),
                  color: 'white',
                  borderRadius: '4px',
                  fontSize: '14px',
                  fontWeight: 'bold',
                }}
              >
                {signal.confidence}%
              </div>
            </div>

            {/* Signal Type */}
            <div
              style={{
                marginBottom: '10px',
                color: '#999',
                fontSize: '14px',
              }}
            >
              {signal.type}
              {signal.source && ` • ${signal.source}`}
            </div>

            {/* Price Levels */}
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(100px, 1fr))',
                gap: '10px',
                marginBottom: '10px',
              }}
            >
              <div>
                <div style={{ color: '#999', fontSize: '12px' }}>Entry</div>
                <div style={{ color: '#fff', fontWeight: 'bold' }}>
                  {signal.entry_price}
                </div>
              </div>
              <div>
                <div style={{ color: '#999', fontSize: '12px' }}>Stop Loss</div>
                <div style={{ color: '#ef5350', fontWeight: 'bold' }}>
                  {signal.stop_loss}
                </div>
              </div>
              <div>
                <div style={{ color: '#999', fontSize: '12px' }}>Take Profit</div>
                <div style={{ color: '#4caf50', fontWeight: 'bold' }}>
                  {signal.take_profit_1}
                </div>
              </div>
              {signal.risk_reward_ratio && (
                <div>
                  <div style={{ color: '#999', fontSize: '12px' }}>R:R</div>
                  <div style={{ color: '#26a69a', fontWeight: 'bold' }}>
                    {signal.risk_reward_ratio}:1
                  </div>
                </div>
              )}
            </div>

            {/* Multiple Take Profits */}
            {(signal.take_profit_2 || signal.take_profit_3) && (
              <div style={{ marginBottom: '10px', color: '#999', fontSize: '12px' }}>
                Additional TPs:{' '}
                {signal.take_profit_2 && `TP2: ${signal.take_profit_2} `}
                {signal.take_profit_3 && `TP3: ${signal.take_profit_3}`}
              </div>
            )}

            {/* Action Button */}
            <button
              onClick={() => executeSignal(signal)}
              disabled={executing === signal.id}
              style={{
                width: '100%',
                padding: '12px',
                background: executing === signal.id ? '#666' : '#1976d2',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                fontSize: '16px',
                fontWeight: 'bold',
                cursor: executing === signal.id ? 'not-allowed' : 'pointer',
                transition: 'background 0.3s',
              }}
              onMouseOver={(e) => {
                if (executing !== signal.id) {
                  e.target.style.background = '#1565c0';
                }
              }}
              onMouseOut={(e) => {
                if (executing !== signal.id) {
                  e.target.style.background = '#1976d2';
                }
              }}
            >
              {executing === signal.id ? '⏳ Executing...' : '🚀 Execute Trade'}
            </button>
          </div>
        ))
      )}
    </div>
  );
};

export default SignalPanel;
