/**
 * Order Manager Component
 * Displays and manages open positions and trade history
 */
import React, { useState, useEffect } from 'react';

const OrderManager = () => {
  const [openPositions, setOpenPositions] = useState([]);
  const [tradeHistory, setTradeHistory] = useState([]);
  const [activeTab, setActiveTab] = useState('open'); // 'open' or 'history'
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadData();

    // Refresh every 5 seconds
    const interval = setInterval(loadData, 5000);
    return () => clearInterval(interval);
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);

      // Load open positions
      const openRes = await fetch('/api/paper-trading/trades/open_positions/');
      const openData = await openRes.json();
      setOpenPositions(openData || []);

      // Load trade history
      const historyRes = await fetch('/api/paper-trading/trades/?status=closed&limit=20');
      const historyData = await historyRes.json();
      setTradeHistory(historyData.results || historyData || []);
    } catch (error) {
      console.error('Error loading data:', error);
    } finally {
      setLoading(false);
    }
  };

  const closePosition = async (tradeId) => {
    if (!window.confirm('Are you sure you want to close this position?')) {
      return;
    }

    try {
      // Get current price
      const position = openPositions.find(p => p.id === tradeId);
      if (!position) return;

      const priceRes = await fetch(`/api/paper-trading/price/realtime/?symbol=${position.pair}`);
      const priceData = await priceRes.json();
      
      const exitPrice = position.order_type === 'buy' ? priceData.bid : priceData.ask;

      // Close position
      const response = await fetch(`/api/paper-trading/trades/${tradeId}/close/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ exit_price: exitPrice })
      });

      if (response.ok) {
        alert('✅ Position closed successfully!');
        loadData();
      } else {
        const error = await response.json();
        alert(`❌ Error: ${error.error}`);
      }
    } catch (error) {
      console.error('Error closing position:', error);
      alert(`❌ Error: ${error.message}`);
    }
  };

  return (
    <div style={{ padding: '20px', background: '#2B2B2B', borderRadius: '8px' }}>
      {/* Header */}
      <h2 style={{ margin: '0 0 20px 0', color: '#D9D9D9' }}>
        📊 Order Manager
      </h2>

      {/* Tabs */}
      <div style={{ display: 'flex', marginBottom: '20px', borderBottom: '1px solid #444' }}>
        <button
          onClick={() => setActiveTab('open')}
          style={{
            padding: '10px 20px',
            background: activeTab === 'open' ? '#1976d2' : 'transparent',
            color: activeTab === 'open' ? 'white' : '#999',
            border: 'none',
            borderBottom: activeTab === 'open' ? '3px solid #1976d2' : 'none',
            cursor: 'pointer',
            fontSize: '16px',
            fontWeight: 'bold',
          }}
        >
          Open Positions ({openPositions.length})
        </button>
        <button
          onClick={() => setActiveTab('history')}
          style={{
            padding: '10px 20px',
            background: activeTab === 'history' ? '#1976d2' : 'transparent',
            color: activeTab === 'history' ? 'white' : '#999',
            border: 'none',
            borderBottom: activeTab === 'history' ? '3px solid #1976d2' : 'none',
            cursor: 'pointer',
            fontSize: '16px',
            fontWeight: 'bold',
          }}
        >
          History ({tradeHistory.length})
        </button>
      </div>

      {/* Content */}
      {loading ? (
        <div style={{ textAlign: 'center', padding: '40px', color: '#999' }}>
          Loading...
        </div>
      ) : activeTab === 'open' ? (
        <OpenPositions positions={openPositions} onClose={closePosition} />
      ) : (
        <TradeHistory history={tradeHistory} />
      )}
    </div>
  );
};

const OpenPositions = ({ positions, onClose }) => {
  if (positions.length === 0) {
    return (
      <div style={{ textAlign: 'center', padding: '40px', color: '#999' }}>
        No open positions
      </div>
    );
  }

  return (
    <div>
      {positions.map((position) => (
        <div
          key={position.id}
          style={{
            padding: '15px',
            margin: '10px 0',
            background: '#1E1E1E',
            borderRadius: '8px',
            borderLeft: `5px solid ${position.order_type === 'buy' ? '#26a69a' : '#ef5350'}`,
          }}
        >
          {/* Header */}
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
            <div>
              <span style={{ fontSize: '18px', fontWeight: 'bold', color: '#D9D9D9' }}>
                {position.pair}
              </span>
              <span
                style={{
                  marginLeft: '10px',
                  padding: '4px 8px',
                  background: position.order_type === 'buy' ? '#26a69a' : '#ef5350',
                  color: 'white',
                  borderRadius: '4px',
                  fontSize: '14px',
                }}
              >
                {position.order_type.toUpperCase()} {position.lot_size} lots
              </span>
            </div>
            <div
              style={{
                fontSize: '18px',
                fontWeight: 'bold',
                color: (position.pips_gained || 0) >= 0 ? '#26a69a' : '#ef5350',
              }}
            >
              {(position.pips_gained || 0) >= 0 ? '+' : ''}
              {(position.pips_gained || 0).toFixed(1)} pips
            </div>
          </div>

          {/* Details Grid */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
              gap: '10px',
              marginBottom: '10px',
            }}
          >
            <div>
              <div style={{ color: '#999', fontSize: '12px' }}>Entry</div>
              <div style={{ color: '#fff', fontWeight: 'bold' }}>{position.entry_price}</div>
            </div>
            <div>
              <div style={{ color: '#999', fontSize: '12px' }}>Stop Loss</div>
              <div style={{ color: '#ef5350', fontWeight: 'bold' }}>{position.stop_loss}</div>
            </div>
            <div>
              <div style={{ color: '#999', fontSize: '12px' }}>Take Profit</div>
              <div style={{ color: '#4caf50', fontWeight: 'bold' }}>{position.take_profit_1}</div>
            </div>
            <div>
              <div style={{ color: '#999', fontSize: '12px' }}>P&L</div>
              <div
                style={{
                  color: (position.profit_loss || 0) >= 0 ? '#26a69a' : '#ef5350',
                  fontWeight: 'bold',
                }}
              >
                ${(position.profit_loss || 0).toFixed(2)}
              </div>
            </div>
          </div>

          {/* Signal Info */}
          {position.signal_type && (
            <div style={{ marginBottom: '10px', color: '#999', fontSize: '12px' }}>
              Signal: {position.signal_type} • {position.signal_source || 'Manual'}
            </div>
          )}

          {/* Close Button */}
          <button
            onClick={() => onClose(position.id)}
            style={{
              width: '100%',
              padding: '10px',
              background: '#d32f2f',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontWeight: 'bold',
            }}
          >
            🔴 Close Position
          </button>
        </div>
      ))}
    </div>
  );
};

const TradeHistory = ({ history }) => {
  if (history.length === 0) {
    return (
      <div style={{ textAlign: 'center', padding: '40px', color: '#999' }}>
        No trade history
      </div>
    );
  }

  return (
    <div>
      {history.map((trade) => (
        <div
          key={trade.id}
          style={{
            padding: '15px',
            margin: '10px 0',
            background: '#1E1E1E',
            borderRadius: '8px',
            borderLeft: `5px solid ${(trade.pips_gained || 0) >= 0 ? '#26a69a' : '#ef5350'}`,
          }}
        >
          {/* Header */}
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
            <div>
              <span style={{ fontSize: '16px', fontWeight: 'bold', color: '#D9D9D9' }}>
                {trade.pair}
              </span>
              <span style={{ marginLeft: '10px', color: '#999', fontSize: '14px' }}>
                {trade.order_type.toUpperCase()} {trade.lot_size} lots
              </span>
            </div>
            <div style={{ color: '#999', fontSize: '14px' }}>
              {new Date(trade.exit_time).toLocaleString()}
            </div>
          </div>

          {/* P&L Summary */}
          <div
            style={{
              padding: '10px',
              background: (trade.pips_gained || 0) >= 0 ? 'rgba(38, 166, 154, 0.1)' : 'rgba(239, 83, 80, 0.1)',
              borderRadius: '4px',
              marginBottom: '10px',
            }}
          >
            <div
              style={{
                fontSize: '20px',
                fontWeight: 'bold',
                color: (trade.pips_gained || 0) >= 0 ? '#26a69a' : '#ef5350',
              }}
            >
              {(trade.pips_gained || 0) >= 0 ? '+' : ''}
              {(trade.pips_gained || 0).toFixed(1)} pips • ${(trade.profit_loss || 0).toFixed(2)}
            </div>
          </div>

          {/* Details */}
          <div style={{ color: '#999', fontSize: '14px' }}>
            Entry: {trade.entry_price} → Exit: {trade.exit_price} • R:R: {trade.risk_reward_ratio}:1
          </div>
        </div>
      ))}
    </div>
  );
};

export default OrderManager;
