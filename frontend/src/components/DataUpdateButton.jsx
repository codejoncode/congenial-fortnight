import React, { useState } from 'react';
import axios from 'axios';

const DataUpdateButton = ({ apiBaseUrl, onUpdateComplete }) => {
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [lastUpdate, setLastUpdate] = useState(null);

  const handleUpdateData = async () => {
    setLoading(true);
    setMessage('');
    setError('');

    try {
      const response = await axios.post(`${apiBaseUrl}/api/data/update/`, {}, {
        headers: { 'Content-Type': 'application/json' }
      });

      if (response.data.status === 'success') {
        const updateTime = new Date().toLocaleString();
        setLastUpdate(updateTime);
        setMessage(`✅ ${response.data.message} - Updated: ${response.data.pairs.join(', ')}`);
        
        if (onUpdateComplete) {
          onUpdateComplete(response.data);
        }
      } else {
        setError('❌ Data update completed with warnings');
      }
    } catch (err) {
      console.error('Data update error:', err);
      setError(`❌ Failed to update data: ${err.response?.data?.error || err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      gap: '10px',
      padding: '15px',
      backgroundColor: '#f8f9fa',
      borderRadius: '8px',
      border: '1px solid #dee2e6'
    }}>
      <button
        onClick={handleUpdateData}
        disabled={loading}
        style={{
          padding: '12px 24px',
          fontSize: '16px',
          fontWeight: '600',
          color: 'white',
          backgroundColor: loading ? '#6c757d' : '#28a745',
          border: 'none',
          borderRadius: '6px',
          cursor: loading ? 'not-allowed' : 'pointer',
          transition: 'all 0.3s ease',
          boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '8px'
        }}
        onMouseOver={(e) => {
          if (!loading) e.currentTarget.style.backgroundColor = '#218838';
        }}
        onMouseOut={(e) => {
          if (!loading) e.currentTarget.style.backgroundColor = '#28a745';
        }}
      >
        {loading ? (
          <>
            <div style={{
              width: '16px',
              height: '16px',
              border: '2px solid #ffffff',
              borderTop: '2px solid transparent',
              borderRadius: '50%',
              animation: 'spin 1s linear infinite'
            }} />
            Updating Data...
          </>
        ) : (
          <>
            🔄 Update Market Data
          </>
        )}
      </button>

      {message && (
        <div style={{
          padding: '10px',
          backgroundColor: '#d4edda',
          color: '#155724',
          borderRadius: '4px',
          fontSize: '14px',
          border: '1px solid #c3e6cb'
        }}>
          {message}
        </div>
      )}

      {error && (
        <div style={{
          padding: '10px',
          backgroundColor: '#f8d7da',
          color: '#721c24',
          borderRadius: '4px',
          fontSize: '14px',
          border: '1px solid #f5c6cb'
        }}>
          {error}
        </div>
      )}

      {lastUpdate && (
        <div style={{
          fontSize: '12px',
          color: '#6c757d',
          textAlign: 'center'
        }}>
          Last updated: {lastUpdate}
        </div>
      )}

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
};

export default DataUpdateButton;
