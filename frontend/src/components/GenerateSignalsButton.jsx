import React, { useState } from 'react';
import axios from 'axios';

const GenerateSignalsButton = ({ apiBaseUrl, onSignalsGenerated }) => {
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  const handleGenerateSignals = async () => {
    setLoading(true);
    setMessage('');
    setError('');

    try {
      const response = await axios.post(`${apiBaseUrl}/api/signals/generate/`, {}, {
        headers: { 'Content-Type': 'application/json' }
      });

      if (response.data.status === 'success') {
        setMessage(`✅ ${response.data.message}`);
        
        if (onSignalsGenerated && response.data.signals) {
          onSignalsGenerated(response.data.signals);
        }

        // Auto-clear success message after 5 seconds
        setTimeout(() => setMessage(''), 5000);
      } else {
        setError('❌ Signal generation completed with warnings');
      }
    } catch (err) {
      console.error('Signal generation error:', err);
      setError(`❌ Failed to generate signals: ${err.response?.data?.error || err.message}`);
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
        onClick={handleGenerateSignals}
        disabled={loading}
        style={{
          padding: '14px 28px',
          fontSize: '18px',
          fontWeight: '700',
          color: 'white',
          backgroundColor: loading ? '#6c757d' : '#007bff',
          border: 'none',
          borderRadius: '8px',
          cursor: loading ? 'not-allowed' : 'pointer',
          transition: 'all 0.3s ease',
          boxShadow: '0 4px 6px rgba(0,0,0,0.2)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '10px',
          textTransform: 'uppercase',
          letterSpacing: '0.5px'
        }}
        onMouseOver={(e) => {
          if (!loading) {
            e.currentTarget.style.backgroundColor = '#0056b3';
            e.currentTarget.style.transform = 'translateY(-2px)';
            e.currentTarget.style.boxShadow = '0 6px 8px rgba(0,0,0,0.3)';
          }
        }}
        onMouseOut={(e) => {
          if (!loading) {
            e.currentTarget.style.backgroundColor = '#007bff';
            e.currentTarget.style.transform = 'translateY(0)';
            e.currentTarget.style.boxShadow = '0 4px 6px rgba(0,0,0,0.2)';
          }
        }}
      >
        {loading ? (
          <>
            <div style={{
              width: '18px',
              height: '18px',
              border: '3px solid #ffffff',
              borderTop: '3px solid transparent',
              borderRadius: '50%',
              animation: 'spin 0.8s linear infinite'
            }} />
            Generating Signals...
          </>
        ) : (
          <>
            ⚡ Generate Trading Signals
          </>
        )}
      </button>

      {message && (
        <div style={{
          padding: '12px',
          backgroundColor: '#d4edda',
          color: '#155724',
          borderRadius: '6px',
          fontSize: '14px',
          fontWeight: '500',
          border: '1px solid #c3e6cb',
          animation: 'slideIn 0.3s ease-out'
        }}>
          {message}
        </div>
      )}

      {error && (
        <div style={{
          padding: '12px',
          backgroundColor: '#f8d7da',
          color: '#721c24',
          borderRadius: '6px',
          fontSize: '14px',
          fontWeight: '500',
          border: '1px solid #f5c6cb',
          animation: 'slideIn 0.3s ease-out'
        }}>
          {error}
        </div>
      )}

      <div style={{
        fontSize: '12px',
        color: '#6c757d',
        textAlign: 'center',
        fontStyle: 'italic'
      }}>
        {loading ? 'Fetching latest data and generating predictions...' : 'Click to generate fresh trading signals'}
      </div>

      <style>
        {`
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
          @keyframes slideIn {
            from {
              opacity: 0;
              transform: translateY(-10px);
            }
            to {
              opacity: 1;
              transform: translateY(0);
            }
          }
        `}
      </style>
    </div>
  );
};

export default GenerateSignalsButton;
