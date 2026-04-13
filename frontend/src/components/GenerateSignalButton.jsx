import React, { useState } from 'react';
import axios from 'axios';

export default function GenerateSignalButton({ apiBaseUrl, onSignalGenerated }) {
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState(null); // { type: 'success'|'error', text: '' }

  const handleClick = async () => {
    setLoading(true);
    setStatus(null);
    try {
      const res = await axios.post(`${apiBaseUrl}/api/signals/generate/`, {}, {
        headers: { 'Content-Type': 'application/json' },
      });
      const signals = res.data.signals || [];
      setStatus({ type: 'success', text: `Generated ${signals.length} signal(s)` });
      if (onSignalGenerated) onSignalGenerated(signals);
      setTimeout(() => setStatus(null), 6000);
    } catch (err) {
      const msg = err.response?.data?.error || err.message || 'Unknown error';
      setStatus({ type: 'error', text: `Failed: ${msg}` });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10, padding: 15, backgroundColor: '#f8f9fa', borderRadius: 8, border: '1px solid #dee2e6' }}>
      <button
        onClick={handleClick}
        disabled={loading}
        style={{
          padding: '12px 24px',
          fontSize: 16,
          fontWeight: 600,
          color: 'white',
          backgroundColor: loading ? '#6c757d' : '#007bff',
          border: 'none',
          borderRadius: 6,
          cursor: loading ? 'not-allowed' : 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 8,
          transition: 'background 0.2s',
        }}
      >
        {loading ? (
          <>
            <span style={{ display: 'inline-block', width: 16, height: 16, border: '3px solid #fff', borderTop: '3px solid transparent', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
            Generating…
          </>
        ) : (
          '⚡ Generate Daily Signals'
        )}
      </button>

      {status && (
        <div style={{
          padding: '10px 14px',
          borderRadius: 6,
          fontSize: 13,
          fontWeight: 500,
          backgroundColor: status.type === 'success' ? '#d4edda' : '#f8d7da',
          color: status.type === 'success' ? '#155724' : '#721c24',
          border: `1px solid ${status.type === 'success' ? '#c3e6cb' : '#f5c6cb'}`,
        }}>
          {status.type === 'success' ? '✅' : '❌'} {status.text}
        </div>
      )}

      <div style={{ fontSize: 12, color: '#6c757d', textAlign: 'center', fontStyle: 'italic' }}>
        {loading ? 'Fetching latest data and generating predictions…' : 'Click to generate fresh trading signals'}
      </div>

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
