import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';

const DOT = ({ ok }) => (
  <span style={{
    display: 'inline-block', width: 10, height: 10, borderRadius: '50%',
    backgroundColor: ok === true ? '#3fb950' : ok === false ? '#f85149' : '#ffa500',
    marginRight: 8, boxShadow: `0 0 6px ${ok === true ? '#3fb950' : ok === false ? '#f85149' : '#ffa500'}`,
  }} />
);

const Row = ({ label, value, ok, sub }) => (
  <div style={{ display: 'flex', alignItems: 'flex-start', padding: '10px 0', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
    <DOT ok={ok} />
    <div style={{ flex: 1 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
        <span style={{ fontWeight: 600, fontSize: 14 }}>{label}</span>
        <span style={{ fontSize: 14, color: ok === true ? '#3fb950' : ok === false ? '#f85149' : '#ffa500' }}>{value}</span>
      </div>
      {sub && <div style={{ fontSize: 12, color: '#6e7681', marginTop: 2 }}>{sub}</div>}
    </div>
  </div>
);

const OVERALL_STYLE = {
  GREEN:  { bg: 'rgba(63,185,80,0.12)',  border: '#3fb950', label: 'ALL SYSTEMS GO',  icon: '✔' },
  YELLOW: { bg: 'rgba(255,165,0,0.12)',  border: '#ffa500', label: 'ATTENTION NEEDED', icon: '!' },
  RED:    { bg: 'rgba(248,81,73,0.12)',  border: '#f85149', label: 'ACTION REQUIRED',  icon: '✘' },
};

export default function SystemHealth({ apiBaseUrl, darkMode = false }) {
  const [health, setHealth]     = useState(null);
  const [decision, setDecision] = useState(null);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState('');
  const [openPos, setOpenPos]     = useState(0);
  const [dailyPnl, setDailyPnl]   = useState(0);
  const [balance, setBalance]     = useState(500);

  const fetchHealth = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const [hRes, dRes] = await Promise.all([
        axios.get(`${apiBaseUrl}/api/system-health/`),
        axios.get(`${apiBaseUrl}/api/signals/decision/?open_positions=${openPos}&daily_pnl=${dailyPnl}&balance=${balance}`),
      ]);
      setHealth(hRes.data);
      setDecision(dRes.data);
    } catch (err) {
      setError('Could not reach the server. Is Django running on port 8000?');
    } finally {
      setLoading(false);
    }
  }, [apiBaseUrl, openPos, dailyPnl]);

  useEffect(() => {
    fetchHealth();
    const interval = setInterval(fetchHealth, 60_000); // refresh every 60s
    return () => clearInterval(interval);
  }, [fetchHealth]);

  const card = {
    background: darkMode ? 'rgba(13,17,23,0.9)' : '#fff',
    borderRadius: 16,
    padding: '24px 28px',
    marginBottom: 20,
    boxShadow: darkMode ? '0 8px 32px rgba(0,0,0,0.4)' : '0 4px 12px rgba(0,0,0,0.08)',
    color: darkMode ? '#c9d1d9' : '#212529',
  };

  if (loading && !health) return (
    <div style={{ ...card, textAlign: 'center', padding: 48 }}>
      <div style={{ fontSize: 32, marginBottom: 12 }}>⟳</div>
      <p style={{ color: '#6e7681' }}>Checking system status...</p>
    </div>
  );

  if (error) return (
    <div style={{ ...card, borderLeft: '4px solid #f85149' }}>
      <strong style={{ color: '#f85149' }}>Cannot connect to server</strong>
      <p style={{ color: '#6e7681', fontSize: 14, marginTop: 8 }}>{error}</p>
      <button onClick={fetchHealth} style={{ marginTop: 8, padding: '8px 16px', background: '#f85149', color: '#fff', border: 'none', borderRadius: 8, cursor: 'pointer' }}>
        Retry
      </button>
    </div>
  );

  const overall    = health?.overall || 'RED';
  const ovStyle    = OVERALL_STYLE[overall] || OVERALL_STYLE.RED;
  const dataItems  = health?.data    || {};
  const modelItems = health?.models  || {};
  const sigs       = health?.signals || {};
  const pos        = health?.positions || {};

  const fmtAge = (min) => {
    if (min == null) return 'never';
    if (min < 60)   return `${Math.round(min)}m ago`;
    return `${(min / 60).toFixed(1)}h ago`;
  };

  return (
    <div>
      {/* Overall Status Banner */}
      <div style={{
        background: ovStyle.bg,
        border: `2px solid ${ovStyle.border}`,
        borderRadius: 16,
        padding: '20px 28px',
        marginBottom: 20,
        display: 'flex',
        alignItems: 'center',
        gap: 16,
      }}>
        <div style={{ fontSize: 36, color: ovStyle.border }}>{ovStyle.icon}</div>
        <div>
          <div style={{ fontSize: 22, fontWeight: 700, color: ovStyle.border }}>{ovStyle.label}</div>
          <div style={{ fontSize: 12, color: '#6e7681', marginTop: 4 }}>
            Last checked: {health?.timestamp ? new Date(health.timestamp).toLocaleTimeString() : '—'}
            <button onClick={fetchHealth} style={{ marginLeft: 12, background: 'none', border: 'none', color: ovStyle.border, cursor: 'pointer', fontSize: 13, fontWeight: 600 }}>
              Refresh
            </button>
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 20 }}>
        {/* Data Health */}
        <div style={card}>
          <h3 style={{ margin: '0 0 16px', fontSize: 16, fontWeight: 700 }}>Data Files</h3>
          {Object.entries(dataItems).map(([key, v]) => (
            <Row
              key={key}
              label={key}
              value={v.exists ? `${v.rows.toLocaleString()} rows` : 'MISSING'}
              ok={v.fresh}
              sub={v.exists ? `Updated ${fmtAge(v.age_minutes)}` : 'Run: Generate Signals to fetch data'}
            />
          ))}
        </div>

        {/* Model Health */}
        <div style={card}>
          <h3 style={{ margin: '0 0 16px', fontSize: 16, fontWeight: 700 }}>Trained Models</h3>
          {Object.entries(modelItems).map(([pair, v]) => (
            <Row
              key={pair}
              label={pair}
              value={v.ready ? `CV ${v.cv_accuracy ? (v.cv_accuracy * 100).toFixed(1) + '%' : '—'}` : 'NOT TRAINED'}
              ok={v.ready}
              sub={v.trained_at ? `Trained: ${new Date(v.trained_at).toLocaleDateString()}` : 'Run: python manage.py train_models'}
            />
          ))}
        </div>

        {/* Signal Status */}
        <div style={card}>
          <h3 style={{ margin: '0 0 16px', fontSize: 16, fontWeight: 700 }}>Today's Signals</h3>
          <Row
            label="Generated today"
            value={sigs.today_count != null ? `${sigs.today_count} signal(s)` : '—'}
            ok={sigs.today_count > 0}
            sub={sigs.today_pairs?.length ? `Pairs: ${sigs.today_pairs.join(', ')}` : 'No signals yet today'}
          />
          <Row
            label="Total in database"
            value={sigs.total_in_db != null ? sigs.total_in_db : '—'}
            ok={sigs.total_in_db > 0}
          />
          <Row
            label="Last generated"
            value={sigs.last_generated ? new Date(sigs.last_generated).toLocaleString() : 'Never'}
            ok={!!sigs.last_generated}
          />
        </div>

        {/* Positions */}
        <div style={card}>
          <h3 style={{ margin: '0 0 16px', fontSize: 16, fontWeight: 700 }}>Open Positions</h3>
          <Row
            label="Open trades"
            value={pos.open_count ?? 0}
            ok={pos.open_count <= 3}
            sub="Max 3 recommended"
          />
          <Row
            label="Unrealized P&L"
            value={`$${(pos.total_pnl ?? 0).toFixed(2)}`}
            ok={pos.total_pnl >= 0}
          />

          {/* Manual account context for decision engine */}
          <div style={{ marginTop: 16, paddingTop: 12, borderTop: '1px solid rgba(255,255,255,0.08)' }}>
            <div style={{ fontSize: 12, color: '#6e7681', marginBottom: 8 }}>Override for decision engine:</div>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <div>
                <label style={{ fontSize: 11, color: '#6e7681', display: 'block' }}>Account balance ($)</label>
                <input type="number" min={100} step={50} value={balance}
                  onChange={e => setBalance(Number(e.target.value))}
                  style={{ width: 90, padding: '4px 6px', borderRadius: 6, border: '1px solid #30363d', background: darkMode ? '#0d1117' : '#f8f9fa', color: darkMode ? '#c9d1d9' : '#212529', fontSize: 13 }} />
              </div>
              <div>
                <label style={{ fontSize: 11, color: '#6e7681', display: 'block' }}>Open positions</label>
                <input type="number" min={0} max={10} value={openPos}
                  onChange={e => setOpenPos(Number(e.target.value))}
                  style={{ width: 70, padding: '4px 6px', borderRadius: 6, border: '1px solid #30363d', background: darkMode ? '#0d1117' : '#f8f9fa', color: darkMode ? '#c9d1d9' : '#212529', fontSize: 13 }} />
              </div>
              <div>
                <label style={{ fontSize: 11, color: '#6e7681', display: 'block' }}>Daily P&L ($)</label>
                <input type="number" step={10} value={dailyPnl}
                  onChange={e => setDailyPnl(Number(e.target.value))}
                  style={{ width: 90, padding: '4px 6px', borderRadius: 6, border: '1px solid #30363d', background: darkMode ? '#0d1117' : '#f8f9fa', color: darkMode ? '#c9d1d9' : '#212529', fontSize: 13 }} />
              </div>
              <button onClick={fetchHealth} style={{ alignSelf: 'flex-end', padding: '5px 12px', background: '#667eea', color: '#fff', border: 'none', borderRadius: 6, cursor: 'pointer', fontSize: 12, fontWeight: 600 }}>
                Update
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Decision Engine */}
      {decision && (
        <div style={{ ...card, marginTop: 0 }}>
          <h3 style={{ margin: '0 0 16px', fontSize: 16, fontWeight: 700 }}>Trade Decision</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
            {Object.entries(decision).map(([pair, d]) => {
              if (!d?.action) return null;
              const actionColor = d.action === 'EXECUTE' ? '#3fb950' : d.action === 'WAIT' ? '#ffa500' : '#f85149';
              return (
                <div key={pair} style={{ background: `${actionColor}15`, border: `1px solid ${actionColor}40`, borderRadius: 12, padding: 16 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                    <span style={{ fontWeight: 700, fontSize: 16 }}>{pair}</span>
                    <span style={{ fontWeight: 700, fontSize: 18, color: actionColor }}>{d.action}</span>
                  </div>
                  <p style={{ fontSize: 13, color: darkMode ? '#8b949e' : '#495057', margin: '0 0 12px', lineHeight: 1.4 }}>{d.summary}</p>
                  <div style={{ fontSize: 12 }}>
                    {(d.reasons || []).map((r, i) => (
                      <div key={i} style={{ display: 'flex', gap: 6, marginBottom: 4, color: darkMode ? '#8b949e' : '#6c757d' }}>
                        <span style={{ color: r.pass ? '#3fb950' : '#f85149', fontWeight: 700, minWidth: 16 }}>
                          {r.pass ? '✔' : '✘'}
                        </span>
                        <span><strong>{r.rule}:</strong> {r.detail}</span>
                      </div>
                    ))}
                  </div>
                  {/* Position Sizing */}
                  {d.sizing && d.action === 'EXECUTE' && (
                    <div style={{ marginTop: 12, padding: '10px 12px', background: darkMode ? 'rgba(0,0,0,0.3)' : 'rgba(0,0,0,0.04)', borderRadius: 8, fontSize: 13 }}>
                      <div style={{ fontWeight: 700, marginBottom: 6, color: actionColor }}>MT5 Order Setup</div>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px 16px', color: darkMode ? '#8b949e' : '#6c757d' }}>
                        <span>Lot size:</span><span style={{ fontWeight: 700, color: darkMode ? '#c9d1d9' : '#212529' }}>{d.sizing.lot_size}</span>
                        <span>Risk $:</span><span style={{ fontWeight: 700, color: '#f85149' }}>${d.sizing.risk_usd} ({d.sizing.risk_pct}%)</span>
                        <span>Pips at risk:</span><span style={{ fontWeight: 700, color: darkMode ? '#c9d1d9' : '#212529' }}>{d.sizing.pip_risk}</span>
                        <span>Potential gain:</span><span style={{ fontWeight: 700, color: '#3fb950' }}>${d.sizing.potential_reward}</span>
                      </div>
                    </div>
                  )}
                  <div style={{ marginTop: 10, fontSize: 11, color: '#6e7681' }}>
                    Score: {d.score}/100
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
