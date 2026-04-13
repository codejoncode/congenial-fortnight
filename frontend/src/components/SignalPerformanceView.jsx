import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';

const PAIR_META = {
  EURUSD: { label: 'EUR/USD', emoji: '🇪🇺', baseline: 50.6 },
  XAUUSD: { label: 'XAU/USD (Gold)', emoji: '🥇', baseline: 52.4 },
};

const TIER_BANDS = [
  { min: 75, label: 'Excellent', color: '#00e5ff', bg: 'rgba(0,229,255,0.08)' },
  { min: 65, label: 'Very Good', color: '#00ff87', bg: 'rgba(0,255,135,0.08)' },
  { min: 60, label: 'Good',      color: '#b8ff45', bg: 'rgba(184,255,69,0.08)' },
  { min: 55, label: 'Marginal',  color: '#ffd700', bg: 'rgba(255,215,0,0.08)' },
  { min:  0, label: 'Weak',      color: '#ff6b6b', bg: 'rgba(255,107,107,0.08)' },
];

function tier(acc_pct) {
  return TIER_BANDS.find(b => acc_pct >= b.min) || TIER_BANDS[TIER_BANDS.length - 1];
}

function AccuracyBar({ value, baseline }) {
  const pct = Math.min(100, Math.max(0, value));
  const t = tier(pct);
  return (
    <div style={{ position: 'relative', height: 10, background: 'rgba(255,255,255,0.08)', borderRadius: 6, overflow: 'hidden', minWidth: 120 }}>
      <div style={{
        position: 'absolute', top: 0, bottom: 0,
        left: `${baseline}%`, width: 2,
        background: 'rgba(255,255,255,0.35)', zIndex: 2,
      }} title={`Baseline ${baseline}%`} />
      <div style={{
        height: '100%', width: `${pct}%`,
        background: `linear-gradient(90deg, ${t.color}88, ${t.color})`,
        borderRadius: 6, transition: 'width 0.8s ease',
      }} />
    </div>
  );
}

function GradeBadge({ grade, color }) {
  return (
    <span style={{
      display: 'inline-block',
      padding: '2px 10px', borderRadius: 20,
      fontSize: 12, fontWeight: 700,
      color: '#0d1117', background: color,
      letterSpacing: '0.5px', minWidth: 36, textAlign: 'center',
    }}>
      {grade}
    </span>
  );
}

function DirectionBadge({ signal }) {
  if (!signal) return <span style={{ color: '#6c757d', fontSize: 13 }}>—</span>;
  const bullish = signal === 'bullish';
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 6,
      padding: '5px 14px', borderRadius: 20,
      background: bullish ? 'rgba(63,185,80,0.15)' : 'rgba(248,81,73,0.15)',
      border: `1px solid ${bullish ? '#3fb950' : '#f85149'}`,
      color: bullish ? '#3fb950' : '#f85149',
      fontSize: 13, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '1px',
    }}>
      {bullish ? '▲' : '▼'} {signal}
    </span>
  );
}

function SignalTable({ signals, baseline, darkMode }) {
  if (!signals || signals.length === 0) {
    return (
      <div style={{ padding: '24px', textAlign: 'center', color: darkMode ? '#8b949e' : '#6c757d' }}>
        No signals meet the current filter.
      </div>
    );
  }

  const labelStyle = {
    fontSize: 11, fontWeight: 600, textTransform: 'uppercase',
    letterSpacing: '0.5px', color: darkMode ? '#8b949e' : '#6c757d', padding: '10px 16px',
  };
  const cellStyle = {
    padding: '12px 16px', fontSize: 14,
    borderBottom: `1px solid ${darkMode ? '#21262d' : '#e9ecef'}`, verticalAlign: 'middle',
  };

  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ borderBottom: `2px solid ${darkMode ? '#30363d' : '#dee2e6'}` }}>
            <th style={{ ...labelStyle, textAlign: 'left' }}>Signal</th>
            <th style={{ ...labelStyle, textAlign: 'center' }}>Grade</th>
            <th style={{ ...labelStyle, textAlign: 'right' }}>Accuracy</th>
            <th style={{ ...labelStyle, textAlign: 'right' }}>Hit Rate</th>
            <th style={{ ...labelStyle, textAlign: 'right' }}>Correlation</th>
            <th style={{ ...labelStyle, textAlign: 'left', minWidth: 160 }}>vs Baseline</th>
          </tr>
        </thead>
        <tbody>
          {signals.map((s, i) => {
            const t = tier(s.accuracy_pct);
            const aboveBaseline = (s.accuracy_pct - baseline).toFixed(2);
            const positive = parseFloat(aboveBaseline) >= 0;
            return (
              <tr
                key={s.feature}
                style={{
                  background: i % 2 === 0
                    ? (darkMode ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0.015)')
                    : 'transparent',
                  transition: 'background 0.2s',
                }}
                onMouseOver={e => e.currentTarget.style.background = t.bg}
                onMouseOut={e => e.currentTarget.style.background = i % 2 === 0
                  ? (darkMode ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0.015)')
                  : 'transparent'}
              >
                <td style={{ ...cellStyle, color: darkMode ? '#c9d1d9' : '#212529', fontWeight: 600 }}>
                  {s.label}
                  {i === 0 && <span style={{ marginLeft: 8, fontSize: 11, color: '#ffd700' }}>⭐ BEST</span>}
                </td>
                <td style={{ ...cellStyle, textAlign: 'center' }}>
                  <GradeBadge grade={s.grade} color={s.grade_color} />
                </td>
                <td style={{ ...cellStyle, textAlign: 'right', fontFamily: 'monospace', color: t.color, fontWeight: 700, fontSize: 15 }}>
                  {s.accuracy_pct.toFixed(2)}%
                </td>
                <td style={{ ...cellStyle, textAlign: 'right', fontFamily: 'monospace', color: darkMode ? '#8b949e' : '#6c757d' }}>
                  {(s.hit_rate * 100).toFixed(1)}%
                </td>
                <td style={{ ...cellStyle, textAlign: 'right', fontFamily: 'monospace', color: s.correlation > 0 ? '#3fb950' : '#f85149' }}>
                  {s.correlation > 0 ? '+' : ''}{s.correlation.toFixed(3)}
                </td>
                <td style={cellStyle}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <AccuracyBar value={s.accuracy_pct} baseline={baseline} />
                    <span style={{
                      fontFamily: 'monospace', fontSize: 12, fontWeight: 700,
                      color: positive ? '#3fb950' : '#f85149',
                      minWidth: 52, textAlign: 'right',
                    }}>
                      {positive ? '+' : ''}{aboveBaseline}%
                    </span>
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function PairSection({ pair, data, currentSignal, onTrade, tradingState, darkMode }) {
  const meta = PAIR_META[pair] || { label: pair, emoji: '📊', baseline: 50 };

  if (data && data.error && (!data.signals || data.signals.length === 0)) {
    return (
      <div style={{ padding: 16, color: '#f85149', borderRadius: 8, background: 'rgba(248,81,73,0.08)', marginBottom: 24 }}>
        {meta.emoji} {meta.label}: {data.error}
      </div>
    );
  }

  const signals = data ? data.signals : [];
  const topSignal = signals[0];
  const t = topSignal ? tier(topSignal.accuracy_pct) : null;

  const hasCurrent = currentSignal && currentSignal.signal;
  const bullish = hasCurrent && currentSignal.signal === 'bullish';
  const signalColor = bullish ? '#3fb950' : '#f85149';
  const tradeKey = `${pair}-${currentSignal?.signal}`;
  const tradeStatus = tradingState[tradeKey] || {};

  return (
    <div style={{
      background: darkMode
        ? 'linear-gradient(145deg, rgba(22,27,34,0.98), rgba(13,17,23,0.98))'
        : '#ffffff',
      borderRadius: 16,
      border: `1px solid ${darkMode ? '#30363d' : '#e9ecef'}`,
      marginBottom: 32, overflow: 'hidden',
      boxShadow: darkMode ? '0 8px 32px rgba(0,0,0,0.4)' : '0 4px 12px rgba(0,0,0,0.08)',
    }}>
      {/* Header */}
      <div style={{
        padding: '20px 24px',
        borderBottom: `1px solid ${darkMode ? '#21262d' : '#e9ecef'}`,
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        flexWrap: 'wrap', gap: 12,
        background: darkMode ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0.02)',
      }}>
        {/* Left: pair info */}
        <div>
          <h3 style={{ margin: 0, fontSize: 22, fontWeight: 700, color: darkMode ? '#c9d1d9' : '#212529' }}>
            {meta.emoji} {meta.label}
          </h3>
          <div style={{ marginTop: 4, fontSize: 13, color: darkMode ? '#8b949e' : '#6c757d' }}>
            Baseline: <strong>{meta.baseline}%</strong> &nbsp;|&nbsp; {signals.length} signals shown
          </div>
        </div>

        {/* Center: current live signal */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
          <div style={{ fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', color: darkMode ? '#8b949e' : '#6c757d' }}>
            Current Signal
          </div>
          {hasCurrent ? (
            <>
              <DirectionBadge signal={currentSignal.signal} />
              <div style={{ fontSize: 12, color: darkMode ? '#8b949e' : '#6c757d', fontFamily: 'monospace' }}>
                {typeof currentSignal.probability === 'number'
                  ? `${(currentSignal.probability * 100).toFixed(1)}% confidence`
                  : 'confidence N/A'}
              </div>
              {currentSignal.stop_loss && (
                <div style={{ fontSize: 11, color: darkMode ? '#8b949e' : '#6c757d', fontFamily: 'monospace' }}>
                  SL: {typeof currentSignal.stop_loss === 'number' ? currentSignal.stop_loss.toFixed(4) : currentSignal.stop_loss}
                </div>
              )}
              {currentSignal.date && (
                <div style={{ fontSize: 11, color: darkMode ? '#6e7681' : '#adb5bd' }}>
                  {new Date(currentSignal.date).toLocaleDateString()}
                </div>
              )}
            </>
          ) : (
            <div style={{ fontSize: 13, color: darkMode ? '#6e7681' : '#adb5bd', fontStyle: 'italic' }}>
              No signal in DB yet
            </div>
          )}
        </div>

        {/* Right: top signal card + trade button */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 10 }}>
          {topSignal && (
            <div style={{
              padding: '12px 20px', borderRadius: 12, textAlign: 'center',
              background: `linear-gradient(135deg, ${t.color}20, ${t.color}10)`,
              border: `1px solid ${t.color}50`,
            }}>
              <div style={{ fontSize: 11, color: darkMode ? '#8b949e' : '#6c757d', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 4 }}>
                Best Historical Signal
              </div>
              <div style={{ fontSize: 13, fontWeight: 600, color: darkMode ? '#c9d1d9' : '#212529' }}>
                {topSignal.label}
              </div>
              <div style={{ fontSize: 22, fontWeight: 800, color: t.color, fontFamily: 'monospace' }}>
                {topSignal.accuracy_pct.toFixed(2)}%
              </div>
              <GradeBadge grade={topSignal.grade} color={topSignal.grade_color} />
            </div>
          )}

          {/* Trade button — only when there's a current signal */}
          {hasCurrent && (
            <div style={{ width: '100%' }}>
              <button
                onClick={() => onTrade(pair, currentSignal)}
                disabled={tradeStatus.loading}
                style={{
                  width: '100%', padding: '12px 20px',
                  background: tradeStatus.loading
                    ? 'rgba(108,117,125,0.5)'
                    : bullish
                      ? 'linear-gradient(135deg, #3fb950, #2ea043)'
                      : 'linear-gradient(135deg, #f85149, #da3633)',
                  color: 'white', border: 'none', borderRadius: 10,
                  cursor: tradeStatus.loading ? 'not-allowed' : 'pointer',
                  fontSize: 14, fontWeight: 700, textTransform: 'uppercase',
                  letterSpacing: '0.5px',
                  boxShadow: tradeStatus.loading ? 'none' : `0 4px 12px ${signalColor}50`,
                  opacity: tradeStatus.loading ? 0.6 : 1,
                  transition: 'all 0.3s ease',
                }}
              >
                {tradeStatus.loading
                  ? '⏳ Executing…'
                  : `📊 Paper Trade ${bullish ? 'BUY' : 'SELL'} ${pair}`}
              </button>

              {tradeStatus.message && (
                <div style={{
                  marginTop: 8, padding: '8px 12px', borderRadius: 6, fontSize: 12,
                  fontWeight: 600, textAlign: 'center',
                  background: tradeStatus.success ? 'rgba(63,185,80,0.1)' : 'rgba(248,81,73,0.1)',
                  color: tradeStatus.success ? '#3fb950' : '#f85149',
                  border: `1px solid ${tradeStatus.success ? '#3fb950' : '#f85149'}`,
                }}>
                  {tradeStatus.message}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Accuracy table */}
      <SignalTable signals={signals} baseline={meta.baseline} darkMode={darkMode} />
    </div>
  );
}

export default function SignalPerformanceView({ apiBaseUrl, darkMode = false }) {
  const [perfData, setPerfData] = useState(null);
  const [currentSignals, setCurrentSignals] = useState({});  // { EURUSD: {...}, XAUUSD: {...} }
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [minAcc, setMinAcc] = useState(50);
  const [limit, setLimit] = useState(20);
  const [selectedPair, setSelectedPair] = useState('all');
  const [tradingState, setTradingState] = useState({}); // { 'EURUSD-bullish': { loading, success, message } }

  // ── fetch performance data ─────────────────────────────────────────────
  const fetchPerformance = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const params = new URLSearchParams({
        pair: selectedPair,
        min_acc: (minAcc / 100).toFixed(2),
        limit,
      });
      const res = await axios.get(`${apiBaseUrl}/api/signal-performance/?${params}`);
      setPerfData(res.data);
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Failed to load signal performance data');
    } finally {
      setLoading(false);
    }
  }, [apiBaseUrl, selectedPair, minAcc, limit]);

  // ── fetch current live signal per pair from DB ─────────────────────────
  const fetchCurrentSignals = useCallback(async () => {
    const pairs = selectedPair === 'all' ? ['EURUSD', 'XAUUSD'] : [selectedPair];
    const results = {};
    await Promise.all(pairs.map(async (p) => {
      try {
        // GET /api/signals/{pair}/ returns an array; take the first (most recent)
        const res = await axios.get(`${apiBaseUrl}/api/signals/${p}/`);
        const data = Array.isArray(res.data) ? res.data : (res.data.results || []);
        results[p] = data.length > 0 ? data[0] : null;
      } catch {
        results[p] = null;
      }
    }));
    setCurrentSignals(results);
  }, [apiBaseUrl, selectedPair]);

  useEffect(() => {
    fetchPerformance();
    fetchCurrentSignals();
  }, [fetchPerformance, fetchCurrentSignals]);

  // ── execute paper trade ────────────────────────────────────────────────
  const executeTrade = async (pair, signal) => {
    const key = `${pair}-${signal.signal}`;
    setTradingState(prev => ({ ...prev, [key]: { loading: true, message: '', success: false } }));
    try {
      const res = await axios.post(`${apiBaseUrl}/api/paper-trades/execute/`, {
        pair,
        signal: signal.signal,
        stop_loss: signal.stop_loss,
        probability: signal.probability,
        lot_size: 0.1,
      });
      const entry = res.data.entry_price;
      const msg = `✅ ${pair} ${signal.signal.toUpperCase()} executed${entry ? ` @ ${Number(entry).toFixed(4)}` : ''}`;
      setTradingState(prev => ({ ...prev, [key]: { loading: false, message: msg, success: true } }));
      setTimeout(() => setTradingState(prev => ({ ...prev, [key]: {} })), 8000);
    } catch (err) {
      const msg = `❌ ${err.response?.data?.error || err.message}`;
      setTradingState(prev => ({ ...prev, [key]: { loading: false, message: msg, success: false } }));
      setTimeout(() => setTradingState(prev => ({ ...prev, [key]: {} })), 8000);
    }
  };

  // ── derived ────────────────────────────────────────────────────────────
  const bg = darkMode ? '#0d1117' : '#f5f5f5';
  const cardBg = darkMode ? 'rgba(22,27,34,0.98)' : '#ffffff';
  const textPrimary = darkMode ? '#c9d1d9' : '#212529';
  const textMuted = darkMode ? '#8b949e' : '#6c757d';
  const borderColor = darkMode ? '#30363d' : '#e9ecef';
  const pairsToRender = perfData ? Object.keys(perfData) : (selectedPair === 'all' ? ['EURUSD', 'XAUUSD'] : [selectedPair]);

  return (
    <div style={{ background: bg, minHeight: '100vh', padding: '0 0 40px' }}>
      {/* Controls bar */}
      <div style={{
        background: cardBg, border: `1px solid ${borderColor}`, borderRadius: 16,
        padding: '20px 24px', marginBottom: 28,
        display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 16, justifyContent: 'space-between',
      }}>
        <div>
          <h2 style={{ margin: 0, fontSize: 26, fontWeight: 700, color: textPrimary }}>
            📊 Signal Performance Dashboard
          </h2>
          <p style={{ margin: '4px 0 0', fontSize: 13, color: textMuted }}>
            Historical accuracy per signal · current live direction · one-click paper trading
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
          {/* Pair filter */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <label style={{ fontSize: 13, color: textMuted }}>Pair</label>
            <select value={selectedPair} onChange={e => setSelectedPair(e.target.value)}
              style={{ padding: '7px 12px', borderRadius: 8, border: `1px solid ${borderColor}`, background: darkMode ? '#21262d' : '#fff', color: textPrimary, fontSize: 13, cursor: 'pointer' }}>
              <option value="all">All Pairs</option>
              <option value="EURUSD">EUR/USD</option>
              <option value="XAUUSD">XAU/USD</option>
            </select>
          </div>

          {/* Min accuracy */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <label style={{ fontSize: 13, color: textMuted }}>Min Accuracy</label>
            <select value={minAcc} onChange={e => setMinAcc(Number(e.target.value))}
              style={{ padding: '7px 12px', borderRadius: 8, border: `1px solid ${borderColor}`, background: darkMode ? '#21262d' : '#fff', color: textPrimary, fontSize: 13, cursor: 'pointer' }}>
              <option value={50}>≥ 50%</option>
              <option value={55}>≥ 55%</option>
              <option value={60}>≥ 60%</option>
              <option value={65}>≥ 65%</option>
            </select>
          </div>

          {/* Limit */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <label style={{ fontSize: 13, color: textMuted }}>Show top</label>
            <select value={limit} onChange={e => setLimit(Number(e.target.value))}
              style={{ padding: '7px 12px', borderRadius: 8, border: `1px solid ${borderColor}`, background: darkMode ? '#21262d' : '#fff', color: textPrimary, fontSize: 13, cursor: 'pointer' }}>
              <option value={10}>10</option>
              <option value={20}>20</option>
              <option value={50}>50</option>
            </select>
          </div>

          <button onClick={() => { fetchPerformance(); fetchCurrentSignals(); }} disabled={loading}
            style={{
              padding: '8px 18px', borderRadius: 8, border: 'none',
              cursor: loading ? 'not-allowed' : 'pointer',
              background: 'linear-gradient(135deg, #667eea, #764ba2)',
              color: '#fff', fontSize: 13, fontWeight: 600, opacity: loading ? 0.6 : 1,
            }}>
            {loading ? '⏳ Loading…' : '🔄 Refresh'}
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div style={{ padding: 16, marginBottom: 24, borderRadius: 10, background: 'rgba(248,81,73,0.1)', border: '1px solid rgba(248,81,73,0.3)', color: '#f85149', fontWeight: 600 }}>
          ⚠️ {error}
        </div>
      )}

      {/* Tier legend */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, marginBottom: 24 }}>
        {TIER_BANDS.map(b => (
          <div key={b.label} style={{
            display: 'flex', alignItems: 'center', gap: 6,
            padding: '5px 14px', borderRadius: 20,
            background: darkMode ? 'rgba(255,255,255,0.04)' : '#fff',
            border: `1px solid ${b.color}50`,
            fontSize: 12, color: b.color, fontWeight: 600,
          }}>
            <span style={{ width: 10, height: 10, borderRadius: '50%', background: b.color, display: 'inline-block' }} />
            {b.label} {b.min > 0 ? `(≥${b.min}%)` : '(<55%)'}
          </div>
        ))}
      </div>

      {/* Loading spinner */}
      {loading && !perfData && (
        <div style={{ textAlign: 'center', padding: 60, color: textMuted }}>
          <div style={{
            width: 48, height: 48, margin: '0 auto 16px',
            border: '4px solid rgba(255,255,255,0.1)', borderTop: '4px solid #667eea',
            borderRadius: '50%', animation: 'spin 0.9s linear infinite',
          }} />
          Loading signal performance data…
          <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
        </div>
      )}

      {/* Pair sections */}
      {pairsToRender.map(pair => (
        <PairSection
          key={pair}
          pair={pair}
          data={perfData ? perfData[pair] : null}
          currentSignal={currentSignals[pair]}
          onTrade={executeTrade}
          tradingState={tradingState}
          darkMode={darkMode}
        />
      ))}
    </div>
  );
}
