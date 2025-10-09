/**
 * Performance Dashboard Component
 * Displays trading statistics, equity curve, and analytics
 */
import React, { useState, useEffect } from 'react';

const PerformanceDashboard = ({ days = 30 }) => {
  const [performance, setPerformance] = useState(null);
  const [equityCurve, setEquityCurve] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadPerformanceData();
  }, [days]);

  const loadPerformanceData = async () => {
    try {
      setLoading(true);

      // Load performance summary
      const perfRes = await fetch(`/api/paper-trading/trades/performance/?days=${days}`);
      const perfData = await perfRes.json();
      setPerformance(perfData);

      // Load equity curve
      const equityRes = await fetch(`/api/paper-trading/trades/equity_curve/?days=${days}`);
      const equityData = await equityRes.json();
      setEquityCurve(equityData);
    } catch (error) {
      console.error('Error loading performance data:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div style={{ padding: '20px', background: '#2B2B2B', borderRadius: '8px' }}>
        <div style={{ textAlign: 'center', padding: '40px', color: '#999' }}>
          Loading performance data...
        </div>
      </div>
    );
  }

  if (!performance) {
    return (
      <div style={{ padding: '20px', background: '#2B2B2B', borderRadius: '8px' }}>
        <div style={{ textAlign: 'center', padding: '40px', color: '#999' }}>
          No performance data available
        </div>
      </div>
    );
  }

  return (
    <div style={{ padding: '20px', background: '#2B2B2B', borderRadius: '8px' }}>
      {/* Header */}
      <h2 style={{ margin: '0 0 20px 0', color: '#D9D9D9' }}>
        📈 Performance Dashboard ({days} days)
      </h2>

      {/* Key Metrics Grid */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '15px',
          marginBottom: '30px',
        }}
      >
        <MetricCard
          title="Win Rate"
          value={`${performance.win_rate.toFixed(1)}%`}
          subtitle={`${performance.winning_trades}W / ${performance.losing_trades}L`}
          color={performance.win_rate >= 60 ? '#4caf50' : performance.win_rate >= 50 ? '#ff9800' : '#f44336'}
        />
        <MetricCard
          title="Total Pips"
          value={performance.total_pips >= 0 ? `+${performance.total_pips.toFixed(1)}` : performance.total_pips.toFixed(1)}
          subtitle={`${performance.total_trades} trades`}
          color={performance.total_pips >= 0 ? '#26a69a' : '#ef5350'}
        />
        <MetricCard
          title="P&L"
          value={`$${performance.total_pnl.toFixed(2)}`}
          subtitle={`Balance: $${performance.current_balance.toFixed(2)}`}
          color={performance.total_pnl >= 0 ? '#26a69a' : '#ef5350'}
        />
        <MetricCard
          title="Avg R:R"
          value={`${performance.avg_rr.toFixed(2)}:1`}
          subtitle="Risk:Reward"
          color="#1976d2"
        />
      </div>

      {/* Best/Worst Trades */}
      {(performance.best_trade || performance.worst_trade) && (
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
            gap: '15px',
            marginBottom: '30px',
          }}
        >
          {performance.best_trade && (
            <div
              style={{
                padding: '15px',
                background: '#1E1E1E',
                borderRadius: '8px',
                borderLeft: '5px solid #4caf50',
              }}
            >
              <div style={{ color: '#999', fontSize: '14px', marginBottom: '5px' }}>
                🏆 Best Trade
              </div>
              <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#4caf50' }}>
                +{performance.best_trade.pips.toFixed(1)} pips
              </div>
              <div style={{ color: '#D9D9D9', fontSize: '14px' }}>
                {performance.best_trade.pair} • ${performance.best_trade.pnl.toFixed(2)}
              </div>
            </div>
          )}
          {performance.worst_trade && (
            <div
              style={{
                padding: '15px',
                background: '#1E1E1E',
                borderRadius: '8px',
                borderLeft: '5px solid #f44336',
              }}
            >
              <div style={{ color: '#999', fontSize: '14px', marginBottom: '5px' }}>
                📉 Worst Trade
              </div>
              <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#f44336' }}>
                {performance.worst_trade.pips.toFixed(1)} pips
              </div>
              <div style={{ color: '#D9D9D9', fontSize: '14px' }}>
                {performance.worst_trade.pair} • ${performance.worst_trade.pnl.toFixed(2)}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Equity Curve */}
      {equityCurve.length > 0 && (
        <div>
          <h3 style={{ color: '#D9D9D9', marginBottom: '15px' }}>Equity Curve</h3>
          <EquityCurveChart data={equityCurve} />
        </div>
      )}
    </div>
  );
};

const MetricCard = ({ title, value, subtitle, color }) => (
  <div
    style={{
      padding: '20px',
      background: '#1E1E1E',
      borderRadius: '8px',
      textAlign: 'center',
    }}
  >
    <div style={{ color: '#999', fontSize: '14px', marginBottom: '8px' }}>
      {title}
    </div>
    <div
      style={{
        fontSize: '32px',
        fontWeight: 'bold',
        color: color,
        marginBottom: '8px',
      }}
    >
      {value}
    </div>
    {subtitle && (
      <div style={{ color: '#666', fontSize: '12px' }}>
        {subtitle}
      </div>
    )}
  </div>
);

const EquityCurveChart = ({ data }) => {
  const chartRef = React.useRef(null);

  // Calculate chart dimensions
  const width = chartRef.current?.clientWidth || 800;
  const height = 300;
  const padding = { top: 20, right: 20, bottom: 40, left: 60 };

  // Calculate scales
  const equities = data.map(d => d.equity);
  const minEquity = Math.min(...equities);
  const maxEquity = Math.max(...equities);
  const range = maxEquity - minEquity;

  const xScale = (index) => padding.left + (index / (data.length - 1)) * (width - padding.left - padding.right);
  const yScale = (equity) => height - padding.bottom - ((equity - minEquity) / range) * (height - padding.top - padding.bottom);

  // Generate path
  const pathD = data.map((point, i) => {
    const x = xScale(i);
    const y = yScale(point.equity);
    return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
  }).join(' ');

  // Calculate color (green if above initial, red if below)
  const initialEquity = data[0]?.equity || 0;
  const finalEquity = data[data.length - 1]?.equity || 0;
  const chartColor = finalEquity >= initialEquity ? '#26a69a' : '#ef5350';

  return (
    <div ref={chartRef} style={{ width: '100%', background: '#1E1E1E', borderRadius: '8px', padding: '15px' }}>
      <svg width="100%" height={height} style={{ display: 'block' }}>
        {/* Grid lines */}
        {[0, 0.25, 0.5, 0.75, 1].map((ratio) => {
          const y = height - padding.bottom - ratio * (height - padding.top - padding.bottom);
          const value = minEquity + ratio * range;
          return (
            <g key={ratio}>
              <line
                x1={padding.left}
                y1={y}
                x2={width - padding.right}
                y2={y}
                stroke="#333"
                strokeWidth="1"
                strokeDasharray="4"
              />
              <text x={padding.left - 10} y={y + 5} fill="#666" fontSize="12" textAnchor="end">
                ${value.toFixed(0)}
              </text>
            </g>
          );
        })}

        {/* Equity curve line */}
        <path d={pathD} fill="none" stroke={chartColor} strokeWidth="2" />

        {/* Data points */}
        {data.map((point, i) => (
          <circle
            key={i}
            cx={xScale(i)}
            cy={yScale(point.equity)}
            r="3"
            fill={chartColor}
          />
        ))}

        {/* X-axis labels (show first, middle, last) */}
        {[0, Math.floor(data.length / 2), data.length - 1].map((index) => {
          if (index >= data.length) return null;
          const point = data[index];
          const x = xScale(index);
          const date = new Date(point.date).toLocaleDateString();
          return (
            <text
              key={index}
              x={x}
              y={height - padding.bottom + 20}
              fill="#666"
              fontSize="12"
              textAnchor="middle"
            >
              {date}
            </text>
          );
        })}
      </svg>

      {/* Summary */}
      <div style={{ marginTop: '15px', textAlign: 'center' }}>
        <span style={{ color: '#999', marginRight: '20px' }}>
          Start: ${initialEquity.toFixed(2)}
        </span>
        <span style={{ color: chartColor, fontWeight: 'bold', marginRight: '20px' }}>
          End: ${finalEquity.toFixed(2)}
        </span>
        <span style={{ color: chartColor, fontWeight: 'bold' }}>
          Change: {finalEquity >= initialEquity ? '+' : ''}${(finalEquity - initialEquity).toFixed(2)}
        </span>
      </div>
    </div>
  );
};

export default PerformanceDashboard;
