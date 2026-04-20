import React from 'react';

export default function SignalDashboard({ signals }) {
  if (!signals || signals.length === 0) return <div>No signals available.</div>;
  return (
    <div>
      {signals.map((s, i) => (
        <div key={i} style={{ border: '1px solid #ccc', margin: 8, padding: 8 }}>
          <h3>{s.pair} {s.signal_name}</h3>
          <div>Confidence: {(s.ensemble_confidence * 100).toFixed(1)}%</div>
          <div>RF Model: {(s.rf_pred_proba * 100).toFixed(1)}%</div>
          <div>XGB Model: {(s.xgb_pred_proba * 100).toFixed(1)}%</div>
          <div>Risk/Reward: {s.risk_reward_ratio}</div>
          <div>Entry: {s.entry}</div>
          <div>Stop Loss: {s.stop_loss}</div>
          <div>Take Profit: {s.take_profit}</div>
          <div>Date: {s.date}</div>
        </div>
      ))}
    </div>
  );
}
