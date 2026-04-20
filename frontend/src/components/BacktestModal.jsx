import React from 'react';

export default function BacktestModal({ signalName, history, onClose }) {
  if (!history || history.length === 0) return null;
  return (
    <div style={{ position: 'fixed', top: 0, left: 0, width: '100vw', height: '100vh', background: 'rgba(0,0,0,0.5)', zIndex: 1000 }}>
      <div style={{ background: 'white', margin: '40px auto', padding: 24, borderRadius: 8, maxWidth: 600 }}>
        <h2>{signalName} - Backtest (Last {history.length})</h2>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th>Date</th><th>Signal</th><th>Confidence</th><th>Result</th><th>Pips</th>
            </tr>
          </thead>
          <tbody>
            {history.map((row, i) => (
              <tr key={i}>
                <td>{row.date}</td>
                <td>{row.signal}</td>
                <td>{(row.confidence * 100).toFixed(1)}%</td>
                <td>{row.result}</td>
                <td>{row.pips}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <button onClick={onClose} style={{ marginTop: 16 }}>Close</button>
      </div>
    </div>
  );
}
