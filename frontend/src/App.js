import React, { useState, useEffect } from 'react';
import axios from 'axios';
import CandlestickChart from './CandlestickChart';
import './App.css';

// API configuration
const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? 'https://congenial-fortnight-1034520618737.europe-west1.run.app'
  : 'http://localhost:8000';

function App() {
  const [signals, setSignals] = useState([]);
  const [backtestResults, setBacktestResults] = useState(null);
  const [showBacktest, setShowBacktest] = useState(false);
  const [showChart, setShowChart] = useState(false);
  const [backtestPair, setBacktestPair] = useState('EURUSD');
  const [chartPair, setChartPair] = useState('EURUSD');
  const [backtestDays, setBacktestDays] = useState(30);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchSignals();
  }, []);

  const fetchSignals = () => {
    axios.get(`${API_BASE_URL}/api/signals/`)
      .then(response => {
        setSignals(response.data);
      })
      .catch(error => {
        console.error('Error fetching signals:', error);
      });
  };

  const downloadCSV = () => {
    const url = `${API_BASE_URL}/api/backtest/csv/?pair=${backtestPair}&days=${backtestDays}`;
    window.open(url, '_blank');
  };

  const runBacktest = () => {
    setLoading(true);
    axios.get(`${API_BASE_URL}/api/backtest/?pair=${backtestPair}&days=${backtestDays}`)
      .then(response => {
        setBacktestResults(response.data);
        setLoading(false);
      })
      .catch(error => {
        console.error('Error running backtest:', error);
        setLoading(false);
      });
  };

  const refreshData = () => {
    fetchSignals();
    if (showBacktest && backtestResults) {
      runBacktest();
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Forex Signals</h1>
        
        <button onClick={() => setShowBacktest(!showBacktest)}>
          {showBacktest ? 'Hide Backtest' : 'Show Backtest Results'}
        </button>
        <button onClick={() => setShowChart(!showChart)} style={{marginLeft: '10px'}}>
          {showChart ? 'Hide Chart' : 'Show Chart'}
        </button>
        
        {!showBacktest ? (
          <>
            <h2>Current Signals</h2>
            <ul>
              {signals.map(signal => (
                <li key={signal.id}>
                  {signal.pair}: {signal.signal} (Prob: {(signal.probability * 100).toFixed(1)}%, SL: {signal.stop_loss?.toFixed(4)}, Date: {signal.date})
                </li>
              ))}
            </ul>
          </>
        ) : (
          <div>
            <h2>Backtest Analysis</h2>
            <div>
              <label>Pair: </label>
              <select value={backtestPair} onChange={(e) => setBacktestPair(e.target.value)}>
                <option value="EURUSD">EURUSD</option>
                <option value="XAUUSD">XAUUSD</option>
                <option value="GBPUSD">GBPUSD</option>
                <option value="USDJPY">USDJPY</option>
                <option value="AUDUSD">AUDUSD</option>
                <option value="USDCAD">USDCAD</option>
                <option value="USDCHF">USDCHF</option>
                <option value="NZDUSD">NZDUSD</option>
              </select>
              <label> Days: </label>
              <input 
                type="number" 
                value={backtestDays} 
                onChange={(e) => setBacktestDays(e.target.value)}
                min="10" 
                max="365"
              />
              <button onClick={runBacktest}>Run Backtest</button>
              {backtestResults && !backtestResults.error && (
                <button onClick={downloadCSV} style={{marginLeft: '10px', backgroundColor: '#28a745', color: 'white'}}>
                  📥 Download CSV
                </button>
              )}
            </div>
            
            {backtestResults && !backtestResults.error && (
              <div>
                <h3>{backtestResults.pair} - {backtestResults.days} Days Enhanced Backtest</h3>

                {/* Overall Statistics */}
                <div style={{backgroundColor: '#f0f0f0', padding: '10px', margin: '10px 0', borderRadius: '5px'}}>
                  <h4>Overall Performance</h4>
                  <p>Accuracy: {(backtestResults.overall_accuracy * 100).toFixed(1)}%</p>
                  <p>Total Signals: {backtestResults.total_signals}</p>
                  <p>Wins: {backtestResults.wins} | Losses: {backtestResults.losses}</p>
                </div>

                {/* Pips Analysis */}
                <div style={{backgroundColor: '#e8f4f8', padding: '10px', margin: '10px 0', borderRadius: '5px'}}>
                  <h4>💰 Pips Analysis</h4>
                  <p>Total Pips Won: <span style={{color: 'green'}}>{backtestResults.total_pips_won?.toFixed(1)}</span></p>
                  <p>Total Pips Lost: <span style={{color: 'red'}}>{Math.abs(backtestResults.total_pips_lost)?.toFixed(1)}</span></p>
                  <p><strong>Net Pips: <span style={{color: backtestResults.net_pips >= 0 ? 'green' : 'red'}}>{backtestResults.net_pips?.toFixed(1)}</span></strong></p>
                  <p>Average Win: {backtestResults.avg_win_pips?.toFixed(2)} pips</p>
                  <p>Average Loss: {Math.abs(backtestResults.avg_loss_pips)?.toFixed(2)} pips</p>
                  <p>Profit Factor: {backtestResults.profit_factor?.toFixed(2)}</p>
                  <p>Largest Win: {backtestResults.largest_win?.toFixed(1)} pips</p>
                  <p>Largest Loss: {Math.abs(backtestResults.largest_loss)?.toFixed(1)} pips</p>
                </div>

                {/* Probability Analysis */}
                <h4>🎯 Accuracy by Probability Range</h4>
                <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '10px'}}>
                  {Object.entries(backtestResults.probability_bins).map(([range, data]) => (
                    <div key={range} style={{
                      border: '1px solid #ddd',
                      padding: '8px',
                      borderRadius: '4px',
                      backgroundColor: data.accuracy > 0.6 ? '#d4edda' : data.accuracy > 0.5 ? '#fff3cd' : '#f8d7da'
                    }}>
                      <strong>{range}:</strong><br/>
                      {data.total > 0 ? (data.accuracy * 100).toFixed(1) : 0}% accuracy<br/>
                      ({data.correct}/{data.total} correct)
                    </div>
                  ))}
                </div>

                {/* Recent Trades */}
                <h4>📊 Recent Trade Details</h4>
                <div style={{maxHeight: '300px', overflowY: 'auto', border: '1px solid #ddd', borderRadius: '4px'}}>
                  <table style={{width: '100%', borderCollapse: 'collapse'}}>
                    <thead>
                      <tr style={{backgroundColor: '#f8f9fa'}}>
                        <th style={{padding: '8px', border: '1px solid #ddd'}}>Date</th>
                        <th style={{padding: '8px', border: '1px solid #ddd'}}>Signal</th>
                        <th style={{padding: '8px', border: '1px solid #ddd'}}>Probability</th>
                        <th style={{padding: '8px', border: '1px solid #ddd'}}>Result</th>
                        <th style={{padding: '8px', border: '1px solid #ddd'}}>Pips</th>
                      </tr>
                    </thead>
                    <tbody>
                      {backtestResults.recent_results?.map((result, index) => (
                        <tr key={index} style={{
                          backgroundColor: result.correct ? '#d4edda' : '#f8d7da'
                        }}>
                          <td style={{padding: '8px', border: '1px solid #ddd'}}>{result.date}</td>
                          <td style={{padding: '8px', border: '1px solid #ddd'}}>{result.signal}</td>
                          <td style={{padding: '8px', border: '1px solid #ddd'}}>{(result.probability * 100).toFixed(1)}%</td>
                          <td style={{padding: '8px', border: '1px solid #ddd'}}>
                            {result.correct ? '✅ Win' : '❌ Loss'}
                          </td>
                          <td style={{
                            padding: '8px',
                            border: '1px solid #ddd',
                            color: result.pips >= 0 ? 'green' : 'red',
                            fontWeight: 'bold'
                          }}>
                            {result.pips?.toFixed(1)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
            
            {backtestResults && backtestResults.error && (
              <p>Error: {backtestResults.error}</p>
            )}
          </div>
        )}

        {showChart && (
          <div style={{marginTop: '20px', padding: '20px', border: '1px solid #ddd', borderRadius: '5px'}}>
            <h2>Price Chart</h2>
            <div style={{marginBottom: '10px'}}>
              <label>Pair: </label>
              <select value={chartPair} onChange={(e) => setChartPair(e.target.value)}>
                <option value="EURUSD">EURUSD</option>
                <option value="XAUUSD">XAUUSD</option>
                <option value="GBPUSD">GBPUSD</option>
                <option value="USDJPY">USDJPY</option>
              </select>
            </div>
            <CandlestickChart pair={chartPair} />
          </div>
        )}
        
        <button onClick={fetchSignals} style={{marginTop: '20px'}}>Refresh Signals</button>
      </header>
    </div>
  );
}

export default App;
