import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [signals, setSignals] = useState([]);
  const [backtestResults, setBacktestResults] = useState(null);
  const [showBacktest, setShowBacktest] = useState(false);
  const [backtestPair, setBacktestPair] = useState('EURUSD');
  const [backtestDays, setBacktestDays] = useState(30);

  useEffect(() => {
    fetchSignals();
  }, []);

  const fetchSignals = () => {
    axios.get('http://localhost:8000/api/signals/')
      .then(response => {
        setSignals(response.data);
      })
      .catch(error => {
        console.error('Error fetching signals:', error);
      });
  };

  const runBacktest = () => {
    axios.get(`http://localhost:8000/api/backtest/?pair=${backtestPair}&days=${backtestDays}`)
      .then(response => {
        setBacktestResults(response.data);
      })
      .catch(error => {
        console.error('Error running backtest:', error);
      });
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Forex Signals</h1>
        
        <button onClick={() => setShowBacktest(!showBacktest)}>
          {showBacktest ? 'Hide Backtest' : 'Show Backtest Results'}
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
            </div>
            
            {backtestResults && !backtestResults.error && (
              <div>
                <h3>{backtestResults.pair} - {backtestResults.days} Days Backtest</h3>
                <p>Overall Accuracy: {(backtestResults.overall_accuracy * 100).toFixed(1)}%</p>
                <p>Total Signals: {backtestResults.total_signals}</p>
                
                <h4>Accuracy by Probability Range</h4>
                <ul>
                  {Object.entries(backtestResults.probability_bins).map(([range, data]) => (
                    <li key={range}>
                      {range}: {data.accuracy ? (data.accuracy * 100).toFixed(1) : 0}% ({data.correct}/{data.total} correct)
                    </li>
                  ))}
                </ul>
                
                <h4>Recent Results</h4>
                <ul>
                  {backtestResults.recent_results.map((result, index) => (
                    <li key={index}>
                      {result.date}: Signal {result.signal}, Actual {result.actual}, 
                      Correct: {result.correct ? 'Yes' : 'No'}, Prob: {(result.probability * 100).toFixed(1)}%
                    </li>
                  ))}
                </ul>
              </div>
            )}
            
            {backtestResults && backtestResults.error && (
              <p>Error: {backtestResults.error}</p>
            )}
          </div>
        )}
        
        <button onClick={fetchSignals}>Refresh Signals</button>
      </header>
    </div>
  );
}

export default App;
