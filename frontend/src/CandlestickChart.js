import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { ComposedChart, Line, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// API configuration
const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? 'https://congenial-fortnight-1034520618737.europe-west1.run.app'
  : 'http://localhost:8000';

const CandlestickChart = ({ pair }) => {
  const [data, setData] = useState([]);
  const [signals, setSignals] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchChartData();
  }, [pair]);

  const fetchChartData = async () => {
    try {
      setLoading(true);
      // Fetch signals for the pair
      const signalsResponse = await axios.get(`${API_BASE_URL}/api/signals/`);
      setSignals(signalsResponse.data);

      // For now, use sample data - in production, you'd fetch real price data
      const sampleData = [
        { date: '2024-01-01', open: 1.05, high: 1.07, low: 1.04, close: 1.06, signal: null },
        { date: '2024-01-02', open: 1.06, high: 1.08, low: 1.05, close: 1.07, signal: 'bullish' },
        { date: '2024-01-03', open: 1.07, high: 1.09, low: 1.06, close: 1.08, signal: null },
        { date: '2024-01-04', open: 1.08, high: 1.10, low: 1.07, close: 1.09, signal: 'bearish' },
        { date: '2024-01-05', open: 1.09, high: 1.11, low: 1.08, close: 1.10, signal: null },
      ];
      setData(sampleData);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching chart data:', error);
      setLoading(false);
    }
  };

  if (loading) {
    return <div>Loading chart...</div>;
  }

  return (
    <div style={{ width: '100%', height: 400 }}>
      <ResponsiveContainer>
        <ComposedChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis domain={['dataMin - 0.01', 'dataMax + 0.01']} />
          <Tooltip 
            formatter={(value, name) => [value.toFixed(4), name]}
            labelFormatter={(label) => `Date: ${label}`}
          />
          <Legend />
          <Bar dataKey="high" fill="#8884d8" name="High" />
          <Bar dataKey="low" fill="#82ca9d" name="Low" />
          <Line type="monotone" dataKey="close" stroke="#ff7300" strokeWidth={2} name="Close" />
          <Line type="monotone" dataKey="open" stroke="#00ff00" strokeWidth={1} name="Open" />
        </ComposedChart>
      </ResponsiveContainer>
      
      {signals.length > 0 && (
        <div style={{marginTop: '10px', fontSize: '12px'}}>
          <strong>Recent Signals:</strong>
          <ul>
            {signals.slice(0, 3).map((signal, index) => (
              <li key={index}>
                {signal.pair}: {signal.signal} ({(signal.probability * 100).toFixed(1)}%)
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default CandlestickChart;