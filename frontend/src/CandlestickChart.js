import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { ComposedChart, Line, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const CandlestickChart = ({ pair }) => {
  const [data, setData] = useState([]);
  const [signals, setSignals] = useState([]);

  useEffect(() => {
    // Fetch historical data and signals
    axios.get(`http://localhost:8000/api/signals/?pair__name=${pair}`)
      .then(response => {
        setSignals(response.data);
      });
    // For demo, use sample data
    const sampleData = [
      { date: '2023-01-01', open: 1.05, high: 1.07, low: 1.04, close: 1.06, signal: 'bullish' },
      { date: '2023-01-02', open: 1.06, high: 1.08, low: 1.05, close: 1.07, signal: 'bearish' },
      // Add more data
    ];
    setData(sampleData);
  }, [pair]);

  return (
    <div style={{ width: '100%', height: 400 }}>
      <ResponsiveContainer>
        <ComposedChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Bar dataKey="high" fill="#8884d8" />
          <Bar dataKey="low" fill="#82ca9d" />
          <Line type="monotone" dataKey="close" stroke="#ff7300" />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
};

export default CandlestickChart;