import React, { useEffect, useState, useCallback } from 'react';
import axios from 'axios';
import { ComposedChart, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Bar, ReferenceLine } from 'recharts';

// Custom Candlestick shape
const Candlestick = (props) => {
  const { payload, x, y, width, height, isPrediction } = props;
  
  if (!payload || !payload.open || !payload.high || !payload.low || !payload.close) {
    return null;
  }

  const { open, high, low, close } = payload;
  const isBullish = close > open;
  
  // In recharts Bar custom shape, y and height are already scaled
  // We need to calculate relative positions within the bar's allocated space
  const priceRange = Math.max(open, high, low, close) - Math.min(open, high, low, close);
  if (priceRange === 0) return null;
  
  // Calculate relative positions within the bar
  const highPos = ((Math.max(open, high, low, close) - high) / priceRange) * height;
  const lowPos = ((Math.max(open, high, low, close) - low) / priceRange) * height;
  const openPos = ((Math.max(open, high, low, close) - open) / priceRange) * height;
  const closePos = ((Math.max(open, high, low, close) - close) / priceRange) * height;
  
  const centerX = x + width / 2;
  const wickY1 = y + highPos;
  const wickY2 = y + lowPos;
  
  // Body position and height
  const bodyY = y + Math.min(openPos, closePos);
  const bodyHeight = Math.abs(closePos - openPos) || 1;

  // Prediction candle styling
  const strokeColor = isPrediction ? '#FFD700' : (isBullish ? "#16a34a" : "#dc2626");
  const fillColor = isPrediction ? 'transparent' : (isBullish ? "#22c55e" : "#ef4444");
  const strokeWidth = isPrediction ? 3 : 1;

  return (
    <g>
      {/* Star above prediction candle */}
      {isPrediction && (
        <text
          x={centerX}
          y={y - 10}
          textAnchor="middle"
          fontSize="16"
          fill="#FFD700"
          fontWeight="bold"
        >
          ★
        </text>
      )}
      
      {/* Wick (high to low line) */}
      <line
        x1={centerX}
        y1={wickY1}
        x2={centerX}
        y2={wickY2}
        stroke={strokeColor}
        strokeWidth={strokeWidth}
      />
      {/* Body (open to close rectangle) */}
      <rect
        x={x + width * 0.15}
        y={bodyY}
        width={width * 0.7}
        height={bodyHeight}
        fill={fillColor}
        stroke={strokeColor}
        strokeWidth={strokeWidth}
      />
    </g>
  );
};

// API configuration
const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? 'https://congenial-fortnight-1034520618737.europe-west1.run.app'
  : 'http://localhost:8000';

const CandlestickChart = ({ pair }) => {
  const [data, setData] = useState([]);
  const [signals, setSignals] = useState([]);
  const [loading, setLoading] = useState(true);

  const fetchChartData = useCallback(async () => {
    try {
      setLoading(true);
      
      // Fetch historical price data
      const [priceResponse, signalsResponse] = await Promise.all([
        axios.get(`${API_BASE_URL}/api/historical/?pair=${pair}&days=30`),
        axios.get(`${API_BASE_URL}/api/signals/`)
      ]);

      const priceData = priceResponse.data;
      const signalsData = signalsResponse.data;

      // Filter signals for this pair
      const pairSignals = signalsData.filter(signal => signal.pair === pair);

      // Combine price data with signals and add prediction candle
      const chartData = priceData.map(item => {
        // Find if there's a signal for this date
        const signalForDate = pairSignals.find(signal => 
          signal.date && signal.date.startsWith(item.date)
        );
        
        return {
          ...item,
          signal: signalForDate ? signalForDate.signal : null,
          probability: signalForDate ? signalForDate.probability : null,
          isPrediction: false
        };
      });

      // Add prediction candle for tomorrow
      if (chartData.length > 0) {
        const lastCandle = chartData[chartData.length - 1];
        const lastSignal = pairSignals.find(signal => 
          signal.date && new Date(signal.date) > new Date(lastCandle.date)
        );
        
        if (lastSignal) {
          // Calculate prediction candle based on signal
          const basePrice = lastCandle.close;
          const volatility = Math.abs(lastCandle.high - lastCandle.low) / lastCandle.close;
          const predictionRange = basePrice * volatility * 0.5; // 50% of typical range
          
          let predictedOpen, predictedClose, predictedHigh, predictedLow;
          
          if (lastSignal.signal === 'bullish') {
            predictedOpen = basePrice;
            predictedClose = basePrice + predictionRange;
            predictedHigh = predictedClose + (predictionRange * 0.2);
            predictedLow = Math.max(basePrice - (predictionRange * 0.1), basePrice * 0.995);
          } else {
            predictedOpen = basePrice;
            predictedClose = basePrice - predictionRange;
            predictedLow = predictedClose - (predictionRange * 0.2);
            predictedHigh = Math.min(basePrice + (predictionRange * 0.1), basePrice * 1.005);
          }

          const tomorrow = new Date(lastCandle.date);
          tomorrow.setDate(tomorrow.getDate() + 1);
          
          chartData.push({
            date: tomorrow.toISOString().split('T')[0],
            open: predictedOpen,
            high: predictedHigh,
            low: predictedLow,
            close: predictedClose,
            signal: lastSignal.signal,
            probability: lastSignal.probability,
            isPrediction: true,
            volume: 0
          });
        }
      }

      setData(chartData);
      setSignals(pairSignals);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching chart data:', error);
      // Fallback to sample data if API fails
      const sampleData = [
        { date: '2025-09-24', open: 1.18117, high: 1.18184, low: 1.17279, close: 1.17370, signal: null, isPrediction: false },
        { date: '2025-09-25', open: 1.17358, high: 1.17541, low: 1.16452, close: 1.16645, signal: 'bullish', isPrediction: false },
        { date: '2025-09-26', open: 1.16570, high: 1.17067, low: 1.16510, close: 1.16990, signal: null, isPrediction: false },
        { date: '2025-09-29', open: 1.16993, high: 1.17547, low: 1.16978, close: 1.17261, signal: null, isPrediction: false },
        { date: '2025-10-01', open: 1.08500, high: 1.09200, low: 1.08200, close: 1.08800, signal: 'bearish', isPrediction: false },
        { date: '2025-10-02', open: 1.08800, high: 1.09500, low: 1.08000, close: 1.08200, signal: 'bearish', probability: 0.232, isPrediction: true },
      ];
      setData(sampleData);
      setLoading(false);
    }
  }, [pair]);

  useEffect(() => {
    fetchChartData();
  }, [fetchChartData]);

  if (loading) {
    return <div>Loading chart...</div>;
  }

  return (
    <div style={{ width: '100%', minWidth: '1200px', height: 600, margin: '20px auto', padding: '20px', backgroundColor: '#f8f9fa', borderRadius: '10px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart 
          data={data}
          margin={{ top: 20, right: 80, left: 20, bottom: 60 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
          <XAxis 
            dataKey="date" 
            tick={{ fontSize: 12, fill: '#666' }}
            angle={-45}
            textAnchor="end"
            height={80}
            interval="preserveStartEnd"
          />
          <YAxis 
            domain={[
              (dataMin) => dataMin - (Math.abs(dataMin) * 0.02), // 2% padding below
              (dataMax) => dataMax + (Math.abs(dataMax) * 0.02)  // 2% padding above
            ]}
            tick={{ fontSize: 12, fill: '#666' }}
            tickFormatter={(value) => value.toFixed(4)}
          />
          <Tooltip 
            contentStyle={{
              backgroundColor: '#fff',
              border: '1px solid #ddd',
              borderRadius: '8px',
              boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
            }}
            formatter={(value, name, props) => {
              if (name === 'OHLC') {
                const { payload } = props;
                const signalInfo = payload.signal ? 
                  `\nSignal: ${payload.signal} (${(payload.probability * 100).toFixed(1)}%)` : '';
                const predictionInfo = payload.isPrediction ? '\n⭐ PREDICTION' : '';
                return [
                  `O: ${payload.open?.toFixed(4)}, H: ${payload.high?.toFixed(4)}, L: ${payload.low?.toFixed(4)}, C: ${payload.close?.toFixed(4)}${signalInfo}${predictionInfo}`,
                  'Price'
                ];
              }
              return [value?.toFixed(4), name];
            }}
            labelFormatter={(label) => `Date: ${label}`}
          />
          <Legend 
            wrapperStyle={{ paddingTop: '20px' }}
            iconType="rect"
          />
          <Bar 
            dataKey="close" 
            fill="transparent" 
            shape={(props) => <Candlestick {...props} isPrediction={props.payload?.isPrediction} />}
            name="OHLC"
          />
          {data.length > 0 && data.some(d => d.isPrediction) && (
            <ReferenceLine 
              x={data.find(d => d.isPrediction)?.date} 
              stroke="#FFD700" 
              strokeDasharray="5 5"
              strokeWidth={2}
              label={{ value: "🎯 Prediction", position: "topRight", fill: "#FFD700", fontWeight: "bold" }}
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>
      
      {signals.length > 0 && (
        <div style={{
          marginTop: '20px', 
          padding: '15px', 
          backgroundColor: '#fff', 
          borderRadius: '8px',
          border: '1px solid #e0e0e0',
          fontSize: '14px'
        }}>
          <strong style={{ color: '#333', marginBottom: '10px', display: 'block' }}>📊 Recent Signals for {pair}:</strong>
          <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
            {signals.slice(0, 5).map((signal, index) => (
              <div key={index} style={{
                padding: '8px 12px',
                backgroundColor: signal.signal === 'bullish' ? '#d4edda' : '#f8d7da',
                border: `1px solid ${signal.signal === 'bullish' ? '#c3e6cb' : '#f5c6cb'}`,
                borderRadius: '6px',
                fontSize: '13px'
              }}>
                <div style={{ fontWeight: 'bold', color: signal.signal === 'bullish' ? '#155724' : '#721c24' }}>
                  {signal.signal.toUpperCase()}
                </div>
                <div>{(signal.probability * 100).toFixed(1)}% confidence</div>
                <div style={{ fontSize: '11px', color: '#666' }}>{signal.date}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default CandlestickChart;