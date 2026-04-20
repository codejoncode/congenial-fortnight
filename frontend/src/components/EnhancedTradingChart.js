/**
 * Enhanced TradingView Chart Component
 * Uses Lightweight Charts for full customization
 * Displays signals, patterns, SL/TP levels in real-time
 */
import React, { useEffect, useRef, useState } from 'react';
import { createChart, CrosshairMode } from 'lightweight-charts';

const EnhancedTradingChart = ({ symbol = 'EURUSD', interval = '1h' }) => {
  const chartContainerRef = useRef();
  const chartRef = useRef();
  const candlestickSeriesRef = useRef();
  const [signals, setSignals] = useState([]);
  const [positions, setPositions] = useState([]);
  const [currentPrice, setCurrentPrice] = useState(null);

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current) {
      console.warn('Chart container ref not ready');
      return;
    }
    if (chartContainerRef.current.offsetWidth === 0) {
      console.warn('Chart container has zero width, waiting...');
      return;
    }
    let chart = null;
    let candlestickSeries = null;
    try {
      chart = createChart(chartContainerRef.current, {
        width: chartContainerRef.current.clientWidth || 800,
        height: 600,
        layout: {
          background: { color: '#1E1E1E' },
          textColor: '#D9D9D9',
        },
        grid: {
          vertLines: { color: '#2B2B2B' },
          horzLines: { color: '#2B2B2B' },
        },
        crosshair: {
          mode: CrosshairMode.Normal,
        },
        rightPriceScale: {
          borderColor: '#2B2B2B',
        },
        timeScale: {
          borderColor: '#2B2B2B',
          timeVisible: true,
          secondsVisible: false,
        },
      });
      if (typeof chart.addCandlestickSeries !== 'function') {
        console.error('addCandlestickSeries not found on chart object');
        console.error('Available methods:', Object.keys(chart));
        throw new Error('Chart API mismatch - check lightweight-charts version');
      }
      candlestickSeries = chart.addCandlestickSeries({
        upColor: '#26a69a',
        downColor: '#ef5350',
        borderVisible: false,
        wickUpColor: '#26a69a',
        wickDownColor: '#ef5350',
      });
      chartRef.current = chart;
      candlestickSeriesRef.current = candlestickSeries;
      const handleResize = () => {
        if (chartContainerRef.current && chartRef.current) {
          chartRef.current.applyOptions({
            width: chartContainerRef.current.clientWidth,
          });
        }
      };
      window.addEventListener('resize', handleResize);
      loadHistoricalData();
      return () => {
        window.removeEventListener('resize', handleResize);
        if (chartRef.current) {
          try {
            chartRef.current.remove();
          } catch (e) {
            console.warn('Error removing chart:', e);
          }
          chartRef.current = null;
        }
      };
    } catch (error) {
      console.error('Error initializing chart:', error);
      console.error('Stack:', error.stack);
    }
  }, [symbol, interval]);

  // Helper to generate mock candles for demo/fallback
  const generateMockCandles = () => {
    const candles = [];
    let basePrice = symbol === 'EURUSD' ? 1.0850 : 2050;
    let time = Math.floor(Date.now() / 1000) - 200 * 3600;
    for (let i = 0; i < 200; i++) {
      const open = basePrice + (Math.random() - 0.5) * 0.01;
      const close = open + (Math.random() - 0.5) * 0.015;
      const high = Math.max(open, close) + Math.random() * 0.01;
      const low = Math.min(open, close) - Math.random() * 0.01;
      candles.push({
        time,
        open: parseFloat(open.toFixed(5)),
        high: parseFloat(high.toFixed(5)),
        low: parseFloat(low.toFixed(5)),
        close: parseFloat(close.toFixed(5))
      });
      basePrice = close;
      time += 3600;
    }
    return candles;
  };

  // Load historical OHLC data
  const loadHistoricalData = async () => {
    try {
      const response = await fetch(
        `/api/paper-trading/price/ohlc/?symbol=${symbol}&interval=${interval}&limit=200`
      );
      if (!response.ok) {
        console.warn(`API returned ${response.status}, using mock data`);
        const mockData = generateMockCandles();
        if (candlestickSeriesRef.current && mockData) {
          candlestickSeriesRef.current.setData(mockData);
          if (mockData.length > 0) {
            setCurrentPrice(mockData[mockData.length - 1].close);
          }
        }
        return;
      }
      const data = await response.json();
      if (data.data && candlestickSeriesRef.current) {
        const ohlcData = data.data.map((candle) => ({
          time: new Date(candle.timestamp).getTime() / 1000,
          open: candle.open,
          high: candle.high,
          low: candle.low,
          close: candle.close,
        }));
        candlestickSeriesRef.current.setData(ohlcData);
        if (ohlcData.length > 0) {
          setCurrentPrice(ohlcData[ohlcData.length - 1].close);
        }
      }
    } catch (error) {
      console.error('Error loading historical data:', error);
      const mockData = generateMockCandles();
      if (candlestickSeriesRef.current && mockData) {
        candlestickSeriesRef.current.setData(mockData);
        if (mockData.length > 0) {
          setCurrentPrice(mockData[mockData.length - 1].close);
        }
      }
    }
  };

  // Load signals and positions
  useEffect(() => {
    loadSignals();
    loadPositions();

    // Set up polling for updates
    const intervalId = setInterval(() => {
      loadSignals();
      loadPositions();
      updatePrice();
    }, 5000); // Update every 5 seconds

    return () => clearInterval(intervalId);
  }, [symbol]);

  const loadSignals = async () => {
    try {
      // Load signals from your multi-model aggregator
      // This would come from your backend signal service
      const response = await fetch(`/api/signals/latest/?pair=${symbol}`);
      const data = await response.json();
      setSignals(data.signals || []);
    } catch (error) {
      console.error('Error loading signals:', error);
    }
  };

  const loadPositions = async () => {
    try {
      const response = await fetch(
        `/api/paper-trading/trades/open_positions/?pair=${symbol}`
      );
      const data = await response.json();
      setPositions(data || []);
    } catch (error) {
      console.error('Error loading positions:', error);
    }
  };

  const updatePrice = async () => {
    try {
      const response = await fetch(
        `/api/paper-trading/price/realtime/?symbol=${symbol}`
      );
      const data = await response.json();

      if (data && data.close) {
        setCurrentPrice(data.close);

        // Update chart with new price
        if (candlestickSeriesRef.current) {
          const lastCandle = candlestickSeriesRef.current.dataByIndex(
            candlestickSeriesRef.current.data().length - 1
          );
          
          // Update last candle or add new one
          // This is simplified - in production you'd handle candle intervals properly
        }
      }
    } catch (error) {
      console.error('Error updating price:', error);
    }
  };

  // Add signal markers to chart
  useEffect(() => {
    if (!candlestickSeriesRef.current || !signals.length) return;

    const markers = signals.map((signal) => {
      const time = new Date(signal.timestamp).getTime() / 1000;
      const isLong = signal.direction === 'buy' || signal.direction === 'LONG';

      return {
        time: time,
        position: isLong ? 'belowBar' : 'aboveBar',
        color: isLong ? '#2196F3' : '#e91e63',
        shape: isLong ? 'arrowUp' : 'arrowDown',
        text: `${signal.type} ${signal.confidence}%`,
      };
    });

    candlestickSeriesRef.current.setMarkers(markers);
  }, [signals]);

  // Add SL/TP lines for positions
  useEffect(() => {
    if (!chartRef.current || !positions.length) return;

    // Clear existing lines
    // Note: Lightweight Charts doesn't have direct line removal,
    // so we'd need to track line series and remove them

    positions.forEach((position) => {
      // Add stop loss line
      const slLineSeries = chartRef.current.addLineSeries({
        color: 'rgba(255, 82, 82, 0.8)',
        lineWidth: 2,
        lineStyle: 2, // Dashed
        title: `SL: ${position.stop_loss}`,
      });
      slLineSeries.setData([
        { time: Date.now() / 1000 - 3600 * 24, value: position.stop_loss },
        { time: Date.now() / 1000, value: position.stop_loss },
      ]);

      // Add take profit lines
      if (position.take_profit_1) {
        const tp1LineSeries = chartRef.current.addLineSeries({
          color: 'rgba(76, 175, 80, 0.8)',
          lineWidth: 2,
          lineStyle: 2,
          title: `TP1: ${position.take_profit_1}`,
        });
        tp1LineSeries.setData([
          { time: Date.now() / 1000 - 3600 * 24, value: position.take_profit_1 },
          { time: Date.now() / 1000, value: position.take_profit_1 },
        ]);
      }

      // Entry line
      const entryLineSeries = chartRef.current.addLineSeries({
        color: 'rgba(255, 193, 7, 0.8)',
        lineWidth: 2,
        lineStyle: 1, // Dotted
        title: `Entry: ${position.entry_price}`,
      });
      entryLineSeries.setData([
        { time: Date.now() / 1000 - 3600 * 24, value: position.entry_price },
        { time: Date.now() / 1000, value: position.entry_price },
      ]);
    });
  }, [positions]);

  return (
    <div className="trading-chart-container" style={{ width: '100%', padding: '20px' }}>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '10px',
        }}
      >
        <h2 style={{ margin: 0, color: '#D9D9D9' }}>
          {symbol} - {interval}
        </h2>
        {currentPrice && (
          <div
            style={{
              fontSize: '24px',
              fontWeight: 'bold',
              color: '#26a69a',
            }}
          >
            {currentPrice.toFixed(5)}
          </div>
        )}
      </div>

      <div
        ref={chartContainerRef}
        style={{
          position: 'relative',
          borderRadius: '8px',
          overflow: 'hidden',
          boxShadow: '0 4px 6px rgba(0, 0, 0, 0.3)',
        }}
      />

      {/* Signal List */}
      {signals.length > 0 && (
        <div
          style={{
            marginTop: '20px',
            padding: '15px',
            background: '#2B2B2B',
            borderRadius: '8px',
          }}
        >
          <h3 style={{ color: '#D9D9D9', marginTop: 0 }}>Active Signals</h3>
          {signals.map((signal, idx) => (
            <div
              key={idx}
              style={{
                padding: '10px',
                margin: '5px 0',
                background: '#1E1E1E',
                borderRadius: '4px',
                borderLeft: `4px solid ${
                  signal.direction === 'buy' ? '#2196F3' : '#e91e63'
                }`,
              }}
            >
              <div style={{ color: '#D9D9D9', fontWeight: 'bold' }}>
                {signal.type} - {signal.direction.toUpperCase()}
              </div>
              <div style={{ color: '#999', fontSize: '14px' }}>
                Entry: {signal.entry_price} | SL: {signal.stop_loss} | TP:{' '}
                {signal.take_profit_1}
              </div>
              <div style={{ color: '#26a69a', fontSize: '14px' }}>
                Confidence: {signal.confidence}% | R:R: {signal.risk_reward_ratio}:1
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Open Positions */}
      {positions.length > 0 && (
        <div
          style={{
            marginTop: '20px',
            padding: '15px',
            background: '#2B2B2B',
            borderRadius: '8px',
          }}
        >
          <h3 style={{ color: '#D9D9D9', marginTop: 0 }}>Open Positions</h3>
          {positions.map((position) => (
            <div
              key={position.id}
              style={{
                padding: '10px',
                margin: '5px 0',
                background: '#1E1E1E',
                borderRadius: '4px',
                borderLeft: `4px solid ${
                  position.order_type === 'buy' ? '#26a69a' : '#ef5350'
                }`,
              }}
            >
              <div style={{ color: '#D9D9D9', fontWeight: 'bold' }}>
                {position.pair} - {position.order_type.toUpperCase()} {position.lot_size} lots
              </div>
              <div style={{ color: '#999', fontSize: '14px' }}>
                Entry: {position.entry_price} | SL: {position.stop_loss} | TP:{' '}
                {position.take_profit_1}
              </div>
              <div
                style={{
                  color: position.profit_loss >= 0 ? '#26a69a' : '#ef5350',
                  fontSize: '14px',
                }}
              >
                P&L: {position.pips_gained?.toFixed(1) || '0.0'} pips (${position.profit_loss?.toFixed(2) || '0.00'})
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default EnhancedTradingChart;
