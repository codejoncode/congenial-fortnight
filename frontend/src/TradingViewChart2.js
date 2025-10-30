import React, { useEffect, useRef, useState } from 'react';

/*
TRADINGVIEW-LIKE CHART COMPONENT

This component demonstrates how to integrate TradingView charts in React.
For production use, install: npm install react-tradingview-widget

Usage:
1. Install: npm install react-tradingview-widget
2. Import: import TradingViewWidget from 'react-tradingview-widget';
3. Use: <TradingViewWidget symbol="FX:EURUSD" theme="dark" locale="en" />

For lightweight alternative, consider:
npm install lightweight-charts

This provides TradingView-like functionality with better performance.
*/

const TradingViewChart = ({ pair = 'EURUSD', darkMode = false }) => {
  const containerRef = useRef(null);
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    if (!containerRef.current) return;

    // Clear any existing content
    containerRef.current.innerHTML = '';
    setIsLoaded(false);

    // Create TradingView widget script
    const script = document.createElement("script");
    script.type = "text/javascript";
    script.async = true;
    script.src = "https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js";
    
    const config = {
      "autosize": true,
      "symbol": pair === 'XAUUSD' ? 'FX:XAUUSD' : 'FX:EURUSD',
      "interval": "15",
      "timezone": "Etc/UTC",
      "theme": darkMode ? "dark" : "light",
      "style": "1",
      "locale": "en",
      "toolbar_bg": darkMode ? "#1e222d" : "#f1f3f6",
      "enable_publishing": false,
      "withdateranges": true,
      "range": "3M",
      "hide_side_toolbar": false,
      "allow_symbol_change": true,
      "save_image": false,
      "calendar": false,
      "support_host": "https://www.tradingview.com"
    };

    script.innerHTML = JSON.stringify(config);

    // Create widget container
    const widgetContainer = document.createElement('div');
    widgetContainer.className = 'tradingview-widget-container';
    widgetContainer.style.height = '100%';
    widgetContainer.style.width = '100%';

    const widgetDiv = document.createElement('div');
    widgetDiv.className = 'tradingview-widget-container__widget';
    widgetDiv.style.height = 'calc(100% - 32px)';
    widgetDiv.style.width = '100%';

    widgetContainer.appendChild(widgetDiv);
    widgetContainer.appendChild(script);

    // Show loading state first
    const loadingDiv = document.createElement('div');
    loadingDiv.innerHTML = `
      <div style="
        width: 100%;
        height: 500px;
        background: ${darkMode ? 'linear-gradient(135deg, #1e222d 0%, #2a2e39 100%)' : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'};
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
      ">
        <div>
          <div style="font-size: 48px; margin-bottom: 20px;">📊</div>
          <div>Loading TradingView Chart for ${pair}</div>
          <div style="font-size: 14px; margin-top: 15px; opacity: 0.8; max-width: 400px;">
            Professional trading interface with advanced indicators, 
            drawing tools, and real-time market data integration.
          </div>
          <div style="margin-top: 20px; display: flex; gap: 10px; flex-wrap: wrap; justify-content: center;">
            ${['Candlestick', 'Volume', 'RSI', 'MACD', 'Moving Averages'].map(item => 
              `<span style="
                padding: 5px 12px; 
                background: rgba(255,255,255,0.2); 
                border-radius: 15px; 
                font-size: 12px;
              ">${item}</span>`
            ).join('')}
          </div>
          <div style="font-size: 14px; margin-top: 15px; opacity: 0.6;">
            Loading professional TradingView widget...
          </div>
        </div>
      </div>
    `;

    containerRef.current.appendChild(loadingDiv);

    // Try to load TradingView widget after a delay
    setTimeout(() => {
      if (containerRef.current) {
        try {
          containerRef.current.innerHTML = '';
          containerRef.current.appendChild(widgetContainer);
          setIsLoaded(true);
        } catch (error) {
          console.log('TradingView widget loading in progress...');
        }
      }
    }, 2000);

    return () => {
      if (containerRef.current) {
        containerRef.current.innerHTML = '';
      }
    };
  }, [pair, darkMode]);

  return (
    <div style={{ 
      height: '500px', 
      width: '100%', 
      position: 'relative',
      backgroundColor: darkMode ? '#131722' : '#ffffff',
      borderRadius: '8px',
      overflow: 'hidden'
    }}>
      <div ref={containerRef} style={{ height: '100%', width: '100%' }} />
      
      {/* Overlay with chart info */}
      <div style={{
        position: 'absolute',
        top: '10px',
        left: '10px',
        backgroundColor: darkMode ? 'rgba(0,0,0,0.7)' : 'rgba(255,255,255,0.9)',
        color: darkMode ? '#fff' : '#333',
        padding: '8px 12px',
        borderRadius: '6px',
        fontSize: '12px',
        fontWeight: 'bold',
        zIndex: 10
      }}>
        {pair} • TradingView Integration
      </div>
    </div>
  );
};

export default TradingViewChart;