"""
Complete Financial Freedom Trading System - Data Collection Module
Free API Implementation for Google Cloud Run Deployment
"""

import os
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging
import time
from typing import Dict, List, Optional

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingDataCollector:
    """Complete data collection system using free APIs"""

    def __init__(self):
        self.api_keys = {
            'fred': os.getenv('FRED_API_KEY'),
            'finnhub': os.getenv('FINNHUB_API_KEY'),
            'fmp': os.getenv('FMP_API_KEY'),
            'alpha_vantage': os.getenv('ALPHA_VANTAGE_API_KEY'),
            'api_ninjas': os.getenv('API_NINJAS_API_KEY')
        }

        # Rate limiting trackers
        self.api_calls = {
            'fred': 0,
            'finnhub': 0,
            'fmp': 0,
            'yahoo': 0,
            'ecb': 0
        }

        self.daily_limits = {
            'fred': float('inf'),  # unlimited
            'finnhub': 100,
            'fmp': 250,
            'yahoo': 100,
            'ecb': float('inf')
        }

    def _check_rate_limit(self, api_name: str) -> bool:
        """Check if we're within rate limits"""
        if self.api_calls[api_name] >= self.daily_limits[api_name]:
            logger.warning(f"Rate limit reached for {api_name}")
            return False
        return True

    def _increment_call(self, api_name: str):
        """Increment API call counter"""
        self.api_calls[api_name] += 1

    def collect_fred_data(self, series_ids: List[str]) -> Dict[str, pd.DataFrame]:
        """Collect data from FRED API (unlimited free)"""
        if not self._check_rate_limit('fred'):
            return {}

        base_url = "https://api.stlouisfed.org/fred/series/observations"
        results = {}

        for series_id in series_ids:
            params = {
                'series_id': series_id,
                'api_key': self.api_keys['fred'],
                'file_type': 'json',
                'observation_start': (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
            }

            try:
                response = requests.get(base_url, params=params)
                response.raise_for_status()
                data = response.json()

                df = pd.DataFrame(data['observations'])
                df['date'] = pd.to_datetime(df['date'])
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                df = df.set_index('date')[['value']].rename(columns={'value': series_id})

                results[series_id] = df
                self._increment_call('fred')
                time.sleep(0.1)  # Rate limiting

            except Exception as e:
                logger.error(f"Error collecting FRED data for {series_id}: {e}")

        return results

    def collect_yahoo_finance_data(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """Collect data from Yahoo Finance (100 calls/day free)"""
        if not self._check_rate_limit('yahoo'):
            return {}

        results = {}

        for ticker in tickers:
            try:
                # Download last 2 years of data
                df = yf.download(ticker, period="2y", interval="1d")
                if not df.empty:
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                    results[ticker] = df
                    self._increment_call('yahoo')
                    time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error collecting Yahoo data for {ticker}: {e}")

        return results

    def collect_finnhub_data(self) -> pd.DataFrame:
        """Collect economic calendar from Finnhub (100 calls/day free)"""
        if not self._check_rate_limit('finnhub'):
            return pd.DataFrame()

        base_url = "https://finnhub.io/api/v1/calendar/economic"
        params = {
            'token': self.api_keys['finnhub'],
            'from': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
            'to': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

            if 'economicCalendar' in data:
                df = pd.DataFrame(data['economicCalendar'])
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                self._increment_call('finnhub')
                return df

        except Exception as e:
            logger.error(f"Error collecting Finnhub data: {e}")

        return pd.DataFrame()

    def collect_fmp_data(self, symbol: str) -> pd.DataFrame:
        """Collect COT data from Financial Modeling Prep (250 calls/day free)"""
        if not self._check_rate_limit('fmp'):
            return pd.DataFrame()

        # Note: FMP may not have direct COT data, this is a placeholder
        # In practice, you'd need to find the correct endpoint
        url = f"https://financialmodelingprep.com/api/v4/commitment_of_traders_report/{symbol}"

        try:
            response = requests.get(url, params={'apikey': self.api_keys['fmp']})
            response.raise_for_status()
            data = response.json()

            if data:
                df = pd.DataFrame(data)
                self._increment_call('fmp')
                return df

        except Exception as e:
            logger.error(f"Error collecting FMP data for {symbol}: {e}")

        return pd.DataFrame()

    def collect_ecb_data(self) -> pd.DataFrame:
        """Collect ECB data from ECB Statistical Data Warehouse"""
        try:
            # Use a simpler ECB endpoint that should work
            url = "https://data-api.ecb.europa.eu/service/data/MIR/MIR_1Y_1_0_0_0_0_0_0_0_0_0?format=jsondata&startPeriod=2020-01-01"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            if 'dataSets' in data and data['dataSets']:
                observations = data['dataSets'][0]['observations']
                dates = []
                values = []

                for key, value_data in observations.items():
                    if value_data and len(value_data) > 0:
                        # Parse date from key (ECB format)
                        date_str = key.split(':')[0]
                        if len(date_str) == 6:  # YYYYMM format
                            date = pd.to_datetime(date_str, format='%Y%m')
                            value = float(value_data[0])
                            dates.append(date)
                            values.append(value)

                if dates and values:
                    df = pd.DataFrame({'ecb_rate': values}, index=dates)
                    df = df.sort_index()
                    logger.info(f"Collected {len(df)} ECB rate observations")
                    return df

        except Exception as e:
            logger.error(f"Error collecting ECB data: {e}")

        return pd.DataFrame()

    def collect_all_data(self) -> Dict[str, pd.DataFrame]:
        """Collect all data from free APIs"""
        logger.info("Starting complete data collection...")

        # FRED Economic Data
        fred_series = [
            'FEDFUNDS', 'DFF', 'CPIAUCSL', 'UNRATE', 'PAYEMS',
            'INDPRO', 'DGORDER', 'DEXUSEU', 'DEXJPUS', 'DGS10', 'DGS2'
        ]
        fred_data = self.collect_fred_data(fred_series)

        # Yahoo Finance Data - Fix tickers
        yahoo_tickers = ['DX-Y.NYB', 'GC=F', '^VIX']  # DXY ticker, Gold futures, VIX
        yahoo_data = self.collect_yahoo_finance_data(yahoo_tickers)

        # Load local EURUSD and XAUUSD daily data
        local_data = {}
        for pair in ['EURUSD', 'XAUUSD']:
            try:
                file_path = f'data/raw/{pair}_Daily.csv'
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    # Standardize column names to title case (Open, High, Low, Close, Volume)
                    df.columns = df.columns.str.title()
                    local_data[pair] = df
                    logger.info(f"Loaded {len(df)} records for {pair}")
            except Exception as e:
                logger.error(f"Error loading {pair} data: {e}")

        # Finnhub Economic Calendar
        finnhub_data = self.collect_finnhub_data()

        # ECB Data
        ecb_data = self.collect_ecb_data()

        # Combine all data
        all_data = {
            **fred_data,
            **yahoo_data,
            **local_data,
            'economic_calendar': finnhub_data,
            'ecb_rates': ecb_data
        }

        logger.info(f"Data collection complete. Collected {len(all_data)} datasets.")
        return all_data

# Strategy implementations will go here
class TradingStrategies:
    """Complete strategy implementation system"""

    def __init__(self, data_collector: TradingDataCollector):
        self.data_collector = data_collector

    def asian_range_breakout(self, df: pd.DataFrame) -> pd.Series:
        """Asian Range Breakout Strategy (67% daily accuracy)"""
        # Implementation from guide
        # GMT timezone handling
        df = df.copy()

        # Calculate Asian session range (typically 00:00-08:00 GMT)
        # This is a simplified implementation
        df['asian_high'] = df['High'].rolling(8).max()  # 8 hours
        df['asian_low'] = df['Low'].rolling(8).min()
        df['asian_range'] = df['asian_high'] - df['asian_low']

        # Range percentile filter
        df['range_percentile'] = df['asian_range'].rolling(20).rank(pct=True)

        # Breakout signals
        df['bullish_breakout'] = (df['Close'] > df['asian_high'].shift(1)) & (df['range_percentile'] > 0.5)
        df['bearish_breakout'] = (df['Close'] < df['asian_low'].shift(1)) & (df['range_percentile'] > 0.5)

        # Convert to signal series
        signals = pd.Series(0, index=df.index)
        signals[df['bullish_breakout']] = 1
        signals[df['bearish_breakout']] = -1

        return signals

    def gap_fill_strategy(self, df: pd.DataFrame) -> pd.Series:
        """Gap Fill Strategy (90% fill rate)"""
        df = df.copy()

        # Calculate weekend gaps
        df['weekend_gap'] = df['Open'] - df['Close'].shift(1)
        df['gap_size'] = abs(df['weekend_gap'])

        # Key levels (simplified)
        df['resistance'] = df['High'].rolling(20).max()
        df['support'] = df['Low'].rolling(20).min()

        # Gap fill conditions
        df['gap_up_fill'] = (df['weekend_gap'] > 0) & (df['Low'] <= df['Close'].shift(1))
        df['gap_down_fill'] = (df['weekend_gap'] < 0) & (df['High'] >= df['Close'].shift(1))

        # Size and level analysis
        df['large_gap'] = df['gap_size'] > df['gap_size'].rolling(20).quantile(0.8)
        df['near_resistance'] = abs(df['Close'].shift(1) - df['resistance']) / df['resistance'] < 0.01
        df['near_support'] = abs(df['Close'].shift(1) - df['support']) / df['support'] < 0.01

        # Combined signal
        signals = pd.Series(0, index=df.index)
        signals[df['gap_up_fill'] & df['large_gap']] = -1  # Expect bearish fill
        signals[df['gap_down_fill'] & df['large_gap']] = 1   # Expect bullish fill

        return signals

    def dxy_exy_crossover_strategy(self) -> pd.Series:
        """DXY/EXY Crossover with Resistance/Support Confirmation"""
        # Get DXY and EXY data
        dxy_data = self.data_collector.collect_yahoo_finance_data(['DX-Y.NYB'])
        exy_data = self.data_collector.collect_yahoo_finance_data(['FXE'])

        if 'DX-Y.NYB' not in dxy_data or 'FXE' not in exy_data:
            return pd.Series()

        dxy = dxy_data['DX-Y.NYB']['Close']
        exy = exy_data['FXE']['Close']

        # Normalize for comparison
        dxy_norm = (dxy - dxy.rolling(252).min()) / (dxy.rolling(252).max() - dxy.rolling(252).min()) * 100
        exy_norm = (exy - exy.rolling(252).min()) / (exy.rolling(252).max() - exy.rolling(252).min()) * 100

        # Crossover signals
        dxy_crosses_above_exy = (dxy_norm > exy_norm) & (dxy_norm.shift(1) <= exy_norm.shift(1))
        exy_crosses_above_dxy = (exy_norm > dxy_norm) & (exy_norm.shift(1) <= dxy_norm.shift(1))

        # Resistance/Support levels
        dxy_resistance = dxy_norm.rolling(50).max()
        dxy_support = dxy_norm.rolling(50).min()
        exy_resistance = exy_norm.rolling(50).max()
        exy_support = exy_norm.rolling(50).min()

        # Confirmation signals
        eurusd_bearish = (
            dxy_crosses_above_exy |
            ((dxy_norm >= dxy_support * 1.01) & (dxy_norm > dxy_norm.shift(1))) |
            ((exy_norm >= exy_resistance * 0.99) & (exy_norm < exy_norm.shift(1)))
        )

        eurusd_bullish = (
            exy_crosses_above_dxy |
            ((exy_norm >= exy_support * 1.01) & (exy_norm > exy_norm.shift(1))) |
            ((dxy_norm >= dxy_resistance * 0.99) & (dxy_norm < dxy_norm.shift(1)))
        )

        # Convert to signal series
        signals = pd.Series(0, index=dxy_norm.index)
        signals[eurusd_bullish] = 1
        signals[eurusd_bearish] = -1

        return signals

    def master_signal_system(self, df: pd.DataFrame) -> pd.Series:
        """Intelligent weighting system combining all strategies"""
        # Get individual strategy signals
        asian_signals = self.asian_range_breakout(df)
        gap_signals = self.gap_fill_strategy(df)
        dxy_signals = self.dxy_exy_crossover_strategy()
        
        # Strategy weights based on expected accuracy
        weights = {
            'asian_breakout': 0.40,      # 67% accuracy - increased weight
            'gap_fill': 0.30,            # 90% fill rate - increased weight  
            'dxy_exy_crossover': 0.15,   # Custom edge (when available)
            'fundamental_bias': 0.15     # Rate differentials (placeholder)
        }

        # Combine signals
        master_score = pd.Series(0, index=df.index)

        # Add weighted signals where available
        if not asian_signals.empty:
            master_score += asian_signals * weights['asian_breakout']
        if not gap_signals.empty:
            master_score += gap_signals * weights['gap_fill']
        if not dxy_signals.empty and not dxy_signals.eq(0).all():
            # Align indices and add DXY signals
            aligned_dxy = dxy_signals.reindex(df.index, method='ffill').fillna(0)
            master_score += aligned_dxy * weights['dxy_exy_crossover']

        # Final signals (lower threshold for testing)
        final_signals = pd.Series(0, index=df.index)
        final_signals[master_score >= 0.25] = 1   # Bullish threshold
        final_signals[master_score <= -0.25] = -1 # Bearish threshold

        return final_signals

# Main execution
if __name__ == "__main__":
    # Initialize system
    collector = TradingDataCollector()
    strategies = TradingStrategies(collector)

    # Collect all data
    data = collector.collect_all_data()

    # Example: Run strategies on EURUSD data
    if 'EURUSD=X' in data:
        eurusd_data = data['EURUSD=X']
        signals = strategies.master_signal_system(eurusd_data)
        print(f"Generated {len(signals[signals != 0])} signals for EURUSD")

    logger.info("Trading system execution complete.")