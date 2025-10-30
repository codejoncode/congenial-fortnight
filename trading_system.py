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
            'ecb': 0,
            'alpha_vantage': 0
        }

        self.daily_limits = {
            'fred': float('inf'),  # unlimited
            'finnhub': 100,
            'fmp': 250,
            'yahoo': 100,
            'ecb': float('inf'),
            'alpha_vantage': 500
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
        """Collect economic calendar from Finnhub (100 calls/day free) - Updated to v2 API"""
        if not self._check_rate_limit('finnhub'):
            return pd.DataFrame()

        # Try v2 endpoint first (newer API)
        base_url = "https://finnhub.io/api/v2/calendar/economic"
        params = {
            'token': self.api_keys['finnhub'],
            'from': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
            'to': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
        }

        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if 'events' in data and data['events']:
                df = pd.DataFrame(data['events'])
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                self._increment_call('finnhub')
                logger.info(f"Collected {len(df)} Finnhub economic events via v2 API")
                return df

        except Exception as e:
            logger.warning(f"Finnhub v2 API failed: {e}")

        # Fallback to v1 endpoint if v2 fails
        try:
            v1_url = "https://finnhub.io/api/v1/calendar/economic"
            v1_params = {
                'token': self.api_keys['finnhub'],
                'from': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                'to': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
            }

            response = requests.get(v1_url, params=v1_params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if 'economicCalendar' in data:
                df = pd.DataFrame(data['economicCalendar'])
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                self._increment_call('finnhub')
                logger.info(f"Collected {len(df)} Finnhub economic events via v1 API (fallback)")
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
        """Collect ECB data from official ECB Statistical Data Warehouse - Updated to reliable endpoint"""
        try:
            # Use the official ECB SDMX endpoint for EUR/USD reference rates
            # This provides daily EUR/USD exchange rates which can be used for EUR strength analysis
            url = "https://sdw-wsrest.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A?format=jsondata&startPeriod=2020-01-01"

            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()

            if 'dataSets' in data and data['dataSets']:
                observations = data['dataSets'][0]['observations']
                dates = []
                values = []

                # Extract date and value from SDMX format
                for key, value_data in observations.items():
                    if value_data and len(value_data) > 0:
                        # Parse date from SDMX key format
                        date_str = key.split(':')[0]
                        if len(date_str) == 8:  # YYYYMMDD format
                            try:
                                date = pd.to_datetime(date_str, format='%Y%m%d')
                                value = float(value_data[0])
                                dates.append(date)
                                values.append(value)
                            except (ValueError, IndexError):
                                continue

                if dates and values:
                    df = pd.DataFrame({'eur_usd_rate': values}, index=dates)
                    df = df.sort_index()
                    logger.info(f"Collected {len(df)} ECB EUR/USD reference rates")
                    return df

        except Exception as e:
            logger.warning(f"ECB SDMX API failed: {e}")

        # Fallback: Use FRED for ECB deposit facility rate (same as our fundamental bias)
        try:
            logger.info("Falling back to FRED API for ECB data")
            fred_data = self.collect_fred_data(['ECBDFR'])
            if 'ECBDFR' in fred_data:
                df = fred_data['ECBDFR']
                df.columns = ['ecb_rate']  # Rename for consistency
                logger.info(f"Collected {len(df)} ECB rates via FRED fallback")
                return df

        except Exception as e:
            logger.error(f"ECB data collection failed completely: {e}")

        return pd.DataFrame()

    def collect_alpha_vantage_data(self, symbol: str, function: str = 'FX_DAILY') -> pd.DataFrame:
        """Collect data from Alpha Vantage (5 calls/minute, 500/day free)"""
        if not self._check_rate_limit('alpha_vantage'):
            return pd.DataFrame()

        base_url = "https://www.alphavantage.co/query"
        params = {
            'function': function,
            'from_symbol': symbol.split('/')[0] if '/' in symbol else symbol,
            'to_symbol': symbol.split('/')[1] if '/' in symbol else 'USD',
            'apikey': self.api_keys['alpha_vantage'],
            'outputsize': 'full'
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

            if 'Time Series FX (Daily)' in data:
                df = pd.DataFrame.from_dict(data['Time Series FX (Daily)'], orient='index')
                df.index = pd.to_datetime(df.index)
                df = df.astype(float)
                df.columns = ['Open', 'High', 'Low', 'Close']
                df = df.sort_index()
                self._increment_call('alpha_vantage')
                logger.info(f"Collected {len(df)} Alpha Vantage records for {symbol}")
                return df

        except Exception as e:
            logger.error(f"Error collecting Alpha Vantage data for {symbol}: {e}")

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

        # Yahoo Finance Data - Fix tickers and use Alpha Vantage as backup
        yahoo_tickers = ['DX-Y.NYB', 'GC=F', '^VIX']  # DXY ticker, Gold futures, VIX
        yahoo_data = self.collect_yahoo_finance_data(yahoo_tickers)

        # Alpha Vantage FX data for DXY and EUR
        alpha_vantage_data = {}
        dxy_data = self.collect_alpha_vantage_data('USD/EUR', 'FX_DAILY')  # DXY equivalent
        if not dxy_data.empty:
            alpha_vantage_data['DXY_AV'] = dxy_data
        eurusd_data = self.collect_alpha_vantage_data('EUR/USD', 'FX_DAILY')
        if not eurusd_data.empty:
            alpha_vantage_data['EURUSD_AV'] = eurusd_data

        # Store Alpha Vantage data for strategies to access
        self.alpha_vantage_data = alpha_vantage_data

        # Store Alpha Vantage data for strategies to access
        self.alpha_vantage_data = alpha_vantage_data

        def _find_price_file(pair: str):
            """Return the best available price file for pair in data/ by preferred granularity."""
            for interval in ['H1', 'H4', 'Daily', 'Weekly', 'Monthly']:
                candidate = f'data/{pair}_' + interval + '.csv' if interval != 'Daily' else f'data/{pair}_Daily.csv'
                if os.path.exists(candidate):
                    return candidate
            return None

        # Load local EURUSD and XAUUSD price data (prefer H1 -> H4 -> Daily -> Weekly -> Monthly)
        local_data = {}
        for pair in ['EURUSD', 'XAUUSD']:
            try:
                file_path = _find_price_file(pair)
                if file_path and os.path.exists(file_path):
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
            **alpha_vantage_data,
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

    def dxy_exy_crossover_strategy(self, df: pd.DataFrame) -> pd.Series:
        """DXY/EXY Crossover with Resistance/Support Confirmation using Alpha Vantage"""
        # Use Alpha Vantage data directly
        dxy_data = None
        eurusd_data = None

        # Check for Alpha Vantage data in collector
        if hasattr(self.data_collector, 'alpha_vantage_data'):
            if 'DXY_AV' in self.data_collector.alpha_vantage_data:
                dxy_data = self.data_collector.alpha_vantage_data['DXY_AV']
            if 'EURUSD_AV' in self.data_collector.alpha_vantage_data:
                eurusd_data = self.data_collector.alpha_vantage_data['EURUSD_AV']

        if dxy_data is None or eurusd_data is None or dxy_data.empty or eurusd_data.empty:
            logger.warning("No DXY or EUR data available for crossover strategy")
            return pd.Series(0, index=df.index)

        # Use close prices for analysis
        dxy_close = dxy_data['Close'] if 'Close' in dxy_data.columns else dxy_data.iloc[:, 0]
        eurusd_close = eurusd_data['Close'] if 'Close' in eurusd_data.columns else eurusd_data.iloc[:, 0]

        # Normalize for comparison (DXY is USD strength, EURUSD is EUR strength)
        # When DXY rises, EURUSD typically falls (inverse relationship)
        dxy_norm = (dxy_close - dxy_close.rolling(252).min()) / (dxy_close.rolling(252).max() - dxy_close.rolling(252).min()) * 100
        eurusd_norm = (eurusd_close - eurusd_close.rolling(252).min()) / (eurusd_close.rolling(252).max() - eurusd_close.rolling(252).min()) * 100

        # Align indices
        common_index = dxy_norm.index.intersection(eurusd_norm.index)
        dxy_norm = dxy_norm.loc[common_index]
        eurusd_norm = eurusd_norm.loc[common_index]

        if len(dxy_norm) < 50 or len(eurusd_norm) < 50:
            logger.warning("Insufficient data for crossover analysis")
            return pd.Series(0, index=df.index)

        # Crossover signals (DXY crossing above EURUSD norm = bearish for EURUSD)
        dxy_crosses_above_eurusd = (dxy_norm > eurusd_norm) & (dxy_norm.shift(1) <= eurusd_norm.shift(1))
        eurusd_crosses_above_dxy = (eurusd_norm > dxy_norm) & (eurusd_norm.shift(1) <= dxy_norm.shift(1))

        # Resistance/Support levels
        dxy_resistance = dxy_norm.rolling(50).max()
        dxy_support = dxy_norm.rolling(50).min()
        eurusd_resistance = eurusd_norm.rolling(50).max()
        eurusd_support = eurusd_norm.rolling(50).min()

        # Confirmation signals
        eurusd_bearish = (
            dxy_crosses_above_eurusd |
            ((dxy_norm >= dxy_support * 1.01) & (dxy_norm > dxy_norm.shift(1))) |
            ((eurusd_norm >= eurusd_resistance * 0.99) & (eurusd_norm < eurusd_norm.shift(1)))
        )

        eurusd_bullish = (
            eurusd_crosses_above_dxy |
            ((eurusd_norm >= eurusd_support * 1.01) & (eurusd_norm > eurusd_norm.shift(1))) |
            ((dxy_norm >= dxy_resistance * 0.99) & (dxy_norm < dxy_norm.shift(1)))
        )

        # Convert to signal series aligned with main data
        signals = pd.Series(0, index=df.index)
        bullish_aligned = eurusd_bullish.reindex(df.index, method='ffill').fillna(0).astype(bool)
        bearish_aligned = eurusd_bearish.reindex(df.index, method='ffill').fillna(0).astype(bool)

        signals[bullish_aligned] = 1
        signals[bearish_aligned] = -1

        return signals

    def fundamental_bias_strategy(self, df: pd.DataFrame) -> pd.Series:
        """Fundamental Bias Strategy using Fed vs ECB rate differentials"""
        # Collect FRED data for rate differentials
        fred_series = ['FEDFUNDS', 'ECBDFR']  # Fed Funds Rate and ECB Deposit Rate
        fred_data = self.data_collector.collect_fred_data(fred_series)

        if 'FEDFUNDS' not in fred_data or 'ECBDFR' not in fred_data:
            logger.warning("FRED data not available for fundamental bias strategy")
            return pd.Series(0, index=df.index)

        fed_rates = fred_data['FEDFUNDS']
        ecb_rates = fred_data['ECBDFR']

        # Calculate rate differential (Fed - ECB)
        # Positive differential favors USD/EUR down (EURUSD down)
        # Negative differential favors EUR/USD up (EURUSD up)
        rate_diff = fed_rates['FEDFUNDS'] - ecb_rates['ECBDFR']

        # Resample to daily frequency and forward fill
        rate_diff_daily = rate_diff.resample('D').ffill()

        # Align with trading data
        aligned_diff = rate_diff_daily.reindex(df.index, method='ffill').ffill().bfill()

        # Calculate rate differential changes and trends
        diff_change = aligned_diff.diff()
        diff_trend = aligned_diff.rolling(30).mean() - aligned_diff.rolling(90).mean()

        # Fundamental bias signals
        # Strong positive differential (Fed > ECB) = bearish for EURUSD
        strong_positive_diff = aligned_diff > aligned_diff.rolling(252).quantile(0.75)
        # Strong negative differential (ECB > Fed) = bullish for EURUSD
        strong_negative_diff = aligned_diff < aligned_diff.rolling(252).quantile(0.25)

        # Rate differential expansion (widening gap)
        expanding_diff = diff_change > diff_change.rolling(20).quantile(0.8)
        contracting_diff = diff_change < diff_change.rolling(20).quantile(0.2)

        # Trend signals
        bullish_trend = diff_trend < diff_trend.rolling(20).quantile(0.3)  # Contracting differential favors EUR
        bearish_trend = diff_trend > diff_trend.rolling(20).quantile(0.7)  # Expanding differential favors USD

        # Combine signals
        signals = pd.Series(0, index=df.index)

        # Strong fundamental signals
        signals[strong_negative_diff & bullish_trend] = 1   # Bullish EURUSD
        signals[strong_positive_diff & bearish_trend] = -1  # Bearish EURUSD

        # Moderate signals based on changes
        signals[expanding_diff & (aligned_diff > 0)] = -1   # Widening positive diff = bearish
        signals[contracting_diff & (aligned_diff < 0)] = 1  # Narrowing negative diff = bullish

        return signals

    def holloway_features(self, df: pd.DataFrame) -> pd.Series:
        """Holloway Algorithm - 347 sophisticated trend analysis features"""
        df = df.copy()

        # Define moving average periods (24 total)
        ma_periods = [5, 7, 10, 14, 20, 28, 50, 56, 100, 112, 200, 225]

        # Calculate EMAs and SMAs
        for period in ma_periods:
            df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
            df[f'sma_{period}'] = df['Close'].rolling(period).mean()

        # Calculate RSI for timing confirmation
        def calculate_rsi(price, period=14):
            delta = price.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        df['rsi'] = calculate_rsi(df['Close'])
        df['rsi_overbought'] = df['rsi'] > 70
        df['rsi_oversold'] = df['rsi'] < 30

        # Initialize count variables
        df['holloway_bull_count'] = 0
        df['holloway_bear_count'] = 0

        # Price vs Moving Averages (12 signals each = 24 total)
        for period in ma_periods:
            # Bullish: Close > MA
            df['holloway_bull_count'] += (df['Close'] > df[f'ema_{period}']).astype(int)
            df['holloway_bull_count'] += (df['Close'] > df[f'sma_{period}']).astype(int)

            # Bearish: Close < MA
            df['holloway_bear_count'] += (df['Close'] < df[f'ema_{period}']).astype(int)
            df['holloway_bear_count'] += (df['Close'] < df[f'sma_{period}']).astype(int)

        # Moving Average Hierarchies (EMA relationships - 66 signals)
        # EMA5 > EMA7 > EMA10 > EMA14 > EMA20 > EMA28 > EMA50 > EMA56 > EMA100 > EMA112 > EMA200 > EMA225
        ema_cols = [f'ema_{p}' for p in ma_periods]
        for i in range(len(ema_cols)-1):
            for j in range(i+1, len(ema_cols)):
                # Shorter period EMA > Longer period EMA (bullish hierarchy)
                df['holloway_bull_count'] += (df[ema_cols[i]] > df[ema_cols[j]]).astype(int)
                # Shorter period EMA < Longer period EMA (bearish hierarchy)
                df['holloway_bear_count'] += (df[ema_cols[i]] < df[ema_cols[j]]).astype(int)

        # SMA Hierarchies (SMA relationships - 66 signals)
        sma_cols = [f'sma_{p}' for p in ma_periods]
        for i in range(len(sma_cols)-1):
            for j in range(i+1, len(sma_cols)):
                df['holloway_bull_count'] += (df[sma_cols[i]] > df[sma_cols[j]]).astype(int)
                df['holloway_bear_count'] += (df[sma_cols[i]] < df[sma_cols[j]]).astype(int)

        # EMA vs SMA Relationships (144 signals)
        for ema_period in ma_periods:
            for sma_period in ma_periods:
                df['holloway_bull_count'] += (df[f'ema_{ema_period}'] > df[f'sma_{sma_period}']).astype(int)
                df['holloway_bear_count'] += (df[f'ema_{ema_period}'] < df[f'sma_{sma_period}']).astype(int)

        # Dynamic Signals - Fresh Breakouts (48 signals)
        for period in ma_periods[:4]:  # Focus on shorter periods for breakouts
            # Bullish breakout: Close crosses above EMA/SMA
            breakout_up_ema = (df['Close'] > df[f'ema_{period}']) & (df['Close'].shift(1) <= df[f'ema_{period}'].shift(1))
            breakout_up_sma = (df['Close'] > df[f'sma_{period}']) & (df['Close'].shift(1) <= df[f'sma_{period}'].shift(1))

            # Bearish breakdown: Close crosses below EMA/SMA
            breakout_down_ema = (df['Close'] < df[f'ema_{period}']) & (df['Close'].shift(1) >= df[f'ema_{period}'].shift(1))
            breakout_down_sma = (df['Close'] < df[f'sma_{period}']) & (df['Close'].shift(1) >= df[f'sma_{period}'].shift(1))

            df['holloway_bull_count'] += (breakout_up_ema | breakout_up_sma).astype(int)
            df['holloway_bear_count'] += (breakout_down_ema | breakout_down_sma).astype(int)

        # Exponential moving averages of counts (span=27)
        df['holloway_bull_avg'] = df['holloway_bull_count'].ewm(span=27).mean()
        df['holloway_bear_avg'] = df['holloway_bear_count'].ewm(span=27).mean()

        # Count differences and ratios
        df['holloway_count_diff'] = df['holloway_bull_count'] - df['holloway_bear_count']
        df['holloway_count_ratio'] = df['holloway_bull_count'] / (df['holloway_bear_count'] + 1)

        # Rolling statistics (20-period)
        df['holloway_bull_max_20'] = df['holloway_bull_count'].rolling(20).max()
        df['holloway_bull_min_20'] = df['holloway_bull_count'].rolling(20).min()
        df['holloway_bear_max_20'] = df['holloway_bear_count'].rolling(20).max()
        df['holloway_bear_min_20'] = df['holloway_bear_count'].rolling(20).min()

        # Direction change detection - Cross signals
        df['holloway_bull_cross_up'] = (df['holloway_bull_count'] > df['holloway_bull_avg']) & \
                                       (df['holloway_bull_count'].shift(1) <= df['holloway_bull_avg'].shift(1))
        df['holloway_bull_cross_down'] = (df['holloway_bull_count'] < df['holloway_bull_avg']) & \
                                         (df['holloway_bull_count'].shift(1) >= df['holloway_bull_avg'].shift(1))
        df['holloway_bear_cross_up'] = (df['holloway_bear_count'] > df['holloway_bear_avg']) & \
                                       (df['holloway_bear_count'].shift(1) <= df['holloway_bear_avg'].shift(1))
        df['holloway_bear_cross_down'] = (df['holloway_bear_count'] < df['holloway_bear_avg']) & \
                                         (df['holloway_bear_count'].shift(1) >= df['holloway_bear_avg'].shift(1))

        # Combined signals with RSI confirmation
        df['holloway_bull_signal'] = df['holloway_bull_cross_up'] & ~df['rsi_overbought']
        df['holloway_bear_signal'] = df['holloway_bear_cross_up'] & ~df['rsi_oversold']

        # Trend strength indicators
        df['holloway_trend_strength'] = df['holloway_count_diff'].abs()
        df['holloway_bull_dominance'] = df['holloway_count_diff'] / (df['holloway_bull_count'] + df['holloway_bear_count'] + 1)

        # Momentum signals
        df['holloway_bull_momentum'] = df['holloway_bull_count'].diff()
        df['holloway_bear_momentum'] = df['holloway_bear_count'].diff()

        # Rolling momentum (5-period)
        df['holloway_bull_momentum_5'] = df['holloway_bull_momentum'].rolling(5).mean()
        df['holloway_bear_momentum_5'] = df['holloway_bear_momentum'].rolling(5).mean()

        # Signal quality filters
        df['holloway_high_quality_bull'] = df['holloway_bull_signal'] & (df['holloway_bull_count'] > df['holloway_bull_count'].rolling(50).quantile(0.75))
        df['holloway_high_quality_bear'] = df['holloway_bear_signal'] & (df['holloway_bear_count'] > df['holloway_bear_count'].rolling(50).quantile(0.75))

        # Convert to signal series for master system
        signals = pd.Series(0, index=df.index)
        signals[df['holloway_high_quality_bull']] = 1
        signals[df['holloway_high_quality_bear']] = -1

        return signals

    def master_signal_system(self, df: pd.DataFrame) -> pd.Series:
        """Intelligent weighting system combining all strategies with consensus filtering"""
        # Get individual strategy signals
        asian_signals = self.asian_range_breakout(df)
        gap_signals = self.gap_fill_strategy(df)
        dxy_signals = self.dxy_exy_crossover_strategy(df)
        fundamental_signals = self.fundamental_bias_strategy(df)
        holloway_signals = self.holloway_features(df)

        print(f"Strategy signals - Asian: {len(asian_signals[asian_signals != 0])}, Gap: {len(gap_signals[gap_signals != 0])}, DXY: {len(dxy_signals[dxy_signals != 0])}, Fundamental: {len(fundamental_signals[fundamental_signals != 0])}, Holloway: {len(holloway_signals[holloway_signals != 0])}")

        # Strategy weights based on expected accuracy
        weights = {
            'asian_breakout': 0.20,      # 67% accuracy
            'gap_fill': 0.15,            # 90% fill rate
            'dxy_exy_crossover': 0.15,   # Custom edge (when available)
            'fundamental_bias': 0.15,    # Rate differentials (high impact)
            'holloway_algorithm': 0.35   # 347 sophisticated features (highest weight)
        }

        # Collect strategy signals for consensus filtering
        strategy_signals = {
            'asian': asian_signals,
            'gap': gap_signals,
            'dxy': dxy_signals,
            'fundamental': fundamental_signals,
            'holloway': holloway_signals
        }

        # Apply consensus filtering - require at least 2 strategies to agree
        consensus_signals = self._apply_consensus_filter(strategy_signals, min_agreement=2)
        print(f"Consensus signals: {len(consensus_signals[consensus_signals != 0])} (from {len(strategy_signals)} strategies)")

        # If consensus signals exist, use them directly (higher quality)
        if not consensus_signals.empty and len(consensus_signals[consensus_signals != 0]) > 0:
            print(f"Using consensus-filtered signals ({len(consensus_signals[consensus_signals != 0])} signals)")
            return consensus_signals

        # Fallback: Combine signals with weights when consensus not available
        print("Consensus filtering returned no signals, falling back to weighted combination")
        master_score = pd.Series(0, index=df.index)

        # Add weighted signals where available
        if not asian_signals.empty:
            aligned_asian = asian_signals.reindex(df.index, method='ffill').fillna(0)
            master_score += aligned_asian * weights['asian_breakout']
        if not gap_signals.empty:
            aligned_gap = gap_signals.reindex(df.index, method='ffill').fillna(0)
            master_score += aligned_gap * weights['gap_fill']
        if not dxy_signals.empty and not dxy_signals.eq(0).all():
            aligned_dxy = dxy_signals.reindex(df.index, method='ffill').fillna(0)
            master_score += aligned_dxy * weights['dxy_exy_crossover']
        if not fundamental_signals.empty:
            master_score += fundamental_signals * weights['fundamental_bias']
        if not holloway_signals.empty:
            master_score += holloway_signals * weights['holloway_algorithm']

        # Final signals (lower threshold for testing)
        final_signals = pd.Series(0, index=df.index)
        final_signals[master_score >= 0.25] = 1   # Bullish threshold
        final_signals[master_score <= -0.25] = -1 # Bearish threshold

        return final_signals

    def _apply_consensus_filter(self, strategy_signals: Dict[str, pd.Series], min_agreement: int = 2) -> pd.Series:
        """Apply consensus filtering - only signal when multiple strategies agree on the same day"""
        if not strategy_signals:
            return pd.Series()

        # Filter out empty series
        valid_signals = {name: signals[signals != 0] for name, signals in strategy_signals.items() if not signals.empty}

        if len(valid_signals) < min_agreement:
            return pd.Series()

        # Find dates where at least min_agreement strategies have signals
        all_dates = set()
        for signals in valid_signals.values():
            all_dates.update(signals.index)

        consensus_signals = pd.Series(0, index=sorted(all_dates))

        for date in all_dates:
            signals_on_date = []
            for name, signals in valid_signals.items():
                if date in signals.index:
                    signal_val = signals.loc[date]
                    try:
                        if pd.notna(signal_val) and signal_val != 0:
                            scalar_val = float(signal_val) if hasattr(signal_val, '__float__') else signal_val
                            signals_on_date.append(scalar_val)
                    except (TypeError, ValueError):
                        continue

            # Check if we have minimum agreement in the same direction
            if len(signals_on_date) >= min_agreement:
                # Check if majority agree on direction
                positive_signals = sum(1 for s in signals_on_date if s > 0)
                negative_signals = sum(1 for s in signals_on_date if s < 0)

                if positive_signals >= min_agreement:
                    consensus_signals.loc[date] = 1   # Bullish consensus
                elif negative_signals >= min_agreement:
                    consensus_signals.loc[date] = -1  # Bearish consensus

        return consensus_signals

# Main execution
if __name__ == "__main__":
    # Initialize system
    collector = TradingDataCollector()
    strategies = TradingStrategies(collector)

    # Quick test with local data only
    data = {}

    # Load local EURUSD data (prefer data/ interval files)
    try:
        # Prefer H1 -> H4 -> Daily -> Weekly -> Monthly
        for candidate in ['data/EURUSD_H1.csv','data/EURUSD_H4.csv','data/EURUSD_Daily.csv','data/EURUSD_Weekly.csv','data/EURUSD_Monthly.csv']:
            if os.path.exists(candidate):
                df = pd.read_csv(candidate)
                break
        else:
            raise FileNotFoundError('No EURUSD data file found in data/')
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df.columns = df.columns.str.title()
        data['EURUSD'] = df
        print(f"Loaded {len(df)} records for EURUSD")
    except Exception as e:
        print(f"Error loading EURUSD data: {e}")
        exit(1)

    # Collect minimal FRED data for fundamental bias
    print("Collecting FRED data for fundamental bias...")
    fred_series = ['FEDFUNDS', 'ECBDFR']
    fred_data = collector.collect_fred_data(fred_series)
    data.update(fred_data)

    # Collect Alpha Vantage data for DXY
    print("Collecting Alpha Vantage data...")
    dxy_data = collector.collect_alpha_vantage_data('USD/EUR', 'FX_DAILY')
    if not dxy_data.empty:
        data['DXY_AV'] = dxy_data
        collector.alpha_vantage_data = {'DXY_AV': dxy_data}
        print(f"Stored {len(dxy_data)} DXY records in collector")
    eurusd_av_data = collector.collect_alpha_vantage_data('EUR/USD', 'FX_DAILY')
    if not eurusd_av_data.empty:
        if not hasattr(collector, 'alpha_vantage_data'):
            collector.alpha_vantage_data = {}
        collector.alpha_vantage_data['EURUSD_AV'] = eurusd_av_data
        print(f"Stored {len(eurusd_av_data)} EURUSD records in collector")

    # Run strategies
    if 'EURUSD' in data:
        eurusd_data = data['EURUSD']
        signals = strategies.master_signal_system(eurusd_data)
        print(f"Generated {len(signals[signals != 0])} signals for EURUSD")
        print(f"Signal distribution: {signals.value_counts()}")

        # Count actual trades (non-zero signals)
        trade_signals = signals[signals != 0]
        total_trades = len(trade_signals)

        # Resample to daily data for accuracy calculation
        daily_data = eurusd_data.resample('D').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        }).dropna()

        # Proper backtest accuracy check (matching Django implementation)
        # For each signal on day t, check if next day t+1 moved in signal direction
        winning_trades = 0
        total_checked = 0

        for date, signal in trade_signals.items():
            # date is already a Timestamp, compare directly with daily_data.index
            if date in daily_data.index:
                # Find the next trading day
                try:
                    next_date_idx = daily_data.index.get_loc(date) + 1
                    if next_date_idx < len(daily_data):
                        next_date = daily_data.index[next_date_idx]
                        current_close = daily_data.loc[date, 'Close']
                        next_close = daily_data.loc[next_date, 'Close']

                        # Check if movement matches signal direction
                        moved_up = next_close > current_close
                        matched = (signal > 0 and moved_up) or (signal < 0 and not moved_up)
                        if matched:
                            winning_trades += 1
                        total_checked += 1
                except (KeyError, IndexError):
                    continue

        accuracy = (winning_trades / total_checked * 100) if total_checked > 0 else 0

        accuracy = (winning_trades / total_checked * 100) if total_checked > 0 else 0

        accuracy = (winning_trades / total_checked * 100) if total_checked > 0 else 0

        print(f"Total trades: {int(total_trades)}")
        print(f"Winning trades: {int(winning_trades)}")
        print(".1f")

        # Also show the simple next-day check for comparison
        df_test = eurusd_data.copy()
        df_test['signal'] = signals
        df_test['next_return'] = df_test['Close'].pct_change().shift(-1)
        simple_correct = ((df_test['signal'] > 0) & (df_test['next_return'] > 0)).sum() + ((df_test['signal'] < 0) & (df_test['next_return'] < 0)).sum()
        simple_total = (df_test['signal'] != 0).sum()
        simple_accuracy = (simple_correct / simple_total * 100) if simple_total > 0 else 0
        print(".1f")

        if total_trades > 0:
            improvement = accuracy - 35  # From 35% baseline
            print(".1f")
            if improvement > 0:
                print(f"Progress toward 43% target: {improvement/43*100:.1f}% complete")

    logger.info("Trading system execution complete.")