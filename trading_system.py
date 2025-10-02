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

    def master_signal_system(self, df: pd.DataFrame) -> pd.Series:
        """Intelligent weighting system combining all strategies"""
        # Get individual strategy signals
        asian_signals = self.asian_range_breakout(df)
        gap_signals = self.gap_fill_strategy(df)
        dxy_signals = self.dxy_exy_crossover_strategy(df)
        fundamental_signals = self.fundamental_bias_strategy(df)

        # Strategy weights based on expected accuracy
        weights = {
            'asian_breakout': 0.35,      # 67% accuracy
            'gap_fill': 0.25,            # 90% fill rate
            'dxy_exy_crossover': 0.15,   # Custom edge (when available)
            'fundamental_bias': 0.25     # Rate differentials (high impact)
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
        if not fundamental_signals.empty:
            master_score += fundamental_signals * weights['fundamental_bias']

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

    # Quick test with local data only
    data = {}

    # Load local EURUSD data
    try:
        df = pd.read_csv('data/raw/EURUSD_Daily.csv')
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

        # Proper backtest accuracy check (matching Django implementation)
        returns = eurusd_data['Close'].pct_change()
        signal_returns = signals.shift(1) * returns  # Shift signals to avoid lookahead bias

        total_trades = signals.abs().sum()
        winning_trades = (signal_returns > 0).sum()
        accuracy = (winning_trades / total_trades * 100) if total_trades > 0 else 0

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