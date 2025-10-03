#!/usr/bin/env python3
"""
FundamentalDataPipeline - Enterprise-grade fundamental data collection for forex trading

This module provides comprehensive fundamental data collection from:
- FRED (Federal Reserve Economic Data) API for economic indicators
- CFTC (Commodity Futures Trading Commission) for Commitment of Traders reports
- ECB (European Central Bank) data for EUR-specific indicators

Features:
- Incremental data fetching (only new observations)
- Metadata tracking and validation
- Error handling and retry logic
- Data quality checks
- CSV storage with timestamps
- Update tracking via JSON metadata

Usage:
    # Initial full download
    pipeline = FundamentalDataPipeline()
    pipeline.run_full_update()

    # Daily incremental updates
    pipeline.run_daily_update()

    # Specific series update
    pipeline.update_series('DEXUSEU')  # EUR/USD exchange rate
"""

import os
import json
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
from fredapi import Fred
import zipfile
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FundamentalDataPipeline:
    """
    Enterprise fundamental data pipeline for forex trading signals.

    Collects economic indicators, central bank data, and market positioning
    that influence currency movements.
    """

    def __init__(self, data_dir: str = "data", fred_api_key: Optional[str] = None):
        """
        Initialize the fundamental data pipeline.

        Args:
            data_dir: Directory to store data files
            fred_api_key: FRED API key (can also be set via FRED_API_KEY env var)
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # FRED API setup
        self.fred_api_key = fred_api_key or os.getenv('FRED_API_KEY')
        if not self.fred_api_key:
            logger.warning("FRED API key not found. Fundamental data will be unavailable.")
            self.fred = None
        else:
            self.fred = Fred(api_key=self.fred_api_key)

        # Metadata file for tracking updates
        self.metadata_file = self.data_dir / "update_metadata.json"

        # Define data series to collect
        self.fred_series = self._get_fred_series_config()
        self.cftc_reports = self._get_cftc_config()

        # Load existing metadata
        self.metadata = self._load_metadata()

    def _get_fred_series_config(self) -> Dict[str, Dict]:
        """
        Define FRED economic series to collect for EURUSD and XAUUSD analysis.

        Returns:
            Dictionary mapping series IDs to metadata
        """
        return {
            # USD Strength Indicators
            'DEXUSEU': {
                'name': 'USD/EUR Exchange Rate',
                'description': 'US Dollar to Euro exchange rate',
                'frequency': 'Daily',
                'category': 'currency',
                'importance': 'high'
            },
            'DEXJPUS': {
                'name': 'USD/JPY Exchange Rate',
                'description': 'US Dollar to Japanese Yen exchange rate',
                'frequency': 'Daily',
                'category': 'currency',
                'importance': 'high'
            },
            'DEXCHUS': {
                'name': 'USD/China Exchange Rate',
                'description': 'US Dollar to Chinese Yuan exchange rate',
                'frequency': 'Daily',
                'category': 'currency',
                'importance': 'medium'
            },

            # US Economic Indicators
            'FEDFUNDS': {
                'name': 'Federal Funds Rate',
                'description': 'Federal Funds Effective Rate',
                'frequency': 'Daily',
                'category': 'interest_rate',
                'importance': 'high'
            },
            'DFF': {
                'name': 'Federal Funds Target Rate',
                'description': 'Federal Funds Target Rate (DISCONTINUED)',
                'frequency': 'Daily',
                'category': 'interest_rate',
                'importance': 'high'
            },
            'CPIAUCSL': {
                'name': 'Consumer Price Index',
                'description': 'Consumer Price Index for All Urban Consumers: All Items',
                'frequency': 'Monthly',
                'category': 'inflation',
                'importance': 'high'
            },
            'CPALTT01USM661S': {
                'name': 'Core CPI',
                'description': 'Consumer Price Index: Total: Total for United States',
                'frequency': 'Monthly',
                'category': 'inflation',
                'importance': 'high'
            },
            'UNRATE': {
                'name': 'Unemployment Rate',
                'description': 'Civilian Unemployment Rate',
                'frequency': 'Monthly',
                'category': 'employment',
                'importance': 'high'
            },
            'PAYEMS': {
                'name': 'Nonfarm Payrolls',
                'description': 'All Employees, Total Nonfarm',
                'frequency': 'Monthly',
                'category': 'employment',
                'importance': 'high'
            },
            'INDPRO': {
                'name': 'Industrial Production Index',
                'description': 'Industrial Production: Total Index',
                'frequency': 'Monthly',
                'category': 'production',
                'importance': 'medium'
            },
            'DGORDER': {
                'name': 'Durable Goods Orders',
                'description': 'Manufacturers\' New Orders: Durable Goods',
                'frequency': 'Monthly',
                'category': 'orders',
                'importance': 'medium'
            },

            # Eurozone Indicators
            'ECBDFR': {
                'name': 'ECB Deposit Facility Rate',
                'description': 'European Central Bank Deposit Facility Rate',
                'frequency': 'Daily',
                'category': 'interest_rate',
                'importance': 'high'
            },
            'ECBRR': {
                'name': 'ECB Refinancing Rate',
                'description': 'European Central Bank Refinancing Rate',
                'frequency': 'Daily',
                'category': 'interest_rate',
                'importance': 'high'
            },
            'CP0000EZ19M086NEST': {
                'name': 'Eurozone CPI',
                'description': 'Consumer Price Index: Total for Euro Area (19 Countries)',
                'frequency': 'Monthly',
                'category': 'inflation',
                'importance': 'high'
            },
            'LRHUTTTTDEM156S': {
                'name': 'Eurozone Unemployment',
                'description': 'Unemployment Rate: Total: All Persons for Germany',
                'frequency': 'Monthly',
                'category': 'employment',
                'importance': 'medium'
            },

            # Gold and Commodity Related
            'GOLDAMGBD228NLBM': {
                'name': 'Gold Fixing Price',
                'description': 'Gold Fixing Price 3:00 P.M. (London time) in London Bullion Market, based in U.S. Dollars',
                'frequency': 'Daily',
                'category': 'commodity',
                'importance': 'high'
            },
            'DCOILWTICO': {
                'name': 'WTI Crude Oil Price',
                'description': 'Crude Oil Prices: West Texas Intermediate (WTI)',
                'frequency': 'Daily',
                'category': 'commodity',
                'importance': 'medium'
            },
            'DCOILBRENTEU': {
                'name': 'Brent Crude Oil Price',
                'description': 'Crude Oil Prices: Brent - Europe',
                'frequency': 'Daily',
                'category': 'commodity',
                'importance': 'medium'
            },

            # Risk and Volatility
            'VIXCLS': {
                'name': 'CBOE Volatility Index',
                'description': 'CBOE Volatility Index: VIX',
                'frequency': 'Daily',
                'category': 'volatility',
                'importance': 'high'
            },
            'DGS10': {
                'name': '10-Year Treasury Rate',
                'description': '10-Year Treasury Constant Maturity Rate',
                'frequency': 'Daily',
                'category': 'interest_rate',
                'importance': 'high'
            },
            'DGS2': {
                'name': '2-Year Treasury Rate',
                'description': '2-Year Treasury Constant Maturity Rate',
                'frequency': 'Daily',
                'category': 'interest_rate',
                'importance': 'high'
            },

            # Trade and Current Account
            'BOPGSTB': {
                'name': 'US Trade Balance',
                'description': 'Trade Balance: Goods and Services, Balance of Payments Basis',
                'frequency': 'Monthly',
                'category': 'trade',
                'importance': 'medium'
            }
        }

    def _get_cftc_config(self) -> Dict[str, Dict]:
        """
        Define CFTC Commitment of Traders reports to collect.

        Returns:
            Dictionary mapping report types to metadata
        """
        return {
            'finfut': {
                'name': 'Financial Futures',
                'description': 'Futures-and-Options-Combined Positions of Traders in Financial Futures',
                'url': 'https://www.cftc.gov/files/dea/history/fin_fut_txt_{year}.zip',
                'frequency': 'Weekly',
                'category': 'positioning',
                'importance': 'high'
            },
            'cftc_cot': {
                'name': 'Legacy COT',
                'description': 'Commitments of Traders - Legacy Format',
                'url': 'https://www.cftc.gov/files/dea/history/cot_txt_{year}.zip',
                'frequency': 'Weekly',
                'category': 'positioning',
                'importance': 'high'
            }
        }

    def _load_metadata(self) -> Dict:
        """
        Load update metadata from JSON file.

        Returns:
            Dictionary containing last update timestamps and status
        """
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load metadata file: {e}")
                return {}
        return {}

    def _save_metadata(self):
        """Save current metadata to JSON file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)

    def _get_last_update(self, series_id: str) -> Optional[datetime]:
        """
        Get the last update timestamp for a series.

        Args:
            series_id: Series identifier

        Returns:
            Last update datetime or None if not found
        """
        if series_id in self.metadata:
            timestamp = self.metadata[series_id].get('last_update')
            if timestamp:
                return datetime.fromisoformat(timestamp)
        return None

    def _update_metadata(self, series_id: str, status: str, last_date: Optional[datetime] = None):
        """
        Update metadata for a series.

        Args:
            series_id: Series identifier
            status: Update status ('success', 'error', 'no_update')
            last_date: Last data date (optional)
        """
        if series_id not in self.metadata:
            self.metadata[series_id] = {}

        self.metadata[series_id]['last_update'] = datetime.now().isoformat()
        self.metadata[series_id]['status'] = status

        if last_date:
            self.metadata[series_id]['last_data_date'] = last_date.isoformat()

        self._save_metadata()

    def fetch_fred_series(self, series_id: str, start_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch data for a FRED series with error handling and retries.

        Args:
            series_id: FRED series ID
            start_date: Start date for incremental updates (YYYY-MM-DD)

        Returns:
            DataFrame with date index and value column
        """
        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                logger.info(f"Fetching FRED series {series_id} (attempt {attempt + 1})")

                # Get data from FRED
                if start_date:
                    data = self.fred.get_series(series_id, start_date=start_date)
                else:
                    data = self.fred.get_series(series_id)

                if data is None or data.empty:
                    logger.warning(f"No data returned for series {series_id}")
                    return pd.DataFrame()

                # Convert to DataFrame
                df = pd.DataFrame(data, columns=['value'])
                df.index.name = 'date'
                df = df.reset_index()

                # Clean data
                df = df.dropna()
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')

                logger.info(f"Successfully fetched {len(df)} observations for {series_id}")
                return df

            except Exception as e:
                logger.error(f"Error fetching {series_id} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    raise e

        return pd.DataFrame()

    def save_series_to_csv(self, series_id: str, df: pd.DataFrame):
        """
        Save series data to CSV file.

        Args:
            series_id: Series identifier
            df: DataFrame with data
        """
        if df.empty:
            return

        csv_file = self.data_dir / f"{series_id}.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"Saved {len(df)} rows to {csv_file}")

    def load_series_from_csv(self, series_id: str) -> pd.DataFrame:
        """
        Load series data from CSV file.

        Args:
            series_id: Series identifier

        Returns:
            DataFrame with series data
        """
        csv_file = self.data_dir / f"{series_id}.csv"
        if not csv_file.exists():
            return pd.DataFrame()

        try:
            df = pd.read_csv(csv_file)
            df['date'] = pd.to_datetime(df['date'])
            return df
        except Exception as e:
            logger.error(f"Error loading {csv_file}: {e}")
            return pd.DataFrame()

    def _load_series_from_json_file(self, filepath: str) -> pd.DataFrame:
        """
        Load a simple JSON time-series file into a DataFrame.

        This helper is intended for offline tests and small fixtures under
        data/tests/fundamentals/. The JSON is expected to be a mapping of
        column -> list of values, including a 'date' column.

        Returns a pandas DataFrame with a parsed 'date' column when present.
        """
        try:
            if not os.path.exists(filepath):
                logger.error(f"JSON file not found: {filepath}")
                return pd.DataFrame()

            with open(filepath, 'r') as f:
                j = json.load(f)

            df = pd.DataFrame(j)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            return df
        except Exception as e:
            logger.error(f"Failed to load JSON series {filepath}: {e}")
            return pd.DataFrame()

    def update_fred_series(self, series_id: str) -> bool:
        """
        Update a single FRED series, fetching only new data if available.

        Args:
            series_id: FRED series ID

        Returns:
            True if update was successful
        """
        try:
            # Get existing data
            existing_df = self.load_series_from_csv(series_id)
            start_date = None

            if not existing_df.empty:
                # Find the last date and add one day for incremental update
                last_date = existing_df['date'].max()
                start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
                logger.info(f"Incremental update for {series_id} from {start_date}")

            # Fetch new data
            new_df = self.fetch_fred_series(series_id, start_date)

            if new_df.empty:
                logger.info(f"No new data for {series_id}")
                self._update_metadata(series_id, 'no_update')
                return True

            # Combine with existing data
            if not existing_df.empty:
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                # Remove duplicates based on date
                combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')
                combined_df = combined_df.sort_values('date')
            else:
                combined_df = new_df

            # Save updated data
            self.save_series_to_csv(series_id, combined_df)

            # Update metadata
            last_data_date = combined_df['date'].max() if not combined_df.empty else None
            self._update_metadata(series_id, 'success', last_data_date)

            logger.info(f"Successfully updated {series_id} with {len(new_df)} new observations")
            return True

        except Exception as e:
            logger.error(f"Failed to update {series_id}: {e}")
            self._update_metadata(series_id, 'error')
            return False

    def fetch_cftc_data(self, report_type: str, year: int) -> pd.DataFrame:
        """
        Fetch CFTC Commitment of Traders data for a specific year.

        Args:
            report_type: Type of report ('finfut' or 'cftc_cot')
            year: Year to fetch

        Returns:
            DataFrame with COT data
        """
        try:
            config = self.cftc_reports[report_type]
            url = config['url'].format(year=year)

            logger.info(f"Fetching CFTC {report_type} data for {year}")

            # Download and extract ZIP file
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                # Find the data file (usually has .txt extension)
                txt_files = [f for f in zf.namelist() if f.endswith('.txt')]
                if not txt_files:
                    raise ValueError(f"No .txt file found in {report_type} ZIP for {year}")

                # Read the data file
                with zf.open(txt_files[0]) as f:
                    # CFTC data is tab-delimited
                    df = pd.read_csv(f, delimiter='\t', low_memory=False)

            # Clean column names
            df.columns = df.columns.str.strip()

            # Convert date column if it exists
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'week' in col.lower()]
            if date_cols:
                df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors='coerce')

            logger.info(f"Successfully fetched CFTC {report_type} data for {year}: {len(df)} rows")
            return df

        except Exception as e:
            logger.error(f"Error fetching CFTC {report_type} for {year}: {e}")
            return pd.DataFrame()

    def update_cftc_data(self, report_type: str) -> bool:
        """
        Update CFTC data for current and previous year.

        Args:
            report_type: Type of report to update

        Returns:
            True if update was successful
        """
        try:
            current_year = datetime.now().year
            years_to_fetch = [current_year, current_year - 1]

            for year in years_to_fetch:
                df = self.fetch_cftc_data(report_type, year)
                if not df.empty:
                    csv_file = self.data_dir / f"cftc_{report_type}_{year}.csv"
                    df.to_csv(csv_file, index=False)
                    logger.info(f"Saved CFTC {report_type} data for {year} to {csv_file}")

            self._update_metadata(f"cftc_{report_type}", 'success')
            return True

        except Exception as e:
            logger.error(f"Failed to update CFTC {report_type}: {e}")
            self._update_metadata(f"cftc_{report_type}", 'error')
            return False

    def run_full_update(self) -> Dict[str, bool]:
        """
        Run full update for all FRED series and CFTC data.

        Returns:
            Dictionary mapping series IDs to success status
        """
        logger.info("Starting full fundamental data update")

        results = {}

        # Update FRED series
        for series_id in self.fred_series.keys():
            logger.info(f"Updating FRED series: {series_id}")
            results[series_id] = self.update_fred_series(series_id)
            time.sleep(0.5)  # Rate limiting

        # Update CFTC data
        for report_type in self.cftc_reports.keys():
            logger.info(f"Updating CFTC report: {report_type}")
            results[f"cftc_{report_type}"] = self.update_cftc_data(report_type)
            time.sleep(1)  # Rate limiting

        # Summary
        successful = sum(1 for status in results.values() if status)
        total = len(results)
        logger.info(f"Full update completed: {successful}/{total} successful")

        return results

    def run_daily_update(self) -> Dict[str, bool]:
        """
        Run daily incremental update for all series.

        Returns:
            Dictionary mapping series IDs to success status
        """
        logger.info("Starting daily fundamental data update")

        results = {}

        # Update FRED series (incremental)
        for series_id in self.fred_series.keys():
            logger.info(f"Daily update for FRED series: {series_id}")
            results[series_id] = self.update_fred_series(series_id)
            time.sleep(0.3)  # Rate limiting

        # Update CFTC data (weekly, so check if needed)
        current_weekday = datetime.now().weekday()
        if current_weekday == 4:  # Friday - CFTC releases on Friday
            for report_type in self.cftc_reports.keys():
                logger.info(f"Weekly update for CFTC report: {report_type}")
                results[f"cftc_{report_type}"] = self.update_cftc_data(report_type)
                time.sleep(1)

        # Summary
        successful = sum(1 for status in results.values() if status)
        total = len(results)
        logger.info(f"Daily update completed: {successful}/{total} successful")

        return results

    def get_data_summary(self) -> Dict:
        """
        Get summary of all available fundamental data.

        Returns:
            Dictionary with data summary statistics
        """
        summary = {
            'fred_series': {},
            'cftc_reports': {},
            'last_update': None,
            'total_observations': 0
        }

        # FRED series summary
        for series_id, config in self.fred_series.items():
            df = self.load_series_from_csv(series_id)
            summary['fred_series'][series_id] = {
                'name': config['name'],
                'observations': len(df),
                'last_date': df['date'].max().isoformat() if not df.empty else None,
                'metadata': self.metadata.get(series_id, {})
            }
            summary['total_observations'] += len(df)

        # CFTC reports summary
        for report_type, config in self.cftc_reports.items():
            # Count all yearly files
            total_rows = 0
            last_date = None
            current_year = datetime.now().year

            for year in [current_year, current_year - 1]:
                csv_file = self.data_dir / f"cftc_{report_type}_{year}.csv"
                if csv_file.exists():
                    try:
                        df = pd.read_csv(csv_file)
                        total_rows += len(df)
                        if not df.empty and 'date' in df.columns:
                            df_date = pd.to_datetime(df['date'], errors='coerce').max()
                            if pd.notna(df_date) and (last_date is None or df_date > last_date):
                                last_date = df_date
                    except Exception as e:
                        logger.warning(f"Error reading {csv_file}: {e}")

            summary['cftc_reports'][report_type] = {
                'name': config['name'],
                'observations': total_rows,
                'last_date': last_date.isoformat() if last_date else None,
                'metadata': self.metadata.get(f"cftc_{report_type}", {})
            }
            summary['total_observations'] += total_rows

        # Overall metadata
        if self.metadata:
            update_times = [pd.to_datetime(v.get('last_update', '2000-01-01')) for v in self.metadata.values()]
            summary['last_update'] = max(update_times).isoformat() if update_times else None

        return summary

    def validate_data_quality(self) -> Dict[str, List[str]]:
        """
        Validate data quality for all series.

        Returns:
            Dictionary mapping series IDs to list of issues found
        """
        issues = {}

        # Check FRED series
        for series_id in self.fred_series.keys():
            df = self.load_series_from_csv(series_id)
            series_issues = []

            if df.empty:
                series_issues.append("No data available")
            else:
                # Check for missing values
                missing_pct = df['value'].isnull().mean() * 100
                if missing_pct > 5:
                    series_issues.append(f"High missing values: {missing_pct:.1f}%")

                # Check date continuity (for daily series)
                if not df.empty and len(df) > 1:
                    df = df.sort_values('date')
                    date_diffs = df['date'].diff().dt.days
                    gaps = (date_diffs > 1).sum()
                    if gaps > 0:
                        series_issues.append(f"Date gaps detected: {gaps} missing days")

                # Check for outliers (values more than 5 std dev from mean)
                if not df['value'].isnull().all():
                    mean_val = df['value'].mean()
                    std_val = df['value'].std()
                    outliers = ((df['value'] - mean_val).abs() > 5 * std_val).sum()
                    if outliers > 0:
                        series_issues.append(f"Potential outliers: {outliers} observations")

            if series_issues:
                issues[series_id] = series_issues

        return issues

    def load_all_series_as_df(self) -> pd.DataFrame:
        """
        Loads all FRED and CFTC data from CSVs into a single merged DataFrame.

        This method iterates through all defined FRED series and CFTC reports,
        loads their corresponding CSV files, and merges them into a single
        time-series DataFrame, indexed by date.

        Returns:
            A pandas DataFrame containing all fundamental data, or an empty
            DataFrame if no data is found.
        """
        if not self.fred:
            logger.warning("Cannot load fundamental data because FRED API key is not configured.")
            return pd.DataFrame()

        all_dfs = []
        # Normalize series list: prefer fred_series keys, and try to extract a 'series_name' from cftc_reports entries
        all_series = list(self.fred_series.keys())
        for report in self.cftc_reports.values():
            if isinstance(report, dict):
                # Try common keys that might indicate the series name
                name = report.get('series_name') or report.get('series') or report.get('id') or report.get('name')
                if name:
                    all_series.append(name)
                else:
                    # if it's an unexpected dict, append its string representation to avoid KeyError
                    all_series.append(str(report))
            else:
                # Non-dict report entries (string identifiers)
                all_series.append(str(report))

        for series_id in all_series:
            df = self.load_series_from_csv(series_id)
            if not df.empty:
                all_dfs.append(df)

        if not all_dfs:
            return pd.DataFrame()

        # Merge all DataFrames on date, using outer join to preserve all dates
        merged_df = all_dfs[0]
        for df in all_dfs[1:]:
            merged_df = pd.merge(merged_df, df, on='date', how='outer', suffixes=('', '_dup'))

        # Remove duplicate columns
        merged_df = merged_df.loc[:, ~merged_df.columns.str.endswith('_dup')]

        # Sort by date
        merged_df = merged_df.sort_values('date')

        return merged_df

def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description='Fundamental Data Pipeline for Forex Trading')
    parser.add_argument('--full', action='store_true', help='Run full update (default)')
    parser.add_argument('--daily', action='store_true', help='Run daily incremental update')
    parser.add_argument('--series', help='Update specific FRED series')
    parser.add_argument('--summary', action='store_true', help='Show data summary')
    parser.add_argument('--validate', action='store_true', help='Validate data quality')
    parser.add_argument('--data-dir', default='data', help='Data directory (default: data)')

    args = parser.parse_args()

    # Initialize pipeline
    try:
        pipeline = FundamentalDataPipeline(data_dir=args.data_dir)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    if args.summary:
        summary = pipeline.get_data_summary()
        print(json.dumps(summary, indent=2, default=str))
        return 0

    if args.validate:
        issues = pipeline.validate_data_quality()
        if issues:
            print("Data quality issues found:")
            for series, series_issues in issues.items():
                print(f"  {series}:")
                for issue in series_issues:
                    print(f"    - {issue}")
        else:
            print("No data quality issues found.")
        return 0

    if args.series:
        success = pipeline.update_fred_series(args.series)
        print(f"Update {'successful' if success else 'failed'} for {args.series}")
        return 0 if success else 1

    # Run update
    if args.daily:
        results = pipeline.run_daily_update()
    else:
        results = pipeline.run_full_update()

    # Report results
    successful = sum(1 for status in results.values() if status)
    total = len(results)
    print(f"Update completed: {successful}/{total} successful")

    if successful < total:
        print("Failed series:")
        for series, success in results.items():
            if not success:
                print(f"  - {series}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())