# CRITICAL DATA LOADING FIX FOR FOREX TRADING SYSTEM
# Addresses: Empty DataFrames, Date Column Detection, Data Validation Issues

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import os
from typing import Dict, List, Optional, Tuple
import warnings
import re
warnings.filterwarnings('ignore')

# Auto-load .env if python-dotenv is available so environment keys like FRED_API_KEY are set
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ForexDataLoader:
    """
    Robust forex data loader that handles common issues:
    - Empty DataFrames
    - Missing or unidentifiable date columns  
    - Inconsistent data formats
    - File access problems
    """
    
    def __init__(self):
        self.loaded_data = {}
        self.data_issues = {}
        
    def diagnose_data_file(self, filepath: str, timeframe: str) -> Dict:
        """Comprehensive diagnosis of data file issues"""
        
        diagnosis = {
            'file_exists': False,
            'file_size': 0,
            'is_readable': False,
            'row_count': 0,
            'column_count': 0,
            'columns': [],
            'date_column_found': False,
            'date_column_name': None,
            'date_format_issues': [],
            'data_quality_issues': [],
            'recommendations': []
        }
        
        try:
            # Check file existence
            if not Path(filepath).exists():
                diagnosis['recommendations'].append(f"Create missing file: {filepath}")
                logger.error(f"❌ {timeframe}: File does not exist - {filepath}")
                return diagnosis
                
            diagnosis['file_exists'] = True
            diagnosis['file_size'] = Path(filepath).stat().st_size
            
            if diagnosis['file_size'] == 0:
                diagnosis['recommendations'].append(f"File is empty, populate with data")
                logger.error(f"❌ {timeframe}: File is empty - {filepath}")
                return diagnosis
            
            # Detect delimiter from first non-empty line using a simple heuristic
            def _detect_delimiter(path: str) -> str:
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
                        for raw in fh:
                            line = raw.strip()
                            if line:
                                first = line
                                break
                        else:
                            return '\n'  # empty
                except Exception:
                    return ','

                # Heuristic checks: tab, comma, or multiple spaces
                if '\t' in first or '\\t' in first:
                    return '\t'
                if ',' in first:
                    return ','
                # multiple consecutive spaces -> whitespace
                if re.search(r'\s{2,}', first):
                    return r'\s+'
                # fallback to comma
                return ','

            sep = _detect_delimiter(filepath)
            try:
                if sep == r'\\s+':
                    full_df = pd.read_csv(filepath, sep=r'\\s+', engine='python')
                else:
                    full_df = pd.read_csv(filepath, sep=sep, engine='python')

                # If full_df has a single column and contains embedded separators, try splitting
                # If full_df has a single column and contains embedded separators, try splitting
                if full_df.shape[1] == 1:
                    col0 = full_df.columns[0]
                    sample_vals = full_df.iloc[:, 0].astype(str).head(5).tolist()

                    # First try simple splits on observed characters
                    if any('\t' in s or '\\t' in s for s in sample_vals):
                        split_df = full_df.iloc[:, 0].astype(str).str.split('\t', expand=True)
                        full_df = split_df
                    elif any(re.search(r'\s{2,}', s) for s in sample_vals):
                        split_df = full_df.iloc[:, 0].astype(str).str.split(r'\s+', expand=True)
                        full_df = split_df
                    elif any(',' in s for s in sample_vals):
                        split_df = full_df.iloc[:, 0].astype(str).str.split(',', expand=True)
                        full_df = split_df
                    else:
                        # Header-aware re-read: read the first non-empty line to detect header tokens
                        try:
                            with open(filepath, 'r', encoding='utf-8', errors='ignore') as fh:
                                header_line = None
                                for raw in fh:
                                    line = raw.rstrip('\n')
                                    if line and not line.isspace():
                                        header_line = line
                                        break

                            if header_line:
                                # detect delimiter from header_line
                                if '\t' in header_line or '\\t' in header_line:
                                    hdr_sep = '\t'
                                elif ',' in header_line:
                                    hdr_sep = ','
                                elif re.search(r'\s{2,}', header_line):
                                    hdr_sep = r'\s+'
                                else:
                                    hdr_sep = sep

                                # try re-reading using header-aware sep
                                try:
                                    if hdr_sep == r'\s+':
                                        candidate = pd.read_csv(filepath, sep=r'\s+', engine='python', header=0)
                                    else:
                                        candidate = pd.read_csv(filepath, sep=hdr_sep, engine='python', header=0)

                                    if candidate.shape[1] > 1:
                                        full_df = candidate
                                    else:
                                        # last resort: read with header=None and derive header from header_line
                                        if hdr_sep == r'\s+':
                                            tokens = re.split(r'\s+', header_line.strip())
                                        else:
                                            tokens = header_line.strip().split(hdr_sep)

                                        candidate = pd.read_csv(filepath, sep=hdr_sep, engine='python', header=None, names=tokens, skiprows=1)
                                        if candidate.shape[1] == len(tokens):
                                            full_df = candidate
                                except Exception:
                                    pass
                        except Exception:
                            pass

                diagnosis['is_readable'] = True
                diagnosis['row_count'] = len(full_df)
                diagnosis['column_count'] = len(full_df.columns)
                diagnosis['columns'] = list(full_df.columns)

                logger.info(f"📊 {timeframe}: {diagnosis['row_count']} rows, {diagnosis['column_count']} columns")
                logger.info(f"   Columns: {diagnosis['columns']}")
            except Exception as e:
                diagnosis['recommendations'].append(f"Fix file format issues: {str(e)}")
                logger.error(f"❌ {timeframe}: Cannot read full file - {str(e)}")
                return diagnosis
            
            # Analyze date column detection (use the fully parsed dataframe)
            date_column = self._identify_date_column(full_df.columns)
            if date_column:
                diagnosis['date_column_found'] = True
                diagnosis['date_column_name'] = date_column

                # Test date parsing
                try:
                    sample_dates = full_df[date_column].dropna().head()
                    if len(sample_dates) > 0:
                        pd.to_datetime(sample_dates.iloc[0])
                        logger.info(f"✅ {timeframe}: Date column '{date_column}' found and parseable")
                    else:
                        diagnosis['date_format_issues'].append("No non-null date values found")
                except Exception as e:
                    diagnosis['date_format_issues'].append(f"Date parsing failed: {str(e)}")
                    diagnosis['recommendations'].append("Fix date format in date column")
            else:
                diagnosis['recommendations'].append("Add or rename date/timestamp column")
                logger.error(f"❌ {timeframe}: No date column found in {diagnosis['columns']}")

            # Check for OHLC columns
            required_cols = ['open', 'high', 'low', 'close']
            missing_ohlc = []
            df_cols_lower = [str(col).lower() for col in full_df.columns]

            for req_col in required_cols:
                if req_col not in df_cols_lower:
                    missing_ohlc.append(req_col)
            
            if missing_ohlc:
                diagnosis['data_quality_issues'].append(f"Missing OHLC columns: {missing_ohlc}")
                diagnosis['recommendations'].append("Add or rename OHLC columns")
            
            # Check data completeness
            if diagnosis['row_count'] < 100:
                diagnosis['data_quality_issues'].append(f"Insufficient data: only {diagnosis['row_count']} rows")
                diagnosis['recommendations'].append("Need at least 252 rows (1 year) for reliable training")
            
        except Exception as e:
            diagnosis['recommendations'].append(f"General file access error: {str(e)}")
            logger.error(f"❌ {timeframe}: Diagnosis failed - {str(e)}")
        
        return diagnosis
    
    def _identify_date_column(self, columns: List[str]) -> Optional[str]:
        """Identify date/timestamp column from various naming conventions"""
        
        # Common date column patterns (in order of preference)
        date_patterns = [
            'timestamp', 'date', 'datetime', 'time',
            'Date', 'DateTime', 'Timestamp', 'Time',
            'DATE', 'DATETIME', 'TIMESTAMP', 'TIME'
        ]
        
        # Direct matches first
        for pattern in date_patterns:
            if pattern in columns:
                return pattern
        
        # Partial matches
        for col in columns:
            col_lower = col.lower()
            if any(pattern.lower() in col_lower for pattern in date_patterns[:4]):
                return col
                
        return None
    
    def fix_data_file(self, filepath: str, timeframe: str) -> bool:
        """Attempt to fix common data file issues"""
        
        try:
            if not Path(filepath).exists():
                logger.error(f"❌ {timeframe}: Cannot fix non-existent file {filepath}")
                return False
            
            logger.info(f"🔧 {timeframe}: Attempting to fix data file...")
            
            # Read the problematic file trying multiple separators
            # Use delimiter detection to read file robustly
            def _detect_delimiter(path: str) -> str:
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
                        for raw in fh:
                            line = raw.strip()
                            if line:
                                first = line
                                break
                        else:
                            return ','
                except Exception:
                    return ','

                # Heuristics similar to diagnosis
                if '\t' in first or '\\t' in first:
                    return '\t'
                if ',' in first:
                    return ','
                if re.search(r'\s{2,}', first):
                    return r'\s+'
                return ','

            sep = _detect_delimiter(filepath)
            try:
                if sep == r'\\s+':
                    df = pd.read_csv(filepath, sep=r'\\s+', engine='python')
                else:
                    df = pd.read_csv(filepath, sep=sep, engine='python')

                # If single column with embedded delimiters, split it
                if df.shape[1] == 1:
                    col0 = df.columns[0]
                    sample_vals = df.iloc[:, 0].astype(str).head(5).tolist()

                    # Try splitting by explicit characters
                    if any('\t' in s or '\\t' in s for s in sample_vals):
                        df = df.iloc[:, 0].astype(str).str.split('\t', expand=True)
                    elif any(re.search(r'\s{2,}', s) for s in sample_vals):
                        df = df.iloc[:, 0].astype(str).str.split(r'\s+', expand=True)
                    elif any(',' in s for s in sample_vals):
                        df = df.iloc[:, 0].astype(str).str.split(',', expand=True)
                    else:
                        # Header-aware re-read: extract header line and derive tokens
                        try:
                            with open(filepath, 'r', encoding='utf-8', errors='ignore') as fh:
                                header_line = None
                                for raw in fh:
                                    line = raw.rstrip('\n')
                                    if line and not line.isspace():
                                        header_line = line
                                        break

                            if header_line:
                                if '\t' in header_line or '\\t' in header_line:
                                    hdr_sep = '\t'
                                elif ',' in header_line:
                                    hdr_sep = ','
                                elif re.search(r'\s{2,}', header_line):
                                    hdr_sep = r'\s+'
                                else:
                                    hdr_sep = sep

                                # Try re-reading with header-aware separator
                                try:
                                    if hdr_sep == r'\s+':
                                        candidate = pd.read_csv(filepath, sep=r'\s+', engine='python', header=0)
                                    else:
                                        candidate = pd.read_csv(filepath, sep=hdr_sep, engine='python', header=0)

                                    if candidate.shape[1] > 1:
                                        df = candidate
                                    else:
                                        # Derive tokens and read body separately
                                        if hdr_sep == r'\s+':
                                            tokens = re.split(r'\s+', header_line.strip())
                                        else:
                                            tokens = header_line.strip().split(hdr_sep)

                                        candidate = pd.read_csv(filepath, sep=hdr_sep, engine='python', header=None, names=tokens, skiprows=1)
                                        if candidate.shape[1] == len(tokens):
                                            df = candidate
                                except Exception:
                                    pass
                        except Exception:
                            pass
            except Exception as e:
                logger.error(f"❌ {timeframe}: Could not read the file with detected separator: {e}")
                return False
            
            if len(df) == 0:
                logger.error(f"❌ {timeframe}: Cannot fix empty DataFrame")
                return False
            
            original_cols = list(df.columns)
            logger.info(f"   Original columns: {original_cols}")
            
            # If original columns indicate a single header string, normalize by splitting into tokens
            if len(original_cols) == 1:
                hdr = original_cols[0]
                if isinstance(hdr, str) and ("\t" in hdr or "," in hdr or re.search(r"\s{2,}", hdr)):
                    # derive tokens
                    if '\t' in hdr or '\\t' in hdr:
                        tokens = hdr.split('\t')
                    elif ',' in hdr:
                        tokens = hdr.split(',')
                    else:
                        tokens = re.split(r'\s+', hdr.strip())

                    # if df has one column with tab-joined values, split that column
                    if df.shape[1] == 1:
                        df = df.iloc[:, 0].astype(str).str.split('\t' if '\t' in hdr else (',' if ',' in hdr else r'\s+'), expand=True)

                    # set tokens as column names if lengths match
                    if df.shape[1] == len(tokens):
                        df.columns = [t.strip() for t in tokens]

            # Fix column names
            df = self._standardize_column_names(df)
            
            # Identify and fix date column
            date_col = self._identify_date_column(df.columns)
            if date_col is None:
                # Try to create a date column if we have an index that looks like dates
                if df.index.name and 'date' in df.index.name.lower():
                    df = df.reset_index()
                    date_col = df.columns[0]
                elif len(df.columns) > 0 and any('date' in str(col).lower() for col in original_cols):
                    # Find the original date-like column
                    for col in original_cols:
                        if 'date' in str(col).lower():
                            date_col = col
                            break
            
            if date_col:
                # Standardize date column name
                if date_col != 'timestamp':
                    df = df.rename(columns={date_col: 'timestamp'})
                
                # Fix date format
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    logger.info(f"✅ {timeframe}: Fixed date column")
                except:
                    logger.error(f"❌ {timeframe}: Could not parse dates in column {date_col}")
                    return False
            else:
                logger.error(f"❌ {timeframe}: No date column could be identified or created")
                return False
            
            # Ensure numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove rows with all NaN values
            df = df.dropna(how='all')
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Save fixed file
            backup_path = filepath + '.backup'
            if not Path(backup_path).exists():
                # Create backup first
                import shutil
                shutil.copy2(filepath, backup_path)
                logger.info(f"   Created backup: {backup_path}")
            
            df.to_csv(filepath, index=False)
            logger.info(f"✅ {timeframe}: Fixed and saved file with {len(df)} rows")
            
            # Verify the fix worked
            verification_df = pd.read_csv(filepath)
            if len(verification_df) > 0 and 'timestamp' in verification_df.columns:
                logger.info(f"✅ {timeframe}: Verification successful")
                return True
            else:
                logger.error(f"❌ {timeframe}: Fix verification failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ {timeframe}: Fix attempt failed - {str(e)}")
            return False
    
    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to expected format"""
        # First, apply explicit known mappings (case-sensitive keys)
        column_mapping = {
            'Date': 'timestamp', 'DateTime': 'timestamp', 'Timestamp': 'timestamp', 'Time': 'timestamp',
            'DATE': 'timestamp', 'DATETIME': 'timestamp', 'TIMESTAMP': 'timestamp', 'TIME': 'timestamp',
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
            'O': 'open', 'H': 'high', 'L': 'low', 'C': 'close',
            'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low', 'CLOSE': 'close',
            'Volume': 'volume', 'VOLUME': 'volume', 'Vol': 'volume'
        }

        df = df.rename(columns=column_mapping)

        # Normalize remaining column names: remove angle brackets, non-alphanumerics -> underscore, lowercase
        normalized_map = {}
        for col in list(df.columns):
            if not isinstance(col, str):
                continue
            orig = col
            # remove enclosing angle brackets and whitespace
            stripped = orig.strip()
            stripped = re.sub(r"^[<\[]+|[>\]]+$", '', stripped)
            # replace non-alphanumeric with underscore
            cleaned = re.sub(r'[^0-9A-Za-z_]+', '_', stripped).lower()

            # map common tokens to canonical names
            if cleaned in ['date', 'date_time', 'datetime', 'timestamp']:
                new_name = 'timestamp'
            elif cleaned == 'time':
                # keep time separate for now; we'll combine if a date column exists
                new_name = 'time'
            elif cleaned in ['open', 'o']:
                new_name = 'open'
            elif cleaned in ['high', 'h']:
                new_name = 'high'
            elif cleaned in ['low', 'l']:
                new_name = 'low'
            elif cleaned in ['close', 'c']:
                new_name = 'close'
            elif cleaned in ['volume', 'vol', 'tickvol', 'tick_volume', 'tickvolume']:
                new_name = 'volume'
            elif cleaned in ['spread']:
                new_name = 'spread'
            else:
                # collapse multiple underscores
                new_name = re.sub(r'_+', '_', cleaned).strip('_') if cleaned else orig

            # Avoid collisions: if new_name already present map to a unique name
            if new_name != orig:
                target = new_name
                suffix = 1
                while target in normalized_map.values() or target in df.columns:
                    target = f"{new_name}_{suffix}"
                    suffix += 1
                normalized_map[orig] = target

        if normalized_map:
            df = df.rename(columns=normalized_map)
            logger.info(f"   Standardized column names: {normalized_map}")

        # If we have both a date-like column and a separate time column, combine them
        cols_lower = [c.lower() for c in df.columns if isinstance(c, str)]
        if 'timestamp' not in cols_lower and 'date' in cols_lower and 'time' in cols_lower:
            try:
                # find actual column names (case-sensitive)
                date_col = next(c for c in df.columns if isinstance(c, str) and c.lower() == 'date')
                time_col = next(c for c in df.columns if isinstance(c, str) and c.lower() == 'time')
                df['timestamp'] = df[date_col].astype(str) + ' ' + df[time_col].astype(str)
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df = df.drop(columns=[time_col])
                logger.info(f"   Combined {date_col} and {time_col} into 'timestamp'")
            except Exception:
                pass

        return df

        return df
    
    def load_timeframe_data(self, filepath: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load and validate data for a specific timeframe"""
        
        logger.info(f"\n🔍 Loading {timeframe} data from {filepath}")
        
        # First, diagnose the file
        diagnosis = self.diagnose_data_file(filepath, timeframe)
        self.data_issues[timeframe] = diagnosis
        
        if not diagnosis['file_exists']:
            return None
            
        if not diagnosis['is_readable']:
            return None
            
        if diagnosis['row_count'] == 0:
            logger.warning(f"⚠️  {timeframe}: File contains no data rows")
            return None
        
        # Try to fix issues if found
        needs_fixing = (
            not diagnosis['date_column_found'] or 
            len(diagnosis['date_format_issues']) > 0 or
            len(diagnosis['data_quality_issues']) > 0
        )
        
        if needs_fixing:
            logger.warning(f"⚠️  {timeframe}: Data issues detected, attempting fixes...")
            if not self.fix_data_file(filepath, timeframe):
                logger.error(f"❌ {timeframe}: Could not fix data issues")
                return None
        
        # Load the (potentially fixed) data
        try:
            df = pd.read_csv(filepath)
            
            if len(df) == 0:
                logger.error(f"❌ {timeframe}: DataFrame is empty after loading")
                return None
            
            # Ensure we have a timestamp column
            timestamp_col = self._identify_date_column(df.columns)
            if timestamp_col is None:
                logger.error(f"❌ {timeframe}: No timestamp column found after fixes")
                return None
            
            if timestamp_col != 'timestamp':
                df = df.rename(columns={timestamp_col: 'timestamp'})
            
            # Parse timestamps
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except Exception as e:
                logger.error(f"❌ {timeframe}: Cannot parse timestamps - {str(e)}")
                return None
            
            # Sort and clean
            df = df.sort_values('timestamp').reset_index(drop=True)
            df = df.dropna(how='all')
            
            # Final validation
            if len(df) == 0:
                logger.error(f"❌ {timeframe}: No valid data remaining after cleaning")
                return None
            
            logger.info(f"✅ {timeframe}: Successfully loaded {len(df)} rows")
            logger.info(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            self.loaded_data[timeframe] = df
            return df
            
        except Exception as e:
            logger.error(f"❌ {timeframe}: Loading failed - {str(e)}")
            return None
    
    def load_all_timeframes(self, data_config: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """Load all timeframe data with comprehensive error handling"""
        
        logger.info("🚀 STARTING COMPREHENSIVE DATA LOADING")
        logger.info("=" * 60)
        
        successful_loads = {}
        failed_loads = {}
        
        for timeframe, filepath in data_config.items():
            logger.info(f"\n--- Processing {timeframe} ---")
            
            df = self.load_timeframe_data(filepath, timeframe)
            
            if df is not None:
                successful_loads[timeframe] = df
                logger.info(f"✅ {timeframe}: SUCCESS")
            else:
                failed_loads[timeframe] = filepath
                logger.error(f"❌ {timeframe}: FAILED")
        
        logger.info(f"\n📊 LOADING SUMMARY")
        logger.info(f"✅ Successful: {len(successful_loads)} timeframes")
        logger.info(f"❌ Failed: {len(failed_loads)} timeframes")
        
        if failed_loads:
            logger.error("🚨 FAILED TIMEFRAMES:")
            for tf, path in failed_loads.items():
                logger.error(f"   {tf}: {path}")
                if tf in self.data_issues:
                    recommendations = self.data_issues[tf].get('recommendations', [])
                    for rec in recommendations:
                        logger.error(f"      ➡️  {rec}")
        
        if not successful_loads:
            logger.error("❌ CRITICAL: NO DATA COULD BE LOADED - CHECK FILE PATHS AND FORMATS")
            raise RuntimeError("Data loading completely failed")
        
        logger.info("=" * 60)
        return successful_loads
    
    def generate_sample_data(self, filepath: str, timeframe: str, rows: int = 1000):
        """Generate sample forex data if files are missing/empty"""
        
        logger.info(f"🔧 Generating sample {timeframe} data: {rows} rows")
        
        # Create date range
        if timeframe.lower() == 'daily':
            freq = 'D'
            start_date = pd.Timestamp.now() - pd.Timedelta(days=rows)
        elif timeframe.lower() == 'h4':
            freq = '4H'
            start_date = pd.Timestamp.now() - pd.Timedelta(hours=rows*4)
        elif timeframe.lower() == 'h1':
            freq = '1H'
            start_date = pd.Timestamp.now() - pd.Timedelta(hours=rows)
        elif timeframe.lower() == 'weekly':
            freq = 'W'
            start_date = pd.Timestamp.now() - pd.Timedelta(weeks=rows)
        else:
            freq = 'D'  # Default
            start_date = pd.Timestamp.now() - pd.Timedelta(days=rows)
        
        dates = pd.date_range(start=start_date, periods=rows, freq=freq)
        
        # Generate realistic EURUSD-like data
        np.random.seed(42)  # For reproducibility
        
        base_price = 1.1000
        returns = np.random.normal(0, 0.005, rows)  # 0.5% daily volatility
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC from price series
        opens = prices
        closes = prices * (1 + np.random.normal(0, 0.001, rows))
        highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.002, rows)))
        lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.002, rows)))
        volumes = np.random.randint(1000, 10000, rows)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        
        # Ensure data directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to file
        df.to_csv(filepath, index=False)
        
        logger.info(f"✅ Generated {rows} rows of sample {timeframe} data")
        logger.info(f"   Saved to: {filepath}")
        
        return df

def emergency_data_recovery():
    """Emergency function to create missing data files"""
    
    logger.info("🚨 EMERGENCY DATA RECOVERY MODE")
    
    # Standard data paths
    data_paths = {
        'Daily': 'data/EURUSD_Daily.csv',
        'H4': 'data/EURUSD_H4.csv', 
        'H1': 'data/EURUSD_H1.csv',
        'Weekly': 'data/EURUSD_Weekly.csv'
    }
    
    loader = ForexDataLoader()
    
    # Check which files need recovery
    for timeframe, filepath in data_paths.items():
        if not Path(filepath).exists() or Path(filepath).stat().st_size == 0:
            logger.warning(f"⚠️  {timeframe}: Missing or empty file - generating sample data")
            
            # Generate appropriate amount of data
            rows = {'Daily': 1000, 'H4': 2000, 'H1': 5000, 'Weekly': 260}
            loader.generate_sample_data(filepath, timeframe, rows.get(timeframe, 1000))
    
    logger.info("🔧 Emergency recovery complete")

# Main integration function for your forecasting.py
def robust_data_loading_pipeline():
    """
    Main function to replace your current data loading logic
    Call this BEFORE any feature engineering or training
    """
    
    logger.info("🚀 STARTING ROBUST DATA LOADING PIPELINE")
    
    # Data configuration - adjust paths as needed
    data_config = {
        'Daily': 'data/EURUSD_Daily.csv',
        'H4': 'data/EURUSD_H4.csv',
        'H1': 'data/EURUSD_H1.csv', 
        'Weekly': 'data/EURUSD_Weekly.csv'
    }
    
    try:
        # Initialize loader
        loader = ForexDataLoader()
        
        # Attempt normal loading
        loaded_data = loader.load_all_timeframes(data_config)
        
        if len(loaded_data) == 0:
            logger.error("❌ NORMAL LOADING FAILED - Initiating emergency recovery")
            emergency_data_recovery()
            
            # Try loading again after recovery
            loader = ForexDataLoader()
            loaded_data = loader.load_all_timeframes(data_config)
        
        if len(loaded_data) == 0:
            raise RuntimeError("❌ DATA LOADING COMPLETELY FAILED")
        
        logger.info(f"✅ DATA LOADING SUCCESSFUL: {list(loaded_data.keys())}")
        
        return loaded_data
        
    except Exception as e:
        logger.error(f"❌ CRITICAL DATA LOADING ERROR: {str(e)}")
        raise

if __name__ == "__main__":
    # Test the data loading pipeline
    try:
        data = robust_data_loading_pipeline()
        print(f"✅ Successfully loaded data for: {list(data.keys())}")
        
        for timeframe, df in data.items():
            print(f"\n{timeframe}:")
            print(f"  Rows: {len(df)}")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        print("Check the logs above for detailed error information")
