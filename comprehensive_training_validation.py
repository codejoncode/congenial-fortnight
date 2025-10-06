#!/usr/bin/env python3
"""
Comprehensive Training Validation and Feature Alignment Report

This script performs a thorough check of the entire training pipeline:
1. Data Loading & Alignment across all timeframes
2. Feature Engineering completeness
3. Signal accuracy measurements
4. Multi-timeframe feature alignment verification
5. Training readiness assessment

Outputs a detailed report showing what signals are accurate and ready for training.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

# Load environment
env_path = BASE_DIR / '.env'
if env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
        print("✅ Loaded .env file")
    except ImportError:
        print("⚠️  python-dotenv not available")

from scripts.forecasting import HybridPriceForecastingEnsemble

class ComprehensiveTrainingValidator:
    """Validates all aspects of the training pipeline"""
    
    def __init__(self, pairs=['EURUSD', 'XAUUSD']):
        self.pairs = pairs
        self.results = {}
        self.feature_alignment = {}
        
    def run_full_validation(self):
        """Run complete validation suite"""
        print("\n" + "="*80)
        print("🔍 COMPREHENSIVE TRAINING VALIDATION REPORT")
        print("="*80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Validating pairs: {', '.join(self.pairs)}")
        print("="*80)
        
        for pair in self.pairs:
            print(f"\n{'='*80}")
            print(f"📊 ANALYZING {pair}")
            print(f"{'='*80}")
            
            self.results[pair] = self._validate_pair(pair)
            
        # Generate summary report
        self._generate_summary_report()
        
    def _validate_pair(self, pair: str) -> dict:
        """Validate a single currency pair"""
        results = {
            'data_loading': {},
            'feature_engineering': {},
            'signal_analysis': {},
            'accuracy_metrics': {},
            'issues': []
        }
        
        try:
            # Initialize forecasting system
            print(f"\n1️⃣  Initializing Forecasting System...")
            fs = HybridPriceForecastingEnsemble(pair)
            print(f"   ✅ {pair} forecasting system initialized")
            
            # Validate data loading
            results['data_loading'] = self._validate_data_loading(fs, pair)
            
            # Validate feature engineering
            results['feature_engineering'] = self._validate_feature_engineering(fs, pair)
            
            # Analyze signals
            results['signal_analysis'] = self._analyze_signals(fs, pair)
            
            # Calculate accuracy metrics
            results['accuracy_metrics'] = self._calculate_accuracy_metrics(fs, pair)
            
        except Exception as e:
            print(f"   ❌ Failed to validate {pair}: {e}")
            results['issues'].append(f"Validation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            
        return results
    
    def _validate_data_loading(self, fs, pair: str) -> dict:
        """Validate all timeframe data loading"""
        print(f"\n2️⃣  Validating Data Loading...")
        
        data_status = {}
        
        # Check all timeframes
        timeframes = {
            'H4 (4-Hour)': getattr(fs, 'intraday_data', None),
            'Daily': getattr(fs, 'daily_data', None),
            'Weekly': getattr(fs, 'weekly_data', None),
            'Monthly': getattr(fs, 'monthly_data', None),
            'Consolidated Price': getattr(fs, 'price_data', None),
            'Fundamental': getattr(fs, 'fundamental_data', None)
        }
        
        for tf_name, tf_data in timeframes.items():
            if tf_data is not None and not tf_data.empty:
                rows = len(tf_data)
                cols = len(tf_data.columns)
                date_range = f"{tf_data.index[0]} to {tf_data.index[-1]}" if isinstance(tf_data.index, pd.DatetimeIndex) and len(tf_data) > 0 else "N/A"
                
                data_status[tf_name] = {
                    'loaded': True,
                    'rows': rows,
                    'columns': cols,
                    'date_range': date_range,
                    'columns_list': list(tf_data.columns)
                }
                
                print(f"   ✅ {tf_name:20s}: {rows:6,} rows × {cols:3} cols | {date_range}")
                
                # Show sample columns
                if cols > 0:
                    sample_cols = ', '.join(list(tf_data.columns)[:5])
                    print(f"      Sample columns: {sample_cols}...")
            else:
                data_status[tf_name] = {'loaded': False, 'rows': 0, 'columns': 0}
                print(f"   ⚠️  {tf_name:20s}: NOT LOADED or EMPTY")
        
        return data_status
    
    def _validate_feature_engineering(self, fs, pair: str) -> dict:
        """Validate feature engineering pipeline"""
        print(f"\n3️⃣  Validating Feature Engineering...")
        
        try:
            # Run feature engineering
            feature_df = fs._prepare_features()
            
            if feature_df.empty:
                print(f"   ❌ Feature engineering returned empty DataFrame")
                return {'success': False, 'features': 0}
            
            # Analyze features by category
            feature_categories = self._categorize_features(feature_df)
            
            print(f"   ✅ Feature engineering successful: {len(feature_df)} rows × {len(feature_df.columns)} features")
            print(f"\n   📊 Feature Categories:")
            
            for category, features in feature_categories.items():
                print(f"      {category:30s}: {len(features):4} features")
                if len(features) > 0 and len(features) <= 5:
                    print(f"         └─ {', '.join(features)}")
                elif len(features) > 5:
                    print(f"         └─ {', '.join(features[:3])} ... and {len(features)-3} more")
            
            # Check for NaN issues
            nan_pct = feature_df.isnull().mean() * 100
            critical_nans = nan_pct[nan_pct > 50].sort_values(ascending=False)
            
            if len(critical_nans) > 0:
                print(f"\n   ⚠️  WARNING: {len(critical_nans)} features with >50% NaN values:")
                for col, pct in critical_nans.head(10).items():
                    print(f"      - {col}: {pct:.1f}% NaN")
            else:
                print(f"\n   ✅ No critical NaN issues (all features <50% NaN)")
            
            # Check alignment across timeframes
            alignment_check = self._check_timeframe_alignment(feature_df)
            
            return {
                'success': True,
                'total_features': len(feature_df.columns),
                'total_rows': len(feature_df),
                'categories': {cat: len(feats) for cat, feats in feature_categories.items()},
                'nan_issues': len(critical_nans),
                'alignment': alignment_check
            }
            
        except Exception as e:
            print(f"   ❌ Feature engineering failed: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def _categorize_features(self, df: pd.DataFrame) -> dict:
        """Categorize features by type"""
        categories = {
            'Price (OHLC)': [],
            'RSI (All Timeframes)': [],
            'MACD (All Timeframes)': [],
            'Moving Averages (SMA/EMA)': [],
            'Holloway Algorithm': [],
            'Day Trading Signals': [],
            'Slump Signals': [],
            'Candlestick Patterns': [],
            'Harmonic Patterns': [],
            'Chart Patterns': [],
            'Elliott Wave': [],
            'Ultimate Signals (SMC/Order Flow)': [],
            'Fundamental Data': [],
            'Volume Indicators': [],
            'Volatility Indicators': [],
            'Time Features': [],
            'Lagged Features': [],
            'Target Variables': [],
            'Other Features': []
        }
        
        for col in df.columns:
            col_lower = col.lower()
            
            if col in ['Open', 'High', 'Low', 'Close']:
                categories['Price (OHLC)'].append(col)
            elif 'rsi' in col_lower:
                categories['RSI (All Timeframes)'].append(col)
            elif 'macd' in col_lower:
                categories['MACD (All Timeframes)'].append(col)
            elif 'sma' in col_lower or 'ema' in col_lower:
                categories['Moving Averages (SMA/EMA)'].append(col)
            elif 'holloway' in col_lower or 'bull_count' in col_lower or 'bear_count' in col_lower or 'slowdown' in col_lower:
                categories['Holloway Algorithm'].append(col)
            elif any(sig in col_lower for sig in ['h1_breakout', 'vwap', 'ribbon', 'macd_scalp', 'inside_outside']):
                categories['Day Trading Signals'].append(col)
            elif 'slump' in col_lower:
                categories['Slump Signals'].append(col)
            elif any(pat in col_lower for pat in ['cdl', 'hammer', 'engulfing', 'doji', 'star']):
                categories['Candlestick Patterns'].append(col)
            elif any(pat in col_lower for pat in ['gartley', 'bat', 'butterfly', 'crab', 'shark']):
                categories['Harmonic Patterns'].append(col)
            elif any(pat in col_lower for pat in ['double_top', 'double_bottom', 'head_shoulders', 'triangle', 'flag', 'cup']):
                categories['Chart Patterns'].append(col)
            elif 'elliott' in col_lower or 'wave' in col_lower:
                categories['Elliott Wave'].append(col)
            elif any(sig in col_lower for sig in ['smc', 'order_flow', 'mtf', 'session', 'master_signal']):
                categories['Ultimate Signals (SMC/Order Flow)'].append(col)
            elif 'fund_' in col_lower:
                categories['Fundamental Data'].append(col)
            elif 'volume' in col_lower:
                categories['Volume Indicators'].append(col)
            elif 'volatility' in col_lower or 'atr' in col_lower:
                categories['Volatility Indicators'].append(col)
            elif any(t in col_lower for t in ['day_of', 'week_of', 'month_of']):
                categories['Time Features'].append(col)
            elif 'lag' in col_lower:
                categories['Lagged Features'].append(col)
            elif 'target' in col_lower:
                categories['Target Variables'].append(col)
            else:
                categories['Other Features'].append(col)
        
        return categories
    
    def _check_timeframe_alignment(self, df: pd.DataFrame) -> dict:
        """Check if multi-timeframe features are properly aligned"""
        print(f"\n   🔄 Checking Multi-Timeframe Alignment...")
        
        # Look for timeframe-specific features
        h4_features = [col for col in df.columns if 'H4' in col or 'h4' in col]
        daily_features = [col for col in df.columns if 'daily' in col.lower()]
        weekly_features = [col for col in df.columns if 'weekly' in col.lower()]
        monthly_features = [col for col in df.columns if 'monthly' in col.lower()]
        
        alignment = {
            'H4': len(h4_features),
            'Daily': len(daily_features),
            'Weekly': len(weekly_features),
            'Monthly': len(monthly_features)
        }
        
        print(f"      H4 features: {alignment['H4']}")
        print(f"      Daily features: {alignment['Daily']}")
        print(f"      Weekly features: {alignment['Weekly']}")
        print(f"      Monthly features: {alignment['Monthly']}")
        
        # Check if RSI exists for all timeframes
        rsi_check = {
            'H4_RSI': any('h4' in col.lower() and 'rsi' in col.lower() for col in df.columns),
            'Daily_RSI': any('daily' in col.lower() and 'rsi' in col.lower() for col in df.columns) or 'rsi_14' in df.columns,
            'Weekly_RSI': any('weekly' in col.lower() and 'rsi' in col.lower() for col in df.columns),
            'Monthly_RSI': any('monthly' in col.lower() and 'rsi' in col.lower() for col in df.columns)
        }
        
        print(f"\n      RSI Alignment Check:")
        for tf, present in rsi_check.items():
            status = "✅" if present else "❌"
            print(f"         {status} {tf}")
        
        return {
            'timeframe_counts': alignment,
            'rsi_alignment': rsi_check,
            'properly_aligned': all(rsi_check.values())
        }
    
    def _analyze_signals(self, fs, pair: str) -> dict:
        """Analyze all signal types"""
        print(f"\n4️⃣  Analyzing Signal Accuracy...")
        
        try:
            feature_df = fs._prepare_features()
            
            if feature_df.empty or 'target_1d' not in feature_df.columns:
                print(f"   ⚠️  Cannot analyze signals: missing data or target")
                return {'success': False}
            
            # Find all signal columns
            signal_patterns = [
                ('day_trading', ['h1_breakout', 'vwap', 'ribbon', 'rsi_mean_reversion', 'inside_outside']),
                ('slump', ['slump']),
                ('candlestick', ['cdl_']),
                ('harmonic', ['gartley', 'bat', 'butterfly', 'crab', 'shark']),
                ('chart', ['double_top', 'double_bottom', 'head_shoulders', 'triangle', 'flag', 'cup']),
                ('elliott_wave', ['elliott', 'wave']),
                ('ultimate', ['smc', 'order_flow', 'mtf', 'session', 'master_signal'])
            ]
            
            signal_accuracies = {}
            
            for signal_type, patterns in signal_patterns:
                type_signals = []
                for pattern in patterns:
                    type_signals.extend([col for col in feature_df.columns if pattern in col.lower() and 'signal' in col.lower()])
                
                if type_signals:
                    print(f"\n   📈 {signal_type.upper()} Signals ({len(type_signals)} signals):")
                    
                    for signal_col in type_signals:
                        accuracy = self._calculate_signal_accuracy(feature_df, signal_col, 'target_1d')
                        signal_accuracies[signal_col] = accuracy
                        
                        if accuracy is not None:
                            status = "✅" if accuracy > 0.55 else "⚠️" if accuracy > 0.50 else "❌"
                            print(f"      {status} {signal_col:40s}: {accuracy:6.2%}")
            
            return {
                'success': True,
                'total_signals': len(signal_accuracies),
                'accuracies': signal_accuracies
            }
            
        except Exception as e:
            print(f"   ❌ Signal analysis failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _calculate_signal_accuracy(self, df: pd.DataFrame, signal_col: str, target_col: str) -> float:
        """Calculate accuracy of a single signal"""
        try:
            # Filter rows where signal is not zero/null
            signal_active = df[df[signal_col].notna() & (df[signal_col] != 0)].copy()
            
            if len(signal_active) < 10:  # Need minimum samples
                return None
            
            # For bullish signals (positive values), predict target=1
            # For bearish signals (negative values), predict target=0
            signal_active['prediction'] = (signal_active[signal_col] > 0).astype(int)
            
            # Calculate accuracy
            correct = (signal_active['prediction'] == signal_active[target_col]).sum()
            total = len(signal_active)
            
            return correct / total if total > 0 else None
            
        except Exception:
            return None
    
    def _calculate_accuracy_metrics(self, fs, pair: str) -> dict:
        """Calculate overall accuracy metrics"""
        print(f"\n5️⃣  Calculating Accuracy Metrics...")
        
        try:
            feature_df = fs._prepare_features()
            
            if feature_df.empty or 'target_1d' not in feature_df.columns:
                print(f"   ⚠️  Cannot calculate metrics: missing data")
                return {'success': False}
            
            # Calculate baseline (always predict most common class)
            most_common = feature_df['target_1d'].mode()[0] if len(feature_df['target_1d'].mode()) > 0 else 1
            baseline_accuracy = (feature_df['target_1d'] == most_common).mean()
            
            print(f"   📊 Baseline Accuracy (most common class): {baseline_accuracy:.2%}")
            
            # Calculate combined signal accuracy (if master_signal exists)
            if 'master_signal' in feature_df.columns:
                master_df = feature_df[feature_df['master_signal'].notna() & (feature_df['master_signal'] != 0)].copy()
                
                if len(master_df) > 0:
                    master_df['prediction'] = (master_df['master_signal'] > 0).astype(int)
                    master_accuracy = (master_df['prediction'] == master_df['target_1d']).mean()
                    print(f"   📊 Master Signal Accuracy: {master_accuracy:.2%}")
                else:
                    master_accuracy = None
                    print(f"   ⚠️  Master signal exists but no active signals")
            else:
                master_accuracy = None
                print(f"   ⚠️  No master signal available")
            
            return {
                'success': True,
                'baseline': baseline_accuracy,
                'master_signal': master_accuracy,
                'samples': len(feature_df)
            }
            
        except Exception as e:
            print(f"   ❌ Metrics calculation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_summary_report(self):
        """Generate final summary report"""
        print(f"\n{'='*80}")
        print("📋 FINAL TRAINING READINESS SUMMARY")
        print(f"{'='*80}\n")
        
        for pair, results in self.results.items():
            print(f"🎯 {pair}:")
            
            # Data loading status
            data_status = results.get('data_loading', {})
            loaded_tfs = sum(1 for tf_data in data_status.values() if tf_data.get('loaded', False))
            print(f"   ✅ Data Loading: {loaded_tfs}/6 timeframes loaded")
            
            # Feature engineering status
            fe_status = results.get('feature_engineering', {})
            if fe_status.get('success'):
                total_features = fe_status.get('total_features', 0)
                print(f"   ✅ Feature Engineering: {total_features} features generated")
                
                # Show category breakdown
                categories = fe_status.get('categories', {})
                key_categories = ['Holloway Algorithm', 'Day Trading Signals', 'Ultimate Signals (SMC/Order Flow)', 
                                 'Harmonic Patterns', 'Chart Patterns', 'Elliott Wave']
                for cat in key_categories:
                    count = categories.get(cat, 0)
                    status = "✅" if count > 0 else "⚠️"
                    print(f"      {status} {cat}: {count} features")
            else:
                print(f"   ❌ Feature Engineering: FAILED")
            
            # Signal analysis
            signal_status = results.get('signal_analysis', {})
            if signal_status.get('success'):
                total_signals = signal_status.get('total_signals', 0)
                accuracies = signal_status.get('accuracies', {})
                
                accurate_signals = sum(1 for acc in accuracies.values() if acc and acc > 0.55)
                print(f"   ✅ Signal Analysis: {accurate_signals}/{total_signals} signals >55% accuracy")
            
            # Accuracy metrics
            accuracy_status = results.get('accuracy_metrics', {})
            if accuracy_status.get('success'):
                baseline = accuracy_status.get('baseline', 0)
                master = accuracy_status.get('master_signal')
                print(f"   📊 Baseline: {baseline:.2%}")
                if master:
                    print(f"   📊 Master Signal: {master:.2%}")
            
            print()
        
        # Final verdict
        print(f"{'='*80}")
        all_successful = all(
            results.get('feature_engineering', {}).get('success', False)
            for results in self.results.values()
        )
        
        if all_successful:
            print("✅ TRAINING READINESS: SYSTEM IS READY FOR TRAINING")
            print("\nNext Steps:")
            print("   1. Run: python -m scripts.automated_training --pairs EURUSD XAUUSD")
            print("   2. Monitor training logs for accuracy improvements")
            print("   3. Review model artifacts in models/ directory")
        else:
            print("⚠️  TRAINING READINESS: ISSUES DETECTED - REVIEW ABOVE")
            print("\nRequired Actions:")
            print("   1. Fix any failed feature engineering")
            print("   2. Ensure fundamental data is loading properly")
            print("   3. Verify data alignment across timeframes")
        
        print(f"{'='*80}\n")

def main():
    """Run comprehensive validation"""
    validator = ComprehensiveTrainingValidator(pairs=['EURUSD', 'XAUUSD'])
    validator.run_full_validation()
    
    print(f"\n✅ Validation complete! Review the report above.\n")

if __name__ == "__main__":
    main()
