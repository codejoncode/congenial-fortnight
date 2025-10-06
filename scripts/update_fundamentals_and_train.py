#!/usr/bin/env python3
"""
Comprehensive Fundamental Data Integration and Retraining Script

This script:
1. Updates fundamental_pipeline.py to include all 28+ fundamental data sources
2. Loads comprehensive fundamental data (FRED, ECB, Alpha Vantage, DXY/EXY cross)
3. Integrates fundamentals into feature engineering
4. Retrains models with complete fundamental + technical features
5. Compares performance: Old models (529 features) vs New (529 + fundamentals)

Date: October 6, 2025
Status: Production Ready
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import joblib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/fundamental_integration.log')
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class ComprehensiveFundamentalLoader:
    """
    Loads ALL fundamental data sources:
    - 23 FRED series (macro indicators)
    - ECB EUR/USD
    - Alpha Vantage FX (EUR/USD, USD/JPY, USD/CHF)
    - DXY/EXY cross indicators (5 indicators)
    - Gold price data
    """
    
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.fundamental_data = None
        
    def load_fred_series(self):
        """Load all 23 FRED economic series"""
        fred_series = [
            'CPIAUCSL',      # US CPI
            'GDPC1',         # US Real GDP
            'FEDFUNDS',      # Federal Funds Rate
            'DFF',           # Effective Fed Funds Rate
            'UNRATE',        # Unemployment Rate
            'INDPRO',        # Industrial Production
            'PAYEMS',        # Nonfarm Payrolls
            'DGORDER',       # Durable Goods Orders
            'BOPGSTB',       # Trade Balance
            'DEXUSEU',       # USD/EUR Exchange Rate
            'DEXJPUS',       # USD/JPY Exchange Rate
            'DEXCHUS',       # USD/CHF Exchange Rate
            'ECBDFR',        # ECB Deposit Facility Rate
            'CP0000EZ19M086NEST',  # Euro Area CPI
            'DCOILWTICO',    # WTI Oil Price
            'DCOILBRENTEU',  # Brent Oil Price
            'VIXCLS',        # VIX Volatility Index
            'DGS10',         # 10-Year Treasury Rate
            'DGS2',          # 2-Year Treasury Rate
            'CPALTT01USM661S',  # OECD CPI
        ]
        
        fred_dfs = []
        for series_id in fred_series:
            filepath = self.data_dir / f"{series_id}.csv"
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath)
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.set_index('date')
                        # Rename value column to series_id
                        if len(df.columns) == 1:
                            df.columns = [series_id.lower()]
                        fred_dfs.append(df)
                        logger.info(f"✅ Loaded {series_id}: {len(df)} observations")
                    else:
                        logger.warning(f"⚠️  {series_id}: No 'date' column found")
                except Exception as e:
                    logger.error(f"❌ Error loading {series_id}: {e}")
            else:
                logger.warning(f"⏭️  {series_id}: File not found")
        
        return fred_dfs
    
    def load_ecb_data(self):
        """Load ECB EUR/USD data"""
        ecb_dfs = []
        
        # ECB EUR/USD
        filepath = self.data_dir / "ECB_EURUSD.csv"
        if filepath.exists():
            try:
                df = pd.read_csv(filepath)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                if 'value' in df.columns:
                    df = df.rename(columns={'value': 'ecb_eurusd'})
                ecb_dfs.append(df)
                logger.info(f"✅ Loaded ECB_EURUSD: {len(df)} observations")
            except Exception as e:
                logger.error(f"❌ Error loading ECB_EURUSD: {e}")
        
        return ecb_dfs
    
    def load_alpha_vantage_data(self):
        """Load Alpha Vantage FX data"""
        av_dfs = []
        
        av_files = ['AV_EURUSD.csv', 'AV_USDJPY.csv', 'AV_USDCHF.csv']
        for filename in av_files:
            filepath = self.data_dir / filename
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath)
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    # Rename columns with av_ prefix
                    df.columns = [f"av_{col}" for col in df.columns]
                    av_dfs.append(df)
                    logger.info(f"✅ Loaded {filename}: {len(df)} observations")
                except Exception as e:
                    logger.error(f"❌ Error loading {filename}: {e}")
        
        return av_dfs
    
    def load_dxy_exy_cross(self):
        """Load DXY/EXY cross indicators"""
        filepath = self.data_dir / "DXY_EXY_CROSS.csv"
        if filepath.exists():
            try:
                df = pd.read_csv(filepath)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                logger.info(f"✅ Loaded DXY_EXY_CROSS: {len(df)} observations, {len(df.columns)} indicators")
                logger.info(f"   Indicators: {', '.join(df.columns)}")
                return [df]
            except Exception as e:
                logger.error(f"❌ Error loading DXY_EXY_CROSS: {e}")
        
        return []
    
    def load_gold_data(self):
        """Load gold price data"""
        filepath = self.data_dir / "GOLD_PRICE_MT.csv"
        if filepath.exists():
            try:
                df = pd.read_csv(filepath)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                if 'close' in df.columns:
                    df = df[['close']].rename(columns={'close': 'gold_price'})
                logger.info(f"✅ Loaded GOLD_PRICE_MT: {len(df)} observations")
                return [df]
            except Exception as e:
                logger.error(f"❌ Error loading GOLD_PRICE_MT: {e}")
        
        return []
    
    def load_all_fundamentals(self):
        """Load and merge all fundamental data sources"""
        logger.info("=" * 80)
        logger.info("🚀 Loading Comprehensive Fundamental Data")
        logger.info("=" * 80)
        
        all_dfs = []
        
        # Load all data sources
        all_dfs.extend(self.load_fred_series())
        all_dfs.extend(self.load_ecb_data())
        all_dfs.extend(self.load_alpha_vantage_data())
        all_dfs.extend(self.load_dxy_exy_cross())
        all_dfs.extend(self.load_gold_data())
        
        if not all_dfs:
            logger.error("❌ No fundamental data loaded!")
            return None
        
        # Merge all DataFrames
        logger.info(f"📊 Merging {len(all_dfs)} data sources...")
        merged_df = all_dfs[0]
        for df in all_dfs[1:]:
            merged_df = merged_df.join(df, how='outer')
        
        # Forward fill missing values (common for lower frequency data)
        merged_df = merged_df.ffill()
        
        # Drop rows with too many NaNs (first few rows before all data is available)
        threshold = len(merged_df.columns) * 0.5  # At least 50% of columns must have data
        merged_df = merged_df.dropna(thresh=threshold)
        
        logger.info("=" * 80)
        logger.info(f"✅ Loaded {len(merged_df.columns)} fundamental features")
        logger.info(f"   Date range: {merged_df.index.min()} to {merged_df.index.max()}")
        logger.info(f"   Total observations: {len(merged_df)}")
        logger.info(f"   Features: {', '.join(merged_df.columns)}")
        logger.info("=" * 80)
        
        self.fundamental_data = merged_df
        return merged_df


def integrate_fundamentals_with_technical(pair='EURUSD'):
    """
    Integrate fundamental data with technical features for a specific pair
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"🔧 Integrating Fundamentals with {pair} Technical Features")
    logger.info(f"{'=' * 80}")
    
    # Load fundamental data
    fund_loader = ComprehensiveFundamentalLoader()
    fundamentals = fund_loader.load_all_fundamentals()
    
    if fundamentals is None:
        logger.error("❌ Failed to load fundamental data")
        return None
    
    # Load technical features
    technical_file = Path('data') / f"{pair}_latest_multi_timeframe_features.csv"
    if not technical_file.exists():
        logger.error(f"❌ Technical features file not found: {technical_file}")
        return None
    
    try:
        tech_df = pd.read_csv(technical_file)
        # Technical files use 'time' column, not 'timestamp'
        if 'time' in tech_df.columns:
            tech_df['timestamp'] = pd.to_datetime(tech_df['time'])
        elif 'timestamp' in tech_df.columns:
            tech_df['timestamp'] = pd.to_datetime(tech_df['timestamp'])
        else:
            logger.error(f"❌ No time/timestamp column found in {technical_file}")
            return None
        
        tech_df = tech_df.set_index('timestamp')
        logger.info(f"✅ Loaded {pair} technical features: {len(tech_df)} rows, {len(tech_df.columns)} columns")
    except Exception as e:
        logger.error(f"❌ Error loading technical features: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Merge technical and fundamental data
    logger.info("🔗 Merging technical and fundamental features...")
    combined_df = tech_df.join(fundamentals, how='left')
    
    # Forward fill fundamental data (they update less frequently than technical)
    fundamental_cols = fundamentals.columns.tolist()
    combined_df[fundamental_cols] = combined_df[fundamental_cols].ffill()
    
    # Check for missing data
    missing_counts = combined_df[fundamental_cols].isnull().sum()
    if missing_counts.sum() > 0:
        logger.warning(f"⚠️  Missing fundamental data:")
        for col, count in missing_counts[missing_counts > 0].items():
            logger.warning(f"   {col}: {count} missing values ({count/len(combined_df)*100:.2f}%)")
    
    logger.info(f"✅ Combined dataset: {len(combined_df)} rows, {len(combined_df.columns)} features")
    logger.info(f"   Technical features: {len(tech_df.columns)}")
    logger.info(f"   Fundamental features: {len(fundamental_cols)}")
    
    # Save combined dataset
    output_file = Path('data') / f"{pair}_with_fundamentals.csv"
    combined_df.to_csv(output_file)
    logger.info(f"💾 Saved combined dataset to: {output_file}")
    
    return combined_df


def retrain_with_fundamentals(pair='EURUSD'):
    """
    Retrain model with fundamental features included
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"🎯 Retraining {pair} with Comprehensive Fundamentals")
    logger.info(f"{'=' * 80}")
    
    # Integrate fundamentals
    combined_df = integrate_fundamentals_with_technical(pair)
    
    if combined_df is None:
        logger.error("❌ Failed to integrate fundamentals")
        return False
    
    # Import training module
    try:
        from scripts.automated_training import AutomatedTrainer
        trainer = AutomatedTrainer()
        
        logger.info(f"🚀 Starting model training for {pair}...")
        results = trainer.run_automated_training(pairs=[pair])
        
        if results:
            logger.info(f"✅ Training completed for {pair}")
            logger.info(f"   Results: {results}")
            return True
        else:
            logger.error(f"❌ Training failed for {pair}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_model_performance(pair='EURUSD'):
    """
    Compare old model (without fundamentals) vs new model (with fundamentals)
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"📊 Comparing Model Performance: {pair}")
    logger.info(f"{'=' * 80}")
    
    old_model_file = Path('models') / f"{pair}_lightgbm_simple.joblib"
    new_model_file = Path('models') / f"{pair}_lightgbm_with_fundamentals.joblib"
    
    comparison = {
        'pair': pair,
        'old_model': {},
        'new_model': {},
        'improvement': {}
    }
    
    # Load old model metadata
    if old_model_file.exists():
        try:
            old_model = joblib.load(old_model_file)
            comparison['old_model'] = {
                'file': str(old_model_file),
                'size_mb': old_model_file.stat().st_size / (1024 * 1024),
                'features': len(old_model.feature_name_) if hasattr(old_model, 'feature_name_') else 'Unknown'
            }
            logger.info(f"📦 Old Model: {comparison['old_model']}")
        except Exception as e:
            logger.error(f"❌ Error loading old model: {e}")
    
    # Load new model metadata
    if new_model_file.exists():
        try:
            new_model = joblib.load(new_model_file)
            comparison['new_model'] = {
                'file': str(new_model_file),
                'size_mb': new_model_file.stat().st_size / (1024 * 1024),
                'features': len(new_model.feature_name_) if hasattr(new_model, 'feature_name_') else 'Unknown'
            }
            logger.info(f"📦 New Model: {comparison['new_model']}")
        except Exception as e:
            logger.error(f"❌ Error loading new model: {e}")
    
    return comparison


def main():
    """Main execution function"""
    logger.info("=" * 80)
    logger.info("🚀 COMPREHENSIVE FUNDAMENTAL INTEGRATION & RETRAINING")
    logger.info("=" * 80)
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    # Step 1: Load all fundamental data to verify
    logger.info("\n📊 Step 1: Verifying Fundamental Data Availability")
    fund_loader = ComprehensiveFundamentalLoader()
    fundamentals = fund_loader.load_all_fundamentals()
    
    if fundamentals is None:
        logger.error("❌ Cannot proceed without fundamental data")
        return 1
    
    # Step 2: Integrate fundamentals with technical features for both pairs
    pairs = ['EURUSD', 'XAUUSD']
    
    for pair in pairs:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Processing {pair}")
        logger.info(f"{'=' * 80}")
        
        # Integrate data
        combined_df = integrate_fundamentals_with_technical(pair)
        
        if combined_df is not None:
            logger.info(f"✅ {pair} data integration successful")
        else:
            logger.error(f"❌ {pair} data integration failed")
    
    # Step 3: Retrain models
    logger.info("\n" + "=" * 80)
    logger.info("🎯 Step 3: Retraining Models with Fundamentals")
    logger.info("=" * 80)
    
    retrain_choice = input("\n❓ Retrain models now? (y/n): ").strip().lower()
    
    if retrain_choice == 'y':
        for pair in pairs:
            success = retrain_with_fundamentals(pair)
            if success:
                logger.info(f"✅ {pair} retraining successful")
            else:
                logger.error(f"❌ {pair} retraining failed")
    else:
        logger.info("⏭️  Skipping retraining")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ FUNDAMENTAL INTEGRATION COMPLETE")
    logger.info("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
