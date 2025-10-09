"""
Fundamental & Macro Signal Generator
Implements 10 fundamental signal types for forex using ACTUAL fundamental data.
Based on CFT_0000999_MORE_SIGNALS.md specifications.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def add_fundamental_signals(df: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """
    Add 10 fundamental signal types to the feature dataframe.
    
    Uses actual fundamental data columns:
    - cpiaucsl, gdpc1, fedfunds, dff, unrate, indpro, payems
    - dgorder, bopgstb, dtwexbgs, dexuseu, dexjpus, dexchus
    - dcoilwtico, dcoilbrenteu, vixcls, dgs10, dgs2
    - ecbdfr, cp0000ez19m086nest, lrhuttttdem156s
    """
    df = df.copy()
    
    try:
        # 1. Macro Surprise Momentum (using CPI, NFP, GDP changes)
        if 'cpiaucsl' in fundamentals.columns:
            cpi_change = fundamentals['cpiaucsl'].pct_change()
            df['fund_cpi_surprise_mom_5d'] = cpi_change.rolling(5).sum()
            df['fund_cpi_positive_count_10d'] = (cpi_change > 0).rolling(10).sum()
            df['fund_cpi_negative_count_10d'] = (cpi_change < 0).rolling(10).sum()
        
        if 'payems' in fundamentals.columns:
            nfp_change = fundamentals['payems'].diff()
            df['fund_nfp_surprise_mom_5d'] = nfp_change.rolling(5).sum()
        
        if 'gdpc1' in fundamentals.columns:
            gdp_change = fundamentals['gdpc1'].pct_change()
            df['fund_gdp_surprise_mom'] = gdp_change.rolling(3).mean()
        
        # 2. Interest Rate Differential Trend (USD vs EUR rates)
        if 'fedfunds' in fundamentals.columns and 'ecbdfr' in fundamentals.columns:
            df['fund_carry_spread'] = fundamentals['fedfunds'] - fundamentals['ecbdfr']
            df['fund_carry_spread_ma20'] = df['fund_carry_spread'].rolling(20).mean()
            df['fund_carry_mom_10'] = df['fund_carry_spread'].diff(10)
            df['fund_carry_long'] = ((df['fund_carry_mom_10'] > 0) & (df['fund_carry_spread'] > df['fund_carry_spread_ma20'])).astype(int)
            df['fund_carry_short'] = ((df['fund_carry_mom_10'] < 0) | (df['fund_carry_spread'] < df['fund_carry_spread_ma20'])).astype(int)
        
        # 3. Yield Curve Slope Shifts (2Y vs 10Y)
        if 'dgs2' in fundamentals.columns and 'dgs10' in fundamentals.columns:
            df['fund_curve_slope_usd'] = fundamentals['dgs10'] - fundamentals['dgs2']
            df['fund_curve_steepening'] = (df['fund_curve_slope_usd'].diff() > 0).astype(int)
            df['fund_curve_inversion'] = (df['fund_curve_slope_usd'] < 0).astype(int)
            df['fund_curve_slope_momentum'] = df['fund_curve_slope_usd'].diff(5)
        
        # 4. Central Bank Policy Surprises (rate changes)
        if 'fedfunds' in fundamentals.columns:
            fed_change = fundamentals['fedfunds'].diff()
            df['fund_cbp_surprise_fed'] = fed_change
            df['fund_cbp_tightening_spike'] = (fed_change >= 0.25).astype(int)
            df['fund_cbp_easing_spike'] = (fed_change <= -0.25).astype(int)
        
        if 'ecbdfr' in fundamentals.columns:
            ecb_change = fundamentals['ecbdfr'].diff()
            df['fund_cbp_surprise_ecb'] = ecb_change
            df['fund_cbp_ecb_tightening'] = (ecb_change >= 0.25).astype(int)
            df['fund_cbp_ecb_easing'] = (ecb_change <= -0.25).astype(int)
        
        # 5. Volatility Jump on Event Days (VIX spikes)
        if 'vixcls' in fundamentals.columns:
            vix_mean = fundamentals['vixcls'].rolling(20).mean()
            vix_std = fundamentals['vixcls'].rolling(20).std()
            df['fund_vol_jump_event'] = (fundamentals['vixcls'] > vix_mean + 2 * vix_std).astype(int)
            df['fund_vix_regime'] = pd.cut(fundamentals['vixcls'], bins=[0, 15, 25, 100], labels=['low', 'medium', 'high']).astype(str)
        
        # 6. Leading Indicators Composite (Industrial Production, Employment)
        if 'indpro' in fundamentals.columns and 'payems' in fundamentals.columns:
            indpro_change = fundamentals['indpro'].pct_change(3)
            payems_change = fundamentals['payems'].pct_change(3)
            df['fund_leading_composite'] = (indpro_change + payems_change) / 2
            df['fund_business_cycle_up'] = (df['fund_leading_composite'] > df['fund_leading_composite'].quantile(0.75)).astype(int)
            df['fund_business_cycle_down'] = (df['fund_leading_composite'] < df['fund_leading_composite'].quantile(0.25)).astype(int)
        
        # 7. Money Supply Growth (using GDP as proxy for liquidity)
        if 'gdpc1' in fundamentals.columns:
            gdp_growth = fundamentals['gdpc1'].pct_change(4)
            df['fund_liquidity_growth'] = gdp_growth
            df['fund_liquidity_expansion'] = (gdp_growth > 0.01).astype(int)  # >1% growth
            df['fund_liquidity_contraction'] = (gdp_growth < -0.01).astype(int)  # <-1% growth
        
        # 8. Trade Balance Shock (Balance of Payments)
        if 'bopgstb' in fundamentals.columns:
            trade_change = fundamentals['bopgstb'].diff()
            trade_std = fundamentals['bopgstb'].rolling(12).std()
            df['fund_trade_surprise'] = trade_change / (trade_std + 1)
            df['fund_trade_surplus_bull'] = (df['fund_trade_surprise'] > 1).astype(int)
            df['fund_trade_deficit_bear'] = (df['fund_trade_surprise'] < -1).astype(int)
        
        # 9. Fiscal Sentiment (using unemployment rate as proxy)
        if 'unrate' in fundamentals.columns:
            unrate_change = fundamentals['unrate'].diff(3)
            df['fund_fiscal_sentiment'] = -unrate_change  # Lower unemployment = better sentiment
            df['fund_fiscal_bull'] = (df['fund_fiscal_sentiment'] > 0.2).astype(int)
            df['fund_fiscal_bear'] = (df['fund_fiscal_sentiment'] < -0.2).astype(int)
        
        # 10. Commodity Price Precursor (Oil correlation with USD)
        if 'dcoilwtico' in fundamentals.columns and 'dtwexbgs' in fundamentals.columns:
            oil_change = fundamentals['dcoilwtico'].pct_change()
            usd_change = fundamentals['dtwexbgs'].pct_change()
            oil_usd_corr = oil_change.rolling(20).corr(usd_change)
            # Align with df's index before assignment
            df['fund_oil_usd_correlation'] = oil_usd_corr.reindex(df.index)
            oil_change_aligned = oil_change.reindex(df.index)
            oil_usd_corr_aligned = oil_usd_corr.reindex(df.index)
            df['fund_oil_correlation_signal'] = np.where(
                (oil_usd_corr_aligned < -0.3) & (oil_change_aligned > 0), 1,  # Negative corr + oil up = USD down
                np.where((oil_usd_corr_aligned > 0.3) & (oil_change_aligned > 0), -1, 0)  # Positive corr + oil up = USD up
            )
        
        logger.info(f"Added {len([c for c in df.columns if c.startswith('fund_')])} fundamental signal features")
        
    except Exception as e:
        logger.error(f"Error adding fundamental signals: {e}", exc_info=True)
    
    return df
