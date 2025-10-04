"""
Fundamental & Macro Signal Generator
Implements 10 fundamental signal types for forex.
"""
import pandas as pd
import numpy as np

def add_fundamental_signals(df, fundamentals):
    df = df.copy()
    # Example signals (placeholders, real logic should use actual macro data)
    df['fund_surprise_momentum'] = fundamentals['surprise'].rolling(3).mean()
    df['fund_yield_curve'] = fundamentals['yield_10y'] - fundamentals['yield_2y']
    df['fund_central_bank'] = fundamentals['policy_rate'].diff()
    df['fund_vol_jump'] = (fundamentals['volatility'] > fundamentals['volatility'].rolling(10).mean() + fundamentals['volatility'].rolling(10).std()).astype(int)
    df['fund_inflation_trend'] = fundamentals['cpi'].pct_change(3)
    df['fund_employment_trend'] = fundamentals['employment'].pct_change(3)
    df['fund_trade_balance'] = fundamentals['trade_balance']
    df['fund_credit_spread'] = fundamentals['corp_bond'] - fundamentals['gov_bond']
    df['fund_fx_reserves'] = fundamentals['fx_reserves'].pct_change(3)
    df['fund_macro_regime'] = (fundamentals['gdp'].pct_change(4) > 0).astype(int)
    return df
