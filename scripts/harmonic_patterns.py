"""
Harmonic Pattern Recognition

Implements detection of classic harmonic patterns using Fibonacci ratios:
- Gartley (0.618-0.786 retracement)
- Bat (0.382-0.886 retracement)
- Butterfly (0.786-1.27 extension)
- Crab (0.618-1.618 extension)
- Shark (0.886-1.13 retracement)
"""
import pandas as pd
import numpy as np

def detect_harmonic_patterns(df: pd.DataFrame, lookback: int = 60, tolerance: float = 0.05) -> pd.DataFrame:
    """
    Detect harmonic patterns using pivot points and Fibonacci ratios.
    
    Args:
        df: DataFrame with OHLC data
        lookback: Number of bars to look back for pivot detection
        tolerance: Tolerance for Fibonacci ratio matching (±5% default)
    
    Returns:
        DataFrame with harmonic pattern columns added
    """
    df = df.copy()
    
    # Initialize pattern columns
    df['gartley_bullish'] = 0
    df['gartley_bearish'] = 0
    df['bat_bullish'] = 0
    df['bat_bearish'] = 0
    df['butterfly_bullish'] = 0
    df['butterfly_bearish'] = 0
    df['crab_bullish'] = 0
    df['crab_bearish'] = 0
    df['shark_bullish'] = 0
    df['shark_bearish'] = 0
    
    # Find pivots (simplified pivot detection)
    df['pivot_high'] = (
        (df['High'] > df['High'].shift(1)) & 
        (df['High'] > df['High'].shift(-1))
    )
    df['pivot_low'] = (
        (df['Low'] < df['Low'].shift(1)) & 
        (df['Low'] < df['Low'].shift(-1))
    )
    
    # For each bar, check if we can form a pattern
    for i in range(lookback, len(df)):
        recent_highs = df.iloc[i-lookback:i][df['pivot_high'].iloc[i-lookback:i]]['High']
        recent_lows = df.iloc[i-lookback:i][df['pivot_low'].iloc[i-lookback:i]]['Low']
        
        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            # Get the last pivots for pattern XABCD
            try:
                # Bullish patterns (bottom formation)
                X = recent_highs.iloc[-2] if len(recent_highs) >= 2 else None
                A = recent_lows.iloc[-2] if len(recent_lows) >= 2 else None
                B = recent_highs.iloc[-1] if len(recent_highs) >= 1 else None
                C = recent_lows.iloc[-1] if len(recent_lows) >= 1 else None
                D = df['Low'].iloc[i]  # Current position
                
                if X and A and B and C:
                    # Calculate Fibonacci ratios
                    XA = abs(A - X)
                    AB = abs(B - A)
                    BC = abs(C - B)
                    CD = abs(D - C)
                    
                    if XA > 0:
                        AB_XA = AB / XA
                        AD_XA = abs(D - A) / XA
                        
                        # Gartley: B retraces 0.618 of XA, D completes at 0.786
                        if _is_close(AB_XA, 0.618, tolerance) and _is_close(AD_XA, 0.786, tolerance):
                            df.loc[df.index[i], 'gartley_bullish'] = 1
                        
                        # Bat: B retraces 0.382-0.50 of XA, D completes at 0.886
                        if (0.382 - tolerance) <= AB_XA <= (0.50 + tolerance) and _is_close(AD_XA, 0.886, tolerance):
                            df.loc[df.index[i], 'bat_bullish'] = 1
                        
                        # Butterfly: B retraces 0.786 of XA, D extends to 1.27
                        if _is_close(AB_XA, 0.786, tolerance) and _is_close(AD_XA, 1.27, tolerance):
                            df.loc[df.index[i], 'butterfly_bullish'] = 1
                        
                        # Crab: B retraces 0.618 of XA, D extends to 1.618
                        if _is_close(AB_XA, 0.618, tolerance) and _is_close(AD_XA, 1.618, tolerance):
                            df.loc[df.index[i], 'crab_bullish'] = 1
                        
                        # Shark: B retraces 0.886 of XA, D extends to 1.13
                        if _is_close(AB_XA, 0.886, tolerance) and _is_close(AD_XA, 1.13, tolerance):
                            df.loc[df.index[i], 'shark_bullish'] = 1
                
                # Bearish patterns (top formation) - mirror logic
                X_bear = recent_lows.iloc[-2] if len(recent_lows) >= 2 else None
                A_bear = recent_highs.iloc[-2] if len(recent_highs) >= 2 else None
                B_bear = recent_lows.iloc[-1] if len(recent_lows) >= 1 else None
                C_bear = recent_highs.iloc[-1] if len(recent_highs) >= 1 else None
                D_bear = df['High'].iloc[i]
                
                if X_bear and A_bear and B_bear and C_bear:
                    XA_bear = abs(A_bear - X_bear)
                    AB_bear = abs(B_bear - A_bear)
                    
                    if XA_bear > 0:
                        AB_XA_bear = AB_bear / XA_bear
                        AD_XA_bear = abs(D_bear - A_bear) / XA_bear
                        
                        if _is_close(AB_XA_bear, 0.618, tolerance) and _is_close(AD_XA_bear, 0.786, tolerance):
                            df.loc[df.index[i], 'gartley_bearish'] = 1
                        if (0.382 - tolerance) <= AB_XA_bear <= (0.50 + tolerance) and _is_close(AD_XA_bear, 0.886, tolerance):
                            df.loc[df.index[i], 'bat_bearish'] = 1
                        if _is_close(AB_XA_bear, 0.786, tolerance) and _is_close(AD_XA_bear, 1.27, tolerance):
                            df.loc[df.index[i], 'butterfly_bearish'] = 1
                        if _is_close(AB_XA_bear, 0.618, tolerance) and _is_close(AD_XA_bear, 1.618, tolerance):
                            df.loc[df.index[i], 'crab_bearish'] = 1
                        if _is_close(AB_XA_bear, 0.886, tolerance) and _is_close(AD_XA_bear, 1.13, tolerance):
                            df.loc[df.index[i], 'shark_bearish'] = 1
            except (IndexError, KeyError):
                pass
    
    # Create composite harmonic signal
    df['harmonic_signal'] = (
        df['gartley_bullish'] + df['bat_bullish'] + df['butterfly_bullish'] + 
        df['crab_bullish'] + df['shark_bullish'] -
        df['gartley_bearish'] - df['bat_bearish'] - df['butterfly_bearish'] - 
        df['crab_bearish'] - df['shark_bearish']
    )
    
    return df

def _is_close(value: float, target: float, tolerance: float) -> bool:
    """Check if value is within tolerance of target"""
    return abs(value - target) <= tolerance

# Legacy function name for compatibility
def add_harmonic_patterns(df):
    return detect_harmonic_patterns(df)
