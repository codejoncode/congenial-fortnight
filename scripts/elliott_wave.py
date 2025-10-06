"""
Elliott Wave Pattern Recognition

Implements detection of Elliott Wave impulse patterns:
- Wave 3 starts (strongest impulse wave)
- Wave 5 starts (final impulse wave)
- Fibonacci ratio validation between waves

Elliott Wave Rules:
- Wave 2 retraces 50-61.8% of Wave 1
- Wave 3 extends 161.8-261.8% of Wave 1 (never shortest)
- Wave 4 retraces 23.6-38.2% of Wave 3
- Wave 5 extends 61.8-100% of Wave 1
"""
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

def detect_elliott_waves(df: pd.DataFrame, lookback: int = 5, tolerance: float = 0.05) -> pd.DataFrame:
    """
    Detect Elliott Wave impulse patterns using pivot analysis and Fibonacci ratios.
    
    Args:
        df: DataFrame with OHLC data
        lookback: Number of bars to use for local extrema detection
        tolerance: Tolerance for Fibonacci ratio matching (±5% default)
    
    Returns:
        DataFrame with Elliott Wave signal columns added
    """
    df = df.copy()
    
    # Initialize wave columns
    df['elliott_wave_3_start'] = 0
    df['elliott_wave_5_start'] = 0
    df['elliott_impulse_bullish'] = 0
    df['elliott_impulse_bearish'] = 0
    
    # Find local maxima and minima (pivots)
    order = lookback
    df['pivot_high'] = False
    df['pivot_low'] = False
    
    high_indices = argrelextrema(df['High'].values, np.greater_equal, order=order)[0]
    low_indices = argrelextrema(df['Low'].values, np.less_equal, order=order)[0]
    
    df.loc[df.index[high_indices], 'pivot_high'] = True
    df.loc[df.index[low_indices], 'pivot_low'] = True
    
    # Get pivot sequences
    pivot_highs = df[df['pivot_high']]['High'].values
    pivot_lows = df[df['pivot_low']]['Low'].values
    pivot_high_indices = df[df['pivot_high']].index
    pivot_low_indices = df[df['pivot_low']].index
    
    # Detect bullish impulse waves (uptrend)
    for i in range(len(df)):
        # Need at least 5 pivots for a complete 5-wave pattern
        recent_pivot_lows = pivot_lows[pivot_low_indices <= df.index[i]]
        recent_pivot_highs = pivot_highs[pivot_high_indices <= df.index[i]]
        
        if len(recent_pivot_lows) >= 3 and len(recent_pivot_highs) >= 2:
            try:
                # Last 5 pivots for waves: Low(0), High(1), Low(2), High(3), Low(4), High(5)
                # For bullish: starts at low, then high, then low, etc.
                
                # Get last relevant pivots
                P0 = recent_pivot_lows[-3]  # Start of Wave 1
                P1 = recent_pivot_highs[-2] if len(recent_pivot_highs) >= 2 else None  # End of Wave 1
                P2 = recent_pivot_lows[-2]  # End of Wave 2
                P3 = recent_pivot_highs[-1] if len(recent_pivot_highs) >= 1 else None  # End of Wave 3
                P4 = recent_pivot_lows[-1]  # End of Wave 4 (current or recent)
                
                if P1 and P3:
                    # Calculate wave lengths
                    wave_1 = P1 - P0
                    wave_2 = P1 - P2
                    wave_3 = P3 - P2
                    wave_4 = P3 - P4 if P4 < P3 else 0
                    
                    if wave_1 > 0:
                        # Check Fibonacci relationships
                        wave_2_ratio = wave_2 / wave_1
                        wave_3_ratio = wave_3 / wave_1
                        wave_4_ratio = wave_4 / wave_3 if wave_3 > 0 else 0
                        
                        # Wave 2 should retrace 50-61.8% of Wave 1
                        wave_2_valid = (0.50 - tolerance) <= wave_2_ratio <= (0.618 + tolerance)
                        
                        # Wave 3 should extend 161.8-261.8% of Wave 1 (and be longest)
                        wave_3_valid = ((1.618 - tolerance) <= wave_3_ratio <= (2.618 + tolerance) and
                                       wave_3 > wave_1)
                        
                        # Wave 4 should retrace 23.6-38.2% of Wave 3
                        wave_4_valid = (0.236 - tolerance) <= wave_4_ratio <= (0.382 + tolerance) if wave_3 > 0 else False
                        
                        # Detect Wave 3 start (strongest signal)
                        if wave_2_valid and wave_3_valid:
                            # Find index where P2 occurred
                            p2_idx = df[df['Low'] == P2].index
                            if len(p2_idx) > 0 and p2_idx[-1] <= df.index[i]:
                                df.loc[df.index[i], 'elliott_wave_3_start'] = 1
                                df.loc[df.index[i], 'elliott_impulse_bullish'] = 1
                        
                        # Detect Wave 5 start
                        if wave_2_valid and wave_3_valid and wave_4_valid:
                            # Wave 5 typically extends 61.8-100% of Wave 1
                            # Signal at P4 (end of Wave 4)
                            p4_idx = df[df['Low'] == P4].index
                            if len(p4_idx) > 0 and p4_idx[-1] <= df.index[i]:
                                df.loc[df.index[i], 'elliott_wave_5_start'] = 1
                                df.loc[df.index[i], 'elliott_impulse_bullish'] = 1
            
            except (IndexError, ValueError):
                pass
        
        # Detect bearish impulse waves (downtrend) - mirror logic
        if len(recent_pivot_highs) >= 3 and len(recent_pivot_lows) >= 2:
            try:
                P0_bear = recent_pivot_highs[-3]  # Start of Wave 1 down
                P1_bear = recent_pivot_lows[-2] if len(recent_pivot_lows) >= 2 else None  # End of Wave 1
                P2_bear = recent_pivot_highs[-2]  # End of Wave 2
                P3_bear = recent_pivot_lows[-1] if len(recent_pivot_lows) >= 1 else None  # End of Wave 3
                P4_bear = recent_pivot_highs[-1]  # End of Wave 4
                
                if P1_bear and P3_bear:
                    wave_1_bear = P0_bear - P1_bear
                    wave_2_bear = P2_bear - P1_bear
                    wave_3_bear = P2_bear - P3_bear
                    wave_4_bear = P4_bear - P3_bear if P4_bear > P3_bear else 0
                    
                    if wave_1_bear > 0:
                        wave_2_ratio_bear = wave_2_bear / wave_1_bear
                        wave_3_ratio_bear = wave_3_bear / wave_1_bear
                        wave_4_ratio_bear = wave_4_bear / wave_3_bear if wave_3_bear > 0 else 0
                        
                        wave_2_valid_bear = (0.50 - tolerance) <= wave_2_ratio_bear <= (0.618 + tolerance)
                        wave_3_valid_bear = ((1.618 - tolerance) <= wave_3_ratio_bear <= (2.618 + tolerance) and
                                            wave_3_bear > wave_1_bear)
                        wave_4_valid_bear = (0.236 - tolerance) <= wave_4_ratio_bear <= (0.382 + tolerance) if wave_3_bear > 0 else False
                        
                        if wave_2_valid_bear and wave_3_valid_bear:
                            p2_bear_idx = df[df['High'] == P2_bear].index
                            if len(p2_bear_idx) > 0 and p2_bear_idx[-1] <= df.index[i]:
                                df.loc[df.index[i], 'elliott_wave_3_start'] = -1
                                df.loc[df.index[i], 'elliott_impulse_bearish'] = 1
                        
                        if wave_2_valid_bear and wave_3_valid_bear and wave_4_valid_bear:
                            p4_bear_idx = df[df['High'] == P4_bear].index
                            if len(p4_bear_idx) > 0 and p4_bear_idx[-1] <= df.index[i]:
                                df.loc[df.index[i], 'elliott_wave_5_start'] = -1
                                df.loc[df.index[i], 'elliott_impulse_bearish'] = 1
            
            except (IndexError, ValueError):
                pass
    
    # Create composite Elliott Wave signal
    df['elliott_wave_signal'] = df['elliott_wave_3_start'] + df['elliott_wave_5_start']
    
    return df

# Legacy function name for compatibility
def add_elliott_wave_signals(df):
    return detect_elliott_waves(df)
