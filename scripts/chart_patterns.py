"""
Chart Pattern Recognition

Implements detection of classic chart patterns:
- Double Top/Bottom
- Head and Shoulders (and Inverse)
- Triangles (Ascending, Descending, Symmetrical)
- Flags and Pennants
- Cup and Handle
"""
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

def detect_chart_patterns(df: pd.DataFrame, lookback: int = 50, tolerance: float = 0.02) -> pd.DataFrame:
    """
    Detect classic chart patterns using pivot analysis.
    
    Args:
        df: DataFrame with OHLC data
        lookback: Number of bars to look back for pattern detection
        tolerance: Tolerance for level matching (±2% default)
    
    Returns:
        DataFrame with chart pattern columns added
    """
    df = df.copy()
    
    # Initialize pattern columns
    df['double_top'] = 0
    df['double_bottom'] = 0
    df['head_shoulders'] = 0
    df['inv_head_shoulders'] = 0
    df['ascending_triangle'] = 0
    df['descending_triangle'] = 0
    df['symmetrical_triangle'] = 0
    df['bull_flag'] = 0
    df['bear_flag'] = 0
    df['cup_and_handle'] = 0
    
    # Find local maxima and minima
    order = 5  # Number of points on each side to compare
    df['local_max'] = df.iloc[argrelextrema(df['High'].values, np.greater_equal, order=order)[0]]['High']
    df['local_min'] = df.iloc[argrelextrema(df['Low'].values, np.less_equal, order=order)[0]]['Low']
    
    # Iterate through bars to detect patterns
    for i in range(lookback, len(df)):
        window = df.iloc[i-lookback:i+1]
        
        # Get recent pivots
        recent_highs = window[window['local_max'].notna()]['local_max'].values
        recent_lows = window[window['local_min'].notna()]['local_min'].values
        
        # Double Top: Two highs at similar level with a valley between
        if len(recent_highs) >= 2:
            last_two_highs = recent_highs[-2:]
            if _levels_match(last_two_highs[0], last_two_highs[1], tolerance):
                # Check if there's a valley between
                between_idx = window[(window.index > window[window['local_max'] == last_two_highs[0]].index[0]) & 
                                    (window.index < window[window['local_max'] == last_two_highs[1]].index[0])]
                if len(between_idx) > 0 and between_idx['Low'].min() < min(last_two_highs) * (1 - tolerance):
                    df.loc[df.index[i], 'double_top'] = 1
        
        # Double Bottom: Two lows at similar level with a peak between
        if len(recent_lows) >= 2:
            last_two_lows = recent_lows[-2:]
            if _levels_match(last_two_lows[0], last_two_lows[1], tolerance):
                between_idx = window[(window.index > window[window['local_min'] == last_two_lows[0]].index[0]) & 
                                    (window.index < window[window['local_min'] == last_two_lows[1]].index[0])]
                if len(between_idx) > 0 and between_idx['High'].max() > max(last_two_lows) * (1 + tolerance):
                    df.loc[df.index[i], 'double_bottom'] = 1
        
        # Head and Shoulders: Three highs where middle (head) is highest
        if len(recent_highs) >= 3:
            last_three_highs = recent_highs[-3:]
            left_shoulder, head, right_shoulder = last_three_highs
            
            if head > left_shoulder and head > right_shoulder:
                if _levels_match(left_shoulder, right_shoulder, tolerance):
                    df.loc[df.index[i], 'head_shoulders'] = 1
        
        # Inverse Head and Shoulders: Three lows where middle is lowest
        if len(recent_lows) >= 3:
            last_three_lows = recent_lows[-3:]
            left_shoulder, head, right_shoulder = last_three_lows
            
            if head < left_shoulder and head < right_shoulder:
                if _levels_match(left_shoulder, right_shoulder, tolerance):
                    df.loc[df.index[i], 'inv_head_shoulders'] = 1
        
        # Triangles: Use linear regression on highs and lows
        if len(recent_highs) >= 3 and len(recent_lows) >= 3:
            # Get last N highs and lows
            last_highs = recent_highs[-min(5, len(recent_highs)):]
            last_lows = recent_lows[-min(5, len(recent_lows)):]
            
            # Fit lines
            high_slope = _calculate_slope(last_highs)
            low_slope = _calculate_slope(last_lows)
            
            # Ascending Triangle: Flat top, rising bottom
            if abs(high_slope) < tolerance and low_slope > tolerance:
                df.loc[df.index[i], 'ascending_triangle'] = 1
            
            # Descending Triangle: Falling top, flat bottom
            elif high_slope < -tolerance and abs(low_slope) < tolerance:
                df.loc[df.index[i], 'descending_triangle'] = 1
            
            # Symmetrical Triangle: Both converging
            elif high_slope < -tolerance and low_slope > tolerance:
                df.loc[df.index[i], 'symmetrical_triangle'] = 1
        
        # Flags: Strong trend followed by consolidation
        # Bull Flag: Strong up move followed by slight downward consolidation
        recent_close = window['Close'].iloc[-10:] if len(window) >= 10 else window['Close']
        if len(recent_close) >= 10:
            first_half_return = (recent_close.iloc[4] - recent_close.iloc[0]) / recent_close.iloc[0]
            second_half_slope = _calculate_slope(recent_close.iloc[5:].values)
            
            if first_half_return > 0.02 and -0.01 < second_half_slope < 0:
                df.loc[df.index[i], 'bull_flag'] = 1
            
            # Bear Flag: Strong down move followed by slight upward consolidation
            if first_half_return < -0.02 and 0 < second_half_slope < 0.01:
                df.loc[df.index[i], 'bear_flag'] = 1
        
        # Cup and Handle: Rounded bottom followed by small consolidation
        if len(window) >= 30:
            cup_window = window['Close'].iloc[-30:-5]
            handle_window = window['Close'].iloc[-5:]
            
            if len(cup_window) > 0 and len(handle_window) > 0:
                # Cup: U-shaped (low in middle)
                cup_middle_idx = len(cup_window) // 2
                if cup_window.iloc[cup_middle_idx] < cup_window.iloc[0] and \
                   cup_window.iloc[cup_middle_idx] < cup_window.iloc[-1]:
                    # Handle: Small pullback
                    if handle_window.iloc[-1] < handle_window.iloc[0] and \
                       handle_window.iloc[-1] > cup_window.iloc[cup_middle_idx]:
                        df.loc[df.index[i], 'cup_and_handle'] = 1
    
    # Create composite chart pattern signal
    df['chart_pattern_signal'] = (
        df['double_bottom'] + df['inv_head_shoulders'] + df['ascending_triangle'] + 
        df['bull_flag'] + df['cup_and_handle'] -
        df['double_top'] - df['head_shoulders'] - df['descending_triangle'] - df['bear_flag']
    )
    
    return df

def _levels_match(level1: float, level2: float, tolerance: float) -> bool:
    """Check if two price levels match within tolerance"""
    return abs(level1 - level2) / max(level1, level2) <= tolerance

def _calculate_slope(values: np.ndarray) -> float:
    """Calculate slope of values using linear regression"""
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values))
    try:
        slope = np.polyfit(x, values, 1)[0]
        # Normalize by mean value
        return slope / np.mean(values) if np.mean(values) != 0 else 0.0
    except:
        return 0.0

# Legacy function name for compatibility
def add_chart_patterns(df):
    return detect_chart_patterns(df)
