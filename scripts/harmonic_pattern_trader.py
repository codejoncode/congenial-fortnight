#!/usr/bin/env python3
"""
Harmonic Pattern Trading System
Specialized geometric pattern trading using Fibonacci projections

This system:
1. Detects harmonic patterns (Gartley, Bat, Butterfly, Crab, Shark)
2. Calculates entry/stop/target levels using pattern geometry
3. Uses Fibonacci extensions for profit targets
4. Scores pattern quality based on precision
5. Provides multiple target levels for scaling out

Author: Trading System
Date: October 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.signal import argrelextrema
import logging

logger = logging.getLogger(__name__)


class HarmonicPatternTrader:
    """
    Specialized harmonic pattern trading system using geometric analysis
    """
    
    def __init__(
        self,
        lookback: int = 100,
        fib_tolerance: float = 0.05,
        min_pattern_bars: int = 20,
        max_pattern_bars: int = 200,
        min_quality_score: float = 0.70
    ):
        """
        Initialize Harmonic Pattern Trader
        
        Args:
            lookback: Number of bars to look back for patterns
            fib_tolerance: Tolerance for Fibonacci ratio matching (0.05 = 5%)
            min_pattern_bars: Minimum bars for pattern formation
            max_pattern_bars: Maximum bars for pattern formation
            min_quality_score: Minimum quality score to trade (0-1)
        """
        self.lookback = lookback
        self.fib_tolerance = fib_tolerance
        self.min_pattern_bars = min_pattern_bars
        self.max_pattern_bars = max_pattern_bars
        self.min_quality_score = min_quality_score
        
        # Fibonacci levels for target calculation
        self.fib_retracements = [0.382, 0.500, 0.618, 0.786]
        self.fib_extensions = [1.272, 1.414, 1.618, 2.000, 2.618]
        
        logger.info(f"HarmonicPatternTrader initialized with lookback={lookback}, "
                   f"fib_tolerance={fib_tolerance:.2%}, min_quality={min_quality_score:.2%}")
    
    def detect_patterns_with_levels(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect all harmonic patterns and calculate entry/stop/target levels
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            List of pattern dictionaries with complete trading levels
        """
        patterns = []
        
        # Find pivot points
        highs_idx = argrelextrema(df['high'].values, np.greater, order=5)[0]
        lows_idx = argrelextrema(df['low'].values, np.less, order=5)[0]
        
        if len(highs_idx) < 3 or len(lows_idx) < 3:
            logger.debug("Not enough pivot points for pattern detection")
            return patterns
        
        # Try to detect each harmonic pattern type
        for pattern_type in ['gartley', 'bat', 'butterfly', 'crab', 'shark']:
            # Check bullish patterns
            bullish = self._detect_single_pattern(
                df, highs_idx, lows_idx, pattern_type, 'bullish'
            )
            if bullish:
                patterns.extend(bullish)
            
            # Check bearish patterns
            bearish = self._detect_single_pattern(
                df, highs_idx, lows_idx, pattern_type, 'bearish'
            )
            if bearish:
                patterns.extend(bearish)
        
        # Filter by quality score
        quality_patterns = [p for p in patterns if p['quality_score'] >= self.min_quality_score]
        
        logger.info(f"Detected {len(patterns)} patterns, {len(quality_patterns)} above quality threshold")
        
        return quality_patterns
    
    def _detect_single_pattern(
        self,
        df: pd.DataFrame,
        highs_idx: np.ndarray,
        lows_idx: np.ndarray,
        pattern_type: str,
        direction: str
    ) -> List[Dict]:
        """
        Detect a single pattern type and direction
        
        Args:
            df: Price DataFrame
            highs_idx: Indices of price highs
            lows_idx: Indices of price lows
            pattern_type: 'gartley', 'bat', 'butterfly', 'crab', 'shark'
            direction: 'bullish' or 'bearish'
            
        Returns:
            List of detected patterns with levels
        """
        patterns = []
        
        # Get pattern-specific Fibonacci ratios
        fib_ratios = self._get_pattern_ratios(pattern_type)
        
        # For bullish: X(low) - A(high) - B(low) - C(high) - D(low)
        # For bearish: X(high) - A(low) - B(high) - C(low) - D(high)
        
        if direction == 'bullish':
            # Try combinations of low-high-low-high-low
            for i in range(len(lows_idx) - 2):
                X_idx = lows_idx[i]
                
                # Find A (high after X)
                A_candidates = highs_idx[(highs_idx > X_idx) & (highs_idx < X_idx + self.max_pattern_bars)]
                if len(A_candidates) == 0:
                    continue
                A_idx = A_candidates[0]
                
                # Find B (low after A)
                B_candidates = lows_idx[(lows_idx > A_idx) & (lows_idx < A_idx + self.max_pattern_bars)]
                if len(B_candidates) == 0:
                    continue
                B_idx = B_candidates[0]
                
                # Find C (high after B)
                C_candidates = highs_idx[(highs_idx > B_idx) & (highs_idx < B_idx + self.max_pattern_bars)]
                if len(C_candidates) == 0:
                    continue
                C_idx = C_candidates[0]
                
                # Find D (low after C) - this is current/recent
                D_candidates = lows_idx[(lows_idx > C_idx) & (lows_idx < C_idx + self.max_pattern_bars)]
                if len(D_candidates) == 0:
                    continue
                D_idx = D_candidates[0]
                
                # Check if pattern is too old
                if len(df) - D_idx > 10:  # Pattern should be recent
                    continue
                
                # Get prices
                X = df.iloc[X_idx]['low']
                A = df.iloc[A_idx]['high']
                B = df.iloc[B_idx]['low']
                C = df.iloc[C_idx]['high']
                D = df.iloc[D_idx]['low']
                
                # Validate Fibonacci ratios
                if self._validate_fibonacci_ratios(X, A, B, C, D, fib_ratios, direction):
                    pattern_data = self._calculate_pattern_levels(
                        X, A, B, C, D, pattern_type, direction,
                        X_idx, A_idx, B_idx, C_idx, D_idx, df
                    )
                    if pattern_data:
                        patterns.append(pattern_data)
        
        else:  # bearish
            # Try combinations of high-low-high-low-high
            for i in range(len(highs_idx) - 2):
                X_idx = highs_idx[i]
                
                # Find A (low after X)
                A_candidates = lows_idx[(lows_idx > X_idx) & (lows_idx < X_idx + self.max_pattern_bars)]
                if len(A_candidates) == 0:
                    continue
                A_idx = A_candidates[0]
                
                # Find B (high after A)
                B_candidates = highs_idx[(highs_idx > A_idx) & (highs_idx < A_idx + self.max_pattern_bars)]
                if len(B_candidates) == 0:
                    continue
                B_idx = B_candidates[0]
                
                # Find C (low after B)
                C_candidates = lows_idx[(lows_idx > B_idx) & (lows_idx < B_idx + self.max_pattern_bars)]
                if len(C_candidates) == 0:
                    continue
                C_idx = C_candidates[0]
                
                # Find D (high after C) - this is current/recent
                D_candidates = highs_idx[(highs_idx > C_idx) & (highs_idx < C_idx + self.max_pattern_bars)]
                if len(D_candidates) == 0:
                    continue
                D_idx = D_candidates[0]
                
                # Check if pattern is too old
                if len(df) - D_idx > 10:
                    continue
                
                # Get prices
                X = df.iloc[X_idx]['high']
                A = df.iloc[A_idx]['low']
                B = df.iloc[B_idx]['high']
                C = df.iloc[C_idx]['low']
                D = df.iloc[D_idx]['high']
                
                # Validate Fibonacci ratios
                if self._validate_fibonacci_ratios(X, A, B, C, D, fib_ratios, direction):
                    pattern_data = self._calculate_pattern_levels(
                        X, A, B, C, D, pattern_type, direction,
                        X_idx, A_idx, B_idx, C_idx, D_idx, df
                    )
                    if pattern_data:
                        patterns.append(pattern_data)
        
        return patterns
    
    def _get_pattern_ratios(self, pattern_type: str) -> Dict[str, Tuple[float, float]]:
        """
        Get ideal Fibonacci ratios for each pattern type
        
        Returns:
            Dictionary with ratio names and (ideal_value, tolerance) tuples
        """
        ratios = {
            'gartley': {
                'B_retracement': (0.618, self.fib_tolerance),  # B retraces 0.618 of XA
                'D_retracement': (0.786, self.fib_tolerance),  # D retraces 0.786 of XA
                'C_retracement': (0.382, 0.50)  # C retraces 0.382-0.886 of AB (flexible)
            },
            'bat': {
                'B_retracement': (0.382, self.fib_tolerance),  # B retraces 0.382-0.50 of XA
                'D_retracement': (0.886, self.fib_tolerance),  # D retraces 0.886 of XA
                'C_retracement': (0.382, 0.50)
            },
            'butterfly': {
                'B_retracement': (0.786, self.fib_tolerance),  # B retraces 0.786 of XA
                'D_extension': (1.272, self.fib_tolerance),    # D extends 1.27 of XA
                'C_retracement': (0.382, 0.50)
            },
            'crab': {
                'B_retracement': (0.382, self.fib_tolerance),  # B retraces 0.382-0.618 of XA
                'D_extension': (1.618, self.fib_tolerance),    # D extends 1.618 of XA
                'C_retracement': (0.382, 0.50)
            },
            'shark': {
                'B_retracement': (0.886, self.fib_tolerance),  # B retraces 0.886-1.13 of XA
                'D_retracement': (0.886, self.fib_tolerance),  # D retraces 0.886-1.13 of XA
                'C_retracement': (1.13, 0.50)
            }
        }
        return ratios.get(pattern_type, {})
    
    def _validate_fibonacci_ratios(
        self,
        X: float, A: float, B: float, C: float, D: float,
        fib_ratios: Dict,
        direction: str
    ) -> bool:
        """
        Validate if XABCD points match Fibonacci ratios for pattern
        
        Args:
            X, A, B, C, D: Price levels
            fib_ratios: Expected Fibonacci ratios for pattern
            direction: 'bullish' or 'bearish'
            
        Returns:
            True if ratios match within tolerance
        """
        if not fib_ratios:
            return False
        
        # Calculate movements
        XA = abs(A - X)
        AB = abs(B - A)
        BC = abs(C - B)
        CD = abs(D - C)
        
        if XA == 0:
            return False
        
        # Calculate actual ratios
        B_ratio = AB / XA  # How much B retraces XA
        
        # For D, check if it's retracement or extension
        if 'D_retracement' in fib_ratios:
            if direction == 'bullish':
                XD = abs(D - X)
                D_ratio = XD / XA
            else:
                XD = abs(X - D)
                D_ratio = XD / XA
        elif 'D_extension' in fib_ratios:
            if direction == 'bullish':
                XD = abs(X - D)  # D extends beyond X
                D_ratio = XD / XA
            else:
                XD = abs(D - X)
                D_ratio = XD / XA
        else:
            D_ratio = 0
        
        # Check B ratio
        B_ideal, B_tol = fib_ratios.get('B_retracement', (0, 0))
        B_match = abs(B_ratio - B_ideal) <= B_tol
        
        # Check D ratio
        if 'D_retracement' in fib_ratios:
            D_ideal, D_tol = fib_ratios['D_retracement']
        elif 'D_extension' in fib_ratios:
            D_ideal, D_tol = fib_ratios['D_extension']
        else:
            return False
        
        D_match = abs(D_ratio - D_ideal) <= D_tol
        
        return B_match and D_match
    
    def _calculate_pattern_levels(
        self,
        X: float, A: float, B: float, C: float, D: float,
        pattern_type: str,
        direction: str,
        X_idx: int, A_idx: int, B_idx: int, C_idx: int, D_idx: int,
        df: pd.DataFrame
    ) -> Optional[Dict]:
        """
        Calculate complete trading levels for a valid pattern
        
        Returns:
            Dictionary with entry, stop, targets, risk/reward, quality score
        """
        # Entry is at D point completion
        entry = D
        
        # Stop loss beyond X point (pattern invalidation)
        if direction == 'bullish':
            stop_loss = X - (abs(A - X) * 0.10)  # 10% buffer below X
        else:
            stop_loss = X + (abs(X - A) * 0.10)  # 10% buffer above X
        
        # Calculate Fibonacci targets based on pattern
        targets = self._calculate_fibonacci_targets(X, A, B, C, D, direction)
        
        # Calculate risk and rewards
        risk_pips = abs(entry - stop_loss) * 10000  # Convert to pips
        
        rewards = []
        risk_rewards = []
        for target in targets:
            reward_pips = abs(target - entry) * 10000
            rewards.append(reward_pips)
            risk_rewards.append(reward_pips / risk_pips if risk_pips > 0 else 0)
        
        # Calculate pattern quality score
        quality_score = self._calculate_quality_score(
            X, A, B, C, D, pattern_type, direction,
            X_idx, A_idx, B_idx, C_idx, D_idx, df
        )
        
        # Get current price and bar index
        current_price = df.iloc[-1]['close']
        current_bar = len(df) - 1
        
        pattern_data = {
            'pattern_type': f"{pattern_type}_{direction}",
            'direction': 'long' if direction == 'bullish' else 'short',
            
            # XABCD points
            'X': X,
            'A': A,
            'B': B,
            'C': C,
            'D': D,
            
            # XABCD indices
            'X_idx': X_idx,
            'A_idx': A_idx,
            'B_idx': B_idx,
            'C_idx': C_idx,
            'D_idx': D_idx,
            
            # Trading levels
            'entry': entry,
            'stop_loss': stop_loss,
            'target_1': targets[0],
            'target_2': targets[1],
            'target_3': targets[2],
            
            # Risk/Reward
            'risk_pips': risk_pips,
            'reward_pips_t1': rewards[0],
            'reward_pips_t2': rewards[1],
            'reward_pips_t3': rewards[2],
            'risk_reward_t1': risk_rewards[0],
            'risk_reward_t2': risk_rewards[1],
            'risk_reward_t3': risk_rewards[2],
            
            # Pattern metrics
            'quality_score': quality_score,
            'pattern_bars': D_idx - X_idx,
            'current_price': current_price,
            'current_bar': current_bar,
            
            # Pattern completion status
            'completed': True,  # D point exists
            'bars_since_completion': current_bar - D_idx
        }
        
        return pattern_data
    
    def _calculate_fibonacci_targets(
        self,
        X: float, A: float, B: float, C: float, D: float,
        direction: str
    ) -> List[float]:
        """
        Calculate 3 Fibonacci-based profit targets
        
        Target 1: 0.382 retracement of AD (conservative)
        Target 2: 0.618 retracement of AD (primary)
        Target 3: Point C level (aggressive)
        
        Returns:
            List of 3 target prices
        """
        AD = abs(A - D)
        
        if direction == 'bullish':
            # Targets above entry (D)
            target_1 = D + (AD * 0.382)
            target_2 = D + (AD * 0.618)
            target_3 = C  # Return to C level
        else:
            # Targets below entry (D)
            target_1 = D - (AD * 0.382)
            target_2 = D - (AD * 0.618)
            target_3 = C  # Return to C level
        
        return [target_1, target_2, target_3]
    
    def _calculate_quality_score(
        self,
        X: float, A: float, B: float, C: float, D: float,
        pattern_type: str,
        direction: str,
        X_idx: int, A_idx: int, B_idx: int, C_idx: int, D_idx: int,
        df: pd.DataFrame
    ) -> float:
        """
        Calculate pattern quality score (0-1)
        
        Factors:
        1. Fibonacci ratio precision (40%)
        2. Time symmetry (20%)
        3. Volume confirmation (20%)
        4. Prior support/resistance at D (20%)
        
        Returns:
            Quality score 0.0 to 1.0
        """
        scores = []
        
        # 1. Fibonacci precision score (40%)
        fib_ratios = self._get_pattern_ratios(pattern_type)
        XA = abs(A - X)
        AB = abs(B - A)
        
        if XA > 0:
            B_ratio = AB / XA
            B_ideal = fib_ratios.get('B_retracement', (0.618, 0.05))[0]
            B_precision = 1.0 - min(abs(B_ratio - B_ideal) / B_ideal, 1.0)
            scores.append(B_precision * 0.40)
        
        # 2. Time symmetry score (20%)
        time_XA = A_idx - X_idx
        time_AB = B_idx - A_idx
        time_BC = C_idx - B_idx
        time_CD = D_idx - C_idx
        
        avg_time = np.mean([time_XA, time_AB, time_BC, time_CD])
        time_variance = np.std([time_XA, time_AB, time_BC, time_CD]) / avg_time if avg_time > 0 else 1
        time_symmetry = max(0, 1.0 - time_variance)
        scores.append(time_symmetry * 0.20)
        
        # 3. Volume confirmation (20%)
        # Ideal: Volume decreasing as pattern progresses, spike at D
        if 'volume' in df.columns:
            vol_X = df.iloc[X_idx]['volume']
            vol_D = df.iloc[D_idx]['volume']
            
            if vol_X > 0:
                vol_ratio = vol_D / vol_X
                # Good if volume at D is higher (breakout volume)
                vol_score = min(vol_ratio, 2.0) / 2.0  # Cap at 2x
                scores.append(vol_score * 0.20)
            else:
                scores.append(0.10)  # Partial credit if no volume data
        else:
            scores.append(0.10)  # Partial credit if no volume data
        
        # 4. Support/Resistance at D (20%)
        # Check if D is near recent support/resistance
        lookback_bars = min(100, X_idx)
        if lookback_bars > 0:
            recent_highs = df.iloc[max(0, D_idx-lookback_bars):D_idx]['high'].values
            recent_lows = df.iloc[max(0, D_idx-lookback_bars):D_idx]['low'].values
            
            # Check if D is near recent levels
            if direction == 'bullish':
                distance_to_support = min(abs(D - low) for low in recent_lows if low > 0)
                sr_score = 1.0 - min(distance_to_support / D, 1.0) if D > 0 else 0
            else:
                distance_to_resistance = min(abs(D - high) for high in recent_highs if high > 0)
                sr_score = 1.0 - min(distance_to_resistance / D, 1.0) if D > 0 else 0
            
            scores.append(sr_score * 0.20)
        else:
            scores.append(0.10)
        
        final_score = sum(scores)
        return min(max(final_score, 0.0), 1.0)  # Clamp to [0, 1]
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from harmonic patterns
        
        Args:
            df: OHLC DataFrame
            
        Returns:
            DataFrame with pattern signals and levels
        """
        patterns = self.detect_patterns_with_levels(df)
        
        if not patterns:
            logger.info("No quality harmonic patterns detected")
            return pd.DataFrame()
        
        # Convert to DataFrame
        signals_df = pd.DataFrame(patterns)
        
        # Sort by quality score
        signals_df = signals_df.sort_values('quality_score', ascending=False)
        
        logger.info(f"Generated {len(signals_df)} harmonic pattern signals")
        logger.info(f"Best pattern: {signals_df.iloc[0]['pattern_type']} "
                   f"(quality={signals_df.iloc[0]['quality_score']:.2f}, "
                   f"R:R={signals_df.iloc[0]['risk_reward_t2']:.2f})")
        
        return signals_df
    
    def get_active_trades(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter for active trading opportunities
        
        Args:
            signals_df: DataFrame from generate_signals()
            
        Returns:
            DataFrame with actionable trades
        """
        if signals_df.empty:
            return signals_df
        
        # Filter for recent patterns (within last 5 bars)
        active = signals_df[signals_df['bars_since_completion'] <= 5].copy()
        
        # Filter for good risk:reward (at least 1:2 on target 2)
        active = active[active['risk_reward_t2'] >= 2.0]
        
        # Filter for quality
        active = active[active['quality_score'] >= self.min_quality_score]
        
        logger.info(f"Found {len(active)} active trading opportunities")
        
        return active


def main():
    """Test the harmonic pattern trader"""
    import sys
    sys.path.append('/workspaces/congenial-fortnight')
    
    # Load sample data
    try:
        df = pd.read_csv('/workspaces/congenial-fortnight/data/EURUSD_H1.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Loaded {len(df)} bars of EURUSD H1 data")
        
        # Initialize trader
        trader = HarmonicPatternTrader(
            lookback=100,
            fib_tolerance=0.05,
            min_quality_score=0.70
        )
        
        # Generate signals
        signals = trader.generate_signals(df)
        
        if not signals.empty:
            print("\n" + "="*80)
            print("HARMONIC PATTERN SIGNALS")
            print("="*80)
            
            for idx, signal in signals.iterrows():
                print(f"\n🎯 {signal['pattern_type'].upper()}")
                print(f"   Quality Score: {signal['quality_score']:.2%}")
                print(f"   Direction: {signal['direction'].upper()}")
                print(f"\n   XABCD Points:")
                print(f"   X: {signal['X']:.5f} (Bar {signal['X_idx']})")
                print(f"   A: {signal['A']:.5f} (Bar {signal['A_idx']})")
                print(f"   B: {signal['B']:.5f} (Bar {signal['B_idx']})")
                print(f"   C: {signal['C']:.5f} (Bar {signal['C_idx']})")
                print(f"   D: {signal['D']:.5f} (Bar {signal['D_idx']})")
                print(f"\n   Trading Levels:")
                print(f"   Entry:      {signal['entry']:.5f}")
                print(f"   Stop Loss:  {signal['stop_loss']:.5f} ({signal['risk_pips']:.1f} pips)")
                print(f"   Target 1:   {signal['target_1']:.5f} ({signal['reward_pips_t1']:.1f} pips, R:R={signal['risk_reward_t1']:.2f})")
                print(f"   Target 2:   {signal['target_2']:.5f} ({signal['reward_pips_t2']:.1f} pips, R:R={signal['risk_reward_t2']:.2f})")
                print(f"   Target 3:   {signal['target_3']:.5f} ({signal['reward_pips_t3']:.1f} pips, R:R={signal['risk_reward_t3']:.2f})")
                print(f"\n   Pattern Age: {signal['bars_since_completion']} bars since completion")
            
            # Show active trades
            active = trader.get_active_trades(signals)
            if not active.empty:
                print("\n" + "="*80)
                print(f"ACTIVE TRADING OPPORTUNITIES: {len(active)}")
                print("="*80)
                for idx, trade in active.iterrows():
                    print(f"✅ {trade['pattern_type']} | Quality: {trade['quality_score']:.2%} | "
                          f"R:R: {trade['risk_reward_t2']:.2f} | {trade['direction'].upper()}")
        else:
            print("\n❌ No harmonic patterns detected in the data")
            
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
