#!/usr/bin/env python3
"""
Harmonic Pattern Backtest System
Backtest harmonic pattern trading with Fibonacci targets

This system:
1. Tests harmonic patterns historically
2. Simulates scaling out at multiple targets
3. Tracks per-pattern performance
4. Calculates realistic win rates and R:R ratios
5. Compares against ML-based system

Author: Trading System
Date: October 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import json
import logging
from pathlib import Path

import sys
sys.path.append('/workspaces/congenial-fortnight')

from scripts.harmonic_pattern_trader import HarmonicPatternTrader

logger = logging.getLogger(__name__)


class HarmonicPatternBacktest:
    """
    Backtest harmonic pattern trading with multi-target management
    """
    
    def __init__(
        self,
        initial_balance: float = 10000,
        risk_per_trade_pct: float = 0.02,
        scale_out_percents: List[float] = [0.50, 0.30, 0.20]
    ):
        """
        Initialize backtest
        
        Args:
            initial_balance: Starting capital
            risk_per_trade_pct: Risk per trade as % of balance (0.02 = 2%)
            scale_out_percents: % to close at each target [T1, T2, T3]
        """
        self.initial_balance = initial_balance
        self.risk_per_trade_pct = risk_per_trade_pct
        self.scale_out_percents = scale_out_percents
        
        self.balance = initial_balance
        self.trades = []
        self.equity_curve = []
        
        logger.info(f"HarmonicPatternBacktest initialized: ${initial_balance:,.0f}, "
                   f"risk={risk_per_trade_pct:.1%}, scale_out={scale_out_percents}")
    
    def run_backtest(
        self,
        df: pd.DataFrame,
        trader: HarmonicPatternTrader,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict:
        """
        Run backtest on historical data
        
        Args:
            df: OHLC DataFrame with timestamp
            trader: HarmonicPatternTrader instance
            start_date: Start date for backtest (optional)
            end_date: End date for backtest (optional)
            
        Returns:
            Backtest results dictionary
        """
        logger.info("Starting harmonic pattern backtest...")
        
        # Filter date range if provided
        if start_date:
            df = df[df['timestamp'] >= start_date]
        if end_date:
            df = df[df['timestamp'] <= end_date]
        
        logger.info(f"Backtesting {len(df)} bars from {df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']}")
        
        # Reset state
        self.balance = self.initial_balance
        self.trades = []
        self.equity_curve = [(df.iloc[0]['timestamp'], self.balance)]
        
        # Walk forward through data
        lookback_bars = trader.lookback
        
        for i in range(lookback_bars, len(df)):
            # Get data up to current point
            historical_data = df.iloc[:i+1].copy()
            
            # Detect patterns
            patterns = trader.detect_patterns_with_levels(historical_data)
            
            if patterns:
                for pattern in patterns:
                    # Only trade if pattern just completed (within last 3 bars)
                    if pattern['bars_since_completion'] <= 3:
                        # Simulate the trade
                        trade_result = self._simulate_trade(
                            pattern,
                            df.iloc[i:],  # Future data for trade simulation
                            df.iloc[i]['timestamp']
                        )
                        
                        if trade_result:
                            self.trades.append(trade_result)
                            self.balance = trade_result['balance_after']
                            
                            logger.debug(f"Trade {len(self.trades)}: {trade_result['pattern_type']} "
                                       f"| Result: {trade_result['result']} "
                                       f"| P&L: ${trade_result['pnl']:.2f}")
            
            # Record equity
            if i % 100 == 0:  # Every 100 bars
                self.equity_curve.append((df.iloc[i]['timestamp'], self.balance))
        
        # Final equity point
        self.equity_curve.append((df.iloc[-1]['timestamp'], self.balance))
        
        # Calculate results
        results = self._calculate_results(df)
        
        logger.info(f"Backtest complete: {len(self.trades)} trades, "
                   f"Final balance: ${self.balance:,.2f}")
        
        return results
    
    def _simulate_trade(
        self,
        pattern: Dict,
        future_data: pd.DataFrame,
        entry_time: datetime
    ) -> Optional[Dict]:
        """
        Simulate a single trade with multi-target scaling
        
        Args:
            pattern: Pattern dictionary with levels
            future_data: DataFrame of future prices
            entry_time: Timestamp of entry
            
        Returns:
            Trade result dictionary or None if trade not valid
        """
        entry = pattern['entry']
        stop_loss = pattern['stop_loss']
        targets = [pattern['target_1'], pattern['target_2'], pattern['target_3']]
        direction = pattern['direction']
        
        # Calculate position size based on risk
        risk_amount = self.balance * self.risk_per_trade_pct
        risk_pips = pattern['risk_pips']
        
        # Position size = risk_amount / (risk_pips * pip_value)
        # Assuming $10 per pip for forex (standard lot)
        pip_value = 10
        position_size = risk_amount / (risk_pips * pip_value) if risk_pips > 0 else 0
        
        if position_size <= 0:
            return None
        
        # Track partial positions
        remaining_positions = {
            'target_1': position_size * self.scale_out_percents[0],
            'target_2': position_size * self.scale_out_percents[1],
            'target_3': position_size * self.scale_out_percents[2]
        }
        
        total_pnl = 0
        exit_time = None
        exit_reason = None
        targets_hit = []
        
        # Simulate trade progression
        max_bars = min(200, len(future_data))  # Max 200 bars per trade
        
        for i in range(max_bars):
            if i >= len(future_data):
                break
            
            bar = future_data.iloc[i]
            high = bar['high']
            low = bar['low']
            
            if direction == 'long':
                # Check stop loss
                if low <= stop_loss:
                    # Hit stop loss - close all remaining
                    total_position = sum(remaining_positions.values())
                    loss_pips = (stop_loss - entry) * 10000
                    pnl = total_position * loss_pips * pip_value
                    total_pnl += pnl
                    
                    exit_time = bar['timestamp']
                    exit_reason = 'stop_loss'
                    break
                
                # Check targets (in order)
                if 'target_1' in remaining_positions and high >= targets[0]:
                    pnl = remaining_positions['target_1'] * (targets[0] - entry) * 10000 * pip_value
                    total_pnl += pnl
                    targets_hit.append('T1')
                    del remaining_positions['target_1']
                
                if 'target_2' in remaining_positions and high >= targets[1]:
                    pnl = remaining_positions['target_2'] * (targets[1] - entry) * 10000 * pip_value
                    total_pnl += pnl
                    targets_hit.append('T2')
                    del remaining_positions['target_2']
                
                if 'target_3' in remaining_positions and high >= targets[2]:
                    pnl = remaining_positions['target_3'] * (targets[2] - entry) * 10000 * pip_value
                    total_pnl += pnl
                    targets_hit.append('T3')
                    del remaining_positions['target_3']
                
                # If all closed, exit
                if not remaining_positions:
                    exit_time = bar['timestamp']
                    exit_reason = 'all_targets_hit'
                    break
            
            else:  # short
                # Check stop loss
                if high >= stop_loss:
                    total_position = sum(remaining_positions.values())
                    loss_pips = (entry - stop_loss) * 10000
                    pnl = total_position * loss_pips * pip_value
                    total_pnl += pnl
                    
                    exit_time = bar['timestamp']
                    exit_reason = 'stop_loss'
                    break
                
                # Check targets
                if 'target_1' in remaining_positions and low <= targets[0]:
                    pnl = remaining_positions['target_1'] * (entry - targets[0]) * 10000 * pip_value
                    total_pnl += pnl
                    targets_hit.append('T1')
                    del remaining_positions['target_1']
                
                if 'target_2' in remaining_positions and low <= targets[1]:
                    pnl = remaining_positions['target_2'] * (entry - targets[1]) * 10000 * pip_value
                    total_pnl += pnl
                    targets_hit.append('T2')
                    del remaining_positions['target_2']
                
                if 'target_3' in remaining_positions and low <= targets[2]:
                    pnl = remaining_positions['target_3'] * (entry - targets[2]) * 10000 * pip_value
                    total_pnl += pnl
                    targets_hit.append('T3')
                    del remaining_positions['target_3']
                
                if not remaining_positions:
                    exit_time = bar['timestamp']
                    exit_reason = 'all_targets_hit'
                    break
        
        # If still open after max bars, close at breakeven or small loss
        if remaining_positions:
            total_position = sum(remaining_positions.values())
            # Close at current price (or small loss)
            if i < len(future_data):
                current_price = future_data.iloc[i]['close']
                if direction == 'long':
                    pnl = total_position * (current_price - entry) * 10000 * pip_value
                else:
                    pnl = total_position * (entry - current_price) * 10000 * pip_value
                total_pnl += pnl
            
            exit_time = future_data.iloc[min(i, len(future_data)-1)]['timestamp']
            exit_reason = 'timeout'
        
        # Determine result
        if total_pnl > 0:
            result = 'win'
        elif total_pnl < 0:
            result = 'loss'
        else:
            result = 'breakeven'
        
        trade_result = {
            'trade_number': len(self.trades) + 1,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'pattern_type': pattern['pattern_type'],
            'direction': direction,
            'quality_score': pattern['quality_score'],
            
            'entry': entry,
            'stop_loss': stop_loss,
            'targets': targets,
            'targets_hit': targets_hit,
            
            'position_size': position_size,
            'risk_amount': risk_amount,
            'pnl': total_pnl,
            'pnl_pct': (total_pnl / self.balance) * 100,
            
            'result': result,
            'exit_reason': exit_reason,
            
            'balance_before': self.balance,
            'balance_after': self.balance + total_pnl
        }
        
        return trade_result
    
    def _calculate_results(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive backtest results"""
        if not self.trades:
            return {
                'error': 'No trades executed',
                'total_trades': 0
            }
        
        trades_df = pd.DataFrame(self.trades)
        
        # Overall metrics
        total_trades = len(trades_df)
        wins = len(trades_df[trades_df['result'] == 'win'])
        losses = len(trades_df[trades_df['result'] == 'loss'])
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = trades_df['pnl'].sum()
        total_return_pct = ((self.balance - self.initial_balance) / self.initial_balance) * 100
        
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if wins > 0 else 0
        avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if losses > 0 else 0
        profit_factor = (wins * avg_win) / (losses * avg_loss) if losses > 0 and avg_loss > 0 else 0
        
        # Target hit analysis
        target_1_hit = sum(1 for t in self.trades if 'T1' in t['targets_hit'])
        target_2_hit = sum(1 for t in self.trades if 'T2' in t['targets_hit'])
        target_3_hit = sum(1 for t in self.trades if 'T3' in t['targets_hit'])
        
        # Per-pattern analysis
        pattern_stats = {}
        for pattern_type in trades_df['pattern_type'].unique():
            pattern_trades = trades_df[trades_df['pattern_type'] == pattern_type]
            pattern_wins = len(pattern_trades[pattern_trades['result'] == 'win'])
            pattern_stats[pattern_type] = {
                'trades': len(pattern_trades),
                'wins': pattern_wins,
                'win_rate': (pattern_wins / len(pattern_trades) * 100) if len(pattern_trades) > 0 else 0,
                'total_pnl': pattern_trades['pnl'].sum(),
                'avg_quality': pattern_trades['quality_score'].mean()
            }
        
        # Time analysis
        start_date = df.iloc[0]['timestamp']
        end_date = df.iloc[-1]['timestamp']
        days = (end_date - start_date).days
        trades_per_month = (total_trades / days * 30) if days > 0 else 0
        
        results = {
            'summary': {
                'total_trades': total_trades,
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'total_return_pct': total_return_pct,
                'initial_balance': self.initial_balance,
                'final_balance': self.balance,
                'profit_factor': profit_factor
            },
            'averages': {
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'avg_trade': total_pnl / total_trades if total_trades > 0 else 0,
                'avg_quality_score': trades_df['quality_score'].mean()
            },
            'targets': {
                'target_1_hit_rate': (target_1_hit / total_trades * 100) if total_trades > 0 else 0,
                'target_2_hit_rate': (target_2_hit / total_trades * 100) if total_trades > 0 else 0,
                'target_3_hit_rate': (target_3_hit / total_trades * 100) if total_trades > 0 else 0
            },
            'patterns': pattern_stats,
            'time': {
                'start_date': str(start_date),
                'end_date': str(end_date),
                'days': days,
                'trades_per_month': trades_per_month
            },
            'trades': self.trades
        }
        
        return results
    
    def save_results(self, results: Dict, output_dir: str = 'output/harmonic_backtests'):
        """Save backtest results to JSON file"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"harmonic_backtest_{timestamp}.json"
        filepath = output_path / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")
        return filepath
    
    def print_summary(self, results: Dict):
        """Print formatted backtest summary"""
        print("\n" + "="*80)
        print("HARMONIC PATTERN BACKTEST RESULTS")
        print("="*80)
        
        summary = results['summary']
        averages = results['averages']
        targets = results['targets']
        time_data = results['time']
        
        print(f"\n📊 OVERALL PERFORMANCE")
        print(f"   Total Trades: {summary['total_trades']}")
        print(f"   Win Rate: {summary['win_rate']:.1f}% ({summary['wins']}W / {summary['losses']}L)")
        print(f"   Profit Factor: {summary['profit_factor']:.2f}")
        print(f"\n💰 FINANCIAL RESULTS")
        print(f"   Initial Balance: ${summary['initial_balance']:,.2f}")
        print(f"   Final Balance: ${summary['final_balance']:,.2f}")
        print(f"   Total P&L: ${summary['total_pnl']:,.2f}")
        print(f"   Return: {summary['total_return_pct']:.2f}%")
        print(f"\n📈 TRADE METRICS")
        print(f"   Avg Win: ${averages['avg_win']:.2f}")
        print(f"   Avg Loss: ${averages['avg_loss']:.2f}")
        print(f"   Avg Trade: ${averages['avg_trade']:.2f}")
        print(f"   Avg Quality Score: {averages['avg_quality_score']:.2%}")
        print(f"\n🎯 TARGET ANALYSIS")
        print(f"   Target 1 Hit Rate: {targets['target_1_hit_rate']:.1f}%")
        print(f"   Target 2 Hit Rate: {targets['target_2_hit_rate']:.1f}%")
        print(f"   Target 3 Hit Rate: {targets['target_3_hit_rate']:.1f}%")
        print(f"\n📅 TIMEFRAME")
        print(f"   Period: {time_data['start_date']} to {time_data['end_date']}")
        print(f"   Duration: {time_data['days']} days")
        print(f"   Trades/Month: {time_data['trades_per_month']:.1f}")
        
        print(f"\n🎯 PATTERN BREAKDOWN")
        for pattern_type, stats in results['patterns'].items():
            print(f"   {pattern_type}:")
            print(f"      Trades: {stats['trades']} | Win Rate: {stats['win_rate']:.1f}% | "
                  f"P&L: ${stats['total_pnl']:.2f} | Quality: {stats['avg_quality']:.2%}")
        
        print("\n" + "="*80)


def main():
    """Run harmonic pattern backtest"""
    try:
        # Load data
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
        
        # Initialize backtest
        backtest = HarmonicPatternBacktest(
            initial_balance=10000,
            risk_per_trade_pct=0.02,
            scale_out_percents=[0.50, 0.30, 0.20]
        )
        
        # Run backtest
        results = backtest.run_backtest(df, trader)
        
        # Print results
        backtest.print_summary(results)
        
        # Save results
        filepath = backtest.save_results(results)
        print(f"\n✅ Results saved to: {filepath}")
        
    except Exception as e:
        logger.error(f"Error in backtest: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
