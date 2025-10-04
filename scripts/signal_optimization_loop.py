"""
Signal Optimization and Tracking Loop
- Hyperparameter optimization for each signal
- Tracks historical results, hit rate, earnings, last hit
- Removes signals from list when target or max attempts reached
- Reports by signal, trade type, and pair
"""
import pandas as pd
import numpy as np
import time
from scripts.day_trading_signals import DayTradingSignalGenerator
from scripts.signal_backtester import backtest_signal
from scripts.ultimate_signal_repository import UltimateSignalRepository

# Example hyperparameter grid for demonstration
HYPERPARAM_GRID = {
    'ma_cross_signal': [{'fast': 5, 'slow': 20}, {'fast': 9, 'slow': 21}],
    'rsi_signal': [{'period': 14, 'overbought': 70, 'oversold': 30}, {'period': 10, 'overbought': 80, 'oversold': 20}],
    # Add more for each signal
}

TARGET_HIT_RATE = 0.55
MAX_ATTEMPTS = 5

class SignalOptimizer:
    def __init__(self, df, pairs):
        self.df = df
        self.pairs = pairs
        self.signal_results = []
        self.optimization_status = {}

    def optimize_signals(self):
        signals_to_optimize = list(HYPERPARAM_GRID.keys())
        for signal in signals_to_optimize:
            self.optimization_status[signal] = {'attempts': 0, 'done': False}

        while any(not v['done'] for v in self.optimization_status.values()):
            for signal in signals_to_optimize:
                status = self.optimization_status[signal]
                if status['done']:
                    continue
                for params in HYPERPARAM_GRID[signal]:
                    status['attempts'] += 1
                    for pair in self.pairs:
                        df_pair = self.df[self.df['pair'] == pair].copy()
                        gen = DayTradingSignalGenerator(df_pair)
                        # Dynamically call signal method
                        signal_func = getattr(gen, signal.replace('_signal',''))
                        df_pair[signal] = signal_func(**params)
                        # Backtest for both bull and bear
                        for trade_type in ['bull', 'bear']:
                            if trade_type == 'bull':
                                mask = df_pair[signal] == 1
                            else:
                                mask = df_pair[signal] == -1
                            if mask.sum() == 0:
                                continue
                            df_bt = backtest_signal(df_pair[mask], signal)
                            earnings = df_bt['trade_return'].sum()
                            hit_rate = (df_bt['trade_return'] > 0).mean()
                            last_hit = df_bt[df_bt['trade_return'] > 0]['timestamp'].max() if 'timestamp' in df_bt else None
                            self.signal_results.append({
                                'signal': signal,
                                'params': params,
                                'pair': pair,
                                'trade_type': trade_type,
                                'earnings_pct': earnings,
                                'hit_rate': hit_rate,
                                'last_hit': last_hit,
                                'attempt': status['attempts']
                            })
                            if hit_rate >= TARGET_HIT_RATE or status['attempts'] >= MAX_ATTEMPTS:
                                status['done'] = True
                                break
                    if status['done']:
                        break
        return pd.DataFrame(self.signal_results)

    def report(self, results_df):
        print("\n=== Signal Optimization Results ===")
        for signal in results_df['signal'].unique():
            print(f"\nSignal: {signal}")
            for pair in results_df[results_df['signal']==signal]['pair'].unique():
                for trade_type in ['bull','bear']:
                    sub = results_df[(results_df['signal']==signal)&(results_df['pair']==pair)&(results_df['trade_type']==trade_type)]
                    if sub.empty:
                        continue
                    best = sub.sort_values('hit_rate', ascending=False).iloc[0]
                    print(f"  Pair: {pair}, Type: {trade_type}, Best Hit Rate: {best['hit_rate']:.2%}, Earnings: {best['earnings_pct']:.2f}, Last Hit: {best['last_hit']}")

# Usage example (to be run in a notebook or script):
# df = ... # Load your OHLCV+pair data
# optimizer = SignalOptimizer(df, pairs=['EURUSD','XAUUSD'])
# results = optimizer.optimize_signals()
# optimizer.report(results)
