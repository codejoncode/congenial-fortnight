"""
Unit tests for all newly created signal modules.
"""
import unittest
import pandas as pd
import numpy as np
from scripts.day_trading_signals import DayTradingSignalGenerator
from scripts.intraday_features import add_intraday_features
from scripts.signal_backtester import backtest_signal
from scripts.slump_signals import generate_slump_signals
from scripts.fundamental_signals import add_fundamental_signals
from scripts.candlestick_patterns import add_candlestick_patterns
from scripts.chart_patterns import add_chart_patterns
from scripts.harmonic_patterns import add_harmonic_patterns
from scripts.elliott_wave import add_elliott_wave_signals
from scripts.ultimate_signal_repository import UltimateSignalRepository

class TestSignalModules(unittest.TestCase):
    def setUp(self):
        # Minimal synthetic OHLCV data
        self.df = pd.DataFrame({
            'open': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'high': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            'low': [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5],
            'close': [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
            'volume': [100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
            'timestamp': pd.date_range('2020-01-01', periods=10, freq='H')
        })
        self.fundamentals = pd.DataFrame({
            'surprise': np.arange(10),
            'yield_10y': np.linspace(1.5, 2.5, 10),
            'yield_2y': np.linspace(1.0, 2.0, 10),
            'policy_rate': np.linspace(0.5, 1.5, 10),
            'volatility': np.linspace(10, 20, 10),
            'cpi': np.linspace(100, 110, 10),
            'employment': np.linspace(1000, 1100, 10),
            'trade_balance': np.linspace(50, 60, 10),
            'corp_bond': np.linspace(3, 4, 10),
            'gov_bond': np.linspace(2, 3, 10),
            'fx_reserves': np.linspace(200, 300, 10),
            'gdp': np.linspace(10000, 11000, 10)
        })

    def test_day_trading_signals(self):
        gen = DayTradingSignalGenerator(self.df)
        df_signals = gen.generate_all_signals()
        self.assertIn('ma_cross_signal', df_signals)
        self.assertIn('rsi_signal', df_signals)
        self.assertIn('bb_signal', df_signals)
        self.assertIn('macd_signal', df_signals)
        self.assertIn('stoch_signal', df_signals)
        self.assertIn('atr_breakout_signal', df_signals)
        self.assertIn('vol_spike_signal', df_signals)
        self.assertIn('engulf_signal', df_signals)
        self.assertIn('doji_signal', df_signals)
        self.assertIn('session_breakout_signal', df_signals)

    def test_intraday_features(self):
        df_feat = add_intraday_features(self.df)
        self.assertIn('return_1', df_feat)
        self.assertIn('range', df_feat)

    def test_signal_backtester(self):
        gen = DayTradingSignalGenerator(self.df)
        df_signals = gen.generate_all_signals()
        df_bt = backtest_signal(df_signals, 'ma_cross_signal')
        self.assertIn('trade_return', df_bt)
        self.assertIn('cum_return', df_bt)

    def test_slump_signals(self):
        signal = generate_slump_signals(self.df)
        self.assertEqual(len(signal), len(self.df))

    def test_fundamental_signals(self):
        df_fund = add_fundamental_signals(self.df, self.fundamentals)
        self.assertIn('fund_surprise_momentum', df_fund)
        self.assertIn('fund_yield_curve', df_fund)
        self.assertIn('fund_central_bank', df_fund)
        self.assertIn('fund_vol_jump', df_fund)
        self.assertIn('fund_inflation_trend', df_fund)
        self.assertIn('fund_employment_trend', df_fund)
        self.assertIn('fund_trade_balance', df_fund)
        self.assertIn('fund_credit_spread', df_fund)
        self.assertIn('fund_fx_reserves', df_fund)
        self.assertIn('fund_macro_regime', df_fund)

    def test_candlestick_patterns(self):
        try:
            df_candle = add_candlestick_patterns(self.df)
            self.assertIn('hammer', df_candle)
            self.assertIn('engulfing', df_candle)
            self.assertIn('doji', df_candle)
        except ImportError:
            self.skipTest('TA-Lib not installed')

    def test_chart_patterns(self):
        df_chart = add_chart_patterns(self.df)
        self.assertIn('double_top', df_chart)
        self.assertIn('double_bottom', df_chart)
        self.assertIn('head_shoulders', df_chart)
        self.assertIn('triangle', df_chart)

    def test_harmonic_patterns(self):
        df_harm = add_harmonic_patterns(self.df)
        self.assertIn('gartley', df_harm)
        self.assertIn('bat', df_harm)
        self.assertIn('butterfly', df_harm)
        self.assertIn('crab', df_harm)

    def test_elliott_wave(self):
        df_ew = add_elliott_wave_signals(self.df)
        self.assertIn('elliott_wave_start', df_ew)
        self.assertIn('elliott_wave_end', df_ew)

    def test_ultimate_signal_repository(self):
        repo = UltimateSignalRepository(self.df)
        repo.aggregate_signals()
        repo.rank_signals()
        repo.add_risk_management()
        repo.performance_tracking()
        self.assertIn('master_signal', repo.df)
        self.assertIn('signal_rank', repo.df)
        self.assertIn('risk_flag', repo.df)
        self.assertIn('signal_perf', repo.df)

if __name__ == '__main__':
    unittest.main()
