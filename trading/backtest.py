"""
trading/backtest.py — Walk-forward backtest engine.

Methodology:
  - Loads pre-trained models (SignalEngine)
  - Walks bar-by-bar through H1 data in "test window" (last 20% by default)
  - Every signal_interval_bars, generates a signal using data UP TO that point
  - Simulates SL/TP against the next N bars of OHLC — no look-ahead bias
  - Reports win rate, profit factor, max drawdown, Sharpe, expectancy

This is an IN-SAMPLE validation when using the same data the model was
trained on, and an OUT-OF-SAMPLE validation for any data received after
the model was last trained.  The CI gate uses a hold-out window that was
explicitly excluded from training so the result is always honest.
"""

import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Pip multipliers for P&L calculation
PIP_MULT = {
    'EURUSD': 10_000,   # 1 pip = 0.0001
    'GBPUSD': 10_000,
    'XAUUSD':    100,   # 1 pip = 0.01 (gold moves in cents)
    'USDJPY':    100,
}
DEFAULT_PIP_MULT = 10_000


@dataclass
class Trade:
    pair:         str
    direction:    str       # 'bullish' | 'bearish'
    probability:  float
    entry_time:   object
    exit_time:    object
    entry_price:  float
    exit_price:   float
    sl:           float
    tp:           float
    outcome:      str       # 'win' | 'loss' | 'timeout'
    pips:         float
    hold_bars:    int


@dataclass
class BacktestResult:
    pair:              str
    n_trades:          int = 0
    n_wins:            int = 0
    n_losses:          int = 0
    n_timeouts:        int = 0
    win_rate:          float = 0.0
    profit_factor:     float = 0.0
    net_pips:          float = 0.0
    gross_profit_pips: float = 0.0
    gross_loss_pips:   float = 0.0
    avg_win_pips:      float = 0.0
    avg_loss_pips:     float = 0.0
    max_drawdown_pips: float = 0.0
    sharpe_ratio:      float = 0.0
    expectancy_pips:   float = 0.0
    is_profitable:     bool  = False
    verdict:           str   = 'UNKNOWN'
    trades:            List[Trade] = field(default_factory=list)
    error:             str   = ''

    def to_dict(self) -> dict:
        base = {k: v for k, v in self.__dict__.items() if k != 'trades'}
        base['recent_trades'] = [
            {
                'entry_time':  str(t.entry_time),
                'exit_time':   str(t.exit_time),
                'direction':   t.direction,
                'probability': round(t.probability, 3),
                'entry':       round(t.entry_price, 5),
                'exit':        round(t.exit_price, 5),
                'sl':          round(t.sl, 5),
                'tp':          round(t.tp, 5),
                'outcome':     t.outcome,
                'pips':        round(t.pips, 2),
                'hold_bars':   t.hold_bars,
            }
            for t in self.trades[-30:]   # last 30 trades for display
        ]
        return base


def run_backtest(
    pair: str,
    df: pd.DataFrame,
    engine,                        # SignalEngine instance
    lookback: int = 60,            # minimum bars of history needed before first signal
    signal_every: int = 24,        # generate one signal per N bars (24 = daily on H1)
    max_hold: int = 24,            # exit after N bars if SL/TP not hit
    test_from_pct: float = 0.80,   # use only the last (1-test_from_pct) of bars as test window
) -> BacktestResult:
    """
    Walk-forward backtest.

    Steps:
      1. Set test start to test_from_pct * len(df)  (e.g. bar 4000 of 5000)
      2. Walk from test_start to len(df) - max_hold, every signal_every bars
      3. For each position: generate signal on data[:i], enter at bar[i] open
      4. Check OHLC of bars[i..i+max_hold] for SL/TP; otherwise exit at close
      5. Aggregate metrics
    """
    result = BacktestResult(pair=pair)
    pip_mult = PIP_MULT.get(pair, DEFAULT_PIP_MULT)

    if df is None or len(df) < lookback + max_hold + 10:
        result.error = f'Insufficient data: {len(df) if df is not None else 0} rows'
        return result

    # Ensure numeric OHLC
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['open', 'high', 'low', 'close'])

    test_start = max(lookback, int(len(df) * test_from_pct))
    trades: List[Trade] = []

    for i in range(test_start, len(df) - max_hold, signal_every):
        hist = df.iloc[:i]
        if len(hist) < lookback:
            continue

        try:
            pred = engine.predict(pair, hist)
        except Exception as exc:
            logger.debug('[backtest][%s] predict failed at bar %d: %s', pair, i, exc)
            continue

        if pred['signal'] == 'no_signal':
            continue

        # Entry: next bar's open (realistic — you can't enter at the signal bar's close)
        entry_bar  = df.iloc[i]
        entry_price = float(entry_bar['open'])
        entry_time  = df.index[i]
        sl = pred['stop_loss']
        tp = pred['take_profit']
        direction = 1 if pred['signal'] == 'bullish' else -1

        # Walk next max_hold bars for SL/TP
        future = df.iloc[i: i + max_hold]
        outcome, exit_price, hold_bars = 'timeout', float(future.iloc[-1]['close']), max_hold - 1

        for j, (_, bar) in enumerate(future.iterrows()):
            hi = float(bar['high'])
            lo = float(bar['low'])

            if direction == 1:   # bullish
                if lo <= sl:
                    outcome, exit_price, hold_bars = 'loss', sl, j
                    break
                if hi >= tp:
                    outcome, exit_price, hold_bars = 'win', tp, j
                    break
            else:                # bearish
                if hi >= sl:
                    outcome, exit_price, hold_bars = 'loss', sl, j
                    break
                if lo <= tp:
                    outcome, exit_price, hold_bars = 'win', tp, j
                    break

        # If timeout, classify as win/loss based on direction vs actual move
        if outcome == 'timeout':
            pnl_raw = (exit_price - entry_price) * direction
            outcome = 'win' if pnl_raw > 0 else 'loss'

        pips = (exit_price - entry_price) * direction * pip_mult

        trades.append(Trade(
            pair=pair,
            direction=pred['signal'],
            probability=pred['probability'],
            entry_time=entry_time,
            exit_time=future.index[min(hold_bars, len(future) - 1)],
            entry_price=entry_price,
            exit_price=exit_price,
            sl=sl,
            tp=tp,
            outcome=outcome,
            pips=round(pips, 2),
            hold_bars=hold_bars,
        ))

    if not trades:
        result.error = 'No trades generated in test window (model may output no_signal for all bars)'
        return result

    wins    = [t for t in trades if t.outcome == 'win']
    losses  = [t for t in trades if t.outcome == 'loss']

    gross_profit = sum(t.pips for t in wins)
    gross_loss   = abs(sum(t.pips for t in losses))

    # Cumulative equity curve for drawdown
    equity   = 0.0
    peak     = 0.0
    max_dd   = 0.0
    for t in trades:
        equity += t.pips
        peak    = max(peak, equity)
        max_dd  = max(max_dd, peak - equity)

    pip_series  = np.array([t.pips for t in trades])
    mean_pip    = float(np.mean(pip_series))
    std_pip     = float(np.std(pip_series))
    sharpe      = (mean_pip / std_pip * np.sqrt(len(pip_series))) if std_pip > 0 else 0.0

    avg_win  = float(np.mean([t.pips for t in wins]))  if wins   else 0.0
    avg_loss = float(np.mean([abs(t.pips) for t in losses])) if losses else 0.0
    pf       = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    wr       = len(wins) / len(trades)
    exp      = wr * avg_win - (1 - wr) * avg_loss

    verdict  = ('PROFITABLE' if pf >= 1.5 else
                ('MARGINAL'  if pf >= 1.0 else 'UNPROFITABLE'))

    result.n_trades          = len(trades)
    result.n_wins            = len(wins)
    result.n_losses          = len(losses)
    result.n_timeouts        = sum(1 for t in trades if t.outcome == 'timeout')
    result.win_rate          = round(wr, 4)
    result.profit_factor     = round(pf, 3)
    result.net_pips          = round(float(np.sum(pip_series)), 2)
    result.gross_profit_pips = round(gross_profit, 2)
    result.gross_loss_pips   = round(gross_loss, 2)
    result.avg_win_pips      = round(avg_win, 2)
    result.avg_loss_pips     = round(avg_loss, 2)
    result.max_drawdown_pips = round(max_dd, 2)
    result.sharpe_ratio      = round(sharpe, 3)
    result.expectancy_pips   = round(exp, 2)
    result.is_profitable     = pf > 1.0
    result.verdict           = verdict
    result.trades            = trades

    return result
