"""
signals/decision_engine.py — Rule-based trade decision support.

Configured for a micro account ($200–$750) with:
  - Dynamic risk % based on signal confidence (1% / 2% / 3%)
  - Micro-lot position sizing (0.01 lot base)
  - Daily signal framing ("Tomorrow's trade")
  - MetaTrader 5 paper-trade compatible output

Returns:
  {
    'action':    'EXECUTE' | 'WAIT' | 'SKIP',
    'reasons':   [{'pass': bool, 'rule': str, 'detail': str}],
    'score':     0-100,
    'summary':   str,
    'sizing':    { risk_pct, risk_usd, lot_size, pip_risk, pip_value }
  }
"""

from __future__ import annotations

# ── Configurable thresholds ───────────────────────────────────────────────────
DECISION_RULES = {
    'min_probability':    0.58,    # minimum to consider trading
    'min_risk_reward':    1.5,     # minimum R:R ratio
    'max_open_positions': 3,       # max concurrent positions
    'max_daily_loss_pct': 5.0,     # stop trading at 5% daily drawdown
    'min_atr_pips_eurusd': 5.0,    # skip if market too quiet
    'min_atr_pips_xauusd': 150.0,  # skip if gold market too quiet
}

# Dynamic risk % by confidence tier
RISK_TIERS = [
    (0.75, 3.0),   # prob >= 75% → risk 3% of account
    (0.65, 2.0),   # prob >= 65% → risk 2%
    (0.58, 1.0),   # prob >= 58% → risk 1%
]

# Pip multipliers and micro-lot pip values (per 0.01 lot)
PIP_MULT = {'EURUSD': 10_000, 'XAUUSD': 100}

# Pip value per 0.01 micro lot in USD (approximate, account in USD)
PIP_VALUE_PER_MICRO = {
    'EURUSD': 0.10,   # $0.10 per pip per 0.01 lot
    'XAUUSD': 1.00,   # $1.00 per pip per 0.01 lot (gold = $1/pip/0.01lot)
}

MIN_LOT  = 0.01   # micro lot
MAX_LOT  = 1.00   # cap for micro account safety


def _risk_pct(prob: float) -> float:
    for threshold, pct in RISK_TIERS:
        if prob >= threshold:
            return pct
    return 1.0


def _atr_pips(signal: dict) -> float:
    atr  = float(signal.get('atr', 0) or 0)
    pair = signal.get('pair', 'EURUSD')
    return atr * PIP_MULT.get(pair, 10_000)


def _position_sizing(signal: dict, account_balance: float) -> dict:
    """
    Calculate lot size for a micro account using fixed-fractional risk.

    lot_size = risk_usd / (pip_risk × pip_value_per_micro_lot)
    Rounded DOWN to nearest 0.01 lot.
    """
    prob    = float(signal.get('probability', 0))
    pair    = signal.get('pair', 'EURUSD')
    entry   = float(signal.get('entry_price') or signal.get('entry') or 0)
    sl      = float(signal.get('stop_loss') or 0)

    risk_pct   = _risk_pct(prob)
    risk_usd   = round(account_balance * risk_pct / 100, 2)

    pip_mult   = PIP_MULT.get(pair, 10_000)
    pip_value  = PIP_VALUE_PER_MICRO.get(pair, 0.10)

    pip_risk   = abs(entry - sl) * pip_mult if (entry and sl) else 0.0

    if pip_risk > 0:
        raw_lots = risk_usd / (pip_risk * pip_value)
        lot_size = max(MIN_LOT, min(MAX_LOT, round(raw_lots - (raw_lots % MIN_LOT), 2)))
    else:
        lot_size = MIN_LOT

    potential_reward = lot_size * abs(float(signal.get('risk_reward', 0)) * pip_risk * pip_value)

    return {
        'risk_pct':         risk_pct,
        'risk_usd':         risk_usd,
        'pip_risk':         round(pip_risk, 1),
        'pip_value_micro':  pip_value,
        'lot_size':         lot_size,
        'potential_reward': round(potential_reward, 2),
        'account_balance':  account_balance,
    }


def evaluate(
    signal: dict,
    open_positions: int  = 0,
    daily_pnl_usd: float = 0.0,
    account_balance: float = 500.0,   # default micro account size
) -> dict:
    """
    Evaluate whether tomorrow's signal should be traded.

    Parameters
    ----------
    signal          : dict from SignalEngine.predict() or Signal model
    open_positions  : current open trades count
    daily_pnl_usd   : today's realised P&L in USD (negative = loss)
    account_balance : current account equity in USD

    Returns
    -------
    dict with action, reasons, score, summary, sizing
    """
    rules   = DECISION_RULES
    checks  = []
    sig_type = signal.get('signal', 'no_signal')

    # ── 0. Signal exists ──────────────────────────────────────────────────────
    if sig_type == 'no_signal':
        return {
            'action':  'SKIP',
            'reasons': [{'pass': False, 'rule': 'Signal direction', 'detail': 'Model output: no_signal — market is ranging or uncertain'}],
            'score':   0,
            'summary': 'No trade tomorrow — model sees no clear direction.',
            'sizing':  _position_sizing(signal, account_balance),
        }

    # ── 1. Probability threshold ──────────────────────────────────────────────
    prob   = float(signal.get('probability', 0))
    min_p  = rules['min_probability']
    checks.append({
        'pass':   prob >= min_p,
        'rule':   'Signal confidence',
        'detail': f'{prob:.1%} (minimum {min_p:.0%})',
    })

    # ── 2. Risk:Reward ratio ──────────────────────────────────────────────────
    rr     = float(signal.get('risk_reward', 0))
    min_rr = rules['min_risk_reward']
    checks.append({
        'pass':   rr >= min_rr,
        'rule':   'Risk:Reward ratio',
        'detail': f'1:{rr:.2f} (minimum 1:{min_rr:.1f})',
    })

    # ── 3. Position limit ─────────────────────────────────────────────────────
    max_pos = rules['max_open_positions']
    checks.append({
        'pass':   open_positions < max_pos,
        'rule':   'Position limit',
        'detail': f'{open_positions} of {max_pos} max open',
    })

    # ── 4. Daily loss limit (% of account) ───────────────────────────────────
    max_loss_pct  = rules['max_daily_loss_pct']
    daily_loss_pct = abs(min(0, daily_pnl_usd)) / account_balance * 100 if account_balance > 0 else 0
    checks.append({
        'pass':   daily_loss_pct < max_loss_pct,
        'rule':   'Daily loss limit',
        'detail': f'{daily_loss_pct:.1f}% of account lost today (limit {max_loss_pct:.0f}%)',
    })

    # ── 5. Market activity ────────────────────────────────────────────────────
    pair    = signal.get('pair', 'EURUSD')
    atr_pip = _atr_pips(signal)
    min_atr = rules.get(f'min_atr_pips_{pair.lower()}', 5.0)
    checks.append({
        'pass':   atr_pip >= min_atr,
        'rule':   'Market activity',
        'detail': f'ATR {atr_pip:.1f} pips (minimum {min_atr:.0f})',
    })

    # ── 6. SL/TP sanity ──────────────────────────────────────────────────────
    entry = float(signal.get('entry_price') or signal.get('entry') or 0)
    sl    = float(signal.get('stop_loss')   or 0)
    tp    = float(signal.get('take_profit') or 0)
    sl_ok = entry > 0 and sl > 0 and tp > 0
    if sl_ok:
        sl_ok = (sl < entry and tp > entry) if sig_type == 'bullish' else (sl > entry and tp < entry)
    checks.append({
        'pass':   sl_ok,
        'rule':   'Price levels valid',
        'detail': f'Entry {entry:.5g}  SL {sl:.5g}  TP {tp:.5g}',
    })

    # ── Sizing ────────────────────────────────────────────────────────────────
    sizing = _position_sizing(signal, account_balance)

    # ── Decision ──────────────────────────────────────────────────────────────
    hard_fails = [c for c in checks if not c['pass'] and c['rule'] in
                  ('Position limit', 'Daily loss limit', 'Price levels valid')]
    soft_fails = [c for c in checks if not c['pass'] and c not in hard_fails]
    score = round(sum(1 for c in checks if c['pass']) / len(checks) * 100)

    label = 'BUY' if sig_type == 'bullish' else 'SELL'

    if hard_fails:
        action  = 'SKIP'
        summary = f'SKIP — {hard_fails[0]["rule"]}. {hard_fails[0]["detail"]}'
    elif soft_fails:
        action  = 'WAIT'
        summary = f'WAIT — {soft_fails[0]["rule"]}. {soft_fails[0]["detail"]}'
    else:
        action  = 'EXECUTE'
        summary = (
            f'YES — {label} {pair} tomorrow @ {entry:.5g} | '
            f'SL {sl:.5g} | TP {tp:.5g} | '
            f'{sizing["lot_size"]} lots | '
            f'Risk ${sizing["risk_usd"]:.2f} ({sizing["risk_pct"]:.0f}%) | '
            f'R:R 1:{rr:.2f}'
        )

    return {
        'action':  action,
        'reasons': checks,
        'score':   score,
        'summary': summary,
        'sizing':  sizing,
    }
