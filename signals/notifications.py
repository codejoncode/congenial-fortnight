"""
signals/notifications.py — Email alert system for trading signals.

Uses Django's built-in EmailBackend (SMTP).
Configure via environment variables or settings.py — zero cost with Gmail.

Setup (once):
  1. In Gmail → Account → Security → 2-Step Verification → App Passwords
  2. Create an App Password for "Mail" + "Windows Computer"
  3. Copy the 16-char password into your .env file:
       ALERT_EMAIL_HOST_USER=your@gmail.com
       ALERT_EMAIL_HOST_PASSWORD=xxxx xxxx xxxx xxxx
       ALERT_EMAIL_TO=your@gmail.com

The system sends emails only for signals where probability >= NOTIFY_THRESHOLD.
"""

import logging
import os

logger = logging.getLogger(__name__)

NOTIFY_THRESHOLD = float(os.getenv('SIGNAL_NOTIFY_THRESHOLD', '0.60'))
ALERT_TO         = os.getenv('ALERT_EMAIL_TO', '')


def _settings_ok() -> bool:
    return bool(
        os.getenv('ALERT_EMAIL_HOST_USER')
        and os.getenv('ALERT_EMAIL_HOST_PASSWORD')
        and ALERT_TO
    )


def _subject(signal: dict) -> str:
    label = 'BUY' if signal.get('signal') == 'bullish' else 'SELL'
    prob  = float(signal.get('probability', 0))
    pair  = signal.get('pair', '')
    return f"[TOMORROW] {label} {pair} | {prob:.1%} confidence"


def _body(signal: dict, decision: dict | None = None) -> str:
    from datetime import datetime, timezone, timedelta
    label    = 'BUY'  if signal.get('signal') == 'bullish' else 'SELL'
    pair     = signal.get('pair', '')
    prob     = float(signal.get('probability', 0))
    entry    = signal.get('entry_price') or signal.get('entry') or '—'
    sl       = signal.get('stop_loss', '—')
    tp       = signal.get('take_profit', '—')
    rr       = signal.get('risk_reward', '—')
    tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).strftime('%A %b %d')

    sizing   = decision.get('sizing', {}) if decision else {}
    lot_size = sizing.get('lot_size', '—')
    risk_usd = sizing.get('risk_usd', '—')
    risk_pct = sizing.get('risk_pct', '—')
    pip_risk = sizing.get('pip_risk', '—')

    action_icon = {'EXECUTE': 'GO', 'WAIT': 'WAIT', 'SKIP': 'SKIP'}.get(
        decision.get('action', ''), '?') if decision else '?'

    lines = [
        f"TOMORROW'S TRADING SIGNAL",
        f"{'=' * 50}",
        f"",
        f"  Date      : {tomorrow}",
        f"  Pair      : {pair}",
        f"  Direction : {label}",
        f"",
        f"  Entry Price  : {entry}",
        f"  Stop Loss    : {sl}",
        f"  Take Profit  : {tp}",
        f"  R:R Ratio    : 1:{rr}",
        f"",
        f"  Confidence   : {prob:.1%}",
        f"",
        f"  -- POSITION SIZING --",
        f"  Lot Size     : {lot_size} lots",
        f"  Risk         : ${risk_usd} ({risk_pct}% of account)",
        f"  Pips at Risk : {pip_risk}",
        f"",
    ]

    if decision:
        lines += [
            f"  DECISION  : [{action_icon}] {decision.get('action', '?')}",
            f"  {decision.get('summary', '')}",
            f"",
            f"  Checklist:",
        ]
        for r in decision.get('reasons', []):
            icon = 'OK  ' if r['pass'] else 'FAIL'
            lines.append(f"    [{icon}] {r['rule']}: {r['detail']}")
        lines.append('')

    lines += [
        f"{'=' * 50}",
        f"  HOW TO SET THIS UP IN MT5:",
        f"  1. Open MetaTrader 5 → New Order",
        f"  2. Symbol: {pair}",
        f"  3. Type: {'Buy Limit' if label=='BUY' else 'Sell Limit'} @ {entry}",
        f"  4. Stop Loss: {sl}",
        f"  5. Take Profit: {tp}",
        f"  6. Volume: {lot_size} lots",
        f"",
        f"  Dashboard: http://localhost:3000",
        f"{'=' * 50}",
        f"  Automated alert — do not reply.",
    ]

    return '\n'.join(lines)


def send_signal_alert(signal: dict, decision: dict | None = None) -> bool:
    """
    Send an email alert for a new signal.

    Returns True if sent, False if skipped (low confidence or email not configured).
    """
    prob = float(signal.get('probability', 0))
    if prob < NOTIFY_THRESHOLD:
        logger.debug(f"Signal probability {prob:.1%} below threshold {NOTIFY_THRESHOLD:.1%} — skipping email")
        return False

    if not _settings_ok():
        logger.debug("Email alert skipped — ALERT_EMAIL_HOST_USER / ALERT_EMAIL_HOST_PASSWORD / ALERT_EMAIL_TO not set in env")
        return False

    try:
        from django.core.mail import send_mail
        from django.conf import settings

        subject = _subject(signal)
        body    = _body(signal, decision)

        send_mail(
            subject=subject,
            message=body,
            from_email=os.getenv('ALERT_EMAIL_HOST_USER'),
            recipient_list=[ALERT_TO],
            fail_silently=False,
        )
        logger.info(f"Signal alert email sent for {signal.get('pair')} {signal.get('signal')}")
        return True

    except Exception as exc:
        logger.warning(f"Signal alert email failed: {exc}")
        return False
