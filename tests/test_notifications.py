"""
tests/test_notifications.py — Unit tests for signals/notifications.py

Covers:
  - _settings_ok(): False without env vars, partial config, True with all set
  - _subject(): correct label (BUY/SELL), contains pair and probability
  - _body(): TOMORROW framing, MT5 steps, lot size, action icon
  - send_signal_alert(): threshold guard, config guard, calls send_mail correctly
"""

import pytest
from unittest.mock import patch, MagicMock


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _bullish_signal(pair='EURUSD', probability=0.72, entry=1.07230, sl=1.06980, tp=1.07800, rr=2.1):
    return {
        'pair':        pair,
        'signal':      'bullish',
        'probability': probability,
        'entry_price': entry,
        'stop_loss':   sl,
        'take_profit': tp,
        'risk_reward': rr,
        'atr':         0.0010,
    }


def _bearish_signal(**kwargs):
    sig = _bullish_signal(**kwargs)
    sig['signal'] = 'bearish'
    return sig


def _decision(action='EXECUTE', lot_size=0.03, risk_usd=10.0, risk_pct=2.0, pip_risk=25.0):
    return {
        'action':  action,
        'summary': f'{action} — trade setup valid',
        'reasons': [{'pass': True, 'rule': 'Signal confidence', 'detail': '72.0% (minimum 58%)'}],
        'score':   100,
        'sizing':  {
            'lot_size':         lot_size,
            'risk_usd':         risk_usd,
            'risk_pct':         risk_pct,
            'pip_risk':         pip_risk,
            'potential_reward': 20.0,
        },
    }


# ── _settings_ok ──────────────────────────────────────────────────────────────

class TestSettingsOk:
    def _call(self, user='', password='', to=''):
        env = {}
        if user:     env['ALERT_EMAIL_HOST_USER']     = user
        if password: env['ALERT_EMAIL_HOST_PASSWORD'] = password
        if to:       env['ALERT_EMAIL_TO']            = to
        with patch.dict('os.environ', env, clear=False):
            # Reimport to pick up env changes
            from importlib import reload
            import signals.notifications as mod
            reload(mod)
            return mod._settings_ok()

    def test_all_missing_returns_false(self):
        with patch.dict('os.environ', {}, clear=True):
            from importlib import reload
            import signals.notifications as mod
            reload(mod)
            assert mod._settings_ok() is False

    def test_only_user_returns_false(self):
        with patch.dict('os.environ', {'ALERT_EMAIL_HOST_USER': 'a@b.com'}, clear=True):
            from importlib import reload
            import signals.notifications as mod
            reload(mod)
            assert mod._settings_ok() is False

    def test_user_and_password_missing_to_returns_false(self):
        with patch.dict('os.environ', {
            'ALERT_EMAIL_HOST_USER':     'a@b.com',
            'ALERT_EMAIL_HOST_PASSWORD': 'secret',
        }, clear=True):
            from importlib import reload
            import signals.notifications as mod
            reload(mod)
            assert mod._settings_ok() is False

    def test_all_three_set_returns_true(self):
        with patch.dict('os.environ', {
            'ALERT_EMAIL_HOST_USER':     'a@b.com',
            'ALERT_EMAIL_HOST_PASSWORD': 'secret',
            'ALERT_EMAIL_TO':            'me@me.com',
        }, clear=True):
            from importlib import reload
            import signals.notifications as mod
            reload(mod)
            assert mod._settings_ok() is True


# ── _subject ──────────────────────────────────────────────────────────────────

class TestSubject:
    def setup_method(self):
        from signals.notifications import _subject
        self._subject = _subject

    def test_bullish_label_is_buy(self):
        subject = self._subject(_bullish_signal())
        assert 'BUY' in subject

    def test_bearish_label_is_sell(self):
        subject = self._subject(_bearish_signal())
        assert 'SELL' in subject

    def test_subject_contains_pair(self):
        subject = self._subject(_bullish_signal(pair='XAUUSD'))
        assert 'XAUUSD' in subject

    def test_subject_contains_probability(self):
        subject = self._subject(_bullish_signal(probability=0.723))
        # Probability formatted as % in subject
        assert '72' in subject  # at least 72% visible

    def test_subject_contains_tomorrow(self):
        subject = self._subject(_bullish_signal())
        assert 'TOMORROW' in subject.upper()

    def test_subject_is_string(self):
        assert isinstance(self._subject(_bullish_signal()), str)


# ── _body ─────────────────────────────────────────────────────────────────────

class TestBody:
    def setup_method(self):
        from signals.notifications import _body
        self._body = _body

    def test_body_contains_tomorrow(self):
        body = self._body(_bullish_signal())
        assert "TOMORROW" in body.upper()

    def test_body_contains_pair(self):
        body = self._body(_bullish_signal(pair='EURUSD'))
        assert 'EURUSD' in body

    def test_body_contains_mt5_steps(self):
        body = self._body(_bullish_signal(), _decision())
        assert 'MetaTrader' in body or 'MT5' in body

    def test_body_buy_limit_for_bullish(self):
        body = self._body(_bullish_signal(), _decision())
        assert 'Buy Limit' in body

    def test_body_sell_limit_for_bearish(self):
        body = self._body(_bearish_signal(), _decision())
        assert 'Sell Limit' in body

    def test_body_contains_lot_size(self):
        body = self._body(_bullish_signal(), _decision(lot_size=0.03))
        assert '0.03' in body

    def test_body_contains_stop_loss(self):
        body = self._body(_bullish_signal(sl=1.06980))
        assert '1.0698' in body

    def test_body_contains_take_profit(self):
        body = self._body(_bullish_signal(tp=1.07800))
        assert '1.078' in body

    def test_body_execute_action_shows_go(self):
        body = self._body(_bullish_signal(), _decision(action='EXECUTE'))
        assert 'GO' in body or 'EXECUTE' in body

    def test_body_wait_action_shows_wait(self):
        body = self._body(_bullish_signal(), _decision(action='WAIT'))
        assert 'WAIT' in body

    def test_body_skip_action_shows_skip(self):
        body = self._body(_bullish_signal(), _decision(action='SKIP'))
        assert 'SKIP' in body

    def test_body_without_decision_still_works(self):
        body = self._body(_bullish_signal(), None)
        assert isinstance(body, str)
        assert len(body) > 50

    def test_body_contains_risk_amount(self):
        body = self._body(_bullish_signal(), _decision(risk_usd=10.0))
        assert '10' in body

    def test_body_is_string(self):
        assert isinstance(self._body(_bullish_signal()), str)


# ── send_signal_alert ─────────────────────────────────────────────────────────

class TestSendSignalAlert:
    def _import(self):
        from signals.notifications import send_signal_alert
        return send_signal_alert

    def test_returns_false_below_threshold(self):
        send_signal_alert = self._import()
        sig = _bullish_signal(probability=0.50)  # below 0.60 default threshold
        assert send_signal_alert(sig) is False

    def test_returns_false_exactly_at_threshold_boundary(self):
        # Threshold is 0.60; 0.599 should fail
        with patch.dict('os.environ', {'SIGNAL_NOTIFY_THRESHOLD': '0.60'}):
            from importlib import reload
            import signals.notifications as mod
            reload(mod)
            sig = _bullish_signal(probability=0.599)
            assert mod.send_signal_alert(sig) is False

    def test_returns_false_without_email_config(self):
        send_signal_alert = self._import()
        sig = _bullish_signal(probability=0.75)
        with patch.dict('os.environ', {}, clear=True):
            from importlib import reload
            import signals.notifications as mod
            reload(mod)
            assert mod.send_signal_alert(sig) is False

    def test_calls_send_mail_when_configured(self):
        sig = _bullish_signal(probability=0.75)
        env = {
            'ALERT_EMAIL_HOST_USER':     'a@b.com',
            'ALERT_EMAIL_HOST_PASSWORD': 'secret',
            'ALERT_EMAIL_TO':            'me@me.com',
            'SIGNAL_NOTIFY_THRESHOLD':   '0.60',
        }
        with patch.dict('os.environ', env, clear=True):
            from importlib import reload
            import signals.notifications as mod
            reload(mod)
            # send_mail is imported lazily inside the function, patch at source
            with patch('django.core.mail.send_mail') as mock_send:
                mock_send.return_value = None
                result = mod.send_signal_alert(sig, _decision())
                assert mock_send.called
                assert result is True

    def test_send_mail_receives_correct_recipient(self):
        sig = _bullish_signal(probability=0.75)
        env = {
            'ALERT_EMAIL_HOST_USER':     'sender@b.com',
            'ALERT_EMAIL_HOST_PASSWORD': 'secret',
            'ALERT_EMAIL_TO':            'target@me.com',
            'SIGNAL_NOTIFY_THRESHOLD':   '0.60',
        }
        with patch.dict('os.environ', env, clear=True):
            from importlib import reload
            import signals.notifications as mod
            reload(mod)
            with patch('django.core.mail.send_mail') as mock_send:
                mock_send.return_value = None
                mod.send_signal_alert(sig, _decision())
                _, kwargs = mock_send.call_args
                recipients = kwargs.get('recipient_list') or mock_send.call_args[0][3]
                assert 'target@me.com' in recipients

    def test_returns_false_when_send_mail_raises(self):
        sig = _bullish_signal(probability=0.75)
        env = {
            'ALERT_EMAIL_HOST_USER':     'a@b.com',
            'ALERT_EMAIL_HOST_PASSWORD': 'secret',
            'ALERT_EMAIL_TO':            'me@me.com',
            'SIGNAL_NOTIFY_THRESHOLD':   '0.60',
        }
        with patch.dict('os.environ', env, clear=True):
            from importlib import reload
            import signals.notifications as mod
            reload(mod)
            with patch('django.core.mail.send_mail', side_effect=Exception('SMTP error')):
                result = mod.send_signal_alert(sig, _decision())
                assert result is False
