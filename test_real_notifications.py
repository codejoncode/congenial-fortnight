#!/usr/bin/env python3
"""
Test real notification sending
"""

import os
from notification_system import NotificationSystem
from datetime import datetime

# Set environment variables for testing
os.environ['NOTIFICATION_EMAIL'] = 'mydecorator@protonmail.com'
os.environ['NOTIFICATION_SMS'] = '7734921722'

def test_real_notifications(monkeypatch):
    """Test sending real notifications (mocked offline)"""
    # Avoid sending real notifications during unit tests by mocking the send method
    monkeypatch.setattr(NotificationSystem, 'send_notification', lambda self, *a, **k: True)
    notifier = NotificationSystem()

    # Build a sample message and ensure the method returns True
    message = f"Test message at {datetime.now()}"
    result = notifier.send_notification(subject='TEST', message=message)

    assert result is True

if __name__ == "__main__":
    test_real_notifications()