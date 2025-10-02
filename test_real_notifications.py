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

def test_real_notifications():
    """Test sending real notifications"""

    notifier = NotificationSystem()

    print('🚀 Testing REAL notification sending...')
    print('Email recipient:', os.getenv('NOTIFICATION_EMAIL'))
    print('SMS recipient:', os.getenv('NOTIFICATION_SMS'))

    # Test notification
    message = """URGENT: This is a REAL test from your automated forex system!

✅ System Status:
• EURUSD Model: 61.5% accuracy
• XAUUSD Model: 77.1% accuracy
• Target: 85% accuracy
• Automated training: Active
• Signal generation: Ready

📱 If you receive this message, REPLY with:
EMAIL: "RECEIVED_EMAIL"
SMS: "RECEIVED_SMS"

Your AI forex system is operational! 🎯📈

Time: """ + str(datetime.now())

    result = notifier.send_notification(
        subject='🚀 FOREX SYSTEM - REAL TEST MESSAGE',
        message=message
    )

    print('Notification result:', result)
    return result

if __name__ == "__main__":
    test_real_notifications()