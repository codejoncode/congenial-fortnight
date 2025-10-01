#!/usr/bin/env python3
"""
Test script for notification system
Run this to validate email and SMS notifications work
"""

import os
import sys
sys.path.append('.')

from notification_system import NotificationSystem

def test_notifications():
    """Test the notification system with sample data"""

    print("🧪 Testing Notification System")
    print("=" * 40)

    # Sample signals for testing
    test_signals = [
        {
            'pair': 'EURUSD',
            'signal': 'bullish',
            'probability': 0.82,
            'entry_price': 1.0850,
            'stop_loss': 0.0020
        },
        {
            'pair': 'XAUUSD',
            'signal': 'bearish',
            'probability': 0.75,
            'entry_price': 1950.50,
            'stop_loss': 15.0
        }
    ]

    notifier = NotificationSystem()

    # Test email
    print("\n📧 Testing Email Notification...")
    email_recipient = os.getenv('NOTIFICATION_EMAIL')
    if email_recipient:
        success = notifier.send_email(
            email_recipient,
            "Test Forex Signals",
            "This is a test notification from your Forex Signal System.",
            None
        )
        print(f"Email test: {'✅ Success' if success else '❌ Failed'}")
    else:
        print("❌ No email recipient configured (set NOTIFICATION_EMAIL)")

    # Test SMS
    print("\n📱 Testing SMS Notification...")
    sms_recipient = os.getenv('NOTIFICATION_SMS')
    if sms_recipient:
        success = notifier.send_sms_textbelt(
            sms_recipient,
            "Test: Forex signals generated. Check email for details."
        )
        print(f"SMS test: {'✅ Success' if success else '❌ Failed'}")
    else:
        print("❌ No SMS recipient configured (set NOTIFICATION_SMS)")

    # Test full signal notification
    print("\n📊 Testing Full Signal Notification...")
    recipients = []
    if email_recipient:
        recipients.append(email_recipient)
    if sms_recipient:
        recipients.append(sms_recipient)

    if recipients:
        notifier.send_signal_notification(test_signals, recipients)
        print("✅ Signal notifications sent to configured recipients")
    else:
        print("❌ No recipients configured for signal notifications")

    print("\n" + "=" * 40)
    print("📋 Configuration Status:")
    print(f"Email configured: {'✅ Yes' if email_recipient else '❌ No'}")
    print(f"SMS configured: {'✅ Yes' if sms_recipient else '❌ No'}")
    print(f"Recipients: {len(recipients)} configured")

    if not recipients:
        print("\n💡 To enable notifications:")
        print("1. Set NOTIFICATION_EMAIL environment variable")
        print("2. Set NOTIFICATION_SMS environment variable")
        print("3. For GitHub Actions, add these as repository secrets")

if __name__ == "__main__":
    test_notifications()