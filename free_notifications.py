#!/usr/bin/env python3
"""
Free Email and SMS Notification System
Uses Gmail SMTP for email and carrier email-to-SMS gateways for SMS
"""

import smtplib
import os
from email.message import EmailMessage
from datetime import datetime

class FreeNotificationSystem:
    """Free notification system using Gmail SMTP and carrier gateways"""

    def __init__(self):
        # Gmail credentials (set these environment variables)
        self.gmail_user = os.getenv('GMAIL_USERNAME', '1man2amazing@gmail.com')
        self.gmail_app_password = os.getenv('GMAIL_APP_PASSWORD', 'ajlkyonpbkljeqzc')

        # Carrier email-to-SMS gateways
        self.carrier_gateways = {
            'att': '@txt.att.net',
            'verizon': '@vtext.com',
            'tmobile': '@tmomail.net',
            'sprint': '@messaging.sprintpcs.com',
            'default': '@txt.att.net'  # Default to AT&T
        }

    def send_email(self, subject, body, to_email):
        """Send email via Gmail SMTP"""
        try:
            msg = EmailMessage()
            msg['Subject'] = subject
            msg['From'] = self.gmail_user
            msg['To'] = to_email
            msg.set_content(body)

            # Connect to Gmail SMTP
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                smtp.login(self.gmail_user, self.gmail_app_password)
                smtp.send_message(msg)

            print(f"✅ EMAIL SENT to {to_email}")
            return True

        except Exception as e:
            print(f"❌ Email failed: {e}")
            return False

    def send_sms_via_carrier(self, phone_number, message, carrier='default'):
        """Send SMS via carrier email-to-SMS gateway"""
        try:
            gateway = self.carrier_gateways.get(carrier, self.carrier_gateways['default'])
            sms_email = f"{phone_number}{gateway}"

            # Send as email to carrier gateway
            return self.send_email('', message, sms_email)

        except Exception as e:
            print(f"❌ SMS failed: {e}")
            return False

    def send_notification(self, subject, message, email_recipient=None, sms_recipient=None):
        """Send notification to both email and SMS"""
        success_count = 0

        # Send email
        if email_recipient:
            if self.send_email(subject, message, email_recipient):
                success_count += 1

        # Send SMS
        if sms_recipient:
            # Try multiple carriers if first one fails
            carriers_to_try = ['att', 'verizon', 'tmobile']
            sms_sent = False

            for carrier in carriers_to_try:
                if self.send_sms_via_carrier(sms_recipient, message[:160], carrier):
                    sms_sent = True
                    success_count += 1
                    break

            if not sms_sent:
                print(f"❌ SMS failed for all carriers")

        return success_count > 0

def test_free_notifications():
    """Test the free notification system"""

    print("🚀 Testing FREE Email & SMS Notifications")
    print("=" * 50)

    # Check if credentials are set
    gmail_user = os.getenv('GMAIL_USERNAME')
    gmail_password = os.getenv('GMAIL_APP_PASSWORD')

    if not gmail_user or not gmail_password:
        print("❌ Gmail credentials not found!")
        print()
        print("📧 To send real notifications, you need Gmail credentials:")
        print()
        print("1. Go to https://myaccount.google.com/security")
        print("2. Enable 2-Step Verification if not already enabled")
        print("3. Generate an App Password:")
        print("   - Click 'App passwords' under 'Signing in to Google'")
        print("   - Select 'Mail' and 'Other (custom name)'")
        print("   - Enter 'Forex Notifications' as the name")
        print("   - Copy the 16-character password")
        print()
        print("4. Set environment variables:")
        print("   GMAIL_USERNAME=your-gmail@gmail.com")
        print("   GMAIL_APP_PASSWORD=your-16-char-app-password")
        print()
        print("5. Or run with credentials:")
        print("   GMAIL_USERNAME=your-email@gmail.com GMAIL_APP_PASSWORD=your-password python free_notifications.py")
        print()
        return False

    # Initialize notification system
    notifier = FreeNotificationSystem()

    # Test recipients
    test_email = os.getenv('NOTIFICATION_EMAIL', 'mydecretor@protonmail.com')
    test_sms = os.getenv('NOTIFICATION_SMS', '7084652230')

    print(f"📧 Email recipient: {test_email}")
    print(f"📱 SMS recipient: {test_sms}")
    print(f"📧 Gmail account: {gmail_user}")
    print()

    # Test message
    subject = "🚀 FOREX SYSTEM - FREE NOTIFICATION TEST"
    message = f"""URGENT: Free Notification Test

✅ Your Forex System is Working!

📊 Current Status:
• EURUSD Model: 61.5% accuracy
• XAUUSD Model: 77.1% accuracy
• Target: 85% accuracy
• Automated training: Active
• Signal generation: Ready

📱 If you receive this EMAIL, reply "EMAIL_WORKS"
📱 If you receive this SMS, reply "SMS_WORKS"

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This message was sent using FREE Gmail SMTP + Carrier Gateways! 🎉
"""

    print("📤 Sending test notifications...")
    print()

    # Send notification
    success = notifier.send_notification(
        subject=subject,
        message=message,
        email_recipient=test_email,
        sms_recipient=test_sms
    )

    print()
    print("=" * 50)
    if success:
        print("✅ NOTIFICATION TEST COMPLETED")
        print("📧 Check your email for the test message")
        print("📱 Check your phone for the SMS")
        print("💡 Reply with 'EMAIL_WORKS' or 'SMS_WORKS' to confirm receipt")
        print()
        print("🎉 SUCCESS! Your free notification system is working!")
    else:
        print("❌ NOTIFICATION TEST FAILED")
        print("💡 Check your Gmail credentials and try again")

    return success

if __name__ == "__main__":
    # The script will use environment variables for credentials
    # Set them before running:
    # export GMAIL_USERNAME=your-gmail@gmail.com
    # export GMAIL_APP_PASSWORD=your-16-char-app-password
    # export NOTIFICATION_EMAIL=mydecorator@protonmail.com
    # export NOTIFICATION_SMS=7734921722

    test_free_notifications()