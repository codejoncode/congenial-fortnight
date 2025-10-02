#!/usr/bin/env python3
"""
Alternative Free Notification System using Yahoo Mail
"""

import smtplib
import os
from email.message import EmailMessage
from datetime import datetime

class YahooNotificationSystem:
    """Free notification system using Yahoo SMTP"""

    def __init__(self):
        # Yahoo credentials
        self.yahoo_user = os.getenv('YAHOO_USERNAME', 'your-email@yahoo.com')
        self.yahoo_app_password = os.getenv('YAHOO_APP_PASSWORD', 'your-app-password')

        # Carrier email-to-SMS gateways
        self.carrier_gateways = {
            'att': '@txt.att.net',
            'verizon': '@vtext.com',
            'tmobile': '@tmomail.net',
            'sprint': '@messaging.sprintpcs.com',
            'default': '@txt.att.net'
        }

    def send_email(self, subject, body, to_email):
        """Send email via Yahoo SMTP"""
        try:
            msg = EmailMessage()
            msg['Subject'] = subject
            msg['From'] = self.yahoo_user
            msg['To'] = to_email
            msg.set_content(body)

            # Connect to Yahoo SMTP
            with smtplib.SMTP_SSL('smtp.mail.yahoo.com', 465) as smtp:
                smtp.login(self.yahoo_user, self.yahoo_app_password)
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

def test_yahoo_notifications():
    """Test the Yahoo notification system"""

    print("🚀 Testing FREE Yahoo Email & SMS Notifications")
    print("=" * 55)

    # Check if credentials are set
    yahoo_user = os.getenv('YAHOO_USERNAME')
    yahoo_password = os.getenv('YAHOO_APP_PASSWORD')

    if not yahoo_user or not yahoo_password:
        print("❌ Yahoo credentials not found!")
        print()
        print("📧 To send real notifications, you need Yahoo credentials:")
        print()
        print("1. Go to: https://mail.yahoo.com")
        print("2. Sign in to your Yahoo account (or create a free one)")
        print("3. Click your profile picture > Account Info")
        print("4. Go to 'Account Security' on the left")
        print("5. Click 'Generate app password' or 'Manage app passwords'")
        print("6. Select 'Other App' and enter 'Forex Notifications'")
        print("7. Copy the generated password")
        print()
        print("8. Set environment variables:")
        print("   YAHOO_USERNAME=your-yahoo-email@yahoo.com")
        print("   YAHOO_APP_PASSWORD=your-yahoo-app-password")
        print()
        print("9. Or run with credentials:")
        print("   YAHOO_USERNAME=your-email@yahoo.com YAHOO_APP_PASSWORD=your-password python yahoo_notifications.py")
        print()
        return False

    # Initialize notification system
    notifier = YahooNotificationSystem()

    # Test recipients
    test_email = os.getenv('NOTIFICATION_EMAIL', 'mydecretor@protonmail.com')
    test_sms = os.getenv('NOTIFICATION_SMS', '7734921722')

    print(f"📧 Email recipient: {test_email}")
    print(f"📱 SMS recipient: {test_sms}")
    print(f"📧 Yahoo account: {yahoo_user}")
    print()

    # Test message
    subject = "🚀 FOREX SYSTEM - YAHOO NOTIFICATION TEST"
    message = f"""URGENT: Yahoo Notification Test

✅ Your Forex System is Working!

📊 Current Status:
• EURUSD Model: 61.5% accuracy
• XAUUSD Model: 77.1% accuracy
• Target: 85% accuracy
• Automated training: Active
• Signal generation: Ready

📱 If you receive this EMAIL, reply "YAHOO_EMAIL_WORKS"
📱 If you receive this SMS, reply "YAHOO_SMS_WORKS"

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This message was sent using FREE Yahoo SMTP + Carrier Gateways! 🎉
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
    print("=" * 55)
    if success:
        print("✅ YAHOO NOTIFICATION TEST COMPLETED")
        print("📧 Check your email for the test message")
        print("📱 Check your phone for the SMS")
        print("💡 Reply with 'YAHOO_EMAIL_WORKS' or 'YAHOO_SMS_WORKS' to confirm receipt")
        print()
        print("🎉 SUCCESS! Your free Yahoo notification system is working!")
    else:
        print("❌ YAHOO NOTIFICATION TEST FAILED")
        print("💡 Check your Yahoo credentials and try again")

    return success

if __name__ == "__main__":
    test_yahoo_notifications()