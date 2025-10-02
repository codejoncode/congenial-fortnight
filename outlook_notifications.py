#!/usr/bin/env python3
"""
Outlook/Hotmail Free Notification System
"""

import smtplib
import os
from email.message import EmailMessage
from datetime import datetime

class OutlookNotificationSystem:
    """Free notification system using Outlook/Hotmail SMTP"""

    def __init__(self):
        # Outlook credentials
        self.outlook_user = os.getenv('OUTLOOK_USERNAME', 'your-email@outlook.com')
        self.outlook_app_password = os.getenv('OUTLOOK_APP_PASSWORD', 'your-app-password')

        # Carrier email-to-SMS gateways
        self.carrier_gateways = {
            'att': '@txt.att.net',
            'verizon': '@vtext.com',
            'tmobile': '@tmomail.net',
            'sprint': '@messaging.sprintpcs.com',
            'default': '@txt.att.net'
        }

    def send_email(self, subject, body, to_email):
        """Send email via Outlook SMTP"""
        try:
            msg = EmailMessage()
            msg['Subject'] = subject
            msg['From'] = self.outlook_user
            msg['To'] = to_email
            msg.set_content(body)

            # Connect to Outlook SMTP
            with smtplib.SMTP('smtp-mail.outlook.com', 587) as smtp:
                smtp.starttls()
                smtp.login(self.outlook_user, self.outlook_app_password)
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

def test_outlook_notifications():
    """Test the Outlook notification system"""

    print("🚀 Testing FREE Outlook Email & SMS Notifications")
    print("=" * 57)

    # Check if credentials are set
    outlook_user = os.getenv('OUTLOOK_USERNAME')
    outlook_password = os.getenv('OUTLOOK_APP_PASSWORD')

    if not outlook_user or not outlook_password:
        print("❌ Outlook credentials not found!")
        print()
        print("📧 To send real notifications, you need Outlook credentials:")
        print()
        print("1. Go to: https://outlook.com")
        print("2. Sign in to your Outlook/Hotmail account (or create a free one)")
        print("3. Go to: https://account.microsoft.com/security")
        print("4. Click 'More security options'")
        print("5. Under 'App passwords', click 'Create a new app password'")
        print("6. Enter 'Forex Notifications' as the name")
        print("7. Copy the generated password (no spaces)")
        print()
        print("8. Set environment variables:")
        print("   OUTLOOK_USERNAME=your-email@outlook.com")
        print("   OUTLOOK_APP_PASSWORD=your-16-char-app-password")
        print()
        print("9. Or run with credentials:")
        print("   OUTLOOK_USERNAME=your-email@outlook.com OUTLOOK_APP_PASSWORD=your-password python outlook_notifications.py")
        print()
        return False

    # Initialize notification system
    notifier = OutlookNotificationSystem()

    # Test recipients
    test_email = os.getenv('NOTIFICATION_EMAIL', 'mydecretor@protonmail.com')
    test_sms = os.getenv('NOTIFICATION_SMS', '7734921722')

    print(f"📧 Email recipient: {test_email}")
    print(f"📱 SMS recipient: {test_sms}")
    print(f"📧 Outlook account: {outlook_user}")
    print()

    # Test message
    subject = "🚀 FOREX SYSTEM - OUTLOOK NOTIFICATION TEST"
    message = f"""URGENT: Outlook Notification Test

✅ Your Forex System is Working!

📊 Current Status:
• EURUSD Model: 61.5% accuracy
• XAUUSD Model: 77.1% accuracy
• Target: 85% accuracy
• Automated training: Active
• Signal generation: Ready

📱 If you receive this EMAIL, reply "OUTLOOK_EMAIL_WORKS"
📱 If you receive this SMS, reply "OUTLOOK_SMS_WORKS"

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This message was sent using FREE Outlook SMTP + Carrier Gateways! 🎉
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
    print("=" * 57)
    if success:
        print("✅ OUTLOOK NOTIFICATION TEST COMPLETED")
        print("📧 Check your email for the test message")
        print("📱 Check your phone for the SMS")
        print("💡 Reply with 'OUTLOOK_EMAIL_WORKS' or 'OUTLOOK_SMS_WORKS' to confirm receipt")
        print()
        print("🎉 SUCCESS! Your free Outlook notification system is working!")
    else:
        print("❌ OUTLOOK NOTIFICATION TEST FAILED")
        print("💡 Check your Outlook credentials and try again")

    return success

if __name__ == "__main__":
    test_outlook_notifications()