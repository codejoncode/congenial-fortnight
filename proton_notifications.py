#!/usr/bin/env python3
"""
ProtonMail Notification System (uses your existing ProtonMail account)
"""

import smtplib
import os
from email.message import EmailMessage
from datetime import datetime

class ProtonMailNotificationSystem:
    """Notification system using ProtonMail SMTP"""

    def __init__(self):
        # ProtonMail credentials (you'll need to set these)
        self.proton_user = os.getenv('PROTON_USERNAME', 'mydecretor@protonmail.com')
        self.proton_password = os.getenv('PROTON_PASSWORD', 'your-proton-password')

        # Carrier email-to-SMS gateways
        self.carrier_gateways = {
            'att': '@txt.att.net',
            'verizon': '@vtext.com',
            'tmobile': '@tmomail.net',
            'sprint': '@messaging.sprintpcs.com',
            'default': '@txt.att.net'
        }

    def send_email(self, subject, body, to_email):
        """Send email via ProtonMail SMTP"""
        try:
            msg = EmailMessage()
            msg['Subject'] = subject
            msg['From'] = self.proton_user
            msg['To'] = to_email
            msg.set_content(body)

            # Connect to ProtonMail SMTP
            with smtplib.SMTP_SSL('mail.protonmail.com', 465) as smtp:
                smtp.login(self.proton_user, self.proton_password)
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

def test_proton_notifications():
    """Test the ProtonMail notification system"""

    print("🚀 Testing ProtonMail Email & SMS Notifications")
    print("=" * 52)

    # Check if credentials are set
    proton_user = os.getenv('PROTON_USERNAME')
    proton_password = os.getenv('PROTON_PASSWORD')

    if not proton_user or not proton_password:
        print("❌ ProtonMail credentials not found!")
        print()
        print("📧 To send notifications using your ProtonMail account:")
        print()
        print("1. You already have a ProtonMail account: mydecretor@protonmail.com")
        print("2. For ProtonMail SMTP, you need to:")
        print("   - Go to ProtonMail settings")
        print("   - Enable 'SMTP submission' in All Mail settings")
        print("   - Or use your regular ProtonMail password")
        print()
        print("3. Set environment variables:")
        print("   PROTON_USERNAME=mydecretor@protonmail.com")
        print("   PROTON_PASSWORD=your-protonmail-password")
        print()
        print("4. Or run with credentials:")
        print("   PROTON_USERNAME=mydecretor@protonmail.com PROTON_PASSWORD=your-password python proton_notifications.py")
        print()
        print("⚠️  Note: ProtonMail SMTP might require Bridge setup for automated sending")
        print("💡 If this doesn't work, try Yahoo or Outlook instead")
        print()
        return False

    # Initialize notification system
    notifier = ProtonMailNotificationSystem()

    # Test recipients
    test_email = os.getenv('NOTIFICATION_EMAIL', 'test@example.com')  # Send to a different email for testing
    test_sms = os.getenv('NOTIFICATION_SMS', '7734921722')

    print(f"📧 Email recipient: {test_email}")
    print(f"📱 SMS recipient: {test_sms}")
    print(f"📧 ProtonMail account: {proton_user}")
    print()

    # Test message
    subject = "🚀 FOREX SYSTEM - PROTONMAIL NOTIFICATION TEST"
    message = f"""URGENT: ProtonMail Notification Test

✅ Your Forex System is Working!

📊 Current Status:
• EURUSD Model: 61.5% accuracy
• XAUUSD Model: 77.1% accuracy
• Target: 85% accuracy
• Automated training: Active
• Signal generation: Ready

📱 If you receive this EMAIL, reply "PROTON_EMAIL_WORKS"
📱 If you receive this SMS, reply "PROTON_SMS_WORKS"

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This message was sent using ProtonMail SMTP + Carrier Gateways! 🎉
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
    print("=" * 52)
    if success:
        print("✅ PROTONMAIL NOTIFICATION TEST COMPLETED")
        print("📧 Check your email for the test message")
        print("📱 Check your phone for the SMS")
        print("💡 Reply with 'PROTON_EMAIL_WORKS' or 'PROTON_SMS_WORKS' to confirm receipt")
        print()
        print("🎉 SUCCESS! Your ProtonMail notification system is working!")
    else:
        print("❌ PROTONMAIL NOTIFICATION TEST FAILED")
        print("💡 Try Yahoo or Outlook instead, or check ProtonMail SMTP settings")

    return success

if __name__ == "__main__":
    test_proton_notifications()