#!/usr/bin/env python3
"""
Test notifications using web-based services (no authentication required)
"""

import requests
import os
from datetime import datetime

class WebNotificationSystem:
    """Notification system using web-based services"""

    def send_email_via_web(self, to_email, subject, message):
        """Try various web-based email services"""
        try:
            # Try different web email services
            services = [
                {
                    'name': 'MailThis.to',
                    'url': 'https://mailthis.to/your-email@example.com',  # Replace with actual service
                    'method': 'POST',
                    'data': {
                        'to': to_email,
                        'subject': subject,
                        'message': message
                    }
                },
                {
                    'name': 'Formspree',
                    'url': 'https://formspree.io/f/your-form-id',  # Would need actual form
                    'method': 'POST',
                    'data': {
                        'email': to_email,
                        'subject': subject,
                        'message': message
                    }
                }
            ]

            for service in services:
                try:
                    if service['method'] == 'POST':
                        response = requests.post(service['url'], data=service['data'], timeout=10)
                    else:
                        response = requests.get(service['url'], params=service['data'], timeout=10)

                    if response.status_code in [200, 201, 202]:
                        print(f"✅ EMAIL SENT via {service['name']} to {to_email}")
                        return True
                    else:
                        print(f"{service['name']} returned: {response.status_code}")

                except Exception as e:
                    print(f"{service['name']} failed: {e}")
                    continue

            # If no web services work, show what would be sent
            print(f"📧 Would send email to: {to_email}")
            print(f"📝 Subject: {subject}")
            print(f"📝 Message: {message[:100]}...")
            return False

        except Exception as e:
            print(f"Web email failed: {e}")
            return False

    def send_sms_via_web(self, phone_number, message):
        """Try web-based SMS services"""
        try:
            # Try free SMS services
            services = [
                {
                    'name': 'Textbelt (Free)',
                    'url': 'https://textbelt.com/text',
                    'data': {
                        'phone': phone_number,
                        'message': message,
                        'key': 'textbelt'
                    }
                }
            ]

            for service in services:
                try:
                    response = requests.post(service['url'], data=service['data'], timeout=10)
                    result = response.json()

                    if service['name'] == 'Textbelt (Free)':
                        if result.get('success'):
                            print(f"✅ SMS SENT to {phone_number} via Textbelt")
                            return True
                        else:
                            print(f"Textbelt failed: {result.get('error', 'Unknown error')}")

                except Exception as e:
                    print(f"{service['name']} failed: {e}")
                    continue

            # Show what would be sent
            print(f"📱 Would send SMS to: {phone_number}")
            print(f"📝 Message: {message[:100]}...")
            return False

        except Exception as e:
            print(f"Web SMS failed: {e}")
            return False

    def send_notification(self, subject, message, email_recipient=None, sms_recipient=None):
        """Send notification via web services"""
        success_count = 0

        if email_recipient:
            if self.send_email_via_web(email_recipient, subject, message):
                success_count += 1

        if sms_recipient:
            if self.send_sms_via_web(sms_recipient, message[:160]):
                success_count += 1

        return success_count > 0

def test_web_notifications():
    """Test web-based notifications"""

    print("🚀 Testing Web-Based Notifications (No Auth Required)")
    print("=" * 58)

    notifier = WebNotificationSystem()

    test_email = 'mydecretor@protonmail.com'
    test_sms = '7734921722'

    print(f"📧 Email recipient: {test_email}")
    print(f"📱 SMS recipient: {test_sms}")
    print()

    subject = "🚀 FOREX SYSTEM - WEB TEST"
    message = f"""URGENT: Web-Based Notification Test

✅ Your Forex System is Working!

📊 Current Status:
• EURUSD Model: 61.5% accuracy
• XAUUSD Model: 77.1% accuracy
• Target: 85% accuracy
• Automated training: Active
• Signal generation: Ready

📧 If you receive this EMAIL, reply "WEB_EMAIL_WORKS"
📱 If you receive this SMS, reply "WEB_SMS_WORKS"

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This message was sent using web services! 🌐
"""

    print("📤 Sending test notifications via web services...")
    print()

    success = notifier.send_notification(
        subject=subject,
        message=message,
        email_recipient=test_email,
        sms_recipient=test_sms
    )

    print()
    print("=" * 58)
    if success:
        print("✅ WEB NOTIFICATION TEST COMPLETED")
        print("📧 Check your email for the test message")
        print("📱 Check your phone for the SMS")
        print("💡 Reply with 'WEB_EMAIL_WORKS' or 'WEB_SMS_WORKS' to confirm receipt")
        print()
        print("🎉 SUCCESS! Web-based notifications are working!")
    else:
        print("❌ WEB NOTIFICATION TEST - NO REAL MESSAGES SENT")
        print("💡 Web services may not be available or may require API keys")
        print()
        print("📋 The messages that WOULD be sent:")
        print(f"📧 Email to: {test_email}")
        print(f"📱 SMS to: {test_sms}")
        print(f"📝 Subject: {subject}")
        print(f"📝 Message: {message[:200]}...")
        print()
        print("💡 To send real notifications, you need:")
        print("   1. Gmail + App Password (after enabling 2-Step Verification)")
        print("   2. Yahoo Mail + App Password")
        print("   3. Outlook + App Password")
        print("   4. Or use a paid SMS service like Twilio")

    return success

if __name__ == "__main__":
    test_web_notifications()