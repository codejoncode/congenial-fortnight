import os
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
from datetime import datetime

class NotificationSystem:
    """Free notification system for signals and alerts"""

    def __init__(self):
        self.email_config = {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': os.getenv('EMAIL_USERNAME'),
            'password': os.getenv('EMAIL_PASSWORD'),
            'from_email': os.getenv('EMAIL_FROM', os.getenv('EMAIL_USERNAME'))
        }

        # Free SMS services (limited usage)
        self.sms_config = {
            'textbelt_api': 'https://textbelt.com/text',
            'twilio_free': os.getenv('TWILIO_FREE_KEY')  # If available
        }

    def send_email(self, to_email, subject, message, html_content=None):
        """Send email using SMTP (works with Gmail, Outlook, etc.)"""
        try:
            if not self.email_config['username'] or not self.email_config['password']:
                print("Email credentials not configured")
                return False

            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.email_config['from_email']
            msg['To'] = to_email

            # Plain text version
            msg.attach(MIMEText(message, 'plain'))

            # HTML version if provided
            if html_content:
                msg.attach(MIMEText(html_content, 'html'))

            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            server.sendmail(self.email_config['from_email'], to_email, msg.as_string())
            server.quit()

            print(f"Email sent to {to_email}")
            return True

        except Exception as e:
            print(f"Email sending failed: {e}")
            return False

    def send_sms_textbelt(self, phone_number, message):
        """Send SMS using Textbelt (free tier: 1 SMS/day)"""
        try:
            payload = {
                'phone': phone_number,
                'message': message,
                'key': 'textbelt'  # Free tier key
            }

            response = requests.post(self.sms_config['textbelt_api'], data=payload)
            result = response.json()

            if result.get('success'):
                print(f"SMS sent to {phone_number}")
                return True
            else:
                print(f"SMS failed: {result.get('error', 'Unknown error')}")
                return False

        except Exception as e:
            print(f"SMS sending failed: {e}")
            return False

    def send_signal_notification(self, signals, recipients):
        """Send signal notifications to multiple recipients"""
        if not signals:
            print("No signals to notify")
            return

        # Format signal message
        signal_text = self.format_signal_message(signals)
        signal_html = self.format_signal_html(signals)

        subject = f"Forex Signals - {datetime.now().strftime('%Y-%m-%d')}"

        for recipient in recipients:
            if '@' in recipient:  # Email
                self.send_email(recipient, subject, signal_text, signal_html)
            else:  # Phone number
                self.send_sms_textbelt(recipient, signal_text[:160])  # SMS limit

    def format_signal_message(self, signals):
        """Format signals for text message"""
        lines = [f"📊 Forex Signals - {datetime.now().strftime('%Y-%m-%d %H:%M')}"]

        for signal in signals:
            direction = "🟢 BULLISH" if signal.get('signal') == 'bullish' else "🔴 BEARISH"
            pair = signal.get('pair', 'EURUSD')
            confidence = signal.get('probability', 0)
            entry = signal.get('entry_price', 0)
            stop = signal.get('stop_loss', 0)

            lines.append(f"{pair}: {direction}")
            lines.append(f"Entry: {entry:.5f}")
            lines.append(f"Stop: {stop:.5f}")
            lines.append(f"Confidence: {confidence:.1%}")
            lines.append("")

        return "\n".join(lines)

    def format_signal_html(self, signals):
        """Format signals for HTML email"""
        html = f"""
        <html>
        <body>
            <h2>📊 Forex Signals - {datetime.now().strftime('%Y-%m-%d %H:%M')}</h2>
            <table border="1" style="border-collapse: collapse;">
                <tr>
                    <th>Pair</th>
                    <th>Signal</th>
                    <th>Entry Price</th>
                    <th>Stop Loss</th>
                    <th>Confidence</th>
                </tr>
        """

        for signal in signals:
            direction = "🟢 BULLISH" if signal.get('signal') == 'bullish' else "🔴 BEARISH"
            pair = signal.get('pair', 'EURUSD')
            confidence = signal.get('probability', 0)
            entry = signal.get('entry_price', 0)
            stop = signal.get('stop_loss', 0)

            html += f"""
                <tr>
                    <td>{pair}</td>
                    <td>{direction}</td>
                    <td>{entry:.5f}</td>
                    <td>{stop:.5f}</td>
                    <td>{confidence:.1%}</td>
                </tr>
            """

        html += """
            </table>
            <p><small>This is an automated message from your Forex Signal System.</small></p>
        </body>
        </html>
        """

        return html

    def send_daily_report(self, report_data, recipients):
        """Send daily performance report"""
        subject = f"Daily Forex Report - {datetime.now().strftime('%Y-%m-%d')}"

        # Format report
        message = f"""
Daily Forex System Report
{'='*30}

Training Status: {report_data.get('training_status', 'Unknown')}
Backtest Results: {report_data.get('backtest_accuracy', 'N/A')} accuracy
New Signals: {len(report_data.get('signals', []))}

Performance Metrics:
- Total Pips: {report_data.get('total_pips', 0):.1f}
- Win Rate: {report_data.get('win_rate', 0):.1%}
- Profit Factor: {report_data.get('profit_factor', 0):.2f}

Next Update: Tomorrow at 2 AM UTC
        """

        for recipient in recipients:
            if '@' in recipient:
                self.send_email(recipient, subject, message)

# Usage example
if __name__ == "__main__":
    notifier = NotificationSystem()

    # Example signals
    signals = [
        {
            'pair': 'EURUSD',
            'signal': 'bullish',
            'probability': 0.82,
            'entry_price': 1.0850,
            'stop_loss': 1.0820
        }
    ]

    # Send to email and SMS
    recipients = [
        'your-email@example.com',
        '+1234567890'  # SMS
    ]

    notifier.send_signal_notification(signals, recipients)