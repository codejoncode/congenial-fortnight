#!/usr/bin/env python3
"""
Notification Service for Trading Signals
Sends email and SMS notifications when new quality trading signals are generated
Integrates with both ML Pip-Based and Harmonic Pattern systems
"""

import os
import sys
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NotificationService:
    """
    Service for sending trading signal notifications via email and SMS
    
    Configuration from environment variables:
    - EMAIL_USER: Gmail sender (1man2amazing@gmail.com)
    - EMAIL_PASSWORD: Gmail app password
    - NOTIFICATION_EMAIL: Recipient email (mydecretor@protonmail.com)
    - SMS_NUMBER: T-Mobile phone number (7084652230)
    """
    
    def __init__(self):
        self.email_user = os.getenv('EMAIL_USER', '1man2amazing@gmail.com')
        # IMPORTANT: This must be a Gmail App Password, not your regular password
        # To generate: Google Account → Security → 2-Step Verification → App Passwords
        self.email_password = os.getenv('EMAIL_PASSWORD', 'ajlkyonpbkljeqzc')
        self.notification_email = os.getenv('NOTIFICATION_EMAIL', 'mydecretor@protonmail.com')
        self.sms_number = os.getenv('SMS_NUMBER', '7084652230')
        
        # T-Mobile email-to-SMS gateway
        self.sms_gateway = f"{self.sms_number}@tmomail.net"
        
        # SMTP configuration for Gmail
        self.smtp_server = 'smtp.gmail.com'
        self.smtp_port = 587
        
        logger.info("NotificationService initialized")
        logger.info(f"Email notifications to: {self.notification_email}")
        logger.info(f"SMS notifications to: {self.sms_number} (T-Mobile)")
    
    def format_ml_signal_message(self, signal: Dict, pair: str) -> tuple[str, str]:
        """
        Format ML signal for notification
        Returns: (short_message, detailed_message)
        """
        signal_type = signal.get('type', 'N/A').upper()
        entry = signal.get('entry', 0)
        stop_loss = signal.get('stop_loss', 0)
        take_profit = signal.get('take_profit', 0)
        rr_ratio = signal.get('risk_reward_ratio', 0)
        confidence = signal.get('confidence', 0)
        quality = signal.get('quality', 'N/A').upper()
        
        # Short message for SMS (160 char limit for T-Mobile)
        emoji = '🚀' if 'long' in signal_type.lower() or 'bull' in signal_type.lower() else '📉'
        short = (
            f"{emoji} {pair} {signal_type}\n"
            f"Entry: {entry:.5f}\n"
            f"Stop: {stop_loss:.5f}\n"
            f"Target: {take_profit:.5f}\n"
            f"R:R {rr_ratio:.1f}:1 | {quality}"
        )
        
        # Detailed message for email
        detailed = f"""
🤖 ML Pip-Based Trading Signal
═══════════════════════════════

📊 PAIR: {pair}
📍 DIRECTION: {signal_type}
⭐ QUALITY: {quality} ({confidence*100:.1f}% confidence)

💰 TRADE SETUP:
   Entry Price:    {entry:.5f}
   Stop Loss:      {stop_loss:.5f}
   Take Profit:    {take_profit:.5f}

📈 TRADE METRICS:
   Risk Pips:      {signal.get('risk_pips', 'N/A')}
   Reward Pips:    {signal.get('reward_pips', 'N/A')}
   Risk:Reward:    {rr_ratio:.2f}:1

🔍 ANALYSIS:
{signal.get('reasoning', 'No reasoning provided')}

⏰ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
        
        return short, detailed
    
    def format_harmonic_signal_message(self, signal: Dict, pair: str) -> tuple[str, str]:
        """
        Format Harmonic pattern signal for notification
        Returns: (short_message, detailed_message)
        """
        signal_type = signal.get('type', 'N/A').upper()
        pattern = signal.get('pattern', 'N/A').replace('_', ' ').upper()
        entry = signal.get('entry', 0)
        stop_loss = signal.get('stop_loss', 0)
        target_1 = signal.get('target_1', 0)
        target_2 = signal.get('target_2', 0)
        target_3 = signal.get('target_3', 0)
        quality = signal.get('quality', 0)
        
        # Short message for SMS
        emoji = '📐' if 'bull' in pattern.lower() else '📉'
        short = (
            f"{emoji} {pair} {signal_type}\n"
            f"{pattern}\n"
            f"Entry: {entry:.5f}\n"
            f"Stop: {stop_loss:.5f}\n"
            f"T1: {target_1:.5f}"
        )
        
        # Detailed message for email
        detailed = f"""
📐 Harmonic Pattern Trading Signal
═════════════════════════════════

📊 PAIR: {pair}
📍 DIRECTION: {signal_type}
🎯 PATTERN: {pattern}
⭐ QUALITY: {quality*100:.1f}%

💰 TRADE SETUP:
   Entry (D):      {entry:.5f}
   Stop Loss:      {stop_loss:.5f}

🎯 FIBONACCI TARGETS:
   T1 (38.2%):     {target_1:.5f}  [R:R {signal.get('risk_reward_t1', 0):.1f}:1]
   T2 (61.8%):     {target_2:.5f}  [R:R {signal.get('risk_reward_t2', 0):.1f}:1]
   T3 (100%):      {target_3:.5f}  [R:R {signal.get('risk_reward_t3', 0):.1f}:1]

📊 PATTERN POINTS:
   X: {signal.get('X', 0):.5f}
   A: {signal.get('A', 0):.5f}
   B: {signal.get('B', 0):.5f}
   C: {signal.get('C', 0):.5f}
   D: {signal.get('D', 0):.5f}

🔍 ANALYSIS:
{signal.get('reasoning', 'No reasoning provided')}

⏰ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
        
        return short, detailed
    
    def format_unified_signal_message(self, signals: Dict, pair: str) -> tuple[str, str]:
        """
        Format unified signals (ML + Harmonic) for notification
        Returns: (short_message, detailed_message)
        """
        recommendation = signals.get('recommendation', {})
        action = recommendation.get('action', 'WAIT')
        confidence = recommendation.get('confidence', 0)
        reason = recommendation.get('reason', '')
        is_confluence = recommendation.get('confluence', False)
        
        ml_signals = signals.get('ml_signals', [])
        harmonic_signals = signals.get('harmonic_signals', [])
        
        # Short message for SMS
        emoji = '⭐' if is_confluence else '📊'
        short = (
            f"{emoji} {pair} {action}\n"
            f"Confidence: {confidence*100:.0f}%\n"
            f"{'CONFLUENCE!' if is_confluence else reason[:40]}"
        )
        
        # Detailed message for email
        detailed = f"""
📊 UNIFIED TRADING SIGNAL
═══════════════════════════════

{'⭐ CONFLUENCE SIGNAL - BOTH SYSTEMS AGREE! ⭐' if is_confluence else ''}

📊 PAIR: {pair}
🎯 RECOMMENDATION: {action}
💪 CONFIDENCE: {confidence*100:.1f}%
📝 REASON: {reason}

{'═══════════════════════════════' if ml_signals or harmonic_signals else ''}
"""
        
        # Add ML signals
        if ml_signals:
            detailed += "\n🤖 ML PIP-BASED SIGNALS:\n"
            for i, sig in enumerate(ml_signals, 1):
                detailed += f"""
Signal #{i}:
   Direction: {sig.get('type', 'N/A').upper()}
   Entry: {sig.get('entry', 0):.5f}
   Stop: {sig.get('stop_loss', 0):.5f}
   Target: {sig.get('take_profit', 0):.5f}
   R:R: {sig.get('risk_reward_ratio', 0):.2f}:1
   Quality: {sig.get('quality', 'N/A').upper()}
"""
        
        # Add Harmonic signals
        if harmonic_signals:
            detailed += "\n📐 HARMONIC PATTERN SIGNALS:\n"
            for i, sig in enumerate(harmonic_signals, 1):
                pattern = sig.get('pattern', 'N/A').replace('_', ' ').upper()
                detailed += f"""
Signal #{i}:
   Pattern: {pattern}
   Direction: {sig.get('type', 'N/A').upper()}
   Entry: {sig.get('entry', 0):.5f}
   Stop: {sig.get('stop_loss', 0):.5f}
   T1: {sig.get('target_1', 0):.5f}
   T2: {sig.get('target_2', 0):.5f}
   T3: {sig.get('target_3', 0):.5f}
"""
        
        detailed += f"\n⏰ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        
        return short, detailed
    
    def send_email(self, subject: str, message: str) -> bool:
        """
        Send email notification
        """
        try:
            msg = MIMEMultipart('alternative')
            msg['From'] = self.email_user
            msg['To'] = self.notification_email
            msg['Subject'] = subject
            
            # Plain text version
            text_part = MIMEText(message, 'plain')
            msg.attach(text_part)
            
            # Send via Gmail SMTP
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_user, self.email_password)
                server.send_message(msg)
            
            logger.info(f"✅ Email sent successfully to {self.notification_email}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send email: {e}")
            return False
    
    def send_sms(self, message: str) -> bool:
        """
        Send SMS notification via T-Mobile email-to-SMS gateway
        T-Mobile has a 160 character limit for SMS
        """
        try:
            # Truncate message if too long for SMS
            if len(message) > 160:
                message = message[:157] + "..."
                logger.warning(f"Message truncated to 160 chars for SMS")
            
            msg = MIMEText(message, 'plain')
            msg['From'] = self.email_user
            msg['To'] = self.sms_gateway
            msg['Subject'] = ''  # No subject for SMS
            
            # Send via Gmail SMTP to T-Mobile gateway
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_user, self.email_password)
                server.send_message(msg)
            
            logger.info(f"✅ SMS sent successfully to {self.sms_number} (T-Mobile)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send SMS: {e}")
            return False
    
    def notify_ml_signal(self, signal: Dict, pair: str) -> bool:
        """
        Send notifications for ML signal
        """
        try:
            short_msg, detailed_msg = self.format_ml_signal_message(signal, pair)
            
            subject = f"🤖 ML Signal: {pair} {signal.get('type', 'N/A').upper()}"
            
            # Send both email and SMS
            email_sent = self.send_email(subject, detailed_msg)
            sms_sent = self.send_sms(short_msg)
            
            return email_sent and sms_sent
            
        except Exception as e:
            logger.error(f"❌ Error notifying ML signal: {e}")
            return False
    
    def notify_harmonic_signal(self, signal: Dict, pair: str) -> bool:
        """
        Send notifications for Harmonic pattern signal
        """
        try:
            short_msg, detailed_msg = self.format_harmonic_signal_message(signal, pair)
            
            pattern = signal.get('pattern', 'Pattern').replace('_', ' ').title()
            subject = f"📐 Harmonic: {pair} {pattern}"
            
            # Send both email and SMS
            email_sent = self.send_email(subject, detailed_msg)
            sms_sent = self.send_sms(short_msg)
            
            return email_sent and sms_sent
            
        except Exception as e:
            logger.error(f"❌ Error notifying Harmonic signal: {e}")
            return False
    
    def notify_unified_signals(self, signals: Dict, pair: str) -> bool:
        """
        Send notifications for unified signals
        """
        try:
            short_msg, detailed_msg = self.format_unified_signal_message(signals, pair)
            
            recommendation = signals.get('recommendation', {})
            action = recommendation.get('action', 'SIGNAL')
            is_confluence = recommendation.get('confluence', False)
            
            if is_confluence:
                subject = f"⭐ CONFLUENCE: {pair} {action}"
            else:
                subject = f"📊 Unified Signal: {pair} {action}"
            
            # Send both email and SMS
            email_sent = self.send_email(subject, detailed_msg)
            sms_sent = self.send_sms(short_msg)
            
            return email_sent and sms_sent
            
        except Exception as e:
            logger.error(f"❌ Error notifying unified signals: {e}")
            return False
    
    def test_notifications(self) -> bool:
        """
        Test notification system by sending test messages
        """
        logger.info("Testing notification system...")
        
        test_email_subject = "🧪 Test Notification - Trading Signal System"
        test_email_body = """
This is a test notification from your Trading Signal System.

If you receive this message, email notifications are working correctly.

System Configuration:
- Email notifications: ✅ Working
- Recipient: mydecretor@protonmail.com

Generated: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        
        test_sms_body = "🧪 Test SMS\nTrading Signal System\nNotifications Active"
        
        email_result = self.send_email(test_email_subject, test_email_body)
        sms_result = self.send_sms(test_sms_body)
        
        if email_result and sms_result:
            logger.info("✅ All notifications sent successfully")
            return True
        else:
            logger.error("❌ Some notifications failed")
            return False


def main():
    """
    Test the notification service
    """
    print("=" * 60)
    print("Trading Signal Notification Service Test")
    print("=" * 60)
    
    service = NotificationService()
    
    print("\nTesting notification system...")
    result = service.test_notifications()
    
    if result:
        print("\n✅ SUCCESS: Notifications are working!")
        print(f"   - Email sent to: {service.notification_email}")
        print(f"   - SMS sent to: {service.sms_number} (T-Mobile)")
    else:
        print("\n❌ FAILED: Some notifications did not send")
        print("   Check logs for details")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
