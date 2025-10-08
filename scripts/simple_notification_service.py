#!/usr/bin/env python3
"""
Alternative Notification Methods
For testing and backup when Gmail is not available
"""

import os
import sys
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleNotificationService:
    """
    Simple notification service that logs to file and console
    Use this for testing when email/SMS setup is not ready
    """
    
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.notification_log = os.path.join(log_dir, 'notifications.log')
        logger.info("SimpleNotificationService initialized")
        logger.info(f"Notifications will be logged to: {self.notification_log}")
    
    def send_notification(self, subject: str, message: str) -> bool:
        """
        Log notification to file and console
        """
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Format notification
            notification = f"""
{'='*80}
NOTIFICATION
Time: {timestamp}
Subject: {subject}
{'='*80}
{message}
{'='*80}
"""
            
            # Print to console
            print(notification)
            
            # Append to log file
            with open(self.notification_log, 'a') as f:
                f.write(notification + '\n')
            
            logger.info(f"✅ Notification logged: {subject}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to log notification: {e}")
            return False
    
    def notify_ml_signal(self, signal: dict, pair: str) -> bool:
        """Send ML signal notification"""
        subject = f"🤖 ML Signal: {pair} {signal.get('type', 'N/A').upper()}"
        
        message = f"""
📊 PAIR: {pair}
📍 DIRECTION: {signal.get('type', 'N/A').upper()}
💰 ENTRY: {signal.get('entry', 0):.5f}
🛑 STOP LOSS: {signal.get('stop_loss', 0):.5f}
🎯 TAKE PROFIT: {signal.get('take_profit', 0):.5f}
📈 RISK:REWARD: {signal.get('risk_reward_ratio', 0):.2f}:1
⭐ CONFIDENCE: {signal.get('confidence', 0)*100:.1f}%
📊 QUALITY: {signal.get('quality', 'N/A').upper()}
"""
        
        return self.send_notification(subject, message)
    
    def notify_harmonic_signal(self, signal: dict, pair: str) -> bool:
        """Send Harmonic signal notification"""
        pattern = signal.get('pattern', 'N/A').replace('_', ' ').upper()
        subject = f"📐 Harmonic: {pair} {pattern}"
        
        message = f"""
📊 PAIR: {pair}
📍 DIRECTION: {signal.get('type', 'N/A').upper()}
🎯 PATTERN: {pattern}
💰 ENTRY: {signal.get('entry', 0):.5f}
🛑 STOP LOSS: {signal.get('stop_loss', 0):.5f}
🎯 TARGETS:
   T1: {signal.get('target_1', 0):.5f} [R:R {signal.get('risk_reward_t1', 0):.1f}:1]
   T2: {signal.get('target_2', 0):.5f} [R:R {signal.get('risk_reward_t2', 0):.1f}:1]
   T3: {signal.get('target_3', 0):.5f} [R:R {signal.get('risk_reward_t3', 0):.1f}:1]
⭐ QUALITY: {signal.get('quality', 0)*100:.1f}%
"""
        
        return self.send_notification(subject, message)
    
    def notify_unified_signals(self, signals: dict, pair: str) -> bool:
        """Send unified signals notification"""
        recommendation = signals.get('recommendation', {})
        action = recommendation.get('action', 'WAIT')
        is_confluence = recommendation.get('confluence', False)
        
        subject = f"{'⭐ CONFLUENCE' if is_confluence else '📊'}: {pair} {action}"
        
        message = f"""
{'⭐'*40 if is_confluence else ''}
{'CONFLUENCE SIGNAL - BOTH SYSTEMS AGREE!' if is_confluence else 'UNIFIED SIGNAL'}
{'⭐'*40 if is_confluence else ''}

📊 PAIR: {pair}
🎯 RECOMMENDATION: {action}
💪 CONFIDENCE: {recommendation.get('confidence', 0)*100:.1f}%
📝 REASON: {recommendation.get('reason', 'N/A')}

ML SIGNALS: {len(signals.get('ml_signals', []))}
HARMONIC SIGNALS: {len(signals.get('harmonic_signals', []))}
"""
        
        return self.send_notification(subject, message)
    
    def test_notifications(self) -> bool:
        """Test notification system"""
        logger.info("Testing notification system...")
        
        test_subject = "🧪 Test Notification - Trading Signal System"
        test_message = """
This is a test notification from your Trading Signal System.

✅ Notifications are working!

System Status:
- Notification logging: Active
- Log file: logs/notifications.log
- Console output: Enabled

Generated: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        
        result = self.send_notification(test_subject, test_message)
        
        if result:
            logger.info("✅ Test notification successful")
            print("\n" + "="*80)
            print("✅ SUCCESS: Notifications are working!")
            print(f"   Check log file: {self.notification_log}")
            print("="*80)
        else:
            logger.error("❌ Test notification failed")
        
        return result


def main():
    """Test the simple notification service"""
    print("=" * 80)
    print("Simple Notification Service Test")
    print("=" * 80)
    print("\nThis service logs notifications to file and console.")
    print("Use this when Gmail App Password is not set up yet.")
    print("\nFor email/SMS setup, see: GMAIL_APP_PASSWORD_SETUP.md")
    print("=" * 80)
    
    service = SimpleNotificationService()
    
    print("\nTesting notification system...")
    result = service.test_notifications()
    
    if result:
        print("\n✅ Test notification created successfully!")
        print(f"\nView notifications: cat {service.notification_log}")
        print("\nTo use this in your trading system:")
        print("  from scripts.simple_notification_service import SimpleNotificationService")
        print("  service = SimpleNotificationService()")
        print("  service.notify_unified_signals(signals, pair='EURUSD')")
    else:
        print("\n❌ Test notification failed")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
