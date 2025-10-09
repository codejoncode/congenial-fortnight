"""
Enhanced Notification Service
Handles email, SMS, and push notifications with user preferences
"""
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional
from decimal import Decimal
from django.conf import settings
from django.utils import timezone

logger = logging.getLogger(__name__)


class EmailNotificationService:
    """Email notification service using Gmail SMTP"""
    
    def __init__(self):
        self.smtp_host = getattr(settings, 'EMAIL_HOST', 'smtp.gmail.com')
        self.smtp_port = getattr(settings, 'EMAIL_PORT', 587)
        self.smtp_user = getattr(settings, 'EMAIL_HOST_USER', '')
        self.smtp_password = getattr(settings, 'EMAIL_HOST_PASSWORD', '')
        self.from_email = getattr(settings, 'DEFAULT_FROM_EMAIL', self.smtp_user)
    
    def send_signal_notification(
        self,
        to_emails: List[str],
        signal: Dict
    ) -> bool:
        """Send new signal notification"""
        try:
            subject = f"🔔 New {signal.get('direction', '').upper()} Signal: {signal.get('pair', 'N/A')}"
            
            html_body = self._generate_signal_email(signal)
            
            return self._send_email(to_emails, subject, html_body)
        
        except Exception as e:
            logger.error(f"Failed to send signal notification: {e}")
            return False
    
    def send_trade_notification(
        self,
        to_emails: List[str],
        trade: Dict,
        action: str  # 'opened' or 'closed'
    ) -> bool:
        """Send trade opened/closed notification"""
        try:
            if action == 'opened':
                subject = f"📈 Trade Opened: {trade.get('pair', 'N/A')} {trade.get('order_type', '').upper()}"
                html_body = self._generate_trade_opened_email(trade)
            else:
                subject = f"💰 Trade Closed: {trade.get('pair', 'N/A')} - {trade.get('result', 'N/A')}"
                html_body = self._generate_trade_closed_email(trade)
            
            return self._send_email(to_emails, subject, html_body)
        
        except Exception as e:
            logger.error(f"Failed to send trade notification: {e}")
            return False
    
    def send_system_notification(
        self,
        to_emails: List[str],
        status: str,
        message: str
    ) -> bool:
        """Send system status notification"""
        try:
            subject = f"⚙️ System Status: {status.upper()}"
            
            html_body = f"""
            <html>
            <body style="font-family: Arial, sans-serif;">
                <h2 style="color: {'#28a745' if status == 'online' else '#dc3545'};">
                    System Status: {status.upper()}
                </h2>
                <p>{message}</p>
                <p style="color: #6c757d; font-size: 12px;">
                    Time: {timezone.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
                </p>
            </body>
            </html>
            """
            
            return self._send_email(to_emails, subject, html_body)
        
        except Exception as e:
            logger.error(f"Failed to send system notification: {e}")
            return False
    
    def _send_email(
        self,
        to_emails: List[str],
        subject: str,
        html_body: str
    ) -> bool:
        """Send email using SMTP"""
        try:
            msg = MIMEMultipart('alternative')
            msg['From'] = self.from_email
            msg['To'] = ', '.join(to_emails)
            msg['Subject'] = subject
            
            html_part = MIMEText(html_body, 'html')
            msg.attach(html_part)
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            
            logger.info(f"Email sent to {len(to_emails)} recipients: {subject}")
            return True
        
        except Exception as e:
            logger.error(f"SMTP error: {e}")
            return False
    
    def _generate_signal_email(self, signal: Dict) -> str:
        """Generate HTML email for signal notification"""
        direction = signal.get('direction', 'N/A').upper()
        pair = signal.get('pair', signal.get('symbol', 'N/A'))
        confidence = signal.get('confidence', 0)
        entry = signal.get('entry_price', 0)
        sl = signal.get('stop_loss', 0)
        tp1 = signal.get('take_profit_1', 0)
        tp2 = signal.get('take_profit_2', 0)
        tp3 = signal.get('take_profit_3', 0)
        
        # Calculate risk:reward
        if direction in ['BUY', 'LONG']:
            risk = abs(float(entry) - float(sl))
            reward = abs(float(tp1) - float(entry))
        else:
            risk = abs(float(sl) - float(entry))
            reward = abs(float(entry) - float(tp1))
        
        rr_ratio = round(reward / risk, 2) if risk > 0 else 0
        
        color = '#28a745' if direction in ['BUY', 'LONG'] else '#dc3545'
        
        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; background-color: #f8f9fa; padding: 20px;">
            <div style="max-width: 600px; margin: 0 auto; background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h2 style="color: {color}; margin-top: 0;">
                    {direction} Signal: {pair}
                </h2>
                
                <div style="background: {color}; color: white; padding: 10px; border-radius: 4px; margin: 15px 0;">
                    <strong>Confidence: {confidence}%</strong>
                </div>
                
                <table style="width: 100%; border-collapse: collapse; margin: 15px 0;">
                    <tr style="background: #f8f9fa;">
                        <td style="padding: 10px; border: 1px solid #dee2e6;"><strong>Entry Price:</strong></td>
                        <td style="padding: 10px; border: 1px solid #dee2e6;">{entry}</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border: 1px solid #dee2e6;"><strong>Stop Loss:</strong></td>
                        <td style="padding: 10px; border: 1px solid #dee2e6; color: #dc3545;">{sl}</td>
                    </tr>
                    <tr style="background: #f8f9fa;">
                        <td style="padding: 10px; border: 1px solid #dee2e6;"><strong>Take Profit 1:</strong></td>
                        <td style="padding: 10px; border: 1px solid #dee2e6; color: #28a745;">{tp1}</td>
                    </tr>
                    {f'<tr><td style="padding: 10px; border: 1px solid #dee2e6;"><strong>Take Profit 2:</strong></td><td style="padding: 10px; border: 1px solid #dee2e6; color: #28a745;">{tp2}</td></tr>' if tp2 else ''}
                    {f'<tr style="background: #f8f9fa;"><td style="padding: 10px; border: 1px solid #dee2e6;"><strong>Take Profit 3:</strong></td><td style="padding: 10px; border: 1px solid #dee2e6; color: #28a745;">{tp3}</td></tr>' if tp3 else ''}
                    <tr>
                        <td style="padding: 10px; border: 1px solid #dee2e6;"><strong>Risk:Reward:</strong></td>
                        <td style="padding: 10px; border: 1px solid #dee2e6;">1:{rr_ratio}</td>
                    </tr>
                </table>
                
                <p style="color: #6c757d; font-size: 12px; margin-top: 20px; border-top: 1px solid #dee2e6; padding-top: 10px;">
                    Generated: {timezone.now().strftime('%Y-%m-%d %H:%M:%S UTC')}<br>
                    <em>This is an automated trading signal. Trade at your own risk.</em>
                </p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _generate_trade_opened_email(self, trade: Dict) -> str:
        """Generate HTML email for trade opened notification"""
        pair = trade.get('pair', 'N/A')
        order_type = trade.get('order_type', 'N/A').upper()
        entry = trade.get('entry_price', 0)
        sl = trade.get('stop_loss', 0)
        tp = trade.get('take_profit_1', 0)
        lot_size = trade.get('lot_size', 0)
        
        color = '#28a745' if order_type == 'BUY' else '#dc3545'
        
        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; background-color: #f8f9fa; padding: 20px;">
            <div style="max-width: 600px; margin: 0 auto; background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h2 style="color: {color}; margin-top: 0;">
                    📈 Trade Opened: {order_type} {pair}
                </h2>
                
                <table style="width: 100%; border-collapse: collapse; margin: 15px 0;">
                    <tr style="background: #f8f9fa;">
                        <td style="padding: 10px; border: 1px solid #dee2e6;"><strong>Entry:</strong></td>
                        <td style="padding: 10px; border: 1px solid #dee2e6;">{entry}</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border: 1px solid #dee2e6;"><strong>Stop Loss:</strong></td>
                        <td style="padding: 10px; border: 1px solid #dee2e6; color: #dc3545;">{sl}</td>
                    </tr>
                    <tr style="background: #f8f9fa;">
                        <td style="padding: 10px; border: 1px solid #dee2e6;"><strong>Take Profit:</strong></td>
                        <td style="padding: 10px; border: 1px solid #dee2e6; color: #28a745;">{tp}</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border: 1px solid #dee2e6;"><strong>Lot Size:</strong></td>
                        <td style="padding: 10px; border: 1px solid #dee2e6;">{lot_size}</td>
                    </tr>
                </table>
                
                <p style="color: #6c757d; font-size: 12px; margin-top: 20px;">
                    Time: {timezone.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
                </p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _generate_trade_closed_email(self, trade: Dict) -> str:
        """Generate HTML email for trade closed notification"""
        pair = trade.get('pair', 'N/A')
        order_type = trade.get('order_type', 'N/A').upper()
        entry = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price', 0)
        pips = trade.get('pips_gained', 0)
        profit = trade.get('profit_loss', 0)
        exit_reason = trade.get('exit_reason', 'N/A')
        
        is_win = float(pips) > 0
        color = '#28a745' if is_win else '#dc3545'
        result = '✅ WIN' if is_win else '❌ LOSS'
        
        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; background-color: #f8f9fa; padding: 20px;">
            <div style="max-width: 600px; margin: 0 auto; background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h2 style="color: {color}; margin-top: 0;">
                    💰 Trade Closed: {result}
                </h2>
                
                <div style="background: {color}; color: white; padding: 15px; border-radius: 4px; margin: 15px 0; text-align: center;">
                    <h3 style="margin: 0;">{order_type} {pair}</h3>
                    <p style="font-size: 24px; margin: 10px 0;">{pips} pips</p>
                    <p style="margin: 0;">Profit/Loss: ${profit}</p>
                </div>
                
                <table style="width: 100%; border-collapse: collapse; margin: 15px 0;">
                    <tr style="background: #f8f9fa;">
                        <td style="padding: 10px; border: 1px solid #dee2e6;"><strong>Entry:</strong></td>
                        <td style="padding: 10px; border: 1px solid #dee2e6;">{entry}</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border: 1px solid #dee2e6;"><strong>Exit:</strong></td>
                        <td style="padding: 10px; border: 1px solid #dee2e6;">{exit_price}</td>
                    </tr>
                    <tr style="background: #f8f9fa;">
                        <td style="padding: 10px; border: 1px solid #dee2e6;"><strong>Exit Reason:</strong></td>
                        <td style="padding: 10px; border: 1px solid #dee2e6;">{exit_reason}</td>
                    </tr>
                </table>
                
                <p style="color: #6c757d; font-size: 12px; margin-top: 20px;">
                    Time: {timezone.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
                </p>
            </div>
        </body>
        </html>
        """
        
        return html


class SMSNotificationService:
    """SMS notification service using Twilio"""
    
    def __init__(self):
        self.account_sid = getattr(settings, 'TWILIO_ACCOUNT_SID', '')
        self.auth_token = getattr(settings, 'TWILIO_AUTH_TOKEN', '')
        self.from_number = getattr(settings, 'TWILIO_PHONE_NUMBER', '')
    
    def send_signal_notification(
        self,
        to_numbers: List[str],
        signal: Dict
    ) -> bool:
        """Send new signal SMS"""
        try:
            direction = signal.get('direction', 'N/A').upper()
            pair = signal.get('pair', signal.get('symbol', 'N/A'))
            confidence = signal.get('confidence', 0)
            entry = signal.get('entry_price', 0)
            
            message = f"🔔 {direction} {pair}\nConfidence: {confidence}%\nEntry: {entry}"
            
            return self._send_sms(to_numbers, message)
        
        except Exception as e:
            logger.error(f"Failed to send signal SMS: {e}")
            return False
    
    def send_trade_notification(
        self,
        to_numbers: List[str],
        trade: Dict,
        action: str
    ) -> bool:
        """Send trade SMS"""
        try:
            pair = trade.get('pair', 'N/A')
            order_type = trade.get('order_type', 'N/A').upper()
            
            if action == 'opened':
                message = f"📈 Trade Opened: {order_type} {pair}"
            else:
                pips = trade.get('pips_gained', 0)
                result = '✅ WIN' if float(pips) > 0 else '❌ LOSS'
                message = f"💰 Trade Closed: {pair} {result} ({pips} pips)"
            
            return self._send_sms(to_numbers, message)
        
        except Exception as e:
            logger.error(f"Failed to send trade SMS: {e}")
            return False
    
    def _send_sms(self, to_numbers: List[str], message: str) -> bool:
        """Send SMS using Twilio"""
        try:
            # Import here to avoid dependency if not configured
            from twilio.rest import Client
            
            client = Client(self.account_sid, self.auth_token)
            
            for number in to_numbers:
                client.messages.create(
                    to=number,
                    from_=self.from_number,
                    body=message
                )
            
            logger.info(f"SMS sent to {len(to_numbers)} recipients")
            return True
        
        except ImportError:
            logger.warning("Twilio not installed. Install with: pip install twilio")
            return False
        except Exception as e:
            logger.error(f"Twilio error: {e}")
            return False


class NotificationManager:
    """Manages notification delivery based on user preferences"""
    
    def __init__(self):
        self.email_service = EmailNotificationService()
        self.sms_service = SMSNotificationService()
    
    def notify_signal(self, user, signal: Dict):
        """Send signal notification based on user preferences"""
        from .models import NotificationPreferences, NotificationLog
        
        try:
            prefs = NotificationPreferences.objects.get(user=user)
            
            # Check if notification should be sent
            if not prefs.notify_new_signal or not prefs.should_notify_signal(signal):
                return
            
            # Check quiet hours
            if prefs.is_quiet_hours():
                logger.info(f"Skipping notification during quiet hours for {user.username}")
                return
            
            # Send email
            if prefs.enable_email and prefs.email_addresses:
                success = self.email_service.send_signal_notification(
                    prefs.email_addresses,
                    signal
                )
                self._log_notification(
                    user, 'signal', 'email',
                    prefs.email_addresses[0], signal,
                    'sent' if success else 'failed'
                )
            
            # Send SMS
            if prefs.enable_sms and prefs.phone_numbers:
                success = self.sms_service.send_signal_notification(
                    prefs.phone_numbers,
                    signal
                )
                self._log_notification(
                    user, 'signal', 'sms',
                    prefs.phone_numbers[0], signal,
                    'sent' if success else 'failed'
                )
        
        except NotificationPreferences.DoesNotExist:
            logger.warning(f"No notification preferences for user {user.username}")
        except Exception as e:
            logger.error(f"Notification error: {e}")
    
    def notify_trade_opened(self, user, trade: Dict):
        """Send trade opened notification"""
        from .models import NotificationPreferences, NotificationLog
        
        try:
            prefs = NotificationPreferences.objects.get(user=user)
            
            if not prefs.notify_trade_opened or prefs.is_quiet_hours():
                return
            
            if prefs.enable_email and prefs.email_addresses:
                self.email_service.send_trade_notification(
                    prefs.email_addresses,
                    trade,
                    'opened'
                )
            
            if prefs.enable_sms and prefs.phone_numbers:
                self.sms_service.send_trade_notification(
                    prefs.phone_numbers,
                    trade,
                    'opened'
                )
        
        except Exception as e:
            logger.error(f"Trade notification error: {e}")
    
    def notify_trade_closed(self, user, trade: Dict):
        """Send trade closed notification"""
        from .models import NotificationPreferences, NotificationLog
        
        try:
            prefs = NotificationPreferences.objects.get(user=user)
            
            if not prefs.notify_trade_closed or prefs.is_quiet_hours():
                return
            
            if prefs.enable_email and prefs.email_addresses:
                self.email_service.send_trade_notification(
                    prefs.email_addresses,
                    trade,
                    'closed'
                )
            
            if prefs.enable_sms and prefs.phone_numbers:
                self.sms_service.send_trade_notification(
                    prefs.phone_numbers,
                    trade,
                    'closed'
                )
        
        except Exception as e:
            logger.error(f"Trade notification error: {e}")
    
    def notify_system_status(self, user, status: str, message: str):
        """Send system status notification"""
        from .models import NotificationPreferences
        
        try:
            prefs = NotificationPreferences.objects.get(user=user)
            
            if not prefs.notify_system_status:
                return
            
            if prefs.enable_email and prefs.email_addresses:
                self.email_service.send_system_notification(
                    prefs.email_addresses,
                    status,
                    message
                )
        
        except Exception as e:
            logger.error(f"System notification error: {e}")
    
    def _log_notification(
        self,
        user,
        notification_type: str,
        method: str,
        recipient: str,
        metadata: dict,
        status: str
    ):
        """Log notification delivery"""
        from .models import NotificationLog
        
        try:
            NotificationLog.objects.create(
                user=user,
                notification_type=notification_type,
                method=method,
                recipient=recipient,
                subject=f"{notification_type} notification",
                message=str(metadata),
                metadata=metadata,
                status=status
            )
        except Exception as e:
            logger.error(f"Failed to log notification: {e}")
