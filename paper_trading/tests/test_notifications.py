"""
Tests for Notification System

Tests NotificationPreferences, NotificationLog models
and notification delivery services (Email, SMS)
"""
import pytest
from datetime import time
from django.contrib.auth.models import User
from django.utils import timezone
from unittest.mock import patch, MagicMock, Mock

from paper_trading.models import NotificationPreferences, NotificationLog
from paper_trading.notification_service import (
    EmailNotificationService,
    SMSNotificationService,
    NotificationManager
)


@pytest.mark.django_db
class TestNotificationPreferences:
    """Test NotificationPreferences model"""
    
    def test_create_notification_preferences(self):
        """Test creating notification preferences"""
        user = User.objects.create_user('testuser', 'test@example.com', 'password')
        
        prefs = NotificationPreferences.objects.create(
            user=user,
            email_addresses=['test@example.com'],
            phone_numbers=['+1234567890'],
            signal_filter='all',
            pair_filter=['EURUSD', 'GBPUSD'],
            notify_new_signal=True,
            notify_trade_opened=True,
            enable_email=True,
            enable_sms=False,
            min_confidence=75
        )
        
        assert prefs.user == user
        assert 'test@example.com' in prefs.email_addresses
        assert '+1234567890' in prefs.phone_numbers
        assert prefs.signal_filter == 'all'
        assert 'EURUSD' in prefs.pair_filter
        assert prefs.min_confidence == 75
    
    def test_should_notify_signal_confidence_threshold(self):
        """Test signal confidence filtering"""
        user = User.objects.create_user('testuser')
        prefs = NotificationPreferences.objects.create(
            user=user,
            active=True,
            min_confidence=80
        )
        
        # Signal above threshold
        signal_high = {'confidence': 85, 'direction': 'buy', 'pair': 'EURUSD'}
        assert prefs.should_notify_signal(signal_high) == True
        
        # Signal below threshold
        signal_low = {'confidence': 75, 'direction': 'buy', 'pair': 'EURUSD'}
        assert prefs.should_notify_signal(signal_low) == False
    
    def test_should_notify_signal_direction_filter(self):
        """Test signal direction filtering"""
        user = User.objects.create_user('testuser')
        
        # Bullish filter
        prefs_bull = NotificationPreferences.objects.create(
            user=user,
            active=True,
            signal_filter='bullish',
            min_confidence=0
        )
        
        assert prefs_bull.should_notify_signal({'confidence': 80, 'direction': 'buy', 'pair': 'EURUSD'}) == True
        assert prefs_bull.should_notify_signal({'confidence': 80, 'direction': 'sell', 'pair': 'EURUSD'}) == False
        
        # Bearish filter
        prefs_bear = NotificationPreferences.objects.create(
            user=User.objects.create_user('testuser2'),
            active=True,
            signal_filter='bearish',
            min_confidence=0
        )
        
        assert prefs_bear.should_notify_signal({'confidence': 80, 'direction': 'sell', 'pair': 'EURUSD'}) == True
        assert prefs_bear.should_notify_signal({'confidence': 80, 'direction': 'buy', 'pair': 'EURUSD'}) == False
    
    def test_should_notify_signal_pair_filter(self):
        """Test pair filtering"""
        user = User.objects.create_user('testuser')
        prefs = NotificationPreferences.objects.create(
            user=user,
            active=True,
            pair_filter=['EURUSD', 'GBPUSD'],
            min_confidence=0
        )
        
        # Pair in filter
        assert prefs.should_notify_signal({'confidence': 80, 'direction': 'buy', 'pair': 'EURUSD'}) == True
        
        # Pair not in filter
        assert prefs.should_notify_signal({'confidence': 80, 'direction': 'buy', 'pair': 'USDJPY'}) == False
    
    def test_should_notify_signal_inactive(self):
        """Test that inactive preferences don't notify"""
        user = User.objects.create_user('testuser')
        prefs = NotificationPreferences.objects.create(
            user=user,
            active=False,
            min_confidence=0
        )
        
        assert prefs.should_notify_signal({'confidence': 95, 'direction': 'buy', 'pair': 'EURUSD'}) == False
    
    def test_is_quiet_hours(self):
        """Test quiet hours detection"""
        user = User.objects.create_user('testuser')
        
        # Quiet hours from 10pm to 6am
        prefs = NotificationPreferences.objects.create(
            user=user,
            quiet_hours_start=time(22, 0),
            quiet_hours_end=time(6, 0)
        )
        
        # Mock current time to test different scenarios
        with patch('paper_trading.models.timezone') as mock_tz:
            # During quiet hours (11pm)
            mock_tz.now.return_value.time.return_value = time(23, 0)
            assert prefs.is_quiet_hours() == True
            
            # During quiet hours (3am)
            mock_tz.now.return_value.time.return_value = time(3, 0)
            assert prefs.is_quiet_hours() == True
            
            # Outside quiet hours (2pm)
            mock_tz.now.return_value.time.return_value = time(14, 0)
            assert prefs.is_quiet_hours() == False
    
    def test_no_quiet_hours_set(self):
        """Test when quiet hours are not configured"""
        user = User.objects.create_user('testuser')
        prefs = NotificationPreferences.objects.create(
            user=user,
            quiet_hours_start=None,
            quiet_hours_end=None
        )
        
        assert prefs.is_quiet_hours() == False


@pytest.mark.django_db
class TestNotificationLog:
    """Test NotificationLog model"""
    
    def test_create_notification_log(self):
        """Test creating notification log entry"""
        user = User.objects.create_user('testuser')
        
        log = NotificationLog.objects.create(
            user=user,
            notification_type='signal',
            method='email',
            recipient='test@example.com',
            subject='New Signal: EURUSD',
            message='Buy signal detected',
            metadata={'pair': 'EURUSD', 'confidence': 85},
            status='pending'
        )
        
        assert log.user == user
        assert log.notification_type == 'signal'
        assert log.method == 'email'
        assert log.status == 'pending'
        assert log.metadata['confidence'] == 85
    
    def test_mark_sent(self):
        """Test marking notification as sent"""
        user = User.objects.create_user('testuser')
        log = NotificationLog.objects.create(
            user=user,
            notification_type='signal',
            method='email',
            recipient='test@example.com',
            subject='Test',
            message='Test message'
        )
        
        assert log.status == 'pending'
        assert log.sent_at is None
        
        log.mark_sent()
        
        assert log.status == 'sent'
        assert log.sent_at is not None
    
    def test_mark_failed(self):
        """Test marking notification as failed"""
        user = User.objects.create_user('testuser')
        log = NotificationLog.objects.create(
            user=user,
            notification_type='signal',
            method='sms',
            recipient='+1234567890',
            subject='Test',
            message='Test message'
        )
        
        error_msg = 'SMTP connection failed'
        log.mark_failed(error_msg)
        
        assert log.status == 'failed'
        assert log.error_message == error_msg
    
    def test_increment_retry(self):
        """Test incrementing retry counter"""
        user = User.objects.create_user('testuser')
        log = NotificationLog.objects.create(
            user=user,
            notification_type='signal',
            method='email',
            recipient='test@example.com',
            subject='Test',
            message='Test message'
        )
        
        assert log.retry_count == 0
        
        log.increment_retry()
        assert log.retry_count == 1
        assert log.status == 'retry'
        
        log.increment_retry()
        assert log.retry_count == 2


@pytest.mark.django_db
class TestEmailNotificationService:
    """Test Email notification service"""
    
    @patch('paper_trading.notification_service.smtplib.SMTP')
    def test_send_signal_notification(self, mock_smtp):
        """Test sending signal notification email"""
        # Setup mock SMTP
        mock_server = Mock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        service = EmailNotificationService()
        
        signal = {
            'pair': 'EURUSD',
            'direction': 'buy',
            'confidence': 85,
            'entry_price': 1.1000,
            'stop_loss': 1.0950,
            'take_profit': 1.1100
        }
        
        result = service.send_signal_notification('test@example.com', signal)
        
        assert result == True
        mock_server.send_message.assert_called_once()
    
    @patch('paper_trading.notification_service.smtplib.SMTP')
    def test_send_trade_notification(self, mock_smtp):
        """Test sending trade opened notification"""
        mock_server = Mock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        service = EmailNotificationService()
        
        trade = {
            'pair': 'GBPUSD',
            'order_type': 'buy',
            'entry_price': 1.3000,
            'lot_size': 0.01,
            'stop_loss': 1.2950,
            'take_profit': 1.3100
        }
        
        result = service.send_trade_notification('test@example.com', trade, 'opened')
        
        assert result == True
        mock_server.send_message.assert_called_once()
    
    @patch('paper_trading.notification_service.smtplib.SMTP')
    def test_send_system_notification(self, mock_smtp):
        """Test sending system status notification"""
        mock_server = Mock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        service = EmailNotificationService()
        
        result = service.send_system_notification(
            'test@example.com',
            'active',
            'Trading system is now active'
        )
        
        assert result == True
        mock_server.send_message.assert_called_once()
    
    @patch('paper_trading.notification_service.smtplib.SMTP')
    def test_email_failure(self, mock_smtp):
        """Test handling email send failure"""
        # Setup mock to raise exception
        mock_smtp.return_value.__enter__.side_effect = Exception('Connection failed')
        
        service = EmailNotificationService()
        
        signal = {
            'pair': 'EURUSD',
            'direction': 'buy',
            'confidence': 85
        }
        
        result = service.send_signal_notification('test@example.com', signal)
        
        assert result == False


@pytest.mark.django_db
class TestSMSNotificationService:
    """Test SMS notification service"""
    
    def test_sms_service_initialization(self):
        """Test SMS service can be initialized"""
        service = SMSNotificationService()
        assert service is not None
    
    @patch('paper_trading.notification_service.SMSNotificationService._send_sms')
    def test_send_signal_notification_sms(self, mock_send):
        """Test sending signal notification via SMS"""
        mock_send.return_value = True
        
        service = SMSNotificationService()
        
        signal = {
            'pair': 'EURUSD',
            'direction': 'buy',
            'confidence': 85,
            'entry_price': 1.1000
        }
        
        result = service.send_signal_notification('+1234567890', signal)
        
        assert result == True
        mock_send.assert_called_once()
    
    @patch('paper_trading.notification_service.SMSNotificationService._send_sms')
    def test_send_trade_notification_sms(self, mock_send):
        """Test sending trade notification via SMS"""
        mock_send.return_value = True
        
        service = SMSNotificationService()
        
        trade = {
            'pair': 'GBPUSD',
            'order_type': 'buy',
            'entry_price': 1.3000,
            'profit_loss': 50.0
        }
        
        result = service.send_trade_notification('+1234567890', trade, 'closed')
        
        assert result == True
        mock_send.assert_called_once()


@pytest.mark.django_db
class TestNotificationManager:
    """Test NotificationManager orchestration"""
    
    @patch('paper_trading.notification_service.EmailNotificationService.send_signal_notification')
    def test_notify_signal_email_enabled(self, mock_email):
        """Test notifying signal with email enabled"""
        mock_email.return_value = True
        
        user = User.objects.create_user('testuser', 'test@example.com')
        NotificationPreferences.objects.create(
            user=user,
            email_addresses=['test@example.com'],
            active=True,
            enable_email=True,
            enable_sms=False,
            notify_new_signal=True,
            min_confidence=75
        )
        
        manager = NotificationManager()
        
        signal = {
            'pair': 'EURUSD',
            'direction': 'buy',
            'confidence': 85,
            'entry_price': 1.1000
        }
        
        manager.notify_signal(user, signal)
        
        mock_email.assert_called_once()
        
        # Check log was created
        logs = NotificationLog.objects.filter(user=user, notification_type='signal')
        assert logs.count() == 1
        assert logs.first().status == 'sent'
    
    @patch('paper_trading.notification_service.EmailNotificationService.send_signal_notification')
    @patch('paper_trading.notification_service.SMSNotificationService.send_signal_notification')
    def test_notify_signal_both_channels(self, mock_sms, mock_email):
        """Test notifying signal with both email and SMS"""
        mock_email.return_value = True
        mock_sms.return_value = True
        
        user = User.objects.create_user('testuser')
        NotificationPreferences.objects.create(
            user=user,
            email_addresses=['test@example.com'],
            phone_numbers=['+1234567890'],
            active=True,
            enable_email=True,
            enable_sms=True,
            notify_new_signal=True,
            min_confidence=0
        )
        
        manager = NotificationManager()
        
        signal = {
            'pair': 'EURUSD',
            'direction': 'buy',
            'confidence': 85
        }
        
        manager.notify_signal(user, signal)
        
        mock_email.assert_called_once()
        mock_sms.assert_called_once()
        
        # Check logs were created
        logs = NotificationLog.objects.filter(user=user, notification_type='signal')
        assert logs.count() == 2  # One for email, one for SMS
    
    @patch('paper_trading.notification_service.EmailNotificationService.send_signal_notification')
    def test_notify_signal_below_confidence(self, mock_email):
        """Test that signal below confidence threshold is not sent"""
        user = User.objects.create_user('testuser')
        NotificationPreferences.objects.create(
            user=user,
            email_addresses=['test@example.com'],
            active=True,
            enable_email=True,
            notify_new_signal=True,
            min_confidence=90
        )
        
        manager = NotificationManager()
        
        signal = {
            'pair': 'EURUSD',
            'direction': 'buy',
            'confidence': 75  # Below threshold
        }
        
        manager.notify_signal(user, signal)
        
        # Should not send
        mock_email.assert_not_called()
        
        # No log should be created
        logs = NotificationLog.objects.filter(user=user)
        assert logs.count() == 0
    
    @patch('paper_trading.notification_service.EmailNotificationService.send_trade_notification')
    def test_notify_trade_opened(self, mock_email):
        """Test notifying when trade is opened"""
        mock_email.return_value = True
        
        user = User.objects.create_user('testuser')
        NotificationPreferences.objects.create(
            user=user,
            email_addresses=['test@example.com'],
            active=True,
            enable_email=True,
            notify_trade_opened=True
        )
        
        manager = NotificationManager()
        
        trade = {
            'pair': 'GBPUSD',
            'order_type': 'buy',
            'entry_price': 1.3000,
            'lot_size': 0.01
        }
        
        manager.notify_trade_opened(user, trade)
        
        mock_email.assert_called_once()
        
        logs = NotificationLog.objects.filter(user=user, notification_type='trade_opened')
        assert logs.count() == 1
    
    @patch('paper_trading.notification_service.EmailNotificationService.send_trade_notification')
    def test_notify_trade_closed(self, mock_email):
        """Test notifying when trade is closed"""
        mock_email.return_value = True
        
        user = User.objects.create_user('testuser')
        NotificationPreferences.objects.create(
            user=user,
            email_addresses=['test@example.com'],
            active=True,
            enable_email=True,
            notify_trade_closed=True
        )
        
        manager = NotificationManager()
        
        trade = {
            'pair': 'EURUSD',
            'order_type': 'sell',
            'entry_price': 1.1000,
            'exit_price': 1.0950,
            'profit_loss': 50.0,
            'pips_gained': 50
        }
        
        manager.notify_trade_closed(user, trade)
        
        mock_email.assert_called_once()
        
        logs = NotificationLog.objects.filter(user=user, notification_type='trade_closed')
        assert logs.count() == 1
    
    @patch('paper_trading.notification_service.EmailNotificationService.send_system_notification')
    def test_notify_system_status(self, mock_email):
        """Test notifying system status"""
        mock_email.return_value = True
        
        user = User.objects.create_user('testuser')
        NotificationPreferences.objects.create(
            user=user,
            email_addresses=['test@example.com'],
            active=True,
            enable_email=True,
            notify_system_status=True
        )
        
        manager = NotificationManager()
        
        manager.notify_system_status(user, 'active', 'Trading system is now active')
        
        mock_email.assert_called_once()
        
        logs = NotificationLog.objects.filter(user=user, notification_type='system_status')
        assert logs.count() == 1
    
    def test_notify_signal_no_preferences(self):
        """Test notifying signal when user has no preferences"""
        user = User.objects.create_user('testuser')
        
        manager = NotificationManager()
        
        signal = {
            'pair': 'EURUSD',
            'direction': 'buy',
            'confidence': 85
        }
        
        # Should not raise exception
        manager.notify_signal(user, signal)
        
        # No logs should be created
        logs = NotificationLog.objects.filter(user=user)
        assert logs.count() == 0
    
    @patch('paper_trading.notification_service.EmailNotificationService.send_signal_notification')
    def test_notification_failure_logged(self, mock_email):
        """Test that notification failures are logged"""
        mock_email.return_value = False  # Simulate failure
        
        user = User.objects.create_user('testuser')
        NotificationPreferences.objects.create(
            user=user,
            email_addresses=['test@example.com'],
            active=True,
            enable_email=True,
            notify_new_signal=True,
            min_confidence=0
        )
        
        manager = NotificationManager()
        
        signal = {
            'pair': 'EURUSD',
            'direction': 'buy',
            'confidence': 85
        }
        
        manager.notify_signal(user, signal)
        
        # Log should be created with 'failed' status
        logs = NotificationLog.objects.filter(user=user, notification_type='signal')
        assert logs.count() == 1
        assert logs.first().status == 'failed'


@pytest.mark.django_db
class TestNotificationIntegration:
    """Integration tests for notification system"""
    
    def test_create_user_with_preferences(self):
        """Test creating user and notification preferences together"""
        user = User.objects.create_user(
            username='trader1',
            email='trader1@example.com',
            password='secure_password'
        )
        
        prefs = NotificationPreferences.objects.create(
            user=user,
            email_addresses=['trader1@example.com', 'backup@example.com'],
            phone_numbers=['+1234567890'],
            signal_filter='all',
            pair_filter=['EURUSD', 'GBPUSD', 'USDJPY'],
            notify_new_signal=True,
            notify_trade_opened=True,
            notify_trade_closed=True,
            notify_high_confidence=True,
            enable_email=True,
            enable_sms=True,
            min_confidence=80,
            quiet_hours_start=time(22, 0),
            quiet_hours_end=time(6, 0)
        )
        
        assert prefs.user == user
        assert len(prefs.email_addresses) == 2
        assert len(prefs.phone_numbers) == 1
        assert len(prefs.pair_filter) == 3
        
        # Test one-to-one relationship
        retrieved_prefs = NotificationPreferences.objects.get(user=user)
        assert retrieved_prefs == prefs
    
    @patch('paper_trading.notification_service.EmailNotificationService.send_signal_notification')
    def test_high_confidence_signal_workflow(self, mock_email):
        """Test complete workflow for high confidence signal"""
        mock_email.return_value = True
        
        # Create user with preferences for high confidence signals
        user = User.objects.create_user('trader', 'trader@example.com')
        NotificationPreferences.objects.create(
            user=user,
            email_addresses=['trader@example.com'],
            active=True,
            enable_email=True,
            notify_new_signal=True,
            notify_high_confidence=True,
            signal_filter='all',
            min_confidence=80
        )
        
        manager = NotificationManager()
        
        # High confidence signal
        signal = {
            'pair': 'EURUSD',
            'direction': 'buy',
            'confidence': 92,
            'entry_price': 1.1000,
            'stop_loss': 1.0950,
            'take_profit': 1.1100,
            'risk_reward_ratio': 2.0
        }
        
        manager.notify_signal(user, signal)
        
        # Verify email was sent
        mock_email.assert_called_once()
        
        # Verify log entry
        log = NotificationLog.objects.get(user=user, notification_type='signal')
        assert log.status == 'sent'
        assert log.method == 'email'
        assert log.metadata['confidence'] == 92
        assert log.metadata['pair'] == 'EURUSD'
    
    def test_multiple_users_independent_preferences(self):
        """Test multiple users with different preferences"""
        user1 = User.objects.create_user('trader1')
        user2 = User.objects.create_user('trader2')
        
        prefs1 = NotificationPreferences.objects.create(
            user=user1,
            signal_filter='bullish',
            min_confidence=75
        )
        
        prefs2 = NotificationPreferences.objects.create(
            user=user2,
            signal_filter='bearish',
            min_confidence=90
        )
        
        # Different filters
        buy_signal = {'confidence': 85, 'direction': 'buy', 'pair': 'EURUSD'}
        assert prefs1.should_notify_signal(buy_signal) == True
        assert prefs2.should_notify_signal(buy_signal) == False
        
        # Different confidence thresholds
        sell_signal = {'confidence': 80, 'direction': 'sell', 'pair': 'EURUSD'}
        assert prefs1.should_notify_signal(sell_signal) == False  # Wrong direction
        assert prefs2.should_notify_signal(sell_signal) == False  # Below confidence
