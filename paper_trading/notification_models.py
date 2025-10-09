"""
Notification System Models
Handles user notification preferences and delivery tracking
"""
from django.db import models
from django.contrib.auth.models import User
from django.core.validators import EmailValidator, RegexValidator
import json


class NotificationPreferences(models.Model):
    """User notification preferences"""
    
    SIGNAL_FILTER_CHOICES = [
        ('all', 'All Signals'),
        ('bullish', 'Bullish Only'),
        ('bearish', 'Bearish Only'),
    ]
    
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='notification_prefs')
    
    # Contact Methods
    email_addresses = models.JSONField(
        default=list,
        help_text="List of email addresses to notify"
    )
    phone_numbers = models.JSONField(
        default=list,
        help_text="List of phone numbers for SMS (E.164 format: +1234567890)"
    )
    
    # Notification Filters
    signal_filter = models.CharField(
        max_length=10,
        choices=SIGNAL_FILTER_CHOICES,
        default='all',
        help_text="Filter signals by direction"
    )
    pair_filter = models.JSONField(
        default=list,
        help_text="List of pairs to monitor (empty = all pairs)"
    )
    
    # Notification Triggers (True = enabled)
    notify_new_signal = models.BooleanField(default=True)
    notify_trade_opened = models.BooleanField(default=True)
    notify_trade_closed = models.BooleanField(default=True)
    notify_tp_hit = models.BooleanField(default=True)
    notify_sl_hit = models.BooleanField(default=True)
    notify_system_status = models.BooleanField(default=True)
    notify_candle_prediction = models.BooleanField(default=False)
    notify_high_confidence = models.BooleanField(
        default=True,
        help_text="Notify for signals with 80%+ confidence"
    )
    
    # Notification Methods
    enable_email = models.BooleanField(default=True)
    enable_sms = models.BooleanField(default=False)
    enable_push = models.BooleanField(default=False)
    
    # Settings
    active = models.BooleanField(default=True)
    min_confidence = models.IntegerField(
        default=75,
        help_text="Minimum signal confidence to notify (0-100)"
    )
    quiet_hours_start = models.TimeField(
        null=True,
        blank=True,
        help_text="Start of quiet hours (no notifications)"
    )
    quiet_hours_end = models.TimeField(
        null=True,
        blank=True,
        help_text="End of quiet hours"
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'notification_preferences'
        verbose_name = 'Notification Preference'
        verbose_name_plural = 'Notification Preferences'
    
    def __str__(self):
        return f"Notification Preferences for {self.user.username}"
    
    def add_email(self, email: str):
        """Add an email address to the notification list"""
        if email not in self.email_addresses:
            self.email_addresses.append(email)
            self.save()
    
    def remove_email(self, email: str):
        """Remove an email address from the notification list"""
        if email in self.email_addresses:
            self.email_addresses.remove(email)
            self.save()
    
    def add_phone(self, phone: str):
        """Add a phone number to the notification list"""
        if phone not in self.phone_numbers:
            self.phone_numbers.append(phone)
            self.save()
    
    def remove_phone(self, phone: str):
        """Remove a phone number from the notification list"""
        if phone in self.phone_numbers:
            self.phone_numbers.remove(phone)
            self.save()
    
    def add_pair(self, pair: str):
        """Add a pair to the filter list"""
        if pair not in self.pair_filter:
            self.pair_filter.append(pair)
            self.save()
    
    def remove_pair(self, pair: str):
        """Remove a pair from the filter list"""
        if pair in self.pair_filter:
            self.pair_filter.remove(pair)
            self.save()
    
    def should_notify_signal(self, signal: dict) -> bool:
        """Check if a signal matches notification preferences"""
        if not self.active:
            return False
        
        # Check confidence threshold
        confidence = signal.get('confidence', 0)
        if confidence < self.min_confidence:
            return False
        
        # Check signal direction filter
        direction = signal.get('direction', '').lower()
        if self.signal_filter == 'bullish' and direction not in ['buy', 'long']:
            return False
        if self.signal_filter == 'bearish' and direction not in ['sell', 'short']:
            return False
        
        # Check pair filter
        pair = signal.get('pair', signal.get('symbol', ''))
        if self.pair_filter and pair not in self.pair_filter:
            return False
        
        return True
    
    def is_quiet_hours(self) -> bool:
        """Check if current time is within quiet hours"""
        if not self.quiet_hours_start or not self.quiet_hours_end:
            return False
        
        from django.utils import timezone
        now = timezone.now().time()
        
        if self.quiet_hours_start < self.quiet_hours_end:
            # Normal case: quiet hours don't span midnight
            return self.quiet_hours_start <= now <= self.quiet_hours_end
        else:
            # Quiet hours span midnight
            return now >= self.quiet_hours_start or now <= self.quiet_hours_end


class NotificationLog(models.Model):
    """Track notification delivery"""
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('sent', 'Sent'),
        ('failed', 'Failed'),
        ('retry', 'Retrying'),
    ]
    
    TYPE_CHOICES = [
        ('signal', 'New Signal'),
        ('trade_opened', 'Trade Opened'),
        ('trade_closed', 'Trade Closed'),
        ('tp_hit', 'Take Profit Hit'),
        ('sl_hit', 'Stop Loss Hit'),
        ('system_status', 'System Status'),
        ('candle_prediction', 'Candle Prediction'),
    ]
    
    METHOD_CHOICES = [
        ('email', 'Email'),
        ('sms', 'SMS'),
        ('push', 'Push Notification'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='notifications')
    notification_type = models.CharField(max_length=20, choices=TYPE_CHOICES)
    method = models.CharField(max_length=10, choices=METHOD_CHOICES)
    recipient = models.CharField(max_length=255, help_text="Email or phone number")
    
    # Content
    subject = models.CharField(max_length=255)
    message = models.TextField()
    metadata = models.JSONField(default=dict, help_text="Signal/trade data")
    
    # Delivery Status
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='pending')
    sent_at = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(blank=True)
    retry_count = models.IntegerField(default=0)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'notification_log'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', 'created_at']),
            models.Index(fields=['status']),
        ]
    
    def __str__(self):
        return f"{self.notification_type} to {self.recipient} - {self.status}"
    
    def mark_sent(self):
        """Mark notification as successfully sent"""
        from django.utils import timezone
        self.status = 'sent'
        self.sent_at = timezone.now()
        self.save()
    
    def mark_failed(self, error: str):
        """Mark notification as failed"""
        self.status = 'failed'
        self.error_message = error
        self.save()
    
    def increment_retry(self):
        """Increment retry counter"""
        self.retry_count += 1
        self.status = 'retry'
        self.save()
