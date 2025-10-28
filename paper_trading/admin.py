"""
Django Admin configuration for paper trading
"""
from django.contrib import admin
from .models import (
    PaperTrade, PriceCache, PerformanceMetrics, APIUsageTracker,
    NotificationPreferences, NotificationLog
)


@admin.register(PaperTrade)
class PaperTradeAdmin(admin.ModelAdmin):
    """Admin interface for paper trades"""
    
    list_display = [
        'id', 'pair', 'order_type', 'entry_price', 'status',
        'pips_gained', 'profit_loss', 'entry_time', 'signal_type'
    ]
    list_filter = ['status', 'order_type', 'pair', 'signal_type', 'entry_time']
    search_fields = ['pair', 'signal_id', 'notes']
    readonly_fields = ['pips_gained', 'profit_loss', 'created_at', 'updated_at']
    
    fieldsets = (
        ('Trade Information', {
            'fields': ('pair', 'order_type', 'entry_price', 'lot_size', 'status')
        }),
        ('Risk Management', {
            'fields': ('stop_loss', 'take_profit_1', 'take_profit_2', 'take_profit_3', 'risk_reward_ratio')
        }),
        ('Execution', {
            'fields': ('entry_time', 'exit_time', 'exit_price')
        }),
        ('Performance', {
            'fields': ('pips_gained', 'profit_loss')
        }),
        ('Signal Details', {
            'fields': ('signal_id', 'signal_type', 'signal_source', 'notes')
        }),
        ('Metadata', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    actions = ['close_selected_trades']
    
    def close_selected_trades(self, request, queryset):
        """Admin action to close selected trades"""
        from .engine import PaperTradingEngine
        from .data_aggregator import DataAggregator
        
        engine = PaperTradingEngine()
        aggregator = DataAggregator()
        
        closed_count = 0
        
        for trade in queryset.filter(status='open'):
            # Get current price
            price_data = aggregator.get_realtime_price(trade.pair)
            if price_data:
                exit_price = price_data['bid'] if trade.order_type == 'buy' else price_data['ask']
                engine.close_position(trade.id, exit_price)
                closed_count += 1
        
        self.message_user(request, f'{closed_count} trades closed successfully.')
    
    close_selected_trades.short_description = 'Close selected open trades'


@admin.register(PerformanceMetrics)
class PerformanceMetricsAdmin(admin.ModelAdmin):
    """Admin interface for performance metrics"""
    
    list_display = [
        'date', 'pair', 'total_trades', 'win_rate',
        'total_pips', 'total_pnl', 'avg_risk_reward'
    ]
    list_filter = ['date', 'pair']
    search_fields = ['pair']
    readonly_fields = ['created_at', 'updated_at']
    
    date_hierarchy = 'date'


@admin.register(PriceCache)
class PriceCacheAdmin(admin.ModelAdmin):
    """Admin interface for price cache"""
    
    list_display = [
        'symbol', 'timestamp', 'close', 'timeframe', 'source', 'created_at'
    ]
    list_filter = ['symbol', 'timeframe', 'source', 'timestamp']
    search_fields = ['symbol']
    readonly_fields = ['created_at']
    
    date_hierarchy = 'timestamp'


@admin.register(APIUsageTracker)
class APIUsageTrackerAdmin(admin.ModelAdmin):
    """Admin interface for API usage tracking"""
    
    list_display = [
        'api_name', 'date', 'requests_made', 'requests_limit',
        'usage_percentage', 'last_request_time'
    ]
    list_filter = ['api_name', 'date']
    readonly_fields = ['created_at', 'updated_at']
    
    date_hierarchy = 'date'
    
    def usage_percentage(self, obj):
        """Calculate usage percentage"""
        if obj.requests_limit == 0:
            return '0%'
        percentage = (obj.requests_made / obj.requests_limit) * 100
        return f'{percentage:.1f}%'
    
    usage_percentage.short_description = 'Usage %'


@admin.register(NotificationPreferences)
class NotificationPreferencesAdmin(admin.ModelAdmin):
    """Admin interface for notification preferences"""
    
    list_display = [
        'user', 'active', 'enable_email', 'enable_sms',
        'signal_filter', 'min_confidence', 'updated_at'
    ]
    list_filter = ['active', 'enable_email', 'enable_sms', 'signal_filter']
    search_fields = ['user__username', 'user__email']
    readonly_fields = ['created_at', 'updated_at']
    
    fieldsets = (
        ('User', {
            'fields': ('user', 'active')
        }),
        ('Contact Methods', {
            'fields': ('email_addresses', 'phone_numbers')
        }),
        ('Notification Channels', {
            'fields': ('enable_email', 'enable_sms', 'enable_push')
        }),
        ('Filters', {
            'fields': ('signal_filter', 'pair_filter', 'min_confidence')
        }),
        ('Triggers', {
            'fields': (
                'notify_new_signal', 'notify_trade_opened', 'notify_trade_closed',
                'notify_tp_hit', 'notify_sl_hit', 'notify_system_status',
                'notify_candle_prediction', 'notify_high_confidence'
            )
        }),
        ('Quiet Hours', {
            'fields': ('quiet_hours_start', 'quiet_hours_end'),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )


@admin.register(NotificationLog)
class NotificationLogAdmin(admin.ModelAdmin):
    """Admin interface for notification logs"""
    
    list_display = [
        'id', 'user', 'notification_type', 'method', 'recipient',
        'status', 'retry_count', 'created_at'
    ]
    list_filter = ['status', 'notification_type', 'method', 'created_at']
    search_fields = ['user__username', 'recipient', 'subject']
    readonly_fields = ['created_at', 'sent_at']
    
    fieldsets = (
        ('Notification Details', {
            'fields': ('user', 'notification_type', 'method', 'recipient')
        }),
        ('Content', {
            'fields': ('subject', 'message', 'metadata')
        }),
        ('Delivery Status', {
            'fields': ('status', 'sent_at', 'retry_count', 'error_message')
        }),
        ('Metadata', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        }),
    )
    
    date_hierarchy = 'created_at'
    
    actions = ['retry_failed_notifications']
    
    def retry_failed_notifications(self, request, queryset):
        """Admin action to retry failed notifications"""
        from .notification_service import NotificationManager
        
        manager = NotificationManager()
        retry_count = 0
        
        for log in queryset.filter(status='failed'):
            # Retry logic would go here
            log.increment_retry()
            retry_count += 1
        
        self.message_user(request, f'{retry_count} notifications marked for retry.')
    
    retry_failed_notifications.short_description = 'Retry failed notifications'
