"""
Django REST Framework Serializers
For paper trading models
"""
from rest_framework import serializers
from .models import PaperTrade, PerformanceMetrics, PriceCache, APIUsageTracker


class PaperTradeSerializer(serializers.ModelSerializer):
    """Serializer for PaperTrade model"""
    
    class Meta:
        model = PaperTrade
        fields = [
            'id', 'signal_id', 'pair', 'order_type', 'entry_price', 'lot_size',
            'stop_loss', 'take_profit_1', 'take_profit_2', 'take_profit_3',
            'entry_time', 'exit_time', 'exit_price', 'pips_gained', 'profit_loss',
            'risk_reward_ratio', 'status', 'signal_type', 'signal_source',
            'notes', 'created_at', 'updated_at'
        ]
        read_only_fields = [
            'id', 'pips_gained', 'profit_loss', 'created_at', 'updated_at'
        ]


class TradeExecutionSerializer(serializers.Serializer):
    """Serializer for trade execution requests"""
    
    pair = serializers.CharField(max_length=20)
    order_type = serializers.ChoiceField(choices=['buy', 'sell'])
    entry_price = serializers.DecimalField(max_digits=10, decimal_places=5)
    stop_loss = serializers.DecimalField(max_digits=10, decimal_places=5)
    take_profit_1 = serializers.DecimalField(
        max_digits=10, decimal_places=5, required=False, allow_null=True
    )
    take_profit_2 = serializers.DecimalField(
        max_digits=10, decimal_places=5, required=False, allow_null=True
    )
    take_profit_3 = serializers.DecimalField(
        max_digits=10, decimal_places=5, required=False, allow_null=True
    )
    lot_size = serializers.DecimalField(
        max_digits=10, decimal_places=2, default=0.01
    )
    signal_id = serializers.CharField(max_length=100, required=False, allow_null=True)
    signal_type = serializers.CharField(max_length=50, required=False, allow_null=True)
    signal_source = serializers.CharField(max_length=50, required=False, allow_null=True)
    notes = serializers.CharField(required=False, allow_null=True, allow_blank=True)


class PerformanceMetricsSerializer(serializers.ModelSerializer):
    """Serializer for PerformanceMetrics model"""
    
    class Meta:
        model = PerformanceMetrics
        fields = [
            'id', 'date', 'pair', 'total_trades', 'winning_trades', 'losing_trades',
            'win_rate', 'total_pips', 'total_pnl', 'avg_risk_reward',
            'max_drawdown', 'starting_equity', 'ending_equity',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']


class PriceCacheSerializer(serializers.ModelSerializer):
    """Serializer for PriceCache model"""
    
    class Meta:
        model = PriceCache
        fields = [
            'id', 'symbol', 'timestamp', 'open', 'high', 'low', 'close',
            'volume', 'source', 'timeframe', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']


class APIUsageTrackerSerializer(serializers.ModelSerializer):
    """Serializer for APIUsageTracker model"""
    
    usage_percentage = serializers.SerializerMethodField()
    
    class Meta:
        model = APIUsageTracker
        fields = [
            'id', 'api_name', 'date', 'requests_made', 'requests_limit',
            'usage_percentage', 'last_request_time', 'rate_limit_per_minute',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']
    
    def get_usage_percentage(self, obj):
        """Calculate usage as percentage"""
        if obj.requests_limit == 0:
            return 0.0
        return round((obj.requests_made / obj.requests_limit) * 100, 2)
