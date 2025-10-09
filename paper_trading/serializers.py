"""
Django REST Framework Serializers
For paper trading models
"""
from rest_framework import serializers
from .models import PaperTrade, PerformanceMetrics, PriceCache, APIUsageTracker


class PaperTradeSerializer(serializers.ModelSerializer):
    """Serializer for PaperTrade model"""
    
    # Add symbol as an alias for pair (read-only)
    symbol = serializers.CharField(source='pair', read_only=True)
    
    class Meta:
        model = PaperTrade
        fields = [
            'id', 'signal_id', 'pair', 'symbol', 'order_type', 'entry_price', 'lot_size',
            'stop_loss', 'take_profit_1', 'take_profit_2', 'take_profit_3',
            'entry_time', 'exit_time', 'exit_price', 'exit_reason', 'pips_gained', 'profit_loss',
            'risk_reward_ratio', 'status', 'signal_type', 'signal_source',
            'notes', 'created_at', 'updated_at'
        ]
        read_only_fields = [
            'id', 'symbol', 'pips_gained', 'profit_loss', 'created_at', 'updated_at'
        ]


class TradeExecutionSerializer(serializers.Serializer):
    """Serializer for trade execution requests"""
    
    pair = serializers.CharField(max_length=20, required=False)
    symbol = serializers.CharField(max_length=20, required=False)  # Alias for pair
    order_type = serializers.ChoiceField(choices=['buy', 'sell', 'BUY', 'SELL'], required=False)
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
    
    def validate(self, data):
        """Handle field aliasing"""
        # Map symbol -> pair
        if 'symbol' in data and 'pair' not in data:
            data['pair'] = data.pop('symbol')
        elif 'symbol' in data:
            data.pop('symbol')  # Remove duplicate
        
        # Map signal_type (when used as direction) -> order_type  
        if 'signal_type' in data and 'order_type' not in data:
            sig_val = data.get('signal_type')
            if sig_val in ['BUY', 'SELL', 'buy', 'sell']:
                data['order_type'] = sig_val.lower()
        
        # Normalize order_type to lowercase
        if 'order_type' in data:
            data['order_type'] = data['order_type'].lower()
        
        # Ensure required fields are present
        if 'pair' not in data:
            raise serializers.ValidationError({'pair': 'This field is required (can use symbol as alias).'})
        if 'order_type' not in data:
            raise serializers.ValidationError({'order_type': 'This field is required (can use signal_type with BUY/SELL).'})
        
        return data


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
