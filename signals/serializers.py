from rest_framework import serializers
from .models import ForexPair, Signal, Trade

class ForexPairSerializer(serializers.ModelSerializer):
    class Meta:
        model = ForexPair
        fields = '__all__'

class SignalSerializer(serializers.ModelSerializer):
    pair_name = serializers.CharField(source='pair.name', read_only=True)

    class Meta:
        model = Signal
        fields = '__all__'

class TradeSerializer(serializers.ModelSerializer):
    signal_details = SignalSerializer(source='signal', read_only=True)

    class Meta:
        model = Trade
        fields = '__all__'