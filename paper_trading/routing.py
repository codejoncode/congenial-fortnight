"""
WebSocket routing for paper trading
"""
from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/trading/$', consumers.TradingWebSocketConsumer.as_asgi()),
    re_path(r'ws/prices/$', consumers.PriceStreamConsumer.as_asgi()),
]
