"""
Test Configuration for Paper Trading
"""
import os
import sys
import pytest

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Django settings for tests
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'forex_signal.settings')

import django
django.setup()

# Configure channel layers for testing
from channels.testing import ChannelsLiveServerTestCase
from django.conf import settings

# Set up in-memory channel layer for tests
@pytest.fixture(scope='session', autouse=True)
def configure_channel_layers():
    """Configure channel layers for WebSocket testing"""
    if not hasattr(settings, 'CHANNEL_LAYERS'):
        settings.CHANNEL_LAYERS = {}
    
    settings.CHANNEL_LAYERS = {
        'default': {
            'BACKEND': 'channels.layers.InMemoryChannelLayer'
        }
    }
