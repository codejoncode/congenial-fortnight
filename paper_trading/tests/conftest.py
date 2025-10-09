"""
Test Configuration for Paper Trading
"""
import os
import sys

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Django settings for tests
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'forex_signal.settings')

import django
django.setup()
