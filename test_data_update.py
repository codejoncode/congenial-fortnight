#!/usr/bin/env python3
"""
Test the updated data fetch function
"""
import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'forex_signal.settings')
django.setup()

from signals.views import update_data
from rest_framework.test import APIRequestFactory

# Create a fake request
factory = APIRequestFactory()
request = factory.post('/api/data/update/')

print("\n" + "="*60)
print("Testing Data Update Function")
print("="*60 + "\n")

# Call the function
response = update_data(request)

print(f"\nStatus Code: {response.status_code}")
print(f"Response Data:")
print(response.data)
