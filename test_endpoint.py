#!/usr/bin/env python3
"""Quick test of unified signals endpoint"""
import requests
import json

try:
    response = requests.get('http://127.0.0.1:8000/api/signals/unified/?pair=EURUSD&mode=parallel')
    print(f"Status Code: {response.status_code}")
    print(f"\nResponse Content:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
