"""
Quick URL diagnostic - Run this to see all registered API endpoints
"""
import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'forex_signal.settings')
sys.path.insert(0, os.getcwd())
django.setup()

from django.urls import get_resolver

resolver = get_resolver()

print("\n" + "="*70)
print("REGISTERED API ENDPOINTS")
print("="*70 + "\n")

def print_urls(patterns, prefix=''):
    for pattern in patterns:
        if hasattr(pattern, 'url_patterns'):
            # It's an include, recurse
            print_urls(pattern.url_patterns, prefix + str(pattern.pattern))
        else:
            full_path = prefix + str(pattern.pattern)
            if 'api/' in full_path or 'signals' in full_path:
                name = pattern.name if hasattr(pattern, 'name') else 'no-name'
                print(f"  {full_path:50} [{name}]")

print_urls(resolver.url_patterns)

print("\n" + "="*70)
print("Looking for 'signals/unified/' specifically...")
print("="*70 + "\n")

for pattern in resolver.url_patterns:
    if 'api/' in str(pattern.pattern):
        if hasattr(pattern, 'url_patterns'):
            for subpattern in pattern.url_patterns:
                if 'unified' in str(subpattern.pattern):
                    print(f"  FOUND: api/{subpattern.pattern}")
