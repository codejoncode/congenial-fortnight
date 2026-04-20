#!/usr/bin/env python
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'forex_signal.settings')
django.setup()

from django.urls import get_resolver

def list_all_signal_urls():
    """List all URL patterns containing 'signal' or 'unified'"""
    resolver = get_resolver()
    
    def get_patterns(urlpatterns, prefix=''):
        patterns = []
        for pattern in urlpatterns:
            if hasattr(pattern, 'url_patterns'):
                # This is an include()
                new_prefix = prefix + str(pattern.pattern)
                patterns.extend(get_patterns(pattern.url_patterns, new_prefix))
            else:
                # This is a path() or re_path()
                full_path = prefix + str(pattern.pattern)
                patterns.append(full_path)
        return patterns
    
    all_patterns = get_patterns(resolver.url_patterns)
    
    print("\n=== All URL patterns containing 'signal' or 'unified' ===")
    for pattern in sorted(all_patterns):
        if 'signal' in pattern.lower() or 'unified' in pattern.lower():
            print(f"  {pattern}")
    
    print("\n=== Checking specific paths ===")
    test_paths = [
        'api/signals/',
        'api/signals/unified/',
        'api/signals/generate/',
        'api/db/',
    ]
    
    for test_path in test_paths:
        # Try to resolve the path
        try:
            from django.urls import resolve
            match = resolve(f'/{test_path}')
            print(f"✅ /{test_path} -> {match.func.__name__ if hasattr(match.func, '__name__') else match.func}")
        except Exception as e:
            print(f"❌ /{test_path} -> {type(e).__name__}: {e}")

if __name__ == '__main__':
    list_all_signal_urls()
