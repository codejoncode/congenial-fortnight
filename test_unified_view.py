#!/usr/bin/env python
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'forex_signal.settings')
django.setup()

from django.test import RequestFactory
from django.urls import resolve, Resolver404

def test_unified_endpoint():
    """Test if the unified endpoint can be resolved and called"""
    
    path = '/api/signals/unified/'
    print(f"\n=== Testing: {path} ===")
    
    # Try to resolve the URL
    try:
        match = resolve(path)
        print(f"✅ URL resolves to: {match.func}")
        print(f"   View name: {match.view_name}")
        print(f"   Kwargs: {match.kwargs}")
        
        # Try to call the view
        factory = RequestFactory()
        request = factory.get(path, {'pair': 'EURUSD', 'mode': 'parallel'})
        
        print(f"\n📞 Attempting to call view...")
        try:
            response = match.func(request, **match.kwargs)
            print(f"✅ View executed successfully!")
            print(f"   Response status: {response.status_code}")
            print(f"   Response type: {type(response)}")
            if response.status_code == 200:
                content = str(response.content)[:200]
                print(f"   Content preview: {content}")
            else:
                print(f"   Error content: {response.content}")
        except Exception as e:
            print(f"❌ View execution failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            
    except Resolver404 as e:
        print(f"❌ URL does not resolve: {e}")
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_unified_endpoint()
