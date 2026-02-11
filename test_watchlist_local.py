#!/usr/bin/env python3
"""Test watchlist.py handler locally."""

import sys
import json

# Add api dir to path
sys.path.insert(0, 'api')

try:
    from watchlist import handler
    
    # Create a mock request object
    class MockRequest:
        pass
    
    request = MockRequest()
    result = handler(request)
    
    print("✅ Handler executed successfully!")
    print("\nResponse structure:")
    print(f"  statusCode: {result.get('statusCode')}")
    print(f"  headers: {result.get('headers')}")
    
    body = result.get('body')
    if isinstance(body, str):
        body_obj = json.loads(body)
    else:
        body_obj = body
    
    print(f"  body.count: {body_obj.get('count')}")
    print(f"  body.source: {body_obj.get('source')}")
    print(f"  body.tickers: {body_obj.get('tickers')}")
    print("\n✅ All OK!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
