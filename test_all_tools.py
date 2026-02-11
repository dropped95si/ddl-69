#!/usr/bin/env python
"""Test all API tools to see what's working"""
import json

tools = [
    ("live", "api.live"),
    ("watchlist", "api.watchlist"),
    ("calibration", "api.calibration"),
    ("overlays", "api.overlays"),
    ("finviz", "api.finviz"),
    ("forecasts", "api.forecasts"),
    ("walkforward", "api.walkforward"),
    ("health", "api.health"),
    ("status", "api.status"),
    ("news", "api.news"),
    ("events", "api.events"),
]

print("=" * 70)
print("TESTING ALL TOOLS")
print("=" * 70)

for name, module_path in tools:
    try:
        module = __import__(module_path, fromlist=['_handler_impl'])
        if not hasattr(module, '_handler_impl'):
            print(f"❌ {name:15} - No _handler_impl")
            continue
        
        result = module._handler_impl(None)
        if not result or 'statusCode' not in result:
            print(f"❌ {name:15} - No statusCode")
            continue
        
        status = result.get('statusCode')
        if status == 200:
            body = json.loads(result.get('body', '{}'))
            count = body.get('count', len(body.get('rows', [])))
            print(f"✅ {name:15} - Status {status}, Data items: {count if count else '(complex)'}")
        else:
            print(f"❌ {name:15} - Status {status}")
    except Exception as e:
        print(f"❌ {name:15} - {str(e)[:50]}")

print()
print("=" * 70)
print("All tools tested. Now deploying...")
print("=" * 70)
