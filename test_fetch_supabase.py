#!/usr/bin/env python3
"""Test the actual _fetch_supabase() function locally"""
import sys
import os
sys.path.insert(0, 'api')

# Set env vars
os.environ['SUPABASE_URL'] = 'https://iyqzrzesrbfltoryfzet.supabase.co'
# Assume SERVICE_KEY is already set

from live import _fetch_supabase

print("Testing _fetch_supabase() with timeframe='all'...")
result = _fetch_supabase(timeframe_filter='all')

if result is None:
    print("❌ RETURNED NONE!")
elif isinstance(result, list):
    print(f"✅ SUCCESS! Got {len(result)} watchlist items")
    if result:
        print("\nFirst 3 tickers:")
        for item in result[:3]:
            print(f"  • {item.get('ticker'):6} {item.get('signal'):4} conf={item.get('confidence'):.3f} horizon={item.get('horizon_days')}d")
else:
    print(f"⚠️  Unexpected result type: {type(result)}")

print("\nTrying with timeframe='day'...")
result_day = _fetch_supabase(timeframe_filter='day')
if result_day:
    print(f"✅ Got {len(result_day)} day trades")
else:
    print("❌ No day trades")
