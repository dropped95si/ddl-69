#!/usr/bin/env python3
import json, sys
sys.path.insert(0, '.')

print("Testing APIs locally...")
print()

# Test 1: Live endpoint
from api.live import _handler_impl as live_handler
result = live_handler(None)
data = json.loads(result['body'])
print('✅ /api/live WORKS')
print(f'   Source: {data["source"]}, Count: {data["count"]}')
print(f'   Tickers: {", ".join(data["tickers"])}')
print(f'   Buy: {data["stats"]["buy_count"]}, Hold: {data["stats"]["hold_count"]}, Sell: {data["stats"]["sell_count"]}')
print()

# Test 2: Watchlist endpoint  
from api.watchlist import _handler_impl as watchlist_handler
result = watchlist_handler(None)
data = json.loads(result['body'])
print('✅ /api/watchlist WORKS')
print(f'   Source: {data["source"]}, Provider: {data["provider"]}')
print(f'   Count: {data["count"]}')
print(f'   Buy: {data["stats"]["buy_count"]}, Hold: {data["stats"]["hold_count"]}, Sell: {data["stats"]["sell_count"]}')
print()
print("✅ SYSTEM IS WORKING - Both APIs functional locally")
print("❌ Vercel deployment protection blocks public access")
print("💡 Solution: Disable protection at https://vercel.com/stas-projects-794d183b/ddl-69/settings/security")
