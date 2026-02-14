#!/usr/bin/env python3
import requests
import json

endpoints = [
    '/api/live',
    '/api/news', 
    '/api/overlays',
    '/api/forecasts',
    '/api/health',
]

base_url = 'https://ddl-69-e5k5xo8pa-stas-projects-794d183b.vercel.app'

print("=" * 80)
print("TRUTH MODE - ENDPOINT VERIFICATION")
print("=" * 80)

for ep in endpoints:
    try:
        r = requests.get(f'{base_url}{ep}', timeout=5)
        data = r.json() if r.headers.get('content-type') == 'application/json' else {}
        
        if isinstance(data, dict):
            keys = list(data.keys())[:3]
            if 'count' in data:
                print(f'✅ {ep:20} | Status: {r.status_code:3} | Count: {data.get("count", "N/A"):4} | Sample keys: {keys}')
            elif 'message' in data:
                msg = data['message'][:35] if len(data['message']) > 35 else data['message']
                print(f'✅ {ep:20} | Status: {r.status_code:3} | Msg: {msg}')
            else:
                sample_val = str(list(data.values())[0])[:30] if data else "empty"
                print(f'✅ {ep:20} | Status: {r.status_code:3} | Keys: {keys}')
        else:
            print(f'✅ {ep:20} | Status: {r.status_code:3} | Response: OK')
    except Exception as e:
        print(f'❌ {ep:20} | Error: {str(e)[:40]}')

print("=" * 80)
print("\nREAL DATA VALIDATION:")
print("=" * 80)

# Test /api/live specifically
try:
    r = requests.get(f'{base_url}/api/live', timeout=5)
    data = r.json()
    if 'count' in data and data['count'] > 0:
        sample = data['ranked'][0] if 'ranked' in data and data['ranked'] else None
        if sample:
            print(f"✅ Live watchlist: {data['count']} real forecasts from Supabase")
            print(f"   Sample: {sample['ticker']} | Signal: {sample.get('signal', 'N/A')} | P(Accept): {sample.get('p_accept', 'N/A')}")
        else:
            print(f"⚠️  Live watchlist has count={data['count']} but no ranked data")
    else:
        print(f"❌ Live watchlist returned no data: {data}")
except Exception as e:
    print(f"❌ Failed to fetch live data: {e}")

print()
