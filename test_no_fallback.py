"""Test new deployment with no fallback."""
import requests
import json

url = "https://ddl-69-psub3p2q4-stas-projects-794d183b.vercel.app/api/live"

print(f"Testing NEW deployment (no fallback): {url}\n")

try:
    resp = requests.get(url, params={"limit": 2}, timeout=15)
    print(f"Status Code: {resp.status_code}")
    
    if resp.status_code == 503:
        print("\n✓ CORRECT: Returns 503 when Supabase unavailable (no fake fallback)")
        data = resp.json()
        print(f"Error: {data.get('error')}")
        print(f"Message: {data.get('message')}")
    elif resp.status_code == 200:
        data = resp.json()
        print(f"\n✓ SUCCESS: Real Supabase data returned")
        print(f"Source: {data.get('source')}")
        print(f"Count: {data.get('count')}")
        
        if data.get('count', 0) > 0:
            item = data['ranked'][0]
            print(f"\nFirst item check:")
            print(f"  Ticker: {item.get('ticker')}")
            print(f"  Source: {item.get('source')}")
            print(f"  horizon_days: {item.get('horizon_days')} (should NOT be None)")
            print(f"  tp1: ${item.get('tp1')} (should NOT be None)")
            print(f"  tp_pct: {item.get('tp_pct')}")
            
            if item.get('source') == 'market_ta':
                print("\n❌ ERROR: Still using Yahoo fallback (market_ta)!")
            elif item.get('horizon_days') is not None:
                print("\n✓ SUCCESS: Real horizon data present!")
            else:
                print("\n⚠️ WARNING: horizon_days is None (Supabase issue)")
    else:
        print(f"\nUnexpected status: {resp.status_code}")
        print(resp.text[:500])
        
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
