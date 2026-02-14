"""Test production API - raw JSON dump."""
import requests
import json

url = "https://ddl-69-h2l76esod-stas-projects-794d183b.vercel.app/api/live"

print("Testing NEW production deployment...")
print(f"URL: {url}\n")

try:
    resp = requests.get(url, params={"limit": 2}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    
    print(f"Status: {resp.status_code}")
    print(f"Count: {data.get('count', 'N/A')}\n")
    
    if 'ranked' in data and len(data['ranked']) > 0:
        item = data['ranked'][0]
        
        print("=" * 80)
        print("FIRST ITEM - HORIZON DATA CHECK:")
        print("=" * 80)
        print(f"Ticker: {item.get('ticker')}")
        print(f"Signal: {item.get('signal')}")
        print(f"Plan type: {item.get('plan_type')}")
        print(f"Price: ${item.get('price')}")
        print(f"\n--- HORIZON DATA ---")
        print(f"horizon_days: {item.get('horizon_days')} (type: {type(item.get('horizon_days'))})")
        print(f"\n--- TP DATA ---")
        print(f"tp_pct: {item.get('tp_pct')}")
        print(f"sl_pct: {item.get('sl_pct')}")
        print(f"tp1: ${item.get('tp1')}")
        print(f"tp2: ${item.get('tp2')}")
        print(f"tp3: ${item.get('tp3')}")
        print(f"sl1: ${item.get('sl1')}")
        print(f"sl2: ${item.get('sl2')}")        
        print(f"sl3: ${item.get('sl3')}")
        
        # Check meta
        meta = item.get('meta', {})
        print(f"\n--- META HORIZON ---")
        print(f"meta.horizon: {meta.get('horizon')}")
        
        print("\n" + "=" * 80)
        print("RAW JSON (first item):")
        print("=" * 80)
        # Print only the fields we care about
        relevant_fields = {k: v for k, v in item.items() 
                          if k in ['ticker', 'signal', 'price', 'plan_type', 'horizon_days', 
                                  'tp_pct', 'sl_pct', 'tp1', 'tp2', 'tp3', 'sl1', 'sl2', 'sl3']}
        print(json.dumps(relevant_fields, indent=2))
        
    else:
        print("ERROR: No ranked data")
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
