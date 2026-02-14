"""Debug production API response structure."""
import requests
import json

url = "https://ddl-69-h2l76esod-stas-projects-794d183b.vercel.app/api/live"

print(f"Testing: {url}\n")

resp = requests.get(url, params={"limit": 2}, timeout=10)
data = resp.json()

print(f"Count: {data.get('count')}")
print(f"Keys in response: {list(data.keys())}\n")

if 'ranked' in data and len(data['ranked']) > 0:
    item = data['ranked'][0]
    print("=" * 80)
    print("FULL FIRST ITEM (checking ALL fields):")
    print("=" * 80)
    print(json.dumps(item, indent=2, default=str)[:2000])  # First 2000 chars
    
    print("\n" + "=" * 80)
    print("KEY FIELDS CHECK:")
    print("=" * 80)
    for key in ['ticker', 'price', 'horizon_days', 'tp_pct', 'sl_pct', 'tp1', 'tp2', 'tp3', 'sl1', 'sl2', 'sl3']:
        value = item.get(key, 'KEY_MISSING')
        print(f"  {key:15} = {value}")
