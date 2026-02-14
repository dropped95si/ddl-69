"""Quick test of production API."""
import requests

url = "https://ddl-69-h2l76esod-stas-projects-794d183b.vercel.app/api/live"

print(f"Testing: {url}\n")

try:
    resp = requests.get(url, params={"limit": 2}, timeout=10)
    print(f"Status Code: {resp.status_code}")
    print(f"Response Length: {len(resp.text)} bytes")
    print(f"Content-Type: {resp.headers.get('content-type', 'N/A')}\n")
    
    if resp.status_code != 200:
        print("ERROR Response:")
        print(resp.text[:1000])
    else:
        try:
            data = resp.json()
            print(f"JSON OK - Count: {data.get('count', 'N/A')}")
            if 'ranked' in data and len(data['ranked']) > 0:
                item = data['ranked'][0]
                print(f"\nFirst item:")
                print(f"  Ticker: {item.get('ticker')}")
                print(f"  Horizon: {item.get('horizon_days')}")
                print(f"  TP1: {item.get('tp1')}")
        except Exception as e:
            print(f"JSON Parse Failed: {e}")
            print(f"Raw response (first 500 chars):")
            print(resp.text[:500])
            
except Exception as e:
    print(f"Request Failed: {e}")
    import traceback
    traceback.print_exc()
