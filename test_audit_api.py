"""Quick test script for audit API"""
import requests
import json

def test_audit():
    url = "https://ddl69.agilera.ai/api/audit?limit=3"
    
    print(f"Testing: {url}")
    
    try:
        resp = requests.get(url, timeout=10)
        print(f"Status: {resp.status_code}")
        
        if resp.status_code == 200:
            data = resp.json()
            print(f"\n✓ SUCCESS!")
            print(f"Total comparisons: {data['summary']['total_comparisons']}")
            print(f"Strong Buy: {data['summary']['strong_buy']}")
            print(f"Agreement: {data['summary']['avg_agreement']:.2%}")
            
            if data['comparisons']:
                first = data['comparisons'][0]
                print(f"\nFirst ticker: {first['ticker']}")
                print(f"Price: ${first['price']}")
                print(f"Consensus: {first['consensus']['recommendation']}")
                print(f"DDL-69 confidence: {first['models']['ddl69']['confidence']:.1%}")
                print(f"Qlib confidence: {first['models']['qlib']['confidence']:.1%}")
                print(f"Chronos confidence: {first['models']['chronos']['confidence']:.1%}")
        else:
            print(f"Error: {resp.status_code}")
            print(resp.text)
            
    except Exception as e:
        print(f"✗ FAILED: {e}")

if __name__ == "__main__":
    test_audit()
