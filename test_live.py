#!/usr/bin/env python
import requests
import json

url = 'https://ddl-69-3xc7uf2ws-stas-projects-794d183b.vercel.app/api/live'
try:
    r = requests.get(url, timeout=15)
    print(f"Status: {r.status_code}")
    print(f"Content-Type: {r.headers.get('content-type')}")
    print(f"Response length: {len(r.text)}")
    print("\nFirst 500 chars:")
    print(r.text[:500])
    
    # Try to parse as JSON
    try:
        data = r.json()
        print("\nJSON parsed successfully!")
        print(f"Error: {data.get('error')}")
        print(f"Message: {data.get('message')}")
        print(f"Count: {data.get('count')}")
    except Exception as e:
        print(f"\nJSON parse error: {e}")
except Exception as e:
    print(f"Request error: {e}")
