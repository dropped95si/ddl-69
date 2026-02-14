#!/usr/bin/env python
import requests
import json

endpoints = {
    '/api/live': 'Live watchlist',
    '/api/news': 'News feed',
    '/api/overlays': 'Technical overlays',
    '/api/health': 'System health',
}

base = 'https://ddl-69-3xc7uf2ws-stas-projects-794d183b.vercel.app'

for endpoint, desc in endpoints.items():
    url = base + endpoint
    try:
        r = requests.get(url, timeout=10)
        data = r.json() if 'application/json' in r.headers.get('content-type', '') else {}
        count = data.get('count', data.get('items', data.get('data')))
        error = data.get('error')
        msg = data.get('message', '—')
        status = f"✅ {r.status_code}"
        if error or (r.status_code != 200):
            status = f"❌ {r.status_code} - {error or msg}"
        print(f"{status:40} | {desc:20} | Count: {count if isinstance(count, int) else '—'}")
    except Exception as e:
        print(f"❌ ERROR                            | {desc:20} | {str(e)[:40]}")
