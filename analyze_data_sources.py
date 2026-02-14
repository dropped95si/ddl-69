#!/usr/bin/env python3
"""Analyze API response format vs dashboard expectations"""

import requests
import json

print("\n" + "="*80)
print("ANALYZING DATA SOURCES FOR REAL DATA SYNCHRONIZATION")
print("="*80 + "\n")

# 1. Check /api/live current structure
print("1️⃣ CHECKING: /api/live RESPONSE (Currently deployed)")
print("-" * 80)
try:
    r = requests.get("https://ddl-69-e5k5xo8pa-stas-projects-794d183b.vercel.app/api/live", timeout=5)
    data = r.json()
    
    print(f"✅ Status: {r.status_code}")
    print(f"✅ Top-level keys: {list(data.keys())}")
    print(f"✅ Row count: {len(data.get('ranked', []))}")
    
    if data.get('ranked'):
        first = data['ranked'][0]
        print(f"\n📌 CURRENT STRUCTURE (first item):")
        print(json.dumps({
            'ticker': first.get('ticker'),
            'signal': first.get('signal'),
            'price': first.get('price'),
            'p_accept': first.get('p_accept'),
            'confidence': first.get('confidence'),
            'weights': type(first.get('weights')).__name__,
            'all_keys': list(first.keys())[:10]
        }, indent=2))
except Exception as e:
    print(f"❌ Error fetching /api/live: {e}")

# 2. Check what dashboard.html expects
print("\n2️⃣ CHECKING: dashboard.html EXPECTED FORMAT")
print("-" * 80)
print("""
dashboard.html expects:
{
  "data": {
    "watchlist": [
      {
        "symbol": "SPY",
        "price": 485.50,
        "score": 0.75,
        "confidence": 0.82,
        "probability": 0.78,
        "change": +1.25,
        "signal": "BUY",
        "name": "SPDR S&P 500",
        "reasoning": "Strong uptrend",
        "targets": {"tp1": 490, "tp2": 500, "tp3": 510},
        "stoploss": {"sl1": 480, "sl2": 475, "sl3": 470},
        "weights": {"model1": 0.4, "model2": 0.3, ...}
      },
      ...
    ],
    "stats": {
      "total_symbols": 50,
      "avg_score": 0.72,
      "avg_confidence": 0.80,
      "buy_count": 15,
      "hold_count": 20,
      "sell_count": 15
    }
  }
}
""")

# 3. Identify mismatches
print("\n3️⃣ TRANSFORMATION NEEDED:")
print("-" * 80)
print("""
/api/live currently returns:
  - ticker (as top level) → dashboard needs symbol
  - signal (as label) → dashboard needs signal ✓
  - price → dashboard needs price ✓
  - p_accept → dashboard needs probability
  - confidence → dashboard needs confidence ✓
  - weights → dashboard needs weights ✓
  
MISSING for dashboard.html:
  - score (derived from p_accept or averaging models)
  - targets/tp1/tp2/tp3 (need to compute from Supabase meta)
  - stoploss/sl1/sl2/sl3 (need to compute from Supabase meta)
  - reasoning (use method + meta from Supabase)
  - change (need recent price data)
  - name (company name from event or static)

SOLUTION: Wrap /api/live in format adapter for dashboard.html
""")

print("\n" + "="*80 + "\n")
