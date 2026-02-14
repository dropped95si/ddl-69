#!/usr/bin/env python3
"""Test if Supabase has live forecast data"""
import os
from supabase import create_client

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://iyqzrzesrbfltoryfzet.supabase.co")
SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

if not SERVICE_KEY:
    print("❌ SUPABASE_SERVICE_ROLE_KEY not set!")
    exit(1)

print(f"✅ Connecting to: {SUPABASE_URL}")
supa = create_client(SUPABASE_URL, SERVICE_KEY)

# Check v_latest_ensemble_forecasts
resp = supa.table("v_latest_ensemble_forecasts").select("event_id,confidence,created_at,method,probs_json").order("created_at", desc=True).limit(50).execute()
rows = resp.data or []
print(f"\n🔍 Total forecast rows found: {len(rows)}")

print(f"\n📊 v_latest_ensemble_forecasts: {len(rows)} rows")
if rows:
    print("   Recent forecasts:")
    for r in rows[:5]:
        eid = r.get("event_id", "")[:12]
        conf = r.get("confidence", 0)
        method = r.get("method", "unknown")
        ts = r.get("created_at", "")[:19]
        print(f"   • {eid}... {method:15} conf={conf:.3f} @ {ts}")
else:
    print("   ⚠️  NO FORECAST DATA!")

# Check uniqueness and probabilities
if rows:
    probs = [float(r.get("confidence", 0.5)) for r in rows]
    from statistics import pstdev
    unique_probs = len(set(round(p, 4) for p in probs))
    prob_stdev = pstdev(probs) if len(probs) > 1 else 0
    print(f"   Probability diversity: {unique_probs} unique values, stdev={prob_stdev:.4f}")
    if prob_stdev < 0.01:
        print("   ⚠️  STDEV TOO LOW - would be rejected by API!")

# Check events table
event_ids = [r["event_id"] for r in rows[:10] if r.get("event_id")] 
if event_ids:
    ev_resp = supa.table("events").select("event_id,subject_id,horizon_json").in_("event_id", event_ids[:5]).execute()
    events = ev_resp.data or []
    print(f"\n📋 events: {len(events)} linked events")
    for ev in events[:3]:
        ticker = ev.get("subject_id", "???")
        horizon = ev.get("horizon_json")
        print(f"   • {ticker:6} horizon={horizon}")
else:
    print("\n⚠️  No event IDs found")
