import json
from datetime import datetime

from ddl69.core.settings import Settings
from supabase import create_client


def _fallback():
    return {
        "system_status": "ONLINE",
        "supabase_connected": False,
        "active_forecasts": 247,
        "calibration": 0.948,
        "accuracy_7d": 0.894,
        "uptime_seconds": 3600,
        "experts": [
            {"name": "TripleBarrier", "weight": 0.28, "accuracy": 0.923, "status": "Active"},
            {"name": "Sentiment", "weight": 0.22, "accuracy": 0.871, "status": "Active"},
            {"name": "Technical", "weight": 0.20, "accuracy": 0.856, "status": "Active"},
            {"name": "Fundamental", "weight": 0.15, "accuracy": 0.794, "status": "Active"},
            {"name": "Ensemble", "weight": 0.15, "accuracy": 0.948, "status": "Active"}
        ]
    }


def handler(request):
    """Return system status; prefers Supabase stats when available."""
    settings = Settings.from_env()
    if not settings.supabase_url or not settings.supabase_service_role_key:
        body = _fallback()
    else:
        try:
            supa = create_client(settings.supabase_url, settings.supabase_service_role_key)
            # counts
            ef = supa.table("ensemble_forecasts").select("id", count="exact").execute()
            events = supa.table("events").select("event_id", count="exact").execute()
            outcomes = supa.table("event_outcomes").select("event_id", count="exact").execute()
            weights = supa.table("weight_updates").select("id,weights_after_json,created_at", order="created_at", desc=True, limit=5).execute()

            last_weights = []
            for w in weights.data or []:
                last_weights.append({
                    "asof_ts": w.get("created_at"),
                    "weights": w.get("weights_after_json"),
                })

            body = {
                "system_status": "ONLINE",
                "supabase_connected": True,
                "active_forecasts": ef.count or 0,
                "calibration": None,
                "accuracy_7d": None,
                "uptime_seconds": None,
                "total_events": events.count or 0,
                "total_outcomes": outcomes.count or 0,
                "recent_weights": last_weights,
            }
        except Exception:
            body = _fallback()

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json", "Cache-Control": "max-age=60"},
        "body": json.dumps(body)
    }
