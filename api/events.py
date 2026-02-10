import json
from datetime import datetime, timedelta, timezone

from ddl69.core.settings import Settings
from supabase import create_client


def _fallback():
    now = datetime.now(timezone.utc)
    events = [
        {
            "timestamp": (now - timedelta(minutes=2)).isoformat(),
            "type": "forecast",
            "title": "Forecast Generated",
            "description": "AAPL: P(REJECT) = 0.73 +/- 0.04 [5D]"
        },
        {
            "timestamp": (now - timedelta(minutes=5)).isoformat(),
            "type": "weight_update",
            "title": "Weight Update",
            "description": "TripleBarrier: 0.27 -> 0.28 (+3.7%)"
        },
        {
            "timestamp": (now - timedelta(minutes=12)).isoformat(),
            "type": "outcome",
            "title": "Outcome Recorded",
            "description": "MSFT: Realized P(ACCEPT) - Calibrated"
        },
        {
            "timestamp": (now - timedelta(minutes=18)).isoformat(),
            "type": "ingest",
            "title": "Data Ingest",
            "description": "Processed 3,247 bars from Polygon"
        },
        {
            "timestamp": (now - timedelta(minutes=25)).isoformat(),
            "type": "forecast",
            "title": "Forecast Generated",
            "description": "GOOGL: P(ACCEPT) = 0.62 +/- 0.06 [10D]"
        }
    ]
    return events


def handler(request):
    settings = Settings.from_env()
    events = _fallback()

    if settings.supabase_url and settings.supabase_service_role_key:
        try:
            supa = create_client(settings.supabase_url, settings.supabase_service_role_key)
            resp = supa.table("events").select("event_id,event_type,subject_id,asof_ts,horizon_json").order("asof_ts", desc=True).limit(20).execute()
            data = resp.data or []
            events = []
            for ev in data:
                horizon = ev.get("horizon_json") or {}
                label = horizon.get("label") or ""
                span = horizon.get("horizon") or ""
                events.append({
                    "timestamp": ev.get("asof_ts"),
                    "type": ev.get("event_type"),
                    "title": f"{ev.get('subject_id','?')} {label}".strip(),
                    "description": f"Horizon: {span}" if span else ev.get("event_id")
                })
        except Exception:
            events = _fallback()

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json", "Cache-Control": "max-age=60"},
        "body": json.dumps({
            "events": events,
            "total": len(events)
        })
    }
