import json
from datetime import datetime, timezone

from ddl69.core.settings import Settings
from supabase import create_client


def _fallback():
    """Return sample overlay data when Supabase is not configured."""
    overlay = {
        "asof": datetime.utcnow().isoformat(),
        "symbols": {
            "SPY": {
                "series": [
                    {"time": "2026-01-01", "value": 470.2},
                    {"time": "2026-01-02", "value": 472.8},
                    {"time": "2026-01-03", "value": 471.5}
                ],
                "zones": [
                    {"from": 468, "to": 476, "label": "Demand", "color": "rgba(0,229,160,0.12)"}
                ],
                "levels": [
                    {"value": 480, "label": "R1", "color": "#ffd479", "style": "dashed"}
                ],
                "percent_levels": [
                    {"percent": 0.02, "label": "+2%"},
                    {"percent": -0.02, "label": "-2%"}
                ]
            }
        }
    }

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json", "Cache-Control": "max-age=300"},
        "body": json.dumps(overlay)
    }


def handler(request):
    """Return overlay data from Supabase (falls back to sample data)."""
    settings = Settings.from_env()
    if not settings.supabase_url or not settings.supabase_service_role_key:
        return _fallback()

    try:
        supa = create_client(settings.supabase_url, settings.supabase_service_role_key)
        # Query artifacts table for latest overlay
        resp = supa.table("artifacts")\
            .select("payload")\
            .eq("artifact_type", "overlay")\
            .order("created_at", desc=True)\
            .limit(1)\
            .execute()

        if resp.data and len(resp.data) > 0:
            overlay = resp.data[0].get("payload", {})
            return {
                "statusCode": 200,
                "headers": {"Content-Type": "application/json", "Cache-Control": "max-age=120"},
                "body": json.dumps(overlay)
            }
        else:
            return _fallback()
    except Exception as exc:
        return _fallback()
