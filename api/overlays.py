import json
from datetime import datetime, timezone

try:
    from _http_adapter import FunctionHandler
except ModuleNotFoundError:
    from api._http_adapter import FunctionHandler

DEFAULT_SUPABASE_URL = ""
DEFAULT_SUPABASE_SERVICE_ROLE_KEY = ""


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


def _handler_impl(request):
    """Return overlay data from Supabase (falls back to sample data)."""
    import os

    supabase_url = os.getenv("SUPABASE_URL", DEFAULT_SUPABASE_URL).strip()
    supabase_service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", DEFAULT_SUPABASE_SERVICE_ROLE_KEY).strip()
    if not supabase_url or not supabase_service_role_key:
        return _fallback()

    try:
        from supabase import create_client

        supa = create_client(supabase_url, supabase_service_role_key)
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


class handler(FunctionHandler):
    endpoint = staticmethod(_handler_impl)
