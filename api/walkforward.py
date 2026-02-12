"""Walk-forward endpoint - Supabase artifact only (no fallback)."""

import json
import os
from datetime import datetime, timezone

try:
    from _http_adapter import FunctionHandler
except ModuleNotFoundError:
    from api._http_adapter import FunctionHandler

def _fetch_walkforward_artifact():
    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not supabase_url or not service_key:
        return None

    try:
        from supabase import create_client
    except Exception:
        return None

    try:
        supa = create_client(supabase_url, service_key)
        resp = (
            supa.table("artifacts")
            .select("payload,created_at")
            .eq("artifact_type", "walkforward")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            return None
        payload = rows[0].get("payload") or {}
        if isinstance(payload, dict):
            payload.setdefault("artifact_created_at", rows[0].get("created_at"))
        return payload
    except Exception:
        return None


def _handler_impl(request):
    payload = _fetch_walkforward_artifact()
    if payload is None:
        return {
            "statusCode": 503,
            "headers": {
                "Content-Type": "application/json",
                "Cache-Control": "no-store",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps(
                {
                    "error": "supabase_unavailable",
                    "message": "Supabase walk-forward artifact required; no fallback enabled.",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
            ),
        }

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Cache-Control": "max-age=120, public",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(payload),
    }


class handler(FunctionHandler):
    endpoint = staticmethod(_handler_impl)
