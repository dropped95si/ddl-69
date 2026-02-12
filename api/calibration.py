"""Calibration endpoint - real artifacts only."""

import json
import os
from datetime import datetime, timezone

try:
    from _http_adapter import FunctionHandler
except ModuleNotFoundError:
    from api._http_adapter import FunctionHandler


def _handler_impl(request):
    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not supabase_url or not service_key:
        return {
            "statusCode": 503,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(
                {
                    "error": "Calibration artifacts unavailable: missing Supabase credentials.",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
            ),
        }

    try:
        from supabase import create_client

        supa = create_client(supabase_url, service_key)
        resp = (
            supa.table("artifacts")
            .select("payload,created_at")
            .eq("artifact_type", "calibration")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            return {
                "statusCode": 503,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps(
                    {
                        "error": "No calibration artifact found.",
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                    }
                ),
            }

        payload = rows[0].get("payload") or {}
        payload["artifact_created_at"] = rows[0].get("created_at")
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json", "Cache-Control": "max-age=120"},
            "body": json.dumps(payload),
        }
    except Exception as exc:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(exc)}),
        }


class handler(FunctionHandler):
    endpoint = staticmethod(_handler_impl)

