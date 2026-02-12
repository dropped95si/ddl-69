"""Recent events endpoint - Supabase data only."""

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
                    "error": "Supabase unavailable for events feed.",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
            ),
        }

    try:
        from supabase import create_client

        supa = create_client(supabase_url, service_key)
        resp = (
            supa.table("events")
            .select("event_id,event_type,subject_id,asof_ts,horizon_json")
            .order("asof_ts", desc=True)
            .limit(200)
            .execute()
        )
        data = resp.data or []
        events = []
        for ev in data:
            horizon = ev.get("horizon_json") or {}
            label = horizon.get("label") if isinstance(horizon, dict) else None
            days = horizon.get("days") if isinstance(horizon, dict) else None
            events.append(
                {
                    "timestamp": ev.get("asof_ts"),
                    "type": ev.get("event_type"),
                    "title": f"{ev.get('subject_id') or '?'} {label or ''}".strip(),
                    "description": f"{days}d horizon" if days else ev.get("event_id"),
                }
            )

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json", "Cache-Control": "max-age=60"},
            "body": json.dumps({"events": events, "total": len(events), "source": "supabase"}),
        }
    except Exception as exc:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(exc)}),
        }


class handler(FunctionHandler):
    endpoint = staticmethod(_handler_impl)

