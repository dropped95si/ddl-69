"""System status endpoint without synthetic metrics."""

import json
import os
from datetime import datetime, timezone

try:
    from _http_adapter import FunctionHandler
except ModuleNotFoundError:
    from api._http_adapter import FunctionHandler

try:
    from _real_market import build_watchlist
except ModuleNotFoundError:
    from api._real_market import build_watchlist


def _handler_impl(request):
    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()

    body = {
        "system_status": "ONLINE",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "supabase_connected": False,
        "active_forecasts": None,
        "total_events": None,
        "total_outcomes": None,
        "market_feed_rows": None,
    }

    if supabase_url and service_key:
        try:
            from supabase import create_client

            supa = create_client(supabase_url, service_key)
            ef = supa.table("ensemble_forecasts").select("id", count="exact").execute()
            events = supa.table("events").select("event_id", count="exact").execute()
            outcomes = supa.table("event_outcomes").select("event_id", count="exact").execute()
            body.update(
                {
                    "supabase_connected": True,
                    "active_forecasts": ef.count or 0,
                    "total_events": events.count or 0,
                    "total_outcomes": outcomes.count or 0,
                }
            )
        except Exception as exc:
            body["supabase_error"] = str(exc)

    try:
        rows = build_watchlist("swing", 40)
        body["market_feed_rows"] = len(rows)
    except Exception as exc:
        body["market_feed_error"] = str(exc)

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json", "Cache-Control": "max-age=60"},
        "body": json.dumps(body),
    }


class handler(FunctionHandler):
    endpoint = staticmethod(_handler_impl)

