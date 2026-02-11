import json
from datetime import datetime, timedelta, timezone

try:
    from _http_adapter import FunctionHandler
except ModuleNotFoundError:
    from api._http_adapter import FunctionHandler

DEFAULT_SUPABASE_URL = ""
DEFAULT_SUPABASE_SERVICE_ROLE_KEY = ""


def _fallback():
    """Return deterministic sample data when Supabase is not configured."""
    def random_walk(length, start=0.5, volatility=0.08, trend=0.002):
        values = [start]
        for i in range(1, length):
            seed = (i * 73 + 47) % 1000
            noise = (seed - 500) / 5000  # -0.1 to +0.1
            change = noise * volatility + trend
            new_val = max(0, min(1, values[-1] + change))
            values.append(new_val)
        return values

    accept_walk = random_walk(30, start=0.55, volatility=0.08, trend=0.001)
    reject_walk = random_walk(30, start=0.40, volatility=0.07, trend=-0.0005)
    continue_walk = random_walk(30, start=0.05, volatility=0.03, trend=0)

    forecasts = []
    for i in range(30):
        date = (datetime.now(timezone.utc) - timedelta(days=30 - i)).strftime('%Y-%m-%d')
        forecasts.append({
            "event_id": f"sample-{i}",
            "ticker": "SAMPLE",
            "date": date,
            "accept": round(accept_walk[i], 4),
            "reject": round(reject_walk[i], 4),
            "continue": round(continue_walk[i], 4),
            "confidence": 0.9,
            "method": "blended",
        })

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json", "Cache-Control": "max-age=300"},
        "body": json.dumps({
            "forecasts": forecasts,
            "total": len(forecasts),
            "span_days": 30,
            "generated_at": datetime.utcnow().isoformat()
        })
    }


def _handler_impl(request):
    """Return latest ensemble forecasts from Supabase (falls back to sample data)."""
    import os

    supabase_url = os.getenv("SUPABASE_URL", DEFAULT_SUPABASE_URL).strip()
    supabase_service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", DEFAULT_SUPABASE_SERVICE_ROLE_KEY).strip()
    if not supabase_url or not supabase_service_role_key:
        return _fallback()

    try:
        from supabase import create_client

        supa = create_client(supabase_url, supabase_service_role_key)
        # latest ensemble per event (view created in ledger_v1.sql)
        resp = supa.table("v_latest_ensemble_forecasts")\
            .select("event_id,method,probs_json,confidence,created_at,weights_json,explain_json,run_id")\
            .order("created_at", desc=True)\
            .limit(200)\
            .execute()
        rows = resp.data or []

        # fetch event metadata for tickers/labels
        event_ids = [r["event_id"] for r in rows]
        events_map = {}
        if event_ids:
            ev_resp = supa.table("events")\
                .select("event_id,subject_id,asof_ts,horizon_json")\
                .in_("event_id", event_ids)\
                .execute()
            for ev in ev_resp.data or []:
                events_map[ev["event_id"]] = ev

        forecasts = []
        for r in rows:
            probs = r.get("probs_json") or {}
            evt = events_map.get(r["event_id"], {})
            forecasts.append({
                "event_id": r.get("event_id"),
                "ticker": evt.get("subject_id") or r.get("event_id"),
                "date": evt.get("asof_ts") or r.get("created_at"),
                "accept": probs.get("ACCEPT") or probs.get("accept"),
                "reject": probs.get("REJECT") or probs.get("reject"),
                "continue": probs.get("CONTINUE") or probs.get("continue"),
                "confidence": r.get("confidence"),
                "method": r.get("method"),
                "weights": r.get("weights_json"),
                "explain": r.get("explain_json"),
                "horizon": evt.get("horizon_json"),
                "run_id": r.get("run_id"),
            })

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json", "Cache-Control": "max-age=120"},
            "body": json.dumps({
                "forecasts": forecasts,
                "total": len(forecasts),
                "span_days": None,
                "generated_at": datetime.utcnow().isoformat()
            })
        }
    except Exception as exc:  # pragma: no cover
        # fallback so UI keeps working even if Supabase errors
        return _fallback()


class handler(FunctionHandler):
    endpoint = staticmethod(_handler_impl)
