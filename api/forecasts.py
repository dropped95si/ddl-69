"""Forecast stream endpoint (Supabase-only, no fallback)."""

import json
import os
from datetime import datetime, timezone

try:
    from _http_adapter import FunctionHandler
except ModuleNotFoundError:
    from api._http_adapter import FunctionHandler

def _supabase_rows():
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
            supa.table("v_latest_ensemble_forecasts")
            .select("event_id,method,probs_json,confidence,created_at,weights_json,explain_json,run_id")
            .order("created_at", desc=True)
            .limit(400)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            return None

        event_ids = [r["event_id"] for r in rows if r.get("event_id")]
        events_map = {}
        if event_ids:
            ev_resp = (
                supa.table("events")
                .select("event_id,subject_id,asof_ts,horizon_json")
                .in_("event_id", event_ids)
                .execute()
            )
            for ev in ev_resp.data or []:
                events_map[ev["event_id"]] = ev

        out = []
        seen = set()
        for r in rows:
            evt = events_map.get(r.get("event_id"), {})
            ticker = str(evt.get("subject_id") or "").upper().strip()
            if not ticker or ticker in seen:
                continue
            seen.add(ticker)

            probs = r.get("probs_json") or {}
            accept_raw = (
                r.get("p_accept")
                or probs.get("ACCEPT_CONTINUE")
                or probs.get("ACCEPT")
                or probs.get("accept")
            )
            if accept_raw is None:
                continue
            accept = float(accept_raw)
            reject = float(
                r.get("p_reject")
                or probs.get("REJECT")
                or probs.get("reject")
                or (1 - accept)
            )
            cont = float(
                r.get("p_continue")
                or probs.get("BREAK_FAIL")
                or probs.get("CONTINUE")
                or probs.get("continue")
                or 0.0
            )

            out.append(
                {
                    "event_id": r.get("event_id"),
                    "ticker": ticker,
                    "date": evt.get("asof_ts") or r.get("created_at"),
                    "accept": round(accept, 6),
                    "reject": round(reject, 6),
                    "continue": round(cont, 6),
                    "confidence": round(float(r.get("confidence") or 0.5), 6),
                    "method": r.get("method") or "supabase",
                    "weights": r.get("weights_json") or {},
                    "explain": r.get("explain_json") or {},
                    "horizon": evt.get("horizon_json"),
                    "run_id": r.get("run_id"),
                }
            )
        if not out:
            return None
        return out
    except Exception:
        return None


def _handler_impl(request):
    rows = _supabase_rows()
    if not rows:
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
                    "message": "Supabase forecasts required; no fallback enabled.",
                    "forecasts": [],
                    "total": 0,
                    "source": "none",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
            ),
        }

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Cache-Control": "max-age=90, public",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(
            {
                "forecasts": rows,
                "total": len(rows),
                "span_days": None,
                "source": "supabase",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        ),
    }


class handler(FunctionHandler):
    endpoint = staticmethod(_handler_impl)
