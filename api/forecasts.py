"""Forecast stream endpoint.

Returns:
1) Supabase ensemble forecast rows when available
2) Real Yahoo screener + TA derived rows as fallback

Never returns deterministic sample generators.
"""

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


def _market_rows():
    rows = []
    rows.extend(build_watchlist("day", 80))
    rows.extend(build_watchlist("swing", 120))
    rows.extend(build_watchlist("long", 80))
    dedup = {}
    for r in rows:
        sym = r.get("symbol")
        if not sym:
            continue
        if sym not in dedup or float(r.get("score", 0)) > float(dedup[sym].get("score", 0)):
            dedup[sym] = r
    out = []
    for r in dedup.values():
        out.append(
            {
                "event_id": f"mkt:{r.get('symbol')}",
                "ticker": r.get("symbol"),
                "date": r.get("created_at"),
                "accept": r.get("p_accept"),
                "reject": r.get("p_reject"),
                "continue": r.get("p_continue") or 0.0,
                "confidence": r.get("confidence"),
                "method": "yahoo_screener_ta",
                "weights": r.get("weights_json") or {},
                "explain": {"meta": r.get("meta") or {}},
                "horizon": {"label": r.get("plan_type"), "days": (r.get("meta") or {}).get("horizon")},
                "run_id": None,
            }
        )
    out.sort(key=lambda x: float(x.get("accept") or 0), reverse=True)
    return out[:220]


def _handler_impl(request):
    rows = _supabase_rows()
    source = "supabase"
    if not rows:
        rows = _market_rows()
        source = "market_ta"

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
                "source": source,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        ),
    }


class handler(FunctionHandler):
    endpoint = staticmethod(_handler_impl)
