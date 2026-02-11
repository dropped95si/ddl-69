import json
import os
from datetime import datetime, timezone

try:
    from _http_adapter import FunctionHandler
except ModuleNotFoundError:
    from api._http_adapter import FunctionHandler

DEFAULT_SUPABASE_URL = ""
DEFAULT_SUPABASE_SERVICE_ROLE_KEY = ""


def _fallback():
    """Return deterministic sample data when Supabase is not configured."""
    watchlist = [
        {"ticker": "NVDA", "symbol": "NVDA", "label": "BUY", "score": 0.87, "p_accept": 0.78, "signal": "BUY", "weights": {"technical": 0.35, "sentiment": 0.25, "fundamental": 0.25, "ensemble": 0.15}, "weights_json": {"technical": 0.35, "sentiment": 0.25, "fundamental": 0.25, "ensemble": 0.15}},
        {"ticker": "TSLA", "symbol": "TSLA", "label": "HOLD", "score": 0.62, "p_accept": 0.58, "signal": "HOLD", "weights": {"technical": 0.30, "sentiment": 0.25, "fundamental": 0.30, "ensemble": 0.15}, "weights_json": {"technical": 0.30, "sentiment": 0.25, "fundamental": 0.30, "ensemble": 0.15}},
        {"ticker": "SPY", "symbol": "SPY", "label": "HOLD", "score": 0.55, "p_accept": 0.52, "signal": "HOLD", "weights": {"technical": 0.30, "sentiment": 0.25, "fundamental": 0.30, "ensemble": 0.15}, "weights_json": {"technical": 0.30, "sentiment": 0.25, "fundamental": 0.30, "ensemble": 0.15}},
    ]
    
    buy_count = len([w for w in watchlist if w.get("signal") == "BUY"])
    sell_count = len([w for w in watchlist if w.get("signal") == "SELL"])
    hold_count = len([w for w in watchlist if w.get("signal") == "HOLD"])
    
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json", "Cache-Control": "max-age=60"},
        "body": json.dumps({
            "asof": datetime.now(timezone.utc).isoformat(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "Demo Data",
            "provider": "DDL-69 ML Pipeline",
            "count": len(watchlist),
            "ranked": watchlist,
            "tickers": [w.get("ticker") for w in watchlist],
            "stats": {
                "total": len(watchlist),
                "buy_count": buy_count,
                "sell_count": sell_count,
                "hold_count": hold_count,
            },
            "message": f"✅ Loaded {len(watchlist)} predictions",
        })
    }


def _handler_impl(request):
    """Fetch watchlist from Supabase ensemble forecasts."""
    supabase_url = os.getenv("SUPABASE_URL", DEFAULT_SUPABASE_URL).strip()
    supabase_service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", DEFAULT_SUPABASE_SERVICE_ROLE_KEY).strip()
    
    if not supabase_url or not supabase_service_role_key:
        return _fallback()

    try:
        from supabase import create_client

        supa = create_client(supabase_url, supabase_service_role_key)
        resp = supa.table("v_latest_ensemble_forecasts")\
            .select("event_id,method,probs_json,confidence,created_at,weights_json,explain_json,run_id")\
            .order("created_at", desc=True)\
            .limit(100)\
            .execute()
        rows = resp.data or []

        event_ids = [r["event_id"] for r in rows]
        events_map = {}
        if event_ids:
            ev_resp = supa.table("events")\
                .select("event_id,subject_id,asof_ts,horizon_json")\
                .in_("event_id", event_ids)\
                .execute()
            for ev in ev_resp.data or []:
                events_map[ev["event_id"]] = ev

        watchlist = []
        for r in rows:
            probs = r.get("probs_json") or {}
            evt = events_map.get(r["event_id"], {})
            
            accept_prob = probs.get("ACCEPT") or probs.get("accept") or 0.5
            reject_prob = probs.get("REJECT") or probs.get("reject") or 0.3
            
            signal = "BUY" if accept_prob > 0.6 else ("SELL" if reject_prob > 0.5 else "HOLD")
            score = max(accept_prob, reject_prob, 0.5)
            
            watchlist.append({
                "ticker": evt.get("subject_id") or r.get("event_id"),
                "symbol": evt.get("subject_id") or r.get("event_id"),
                "label": signal,
                "score": round(score, 2),
                "p_accept": round(accept_prob, 2),
                "signal": signal,
                "weights": r.get("weights_json") or {},
                "weights_json": r.get("weights_json") or {},
                "confidence": round(r.get("confidence", 0.5), 2),
                "method": r.get("method"),
                "created_at": r.get("created_at"),
                "run_id": r.get("run_id"),
            })

        watchlist = sorted(watchlist, key=lambda x: x.get("score", 0), reverse=True)
        
        buy_count = len([w for w in watchlist if w.get("signal") == "BUY"])
        sell_count = len([w for w in watchlist if w.get("signal") == "SELL"])
        hold_count = len([w for w in watchlist if w.get("signal") == "HOLD"])
        
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json", "Cache-Control": "max-age=60"},
            "body": json.dumps({
                "asof": datetime.now(timezone.utc).isoformat(),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "source": "Supabase Ledger",
                "provider": "DDL-69 ML Pipeline",
                "count": len(watchlist),
                "ranked": watchlist,
                "tickers": [w.get("ticker") for w in watchlist],
                "stats": {
                    "total": len(watchlist),
                    "buy_count": buy_count,
                    "sell_count": sell_count,
                    "hold_count": hold_count,
                },
                "message": f"✅ Loaded {len(watchlist)} predictions",
            })
        }
    except Exception:
        return _fallback()


class handler(FunctionHandler):
    endpoint = staticmethod(_handler_impl)
