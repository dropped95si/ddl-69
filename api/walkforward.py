"""Walk-forward endpoint - real artifact only."""

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
        # Real market-derived proxy summary when artifact is not available.
        rows = build_watchlist("swing", 120)
        weights = {}
        for row in rows:
            for k, v in (row.get("weights_json") or {}).items():
                weights.setdefault(k, []).append(float(v))

        weight_avg = {}
        for k, vals in weights.items():
            if vals:
                weight_avg[k] = sum(vals) / len(vals)

        sorted_weights = sorted(weight_avg.items(), key=lambda kv: abs(kv[1]), reverse=True)
        pos_count = len([v for v in weight_avg.values() if v > 0])
        neg_count = len([v for v in weight_avg.values() if v < 0])
        net = sum(weight_avg.values()) if weight_avg else 0.0

        payload = {
            "run_id": "market_ta_proxy",
            "asof": datetime.now(timezone.utc).isoformat(),
            "horizon": 10,
            "top_rules": min(8, len(sorted_weights)),
            "signals_rows": len(rows),
            "weights": {k: round(v, 6) for k, v in weight_avg.items()},
            "stats": {
                "total_rules": len(weight_avg),
                "pos_count": pos_count,
                "neg_count": neg_count,
                "net_weight": round(net, 6),
                "avg_win_rate": None,
                "avg_return": None,
                "avg_score": round(sum(float(r.get("score", 0)) for r in rows) / len(rows), 6) if rows else None,
            },
            "weights_top": [
                {"rule": k, "weight": round(v, 6)} for k, v in sorted_weights[:8]
            ],
            "source": "market_ta_proxy",
            "note": "Supabase walkforward artifact unavailable; using live TA contribution aggregates.",
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
