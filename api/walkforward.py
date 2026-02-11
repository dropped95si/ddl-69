import json
from datetime import datetime, timezone

try:
    from _http_adapter import FunctionHandler
except ModuleNotFoundError:
    from api._http_adapter import FunctionHandler

DEFAULT_SUPABASE_URL = ""
DEFAULT_SUPABASE_SERVICE_ROLE_KEY = ""


def _fallback():
    """Return sample walk-forward data when Supabase is not configured."""
    wf = {
        "run_id": "sample_run",
        "asof": datetime.utcnow().isoformat(),
        "horizon": 5,
        "top_rules": 8,
        "signals_rows": 42,
        "weights": {
            "SMA_CROSS_5_20": 0.15,
            "RSI_OVERSOLD": 0.12,
            "EMA_CROSS_12_26": 0.11,
            "VOL_SPIKE": 0.10
        },
        "stats": {
            "total_rules": 24,
            "pos_count": 12,
            "neg_count": 12,
            "net_weight": 0.03,
            "avg_win_rate": 0.52,
            "avg_return": 0.003,
            "avg_score": 0.68
        },
        "weights_top": [
            {"rule": "SMA_CROSS_5_20", "weight": 0.15},
            {"rule": "RSI_OVERSOLD", "weight": 0.12},
            {"rule": "EMA_CROSS_12_26", "weight": 0.11},
            {"rule": "VOL_SPIKE", "weight": 0.10}
        ]
    }

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json", "Cache-Control": "max-age=300"},
        "body": json.dumps(wf)
    }


def _handler_impl(request):
    """Return walk-forward summary from Supabase (falls back to sample data)."""
    import os

    supabase_url = os.getenv("SUPABASE_URL", DEFAULT_SUPABASE_URL).strip()
    supabase_service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", DEFAULT_SUPABASE_SERVICE_ROLE_KEY).strip()
    if not supabase_url or not supabase_service_role_key:
        return _fallback()

    try:
        from supabase import create_client

        supa = create_client(supabase_url, supabase_service_role_key)
        # Query artifacts table for latest walk-forward
        resp = supa.table("artifacts")\
            .select("payload")\
            .eq("artifact_type", "walkforward")\
            .order("created_at", desc=True)\
            .limit(1)\
            .execute()

        if resp.data and len(resp.data) > 0:
            wf = resp.data[0].get("payload", {})
            return {
                "statusCode": 200,
                "headers": {"Content-Type": "application/json", "Cache-Control": "max-age=120"},
                "body": json.dumps(wf)
            }
        else:
            return _fallback()
    except Exception as exc:
        return _fallback()


class handler(FunctionHandler):
    endpoint = staticmethod(_handler_impl)
