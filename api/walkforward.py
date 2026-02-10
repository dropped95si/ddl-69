import json
from datetime import datetime, timezone

from ddl69.core.settings import Settings
from supabase import create_client


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


def handler(request):
    """Return walk-forward summary from Supabase (falls back to sample data)."""
    settings = Settings.from_env()
    if not settings.supabase_url or not settings.supabase_service_role_key:
        return _fallback()

    try:
        supa = create_client(settings.supabase_url, settings.supabase_service_role_key)
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
