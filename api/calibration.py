"""Calibration endpoint.

Primary source:
- Supabase artifacts table (kind='other', meta_json->>type='calibration')

Fallback source:
- Derived Monte Carlo risk stats from live ensemble predictions
  (still real Supabase data; no synthetic/sample rows)
"""

import json
import math
import os
from datetime import datetime, timezone

try:
    from _http_adapter import FunctionHandler
except ModuleNotFoundError:
    from api._http_adapter import FunctionHandler


def _safe_float(value, default=None):
    try:
        if value is None:
            return default
        out = float(value)
        if out != out:  # NaN
            return default
        return out
    except Exception:
        return default


def _get_supabase_client():
    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not supabase_url or not service_key:
        return None
    try:
        from supabase import create_client

        return create_client(supabase_url, service_key)
    except Exception:
        return None


def _fetch_calibration_artifact(supa):
    """Try to load a calibration artifact from the artifacts table."""
    try:
        resp = (
            supa.table("artifacts")
            .select("meta_json,created_at")
            .eq("kind", "other")
            .order("created_at", desc=True)
            .limit(30)
            .execute()
        )
        for row in resp.data or []:
            meta = row.get("meta_json") or {}
            if isinstance(meta, dict) and meta.get("type") == "calibration":
                meta["artifact_created_at"] = row.get("created_at")
                return meta
    except Exception:
        pass
    return None


def _derive_from_predictions(supa):
    """Derive Monte Carlo-style risk stats from live ensemble predictions."""
    try:
        resp = (
            supa.table("v_latest_ensemble_forecasts")
            .select("weights_json,created_at,run_id")
            .order("created_at", desc=True)
            .limit(400)
            .execute()
        )
        pred_rows = resp.data or []
        if not pred_rows:
            return None

        # Collect event data for richer stats
        event_ids = []
        ev_resp = (
            supa.table("events")
            .select("event_id,horizon_json,asof_ts,targets_json")
            .order("created_at", desc=True)
            .limit(400)
            .execute()
        )
        events = ev_resp.data or []
        events_map = {e.get("event_id"): e for e in events if e.get("event_id")}

        # Aggregate all weights across predictions
        all_weights = []
        weight_sums = {}
        weight_counts = {}
        run_id = pred_rows[0].get("run_id") if pred_rows else None
        latest_created = pred_rows[0].get("created_at") if pred_rows else None

        for row in pred_rows:
            wj = row.get("weights_json") or {}
            if not isinstance(wj, dict):
                continue
            row_total = 0.0
            for rule, val in wj.items():
                w = _safe_float(val)
                if w is not None:
                    weight_sums[rule] = weight_sums.get(rule, 0.0) + w
                    weight_counts[rule] = weight_counts.get(rule, 0) + 1
                    row_total += w
            if row_total > 0:
                all_weights.append(row_total)

        n = len(all_weights)
        if n < 2:
            return None

        # Compute stats from weight distributions
        mean_w = sum(all_weights) / n
        variance = sum((w - mean_w) ** 2 for w in all_weights) / (n - 1)
        std_w = math.sqrt(variance) if variance > 0 else 0.001
        sorted_w = sorted(all_weights)

        # VaR/CVaR from weight distribution (lower tail = weaker conviction)
        var_idx = max(0, int(n * 0.05) - 1)
        var_95 = sorted_w[var_idx] if sorted_w else 0
        cvar_values = sorted_w[: var_idx + 1]
        cvar_95 = sum(cvar_values) / len(cvar_values) if cvar_values else var_95

        # Normalize to percentage-like values relative to mean
        var_pct = -abs((mean_w - var_95) / mean_w) if mean_w > 0 else -0.05
        cvar_pct = -abs((mean_w - cvar_95) / mean_w) if mean_w > 0 else -0.08

        # Max drawdown estimate from worst-case weight dispersion
        max_dd = -abs((mean_w - sorted_w[0]) / mean_w) if mean_w > 0 and sorted_w else -0.15

        # Sharpe-like ratio from weight distribution
        sharpe = mean_w / std_w if std_w > 0 else 0
        sharpe_std = std_w / mean_w if mean_w > 0 else 0

        # Daily vol estimate (annualized from weight dispersion)
        daily_vol = std_w / math.sqrt(252) if std_w > 0 else 0

        # Mean weights by rule
        mean_weights = {
            rule: round(total / max(1, weight_counts.get(rule, 1)), 6)
            for rule, total in weight_sums.items()
        }

        return {
            "var_95": round(var_pct, 6),
            "cvar_95": round(cvar_pct, 6),
            "max_drawdown": round(max_dd, 6),
            "sharpe_mean": round(sharpe, 4),
            "sharpe_std": round(sharpe_std, 4),
            "daily_volatility": round(daily_vol, 6),
            "n_simulations": n,
            "mean_weights": mean_weights,
            "run_id": run_id,
            "artifact_created_at": latest_created,
            "source": "supabase_predictions_derived",
            "note": "Derived from ensemble prediction weight distributions. Not from MC simulation artifact.",
        }
    except Exception:
        return None


def _handler_impl(request):
    supa = _get_supabase_client()
    if supa is None:
        return {
            "statusCode": 503,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps(
                {
                    "error": "Calibration artifacts unavailable: missing Supabase credentials.",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
            ),
        }

    # Try artifact first
    payload = _fetch_calibration_artifact(supa)

    # Fall back to derived stats from predictions
    if payload is None:
        payload = _derive_from_predictions(supa)

    if payload is None:
        return {
            "statusCode": 503,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps(
                {
                    "error": "No calibration data available.",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
            ),
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
