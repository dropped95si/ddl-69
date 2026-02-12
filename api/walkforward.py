"""Walk-forward endpoint.

Primary source:
- Supabase walkforward artifact

Fallback source:
- Derived summary from latest Supabase ensemble forecasts + events
  (still real Supabase data; no synthetic/sample rows)
"""

import json
import os
from datetime import datetime, timezone
import re

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


def _parse_horizon_days(raw_horizon):
    try:
        if isinstance(raw_horizon, dict):
            days = raw_horizon.get("days") or raw_horizon.get("horizon_days")
            if days is None:
                unit = str(raw_horizon.get("unit") or "").lower().strip()
                value = raw_horizon.get("value")
                if value is not None:
                    value = float(value)
                    if unit in ("", "d", "day", "days"):
                        days = value
                    elif unit in ("w", "wk", "week", "weeks"):
                        days = value * 7.0
                    elif unit in ("mo", "mon", "month", "months", "m"):
                        days = value * 30.0
                    elif unit in ("y", "yr", "year", "years"):
                        days = value * 365.0
            if days is not None:
                return max(1, int(float(days)))
        elif isinstance(raw_horizon, (int, float)):
            return max(1, int(float(raw_horizon)))
        elif isinstance(raw_horizon, str):
            txt = raw_horizon.strip().lower()
            match = re.match(
                r"^([0-9]+(?:\.[0-9]+)?)\s*(d|day|days|w|wk|week|weeks|mo|mon|month|months|m|y|yr|year|years)?$",
                txt,
            )
            if match:
                value = float(match.group(1))
                unit = (match.group(2) or "d").strip()
                if unit in ("d", "day", "days"):
                    days = value
                elif unit in ("w", "wk", "week", "weeks"):
                    days = value * 7.0
                elif unit in ("mo", "mon", "month", "months", "m"):
                    days = value * 30.0
                elif unit in ("y", "yr", "year", "years"):
                    days = value * 365.0
                else:
                    days = value
                return max(1, int(round(days)))
    except Exception:
        return None
    return None


def _classify_timeframe(horizon_days):
    if horizon_days is None:
        return "swing"
    if horizon_days <= 30:
        return "day"
    if horizon_days <= 365:
        return "swing"
    return "long"


def _default_horizon_for_timeframe(timeframe):
    if timeframe == "day":
        return 10
    if timeframe == "long":
        return 400
    return 180


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


def _fetch_walkforward_artifact():
    supa = _get_supabase_client()
    if supa is None:
        return None

    try:
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
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _derive_from_supabase_forecasts(timeframe_filter="all"):
    supa = _get_supabase_client()
    if supa is None:
        return None

    try:
        pred_resp = (
            supa.table("v_latest_ensemble_forecasts")
            .select("event_id,weights_json,created_at,run_id")
            .order("created_at", desc=True)
            .limit(400)
            .execute()
        )
        pred_rows = pred_resp.data or []
        if not pred_rows:
            return None

        event_ids = [r.get("event_id") for r in pred_rows if r.get("event_id")]
        events_map = {}
        if event_ids:
            ev_resp = (
                supa.table("events")
                .select("event_id,horizon_json,asof_ts")
                .in_("event_id", event_ids)
                .execute()
            )
            for ev in ev_resp.data or []:
                eid = ev.get("event_id")
                if eid:
                    events_map[eid] = ev

        latest_run_id = next((r.get("run_id") for r in pred_rows if r.get("run_id")), None)
        rows = [r for r in pred_rows if (not latest_run_id or r.get("run_id") == latest_run_id)] or pred_rows

        tf_counts = {"day": 0, "swing": 0, "long": 0}
        scoped_rows = []
        for row in rows:
            ev = events_map.get(row.get("event_id"), {})
            horizon_days = _parse_horizon_days(ev.get("horizon_json"))
            tf = _classify_timeframe(horizon_days)
            tf_counts[tf] = tf_counts.get(tf, 0) + 1
            if timeframe_filter != "all" and tf != timeframe_filter:
                continue
            row_copy = dict(row)
            row_copy["_wf_horizon_days"] = horizon_days
            row_copy["_wf_asof_ts"] = ev.get("asof_ts")
            scoped_rows.append(row_copy)

        rows = scoped_rows
        if not rows:
            if timeframe_filter != "all":
                fallback_asof = pred_rows[0].get("created_at") or datetime.now(timezone.utc).isoformat()
                return {
                    "summary": {
                        "run_id": latest_run_id or "derived_supabase",
                        "asof": fallback_asof,
                        "horizon": _default_horizon_for_timeframe(timeframe_filter),
                        "top_rules": 0,
                        "signals_rows": 0,
                        "weights": {},
                        "weights_top": [],
                        "stats": {
                            "total_rules": 0,
                            "pos_count": 0,
                            "neg_count": 0,
                            "net_weight": 0.0,
                            "avg_win_rate": None,
                            "avg_return": None,
                        },
                        "source": "supabase_forecasts_derived",
                        "timeframe": timeframe_filter,
                        "timeframe_counts": tf_counts,
                        "note": f"No rows available for timeframe '{timeframe_filter}' in latest Supabase run.",
                    }
                }
            return None

        weights_sum = {}
        weights_count = {}
        horizon_days_values = []
        asof_candidates = []
        created_candidates = []

        for row in rows:
            created_at = row.get("created_at")
            if created_at:
                created_candidates.append(created_at)

            asof_ts = row.get("_wf_asof_ts")
            if asof_ts:
                asof_candidates.append(asof_ts)
            days = row.get("_wf_horizon_days")
            if days is not None:
                horizon_days_values.append(days)

            weights = row.get("weights_json") or {}
            if not isinstance(weights, dict):
                continue
            for rule, value in weights.items():
                w = _safe_float(value, None)
                if w is None:
                    continue
                weights_sum[rule] = weights_sum.get(rule, 0.0) + w
                weights_count[rule] = weights_count.get(rule, 0) + 1

        if not weights_sum:
            return None

        mean_weights = {
            rule: (total / max(1, int(weights_count.get(rule, 1))))
            for rule, total in weights_sum.items()
        }
        top = sorted(
            [{"rule": k, "weight": round(v, 6)} for k, v in mean_weights.items()],
            key=lambda x: abs(x["weight"]),
            reverse=True,
        )

        horizon_days = (
            int(round(sum(horizon_days_values) / len(horizon_days_values)))
            if horizon_days_values
            else (_default_horizon_for_timeframe(timeframe_filter) if timeframe_filter != "all" else None)
        )
        asof = (
            asof_candidates[0]
            if asof_candidates
            else (
                created_candidates[0]
                if created_candidates
                else datetime.now(timezone.utc).isoformat()
            )
        )

        return {
            "summary": {
                "run_id": latest_run_id or "derived_supabase",
                "asof": asof,
                "horizon": horizon_days,
                "top_rules": min(8, len(top)),
                "signals_rows": len(rows),
                "weights": {k: round(v, 6) for k, v in mean_weights.items()},
                "weights_top": top[:8],
                "stats": {
                    "total_rules": len(mean_weights),
                    "pos_count": len([w for w in mean_weights.values() if w > 0]),
                    "neg_count": len([w for w in mean_weights.values() if w < 0]),
                    "net_weight": round(float(sum(mean_weights.values())), 6),
                    "avg_win_rate": None,
                    "avg_return": None,
                },
                "source": "supabase_forecasts_derived",
                "timeframe": timeframe_filter,
                "timeframe_counts": tf_counts,
                "note": (
                    "Walk-forward artifact unavailable; derived from latest Supabase ensemble weights."
                    if timeframe_filter == "all"
                    else f"Walk-forward artifact unavailable; derived from Supabase ensemble weights ({timeframe_filter})."
                ),
            }
        }
    except Exception:
        return None


def _handler_impl(request):
    args = request.args if hasattr(request, "args") else {}
    timeframe = str((args.get("timeframe") if args else "") or "all").strip().lower()
    if timeframe not in ("all", "day", "swing", "long"):
        timeframe = "all"

    allow_derived_arg = str((args.get("allow_derived") if args else "") or "").strip().lower()
    allow_derived = allow_derived_arg in ("1", "true", "yes", "on")
    if not allow_derived:
        allow_derived = str(os.getenv("WALKFORWARD_ALLOW_DERIVED", "0")).strip().lower() in ("1", "true", "yes", "on")

    payload = _fetch_walkforward_artifact()
    if payload is None and allow_derived:
        payload = _derive_from_supabase_forecasts(timeframe_filter=timeframe)
    if payload is None:
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
                    "message": (
                        "Supabase walk-forward artifact required; no fallback enabled."
                        if not allow_derived
                        else (
                            "Supabase walk-forward artifact and derived forecast aggregates are unavailable."
                            if timeframe == "all"
                            else f"Supabase walk-forward artifact and derived forecast aggregates are unavailable for timeframe '{timeframe}'."
                        )
                    ),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
            ),
        }

    summary = payload.get("summary") if isinstance(payload, dict) else None
    if isinstance(summary, dict):
        if allow_derived:
            summary.setdefault("timeframe", timeframe)
        else:
            # Artifact is run-level and not guaranteed to be timeframe-scoped.
            summary.setdefault("timeframe", "all")
            if timeframe != "all":
                note = str(summary.get("note") or "").strip()
                scope_note = f"Requested timeframe '{timeframe}' uses run-level artifact (global scope)."
                summary["note"] = f"{note} {scope_note}".strip() if note else scope_note

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
