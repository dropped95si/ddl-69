"""Run catalog endpoint.

Lists recent forecast runs observed in Supabase forecast rows.
"""

import json
import os
import re
from datetime import datetime, timezone

try:
    from _http_adapter import FunctionHandler
except ModuleNotFoundError:
    from api._http_adapter import FunctionHandler


def _safe_int(raw, default=0, minimum=0, maximum=10000):
    try:
        value = int(raw)
    except Exception:
        value = default
    value = max(minimum, value)
    value = min(maximum, value)
    return value


def _parse_iso_ts(value):
    if not value:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return datetime.fromtimestamp(0, tz=timezone.utc)


def _parse_horizon_days(raw_horizon):
    def _as_float(value):
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None

    if raw_horizon is None:
        return None

    if isinstance(raw_horizon, (int, float)):
        return _as_float(raw_horizon)

    if isinstance(raw_horizon, dict):
        direct = _as_float(raw_horizon.get("days"))
        if direct is None:
            direct = _as_float(raw_horizon.get("horizon_days"))
        if direct is not None:
            return direct

        value = _as_float(raw_horizon.get("value"))
        if value is None:
            return None
        unit = str(raw_horizon.get("unit") or "").strip().lower()
        if unit in ("", "d", "day", "days"):
            return value
        if unit in ("w", "wk", "week", "weeks"):
            return value * 7.0
        if unit in ("mo", "mon", "month", "months", "m"):
            return value * 30.0
        if unit in ("y", "yr", "year", "years"):
            return value * 365.0
        return None

    if isinstance(raw_horizon, str):
        txt = raw_horizon.strip().lower()
        if not txt:
            return None
        match = re.match(
            r"^([0-9]+(?:\.[0-9]+)?)\s*(d|day|days|w|wk|week|weeks|mo|mon|month|months|m|y|yr|year|years)?$",
            txt,
        )
        if not match:
            return None
        value = _as_float(match.group(1))
        unit = (match.group(2) or "d").strip()
        if value is None:
            return None
        if unit in ("d", "day", "days"):
            return value
        if unit in ("w", "wk", "week", "weeks"):
            return value * 7.0
        if unit in ("mo", "mon", "month", "months", "m"):
            return value * 30.0
        if unit in ("y", "yr", "year", "years"):
            return value * 365.0
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


def _chunked(values, size):
    chunk = []
    for value in values:
        chunk.append(value)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _fetch_run_catalog(limit_runs=25, lookback_rows=5000):
    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not supabase_url or not service_key:
        return None, {"error": "missing_credentials"}

    try:
        from supabase import create_client
    except Exception as exc:
        return None, {"error": f"import_error: {exc}"}

    try:
        supa = create_client(supabase_url, service_key)
        resp = (
            supa.table("v_latest_ensemble_forecasts")
            .select("event_id,run_id,method,created_at")
            .order("created_at", desc=True)
            .limit(lookback_rows)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            return None, {"error": "no_forecast_rows"}

        run_order = []
        run_seen = set()
        for row in rows:
            run_id = str(row.get("run_id") or "").strip()
            if not run_id or run_id in run_seen:
                continue
            run_seen.add(run_id)
            run_order.append(run_id)
            if len(run_order) >= limit_runs:
                break
        if not run_order:
            return None, {"error": "no_run_ids"}

        run_set = set(run_order)
        selected_rows = [row for row in rows if str(row.get("run_id") or "").strip() in run_set]
        event_ids = list({row.get("event_id") for row in selected_rows if row.get("event_id")})

        events_map = {}
        for chunk in _chunked(event_ids, 500):
            ev_resp = (
                supa.table("events")
                .select("event_id,subject_id,horizon_json,asof_ts")
                .in_("event_id", chunk)
                .execute()
            )
            for ev in ev_resp.data or []:
                event_id = ev.get("event_id")
                if event_id:
                    events_map[event_id] = ev

        stats_by_run = {
            run_id: {
                "run_id": run_id,
                "rows": 0,
                "unique_tickers": set(),
                "methods": set(),
                "timeframe_counts": {"day": 0, "swing": 0, "long": 0},
                "created_at": None,
                "asof": None,
            }
            for run_id in run_order
        }

        for row in selected_rows:
            run_id = str(row.get("run_id") or "").strip()
            if not run_id or run_id not in stats_by_run:
                continue
            stat = stats_by_run[run_id]
            stat["rows"] += 1

            method = str(row.get("method") or "").strip()
            if method:
                stat["methods"].add(method)

            created_at = row.get("created_at")
            if created_at:
                current_created = _parse_iso_ts(stat["created_at"])
                next_created = _parse_iso_ts(created_at)
                if next_created >= current_created:
                    stat["created_at"] = created_at

            event = events_map.get(row.get("event_id"), {})
            ticker = str(event.get("subject_id") or "").upper().strip()
            if ticker:
                stat["unique_tickers"].add(ticker)

            asof = event.get("asof_ts")
            if asof:
                current_asof = _parse_iso_ts(stat["asof"])
                next_asof = _parse_iso_ts(asof)
                if next_asof >= current_asof:
                    stat["asof"] = asof

            horizon_days = _parse_horizon_days(event.get("horizon_json"))
            timeframe = _classify_timeframe(horizon_days)
            stat["timeframe_counts"][timeframe] = stat["timeframe_counts"].get(timeframe, 0) + 1

        runs = []
        for run_id in run_order:
            stat = stats_by_run[run_id]
            runs.append(
                {
                    "run_id": run_id,
                    "created_at": stat["created_at"],
                    "asof": stat["asof"] or stat["created_at"],
                    "rows": stat["rows"],
                    "unique_tickers": len(stat["unique_tickers"]),
                    "methods": sorted(stat["methods"])[:8],
                    "timeframe_counts": stat["timeframe_counts"],
                }
            )

        runs_table_total = None
        try:
            runs_resp = supa.table("runs").select("run_id", count="exact").limit(1).execute()
            runs_table_total = runs_resp.count
        except Exception:
            runs_table_total = None

        return {
            "runs": runs,
            "latest_run_id": run_order[0] if run_order else None,
            "run_count": len(runs),
            "sampled_forecast_rows": len(rows),
            "runs_table_total": runs_table_total,
        }, None
    except Exception as exc:
        return None, {"error": f"exception: {exc}"}


def _handler_impl(request):
    args = request.args if hasattr(request, "args") else {}
    limit_runs = _safe_int((args.get("limit_runs") if args else None), default=25, minimum=1, maximum=100)
    lookback_rows = _safe_int((args.get("lookback_rows") if args else None), default=5000, minimum=200, maximum=10000)

    payload, error = _fetch_run_catalog(limit_runs=limit_runs, lookback_rows=lookback_rows)
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
                    "message": "Unable to load run catalog from Supabase forecasts.",
                    "details": error or {},
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
            ),
        }

    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    payload["source"] = "supabase_forecasts"
    payload["message"] = f"Loaded {payload.get('run_count', 0)} runs from forecast rows."
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Cache-Control": "max-age=60, public",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(payload),
    }


class handler(FunctionHandler):
    endpoint = staticmethod(_handler_impl)
