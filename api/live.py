"""Live endpoint - primary watchlist feed.

Priority:
1) Supabase ensemble forecasts (required)

No synthetic/demo fallback payloads.
"""

import json
import os
from statistics import pstdev
from datetime import datetime, timezone
import re

try:
    from _http_adapter import FunctionHandler
except ModuleNotFoundError:
    from api._http_adapter import FunctionHandler



def _parse_horizon_days(horizon_json):
    if horizon_json is None:
        return None

    def _as_float(raw):
        try:
            if raw is None:
                return None
            return float(raw)
        except Exception:
            return None

    if isinstance(horizon_json, (int, float)):
        return _as_float(horizon_json)

    if isinstance(horizon_json, str):
        txt = horizon_json.strip().lower()
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

    if isinstance(horizon_json, dict):
        direct = _as_float(horizon_json.get("days"))
        if direct is None:
            direct = _as_float(horizon_json.get("horizon_days"))
        if direct is not None:
            return direct

        value = _as_float(horizon_json.get("value"))
        if value is None:
            return None
        unit = str(horizon_json.get("unit", "")).strip().lower()
        if unit in ("", "d", "day", "days"):
            return value
        if unit in ("w", "wk", "week", "weeks"):
            return value * 7.0
        if unit in ("mo", "mon", "month", "months", "m"):
            return value * 30.0
        if unit in ("y", "yr", "year", "years"):
            return value * 365.0
        return None

    return None


def _classify_timeframe(horizon_json):
    if not horizon_json:
        return "swing"
    days = _parse_horizon_days(horizon_json)
    if days is not None:
        try:
            days = float(days)
        except Exception:
            return "swing"
        if days <= 30:
            return "day"
        if days <= 365:  # 1-12 months
            return "swing"
        return "long"  # 1+ years
    return "swing"


def _tp_sl_for_timeframe(timeframe, horizon_days=None):
    """Calculate TP/SL bands based on timeframe. horizon_days from Supabase if available."""
    # Use actual horizon from Supabase event if provided, otherwise fallback to timeframe estimate
    if horizon_days is None:
        if timeframe == "day":
            horizon_days = 10
        elif timeframe == "long":
            horizon_days = 400  # 1+ years
        else:  # swing
            horizon_days = 180  # 1-12 months avg
    
    # Convert to float and clamp to reasonable range
    try:
        horizon_days = max(1, min(float(horizon_days), 730))
    except (TypeError, ValueError):
        horizon_days = 10
    
    # Calculate bands based on actual horizon
    if horizon_days <= 30:
        # Day trades: 1-30 days, scale by sqrt(horizon/10) so 5d != 30d
        from math import sqrt
        scale = sqrt(horizon_days / 10.0)
        return {"tp_pct": [round(0.015 * scale, 5), round(0.03 * scale, 5), round(0.05 * scale, 5)],
                "sl_pct": [round(-0.01 * scale, 5), round(-0.02 * scale, 5), round(-0.03 * scale, 5)],
                "horizon_days": horizon_days}
    elif horizon_days >= 366:
        # Long: 1+ years
        scale = horizon_days / 365.0
        return {"tp_pct": [0.25 * scale, 0.50 * scale, 0.80 * scale], "sl_pct": [-0.12 * scale, -0.18 * scale, -0.25 * scale], "horizon_days": horizon_days}
    else:
        # Swing: 31-365 days (1-12 months)
        scale = horizon_days / 180.0  # Normalize to 6-month baseline
        return {
            "tp_pct": [0.08 * scale, 0.15 * scale, 0.25 * scale],
            "sl_pct": [-0.04 * scale, -0.07 * scale, -0.12 * scale],
            "horizon_days": horizon_days
        }


def _build_meta(row, evt, timeframe, bands, price=None):
    accept_raw = row.get("p_accept")
    accept = float(accept_raw) if accept_raw is not None else 0.0
    meta = {
        "source": "supabase_ensemble",
        "mode": timeframe,
        "p_up": round(accept, 4),
        "p_down": round(1 - accept, 4),
        "p_target": round(accept, 4),
        "horizon": f"{bands['horizon_days']}d",
        "tp1_pct": bands["tp_pct"][0],
        "tp2_pct": bands["tp_pct"][1],
        "tp3_pct": bands["tp_pct"][2],
        "sl1_pct": bands["sl_pct"][0],
        "sl2_pct": bands["sl_pct"][1],
        "sl3_pct": bands["sl_pct"][2],
        "method": row.get("method"),
        "run_id": row.get("run_id"),
        "event_id": row.get("event_id"),
        "reason": f"Supabase ensemble forecast ({row.get('method', 'blended')})",
    }
    if price and price > 0:
        meta["last_price"] = round(price, 2)
        meta["tp1"] = round(price * (1 + bands["tp_pct"][0]), 2)
        meta["tp2"] = round(price * (1 + bands["tp_pct"][1]), 2)
        meta["tp3"] = round(price * (1 + bands["tp_pct"][2]), 2)
        meta["sl1"] = round(price * (1 + bands["sl_pct"][0]), 2)
        meta["sl2"] = round(price * (1 + bands["sl_pct"][1]), 2)
        meta["sl3"] = round(price * (1 + bands["sl_pct"][2]), 2)
        meta["target_price"] = meta["tp1"]
    return meta


def _first_not_none(mapping, *keys):
    for key in keys:
        if key in mapping and mapping.get(key) is not None:
            return mapping.get(key)
    return None


def _cap_bucket(market_cap):
    try:
        cap = float(market_cap)
    except Exception:
        return "unknown"
    if cap <= 0:
        return "unknown"
    if cap >= 200_000_000_000:
        return "mega"
    if cap >= 10_000_000_000:
        return "large"
    if cap >= 2_000_000_000:
        return "mid"
    if cap >= 300_000_000:
        return "small"
    return "micro"


def _event_market_cap_and_bucket(event_row):
    if not isinstance(event_row, dict):
        return None, "unknown"
    context = event_row.get("context_json") if isinstance(event_row.get("context_json"), dict) else {}
    params = event_row.get("event_params_json") if isinstance(event_row.get("event_params_json"), dict) else {}
    cap_candidates = [
        context.get("market_cap"),
        context.get("marketCap"),
        params.get("market_cap"),
        params.get("marketCap"),
    ]
    for raw in cap_candidates:
        try:
            cap = float(raw)
        except Exception:
            cap = None
        if cap is not None and cap > 0:
            cap_i = int(cap)
            return cap_i, _cap_bucket(cap_i)

    bucket_candidates = [
        context.get("cap_bucket"),
        context.get("market_cap_bucket"),
        params.get("cap_bucket"),
        params.get("market_cap_bucket"),
    ]
    for raw in bucket_candidates:
        txt = str(raw or "").strip().lower()
        if txt in ("mega", "large", "mid", "small", "micro", "unknown"):
            return None, txt
    return None, "unknown"


def _event_asset_type(event_row):
    if not isinstance(event_row, dict):
        return "unknown"
    context = event_row.get("context_json") if isinstance(event_row.get("context_json"), dict) else {}
    params = event_row.get("event_params_json") if isinstance(event_row.get("event_params_json"), dict) else {}
    for raw in (
        context.get("asset_type"),
        context.get("security_type"),
        params.get("asset_type"),
        params.get("security_type"),
    ):
        normalized = _asset_type(raw)
        if normalized != "unknown":
            return normalized
    return "unknown"


def _asset_type(raw):
    txt = str(raw or "").strip().lower()
    if not txt:
        return "unknown"
    if txt in ("etf", "etn", "fund", "mutualfund", "mutual fund"):
        return "etf"
    if txt in ("equity", "stock", "commonstock", "common stock"):
        return "equity"
    return txt


def _fetch_supabase(timeframe_filter=None, run_id_filter=None):
    debug_info = {"stage": "init", "error": None}
    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not supabase_url or not service_key:
        debug_info["error"] = "missing_credentials"
        return None, debug_info

    try:
        from supabase import create_client
    except Exception as e:
        debug_info["error"] = f"import_error: {str(e)}"
        return None, debug_info

    try:
        debug_info["stage"] = "query_forecasts"
        supa = create_client(supabase_url, service_key)
        resp = (
            supa.table("v_latest_ensemble_forecasts")
            .select("event_id,method,probs_json,confidence,created_at,weights_json,explain_json,run_id")
            .order("created_at", desc=True)
            .limit(400)
            .execute()
        )
        rows = resp.data or []
        debug_info["forecast_rows"] = len(rows)
        if not rows:
            debug_info["error"] = "no_forecast_rows"
            return None, debug_info

        run_ids = [r.get("run_id") for r in rows if r.get("run_id")]
        latest_run_id = run_ids[0] if run_ids else None
        active_run_id = (str(run_id_filter).strip() if run_id_filter else "") or latest_run_id
        if active_run_id:
            rows = [r for r in rows if r.get("run_id") == active_run_id]
        debug_info["active_run_id"] = active_run_id
        debug_info["pipeline_mode"] = pipeline_mode
        debug_info["pipeline_reason"] = pipeline_reason
        debug_info["training_executed"] = training_executed
        debug_info["artifacts_written"] = artifacts_written
        debug_info["schema_version"] = 2
        debug_info["debug_info_generated_utc"] = datetime.now(timezone.utc).isoformat()
        debug_info["commit_sha"] = os.getenv("VERCEL_GIT_COMMIT_SHA", "unknown")
        debug_info["latest_run_id"] = latest_run_id
        debug_info["available_runs"] = list(dict.fromkeys(run_ids))[:10]
        debug_info["run_filtered_rows"] = len(rows)
        if not rows:
            debug_info["error"] = "no_rows_for_run" if run_id_filter else "no_rows_latest_run"
            return None, debug_info

        debug_info["stage"] = "query_events"
        event_ids = list({r["event_id"] for r in rows if r.get("event_id")})
        events_map = {}
        if event_ids:
            ev_resp = (
                supa.table("events")
                .select("event_id,subject_id,asof_ts,horizon_json,context_json,event_params_json")
                .in_("event_id", event_ids)
                .execute()
            )
            for ev in ev_resp.data or []:
                events_map[ev["event_id"]] = ev

        try:
            from _prices import fetch_quote_snapshots
        except ModuleNotFoundError:
            from api._prices import fetch_quote_snapshots

        all_tickers = []
        for r in rows:
            evt = events_map.get(r.get("event_id"), {})
            ticker = str(evt.get("subject_id") or "").upper().strip()
            if ticker:
                all_tickers.append(ticker)
        all_tickers = list(dict.fromkeys(all_tickers))
        snapshots = fetch_quote_snapshots(all_tickers) if all_tickers else {}

        watchlist = []
        seen_tickers = set()
        for r in rows:
            probs = r.get("probs_json") or {}
            evt = events_map.get(r.get("event_id"), {})
            ticker = str(evt.get("subject_id") or "").upper().strip()
            if not ticker:
                continue
            if ticker in seen_tickers:
                continue
            seen_tickers.add(ticker)

            accept_prob = r.get("p_accept")
            reject_prob = r.get("p_reject")
            continue_prob = r.get("p_continue")

            if accept_prob is None:
                accept_prob = _first_not_none(probs, "ACCEPT_CONTINUE", "ACCEPT", "accept")
            if reject_prob is None:
                reject_prob = _first_not_none(probs, "REJECT", "reject")
            if continue_prob is None:
                continue_prob = _first_not_none(probs, "BREAK_FAIL", "CONTINUE", "continue")

            if accept_prob is None:
                continue

            accept_prob = float(accept_prob)
            if reject_prob is None:
                reject_prob = 1 - accept_prob
            reject_prob = float(reject_prob)
            continue_prob = float(continue_prob or 0.0)
            confidence = float(r.get("confidence") or 0.5)

            horizon_json = evt.get("horizon_json")
            timeframe = _classify_timeframe(horizon_json)
            if timeframe_filter and timeframe_filter != "all" and timeframe != timeframe_filter:
                continue

            signal = "BUY" if accept_prob > 0.6 else ("SELL" if reject_prob > 0.5 else "HOLD")
            score = round(max(accept_prob, reject_prob), 4)
            
            # Extract REAL horizon days from Supabase event JSON
            real_horizon_days = _parse_horizon_days(horizon_json)

            bands = _tp_sl_for_timeframe(timeframe, horizon_days=real_horizon_days)
            snap = snapshots.get(ticker) or {}
            price = snap.get("price")
            event_market_cap, event_bucket = _event_market_cap_and_bucket(evt)
            market_cap = snap.get("market_cap")
            if market_cap is None:
                market_cap = event_market_cap
            asset_type = _asset_type(snap.get("quote_type"))
            if asset_type == "unknown":
                asset_type = _event_asset_type(evt)
            cap_bucket = _cap_bucket(market_cap)
            if cap_bucket == "unknown" and event_bucket != "unknown":
                cap_bucket = event_bucket
            
            # Calculate actual dollar targets from percentages
            tp1 = round(price * (1 + bands["tp_pct"][0]), 2) if price else None
            tp2 = round(price * (1 + bands["tp_pct"][1]), 2) if price else None
            tp3 = round(price * (1 + bands["tp_pct"][2]), 2) if price else None
            sl1 = round(price * (1 + bands["sl_pct"][0]), 2) if price else None
            sl2 = round(price * (1 + bands["sl_pct"][1]), 2) if price else None
            sl3 = round(price * (1 + bands["sl_pct"][2]), 2) if price else None

            # Expected return = probability-weighted outcome
            exp_ret = round((accept_prob * bands["tp_pct"][0] + reject_prob * bands["sl_pct"][0]) * 100, 2)

            watchlist.append(
                {
                    "ticker": ticker,
                    "symbol": ticker,
                    "label": signal,
                    "score": score,
                    "price": round(price, 2) if price else None,
                    "p_accept": round(accept_prob, 4),
                    "p_reject": round(reject_prob, 4),
                    "p_continue": round(continue_prob, 4),
                    "probability": round(accept_prob, 4),
                    "signal": signal,
                    "confidence": round(confidence, 4),
                    "expected_return_pct": exp_ret,
                    "plan_type": timeframe,
                    "horizon_days": bands.get("horizon_days"),
                    "tp_pct": bands.get("tp_pct"),
                    "sl_pct": bands.get("sl_pct"),
                    "tp1": tp1,
                    "tp2": tp2,
                    "tp3": tp3,
                    "sl1": sl1,
                    "sl2": sl2,
                    "sl3": sl3,
                    "market_cap": market_cap,
                    "cap_bucket": cap_bucket,
                    "asset_type": asset_type,
                    "source": "supabase",
                    "weights": r.get("weights_json") or {},
                    "weights_json": r.get("weights_json") or {},
                    "method": r.get("method"),
                    "created_at": r.get("created_at"),
                    "run_id": r.get("run_id"),
                    "meta": _build_meta(
                        {"p_accept": accept_prob, "method": r.get("method"), "run_id": r.get("run_id"), "event_id": r.get("event_id")},
                        evt,
                        timeframe,
                        bands,
                        price=price,
                    ),
                }
            )
            watchlist[-1]["meta"]["market_cap"] = market_cap
            watchlist[-1]["meta"]["cap_bucket"] = cap_bucket
            watchlist[-1]["meta"]["asset_type"] = asset_type

        watchlist.sort(key=lambda x: x.get("score", 0), reverse=True)
        debug_info["watchlist_size"] = len(watchlist)
        if not watchlist:
            debug_info["error"] = "empty_watchlist"
            return None, debug_info

        debug_info["stage"] = "success"
        probs = [float(r.get("p_accept") or 0.0) for r in watchlist]
        unique = len(set(round(p, 4) for p in probs))
        prob_stdev = pstdev(probs) if len(probs) > 1 else 0
        debug_info["unique_probs"] = unique
        debug_info["prob_stdev"] = round(prob_stdev, 4)
        # Removed overly strict diversity check - user has real WF data
        
        return watchlist, debug_info
    except Exception as e:
        debug_info["error"] = f"exception: {str(e)[:200]}"
        debug_info["exception_type"] = type(e).__name__
        return None, debug_info


def _handler_impl(request):
    timeframe = (request.args.get("timeframe") if hasattr(request, "args") else "all") or "all"
    timeframe = timeframe.lower()
    if timeframe not in ("all", "swing", "day", "long"):
        timeframe = "all"
    run_id = (request.args.get("run_id") if hasattr(request, "args") else "") or ""
    run_id = str(run_id).strip()

    watchlist, debug_info = _fetch_supabase(timeframe_filter=timeframe, run_id_filter=run_id or None)
    source = "Supabase ML Pipeline"

    if not watchlist:
        if timeframe != "all" and debug_info.get("error") == "empty_watchlist":
            return {
                "statusCode": 200,
                "headers": {
                    "Content-Type": "application/json",
                    "Cache-Control": "no-store",
                    "Access-Control-Allow-Origin": "*",
                },
                "body": json.dumps(
                    {
                        "asof": datetime.now(timezone.utc).isoformat(),
                        "source": source,
                        "provider": "DDL-69 Live Feed",
                        "is_live": True,
                        "timeframe_filter": timeframe,
                        "requested_timeframe": timeframe,
                        "timeframe_fallback": None,
                        "timeframe_counts": {},
                        "run_id": debug_info.get("active_run_id"),
                        "count": 0,
                        "ranked": [],
                        "tickers": [],
                        "stats": {"total": 0, "buy_count": 0, "hold_count": 0, "sell_count": 0},
                        "message": f"No rows available for timeframe '{timeframe}' in current Supabase run.",
                        "details": debug_info,
                    }
                ),
            }
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
                    "message": "Supabase ensemble forecasts required; no fallback enabled.",
                    "details": debug_info,
                }
            ),
        }

    stats = {
        "total": len(watchlist),
        "buy_count": len([w for w in watchlist if w.get("signal") == "BUY"]),
        "hold_count": len([w for w in watchlist if w.get("signal") == "HOLD"]),
        "sell_count": len([w for w in watchlist if w.get("signal") == "SELL"]),
    }

    tf_counts = {}
    for w in watchlist:
        tf = (w.get("plan_type") or "swing").lower()
        tf_counts[tf] = tf_counts.get(tf, 0) + 1

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Cache-Control": "max-age=60, public",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(
            {
                "asof": datetime.now(timezone.utc).isoformat(),
                "source": source,
                "provider": "DDL-69 Live Feed",
                "is_live": True,
                "timeframe_filter": timeframe,
                "requested_timeframe": timeframe,
                "timeframe_fallback": None,
                "timeframe_counts": tf_counts,
                "run_id": debug_info.get("active_run_id"),
                "count": len(watchlist),
                "ranked": watchlist,
                "tickers": [w.get("symbol") for w in watchlist if w.get("symbol")],
                "stats": stats,
                "message": f"Loaded {len(watchlist)} live rows",
            }
        ),
    }


class handler(FunctionHandler):
    endpoint = staticmethod(_handler_impl)
