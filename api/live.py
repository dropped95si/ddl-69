"""Live endpoint - primary watchlist feed.

Priority:
1) Supabase ensemble forecasts (if available and query succeeds)
2) Real-time Yahoo screener + TA computed rows

No synthetic/demo fallback payloads.
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


def _classify_timeframe(horizon_json):
    if not horizon_json:
        return "swing"
    days = None
    if isinstance(horizon_json, dict):
        days = horizon_json.get("days") or horizon_json.get("horizon_days")
        if not days and horizon_json.get("unit") == "days":
            days = horizon_json.get("value")
    if isinstance(horizon_json, (int, float)):
        days = horizon_json
    if days is not None:
        try:
            days = float(days)
        except Exception:
            return "swing"
        if days <= 3:
            return "day"
        if days <= 15:
            return "swing"
        return "long"
    return "swing"


def _tp_sl_for_timeframe(timeframe):
    if timeframe == "day":
        return {"tp_pct": [0.015, 0.03, 0.05], "sl_pct": [-0.01, -0.02, -0.03], "horizon_days": 2}
    if timeframe == "long":
        return {"tp_pct": [0.10, 0.20, 0.35], "sl_pct": [-0.05, -0.08, -0.12], "horizon_days": 45}
    return {"tp_pct": [0.04, 0.08, 0.12], "sl_pct": [-0.02, -0.04, -0.06], "horizon_days": 10}


def _build_meta(row, evt, timeframe, bands, price=None):
    accept = float(row.get("p_accept", 0.5))
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


def _fetch_supabase(timeframe_filter=None):
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

        event_ids = list({r["event_id"] for r in rows if r.get("event_id")})
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

        try:
            from _prices import fetch_prices
        except ModuleNotFoundError:
            from api._prices import fetch_prices

        all_tickers = []
        for r in rows:
            evt = events_map.get(r.get("event_id"), {})
            ticker = str(evt.get("subject_id") or "").upper().strip()
            if ticker:
                all_tickers.append(ticker)
        all_tickers = list(dict.fromkeys(all_tickers))
        prices = fetch_prices(all_tickers) if all_tickers else {}

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

            accept_prob = float(probs.get("ACCEPT") or probs.get("accept") or 0.5)
            reject_prob = float(probs.get("REJECT") or probs.get("reject") or (1 - accept_prob))
            continue_prob = float(probs.get("CONTINUE") or probs.get("continue") or 0.0)

            timeframe = _classify_timeframe(evt.get("horizon_json"))
            if timeframe_filter and timeframe_filter != "all" and timeframe != timeframe_filter:
                continue

            signal = "BUY" if accept_prob > 0.6 else ("SELL" if reject_prob > 0.5 else "HOLD")
            score = round(max(accept_prob, reject_prob), 4)
            bands = _tp_sl_for_timeframe(timeframe)
            price = prices.get(ticker)

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
                    "confidence": round(float(r.get("confidence") or 0.5), 4),
                    "plan_type": timeframe,
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

        watchlist.sort(key=lambda x: x.get("score", 0), reverse=True)
        return watchlist if watchlist else None
    except Exception:
        return None


def _fetch_market_ta(timeframe_filter):
    if timeframe_filter == "all":
        rows = []
        rows.extend(build_watchlist("day", 90))
        rows.extend(build_watchlist("swing", 130))
        rows.extend(build_watchlist("long", 90))
        dedup = {}
        for row in rows:
            sym = row.get("symbol") or row.get("ticker")
            if not sym:
                continue
            old = dedup.get(sym)
            if old is None or float(row.get("score", 0)) > float(old.get("score", 0)):
                dedup[sym] = row
        ranked = list(dedup.values())
        ranked.sort(key=lambda r: float(r.get("score", 0)), reverse=True)
        return ranked[:220]
    return build_watchlist(timeframe_filter, 180)


def _handler_impl(request):
    timeframe = (request.args.get("timeframe") if hasattr(request, "args") else "all") or "all"
    timeframe = timeframe.lower()
    if timeframe not in ("all", "swing", "day", "long"):
        timeframe = "all"

    watchlist = _fetch_supabase(timeframe_filter=timeframe)
    source = "Supabase ML Pipeline"
    if not watchlist:
        watchlist = _fetch_market_ta(timeframe_filter=timeframe)
        source = "Yahoo Screener + TA"

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
                "timeframe_counts": tf_counts,
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

