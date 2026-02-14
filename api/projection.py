"""Price projection endpoint - historical prices + forward ML projection with confidence bands."""

import json
import logging
import math
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

try:
    from _http_adapter import FunctionHandler
except ModuleNotFoundError:
    from api._http_adapter import FunctionHandler

try:
    from _prices import fetch_prices
except ModuleNotFoundError:
    from api._prices import fetch_prices

try:
    from _real_market import build_rows_for_symbols, target_profile
except ModuleNotFoundError:
    from api._real_market import build_rows_for_symbols, target_profile


def _fetch_history(ticker, days=60):
    """Fetch historical daily prices from Yahoo Finance v8 chart API."""
    import requests

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}
    params = {"range": f"{days}d", "interval": "1d"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        if resp.status_code != 200:
            return []
        data = resp.json()
        result = data.get("chart", {}).get("result", [{}])[0]
        timestamps = result.get("timestamp", [])
        closes = result.get("indicators", {}).get("quote", [{}])[0].get("close", [])
        if not timestamps or not closes:
            return []
        points = []
        for ts, close in zip(timestamps, closes):
            if close is not None and ts is not None:
                points.append({"time": int(ts), "value": round(float(close), 2)})
        return points
    except Exception as exc:
        logger.warning("projection history fetch failed for %s: %s", ticker, exc)
        return []


def _generate_projection(current_price, p_up, tp_pcts, sl_pcts, horizon_days):
    """Generate forward projection with confidence bands.

    Uses geometric Brownian motion assumptions:
    - Drift based on ML probability (p_up)
    - Volatility estimated from TP/SL spread
    - Returns median path + 80/95% confidence intervals
    """
    if not current_price or current_price <= 0:
        return None

    # Estimate daily drift from probability
    # If p_up = 0.6, expected daily return = (2*p_up - 1) * avg_tp_pct / horizon
    avg_tp = abs(tp_pcts[0]) if tp_pcts else 0.04
    avg_sl = abs(sl_pcts[0]) if sl_pcts else 0.02
    daily_drift = (2 * p_up - 1) * avg_tp / max(horizon_days, 1)

    # Estimate daily volatility from the TP/SL spread
    spread = avg_tp + avg_sl
    daily_vol = spread / max(1, math.sqrt(horizon_days)) * 0.8

    now_ts = int(datetime.now(timezone.utc).timestamp())
    day_sec = 86400

    median = []
    upper_80 = []
    lower_80 = []
    upper_95 = []
    lower_95 = []

    for d in range(1, horizon_days + 1):
        t = now_ts + d * day_sec
        sqrt_d = math.sqrt(d)

        # Expected price (log-normal)
        expected = current_price * math.exp(daily_drift * d)

        # Confidence intervals
        z80 = 1.28  # 80% CI
        z95 = 1.96  # 95% CI

        vol_spread = daily_vol * sqrt_d
        u80 = current_price * math.exp(daily_drift * d + z80 * vol_spread)
        l80 = current_price * math.exp(daily_drift * d - z80 * vol_spread)
        u95 = current_price * math.exp(daily_drift * d + z95 * vol_spread)
        l95 = current_price * math.exp(daily_drift * d - z95 * vol_spread)

        median.append({"time": t, "value": round(expected, 2)})
        upper_80.append({"time": t, "value": round(u80, 2)})
        lower_80.append({"time": t, "value": round(l80, 2)})
        upper_95.append({"time": t, "value": round(u95, 2)})
        lower_95.append({"time": t, "value": round(l95, 2)})

    return {
        "median": median,
        "upper_80": upper_80,
        "lower_80": lower_80,
        "upper_95": upper_95,
        "lower_95": lower_95,
    }


def _get_prediction_data(ticker):
    """Try to get ML prediction data from Supabase for this ticker."""
    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not supabase_url or not service_key:
        return None

    try:
        from supabase import create_client

        supa = create_client(supabase_url, service_key)

        # Find events for this ticker
        ev_resp = (
            supa.table("events")
            .select("event_id,subject_id,horizon_json")
            .eq("subject_id", ticker.upper())
            .order("asof_ts", desc=True)
            .limit(1)
            .execute()
        )
        if not ev_resp.data:
            return None

        event = ev_resp.data[0]
        event_id = event["event_id"]

        # Get latest prediction
        pred_resp = (
            supa.table("predictions")
            .select("p_accept,p_reject,confidence,method,probs_json")
            .eq("event_id", event_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if not pred_resp.data:
            return None

        pred = pred_resp.data[0]
        horizon = event.get("horizon_json") or {}

        return {
            "p_accept": float(pred.get("p_accept") or 0.5),
            "confidence": float(pred.get("confidence") or 0.5),
            "method": pred.get("method"),
            "horizon": horizon,
        }
    except Exception as exc:
        logger.warning("projection prediction fetch failed: %s", exc)
        return None


def _parse_horizon_days(raw_horizon, default=10):
    """Parse Supabase horizon payloads safely (dict, number, or strings like '14d')."""
    try:
        if isinstance(raw_horizon, dict):
            val = raw_horizon.get("days") or raw_horizon.get("horizon_days")
            if val is None:
                unit = str(raw_horizon.get("unit") or "").lower().strip()
                if unit in ("d", "day", "days"):
                    val = raw_horizon.get("value")
            if val is not None:
                return max(1, min(int(float(val)), 730))
        elif isinstance(raw_horizon, (int, float)):
            return max(1, min(int(float(raw_horizon)), 730))
        elif isinstance(raw_horizon, str):
            txt = raw_horizon.strip().lower()
            if txt.endswith("d"):
                txt = txt[:-1]
            return max(1, min(int(float(txt)), 730))
    except Exception:
        pass
    return int(default)


def _handler_impl(request):
    args = request.args if hasattr(request, "args") else {}
    ticker = (
        (args.get("ticker") if args else None) or "SPY"
    ).upper()
    requested_timeframe = str((args.get("timeframe") if args else "") or "").lower().strip()
    if requested_timeframe not in ("day", "swing", "long"):
        requested_timeframe = ""
    days = 60

    # Fetch historical prices
    history = _fetch_history(ticker, days)

    # Get current price
    prices = fetch_prices([ticker])
    current_price = prices.get(ticker)
    if not current_price and history:
        current_price = history[-1]["value"]

    # Get ML prediction data (Supabase first)
    pred_data = _get_prediction_data(ticker)
    market_mode = requested_timeframe or "swing"
    market_rows = build_rows_for_symbols([ticker], mode=market_mode)
    market_row = market_rows[0] if market_rows else None

    if pred_data:
        p_up = pred_data["p_accept"]
        horizon = pred_data.get("horizon")
        horizon_days = _parse_horizon_days(horizon, default=10)
        method = pred_data.get("method") or "supabase"
        confidence = round(pred_data["confidence"], 4) if pred_data else None
    elif market_row:
        p_up = float(market_row.get("p_accept") or 0.5)
        meta = market_row.get("meta") or {}
        horizon_txt = str(meta.get("horizon") or "10d")
        try:
            horizon_days = max(1, int(horizon_txt.replace("d", "")))
        except Exception:
            horizon_days = 10
        method = "yahoo_screener_ta"
        confidence = market_row.get("confidence")
    else:
        p_up = 0.5
        horizon_days = 10
        method = "unavailable"
        confidence = None

    # Explicit timeframe from UI overrides inferred horizon banding.
    timeframe = requested_timeframe or ""

    # TP/SL bands based on horizon (ATR-aware when market TA exists)
    if market_row:
        meta = market_row.get("meta") or {}
        atr_pct = float(meta.get("atr_pct") or 0.02)
        if not timeframe:
            if horizon_days <= 30:
                timeframe = "day"
            elif horizon_days <= 365:
                timeframe = "swing"
            else:
                timeframe = "long"
        profile = target_profile(timeframe, atr_pct)
        tp_pcts = profile["tp_pct"]
        sl_pcts = profile["sl_pct"]
        horizon_days = profile["horizon_days"]
    else:
        if not timeframe:
            if horizon_days <= 30:
                timeframe = "day"
            elif horizon_days <= 365:
                timeframe = "swing"
            else:
                timeframe = "long"
        profile = target_profile(timeframe, 0.02)
        tp_pcts = profile["tp_pct"]
        sl_pcts = profile["sl_pct"]
        horizon_days = profile["horizon_days"]

    # Generate projection
    projection = None
    targets = None
    if current_price:
        projection = _generate_projection(
            current_price, p_up, tp_pcts, sl_pcts, horizon_days
        )
        targets = {
            "tp1": round(current_price * (1 + tp_pcts[0]), 2),
            "tp2": round(current_price * (1 + tp_pcts[1]), 2),
            "tp3": round(current_price * (1 + tp_pcts[2]), 2),
            "sl1": round(current_price * (1 + sl_pcts[0]), 2),
            "sl2": round(current_price * (1 + sl_pcts[1]), 2),
            "sl3": round(current_price * (1 + sl_pcts[2]), 2),
        }

    body = {
        "ticker": ticker,
        "current_price": round(current_price, 2) if current_price else None,
        "history": history,
        "projection": projection,
        "targets": targets,
        "model": {
            "p_up": round(p_up, 4),
            "p_down": round(1 - p_up, 4),
            "horizon_days": horizon_days,
            "timeframe": timeframe,
            "method": method,
            "confidence": confidence,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Cache-Control": "max-age=120",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body),
    }


class handler(FunctionHandler):
    endpoint = staticmethod(_handler_impl)
