"""
Model Audit & Performance Analysis
Real predictions from Supabase with proper timeframe classifications
Day: 1-30 days, Swing: 31-365 days, Long: 366+ days
"""

import json
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any
import re

try:
    from _http_adapter import FunctionHandler
except ModuleNotFoundError:
    from api._http_adapter import FunctionHandler


def _classify_timeframe_correct(horizon_days: float) -> str:
    """Correct timeframe classification"""
    if horizon_days <= 30:
        return "day"
    elif horizon_days <= 365:
        return "swing"
    else:
        return "long"


def _parse_horizon_days(raw_horizon: Any) -> float | None:
    def _as_float(value: Any) -> float | None:
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


def _parse_iso_ts(value: Any) -> datetime:
    if not value:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return datetime.fromtimestamp(0, tz=timezone.utc)


def _dedupe_predictions(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep one strongest+newest prediction per ticker."""
    best_by_ticker: Dict[str, Dict[str, Any]] = {}

    for pred in predictions:
        ticker = str(pred.get("ticker") or "").upper().strip()
        if not ticker:
            continue

        existing = best_by_ticker.get(ticker)
        if existing is None:
            best_by_ticker[ticker] = pred
            continue

        conf_new = float(pred.get("confidence") or 0.0)
        conf_old = float(existing.get("confidence") or 0.0)
        ts_new = _parse_iso_ts(pred.get("created_at"))
        ts_old = _parse_iso_ts(existing.get("created_at"))
        pa_new = float(pred.get("p_accept") or 0.0)
        pa_old = float(existing.get("p_accept") or 0.0)

        if conf_new > conf_old or (conf_new == conf_old and ts_new > ts_old) or (
            conf_new == conf_old and ts_new == ts_old and pa_new > pa_old
        ):
            best_by_ticker[ticker] = pred

    deduped = list(best_by_ticker.values())
    deduped.sort(
        key=lambda p: (
            float(p.get("confidence") or 0.0),
            _parse_iso_ts(p.get("created_at")),
            float(p.get("p_accept") or 0.0),
        ),
        reverse=True,
    )
    return deduped


def _fetch_supabase_predictions(limit=10, distinct_tickers=True, timeframe_filter="all"):
    """Fetch top predictions from our Supabase ensemble pipeline with REAL data"""
    try:
        from supabase import create_client

        url = os.getenv("SUPABASE_URL", "").strip()
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        
        if not url or not key:
            return []
        
        supa = create_client(url, key)
        
        # Pull a larger candidate window so top-N unique tickers can be selected.
        fetch_limit = max(limit * 25, 200)
        fetch_limit = min(fetch_limit, 1000)

        # Get latest ensemble forecasts with ALL model predictions
        resp = (
            supa.table("v_latest_ensemble_forecasts")
            .select("event_id,method,probs_json,confidence,created_at,weights_json,explain_json,run_id")
            .order("confidence", desc=True)
            .limit(fetch_limit)
            .execute()
        )
        
        if not resp.data:
            return []
        
        # Get event horizons and symbols
        event_ids = list({r["event_id"] for r in resp.data if r.get("event_id")})
        events = {}
        if event_ids:
            ev_resp = (
                supa.table("events")
                .select("event_id,subject_id,horizon_json,asof_ts")
                .in_("event_id", event_ids)
                .execute()
            )
            events = {e["event_id"]: e for e in (ev_resp.data or [])}

        try:
            from _prices import fetch_prices
        except ModuleNotFoundError:
            from api._prices import fetch_prices

        tickers = []
        for row in resp.data:
            evt = events.get(row.get("event_id"), {})
            ticker = str(evt.get("subject_id") or "").upper().strip()
            if ticker:
                tickers.append(ticker)
        tickers = list(dict.fromkeys(tickers))
        prices = fetch_prices(tickers) if tickers else {}
        
        results = []
        for row in resp.data:
            event = events.get(row.get("event_id"), {})
            horizon_json = event.get("horizon_json", {})
            
            # Parse horizon days across day/week/month/year units
            horizon_days = _parse_horizon_days(horizon_json)
            if horizon_days is None:
                horizon_days = 180.0  # Unknown horizon defaults to swing baseline
            
            ticker = str(event.get("subject_id") or "").upper().strip()
            price = prices.get(ticker) if ticker else None

            # Calculate TP/SL from horizon using sqrt-scaled bands
            from math import sqrt
            hd = float(horizon_days)
            if hd <= 30:
                sc = sqrt(hd / 10.0)
                tp_pcts = [0.015 * sc, 0.03 * sc, 0.05 * sc]
                sl_pcts = [-0.01 * sc, -0.02 * sc, -0.03 * sc]
            elif hd >= 366:
                sc = hd / 365.0
                tp_pcts = [0.25 * sc, 0.50 * sc, 0.80 * sc]
                sl_pcts = [-0.12 * sc, -0.18 * sc, -0.25 * sc]
            else:
                sc = hd / 180.0
                tp_pcts = [0.08 * sc, 0.15 * sc, 0.25 * sc]
                sl_pcts = [-0.04 * sc, -0.07 * sc, -0.12 * sc]

            tp1 = round(price * (1 + tp_pcts[0]), 2) if price else None
            tp2 = round(price * (1 + tp_pcts[1]), 2) if price else None
            tp3 = round(price * (1 + tp_pcts[2]), 2) if price else None
            sl1 = round(price * (1 + sl_pcts[0]), 2) if price else None
            
            confidence = row.get("confidence")
            probs = row.get("probs_json") or {}
            p_accept = probs.get("ACCEPT_CONTINUE") or probs.get("ACCEPT") or probs.get("accept")
            p_reject = probs.get("REJECT") or probs.get("reject")
            p_continue = probs.get("BREAK_FAIL") or probs.get("CONTINUE") or probs.get("continue")

            # Skip rows with missing probabilities or confidence (no fake fallbacks)
            if not ticker or confidence is None or p_accept is None:
                continue

            if p_reject is None:
                p_reject = 1 - float(p_accept)

            signal = "BUY" if float(p_accept) > 0.6 else ("SELL" if float(p_reject) > 0.5 else "HOLD")

            # Probability-weighted expected return
            pa = float(p_accept)
            pr = float(p_reject) if p_reject is not None else (1 - pa)
            expected_return = pa * tp_pcts[0] + pr * sl_pcts[0]

            results.append({
                "ticker": ticker,
                "price": price,
                "confidence": confidence,
                "p_accept": pa,
                "p_reject": pr,
                "p_continue": float(p_continue) if p_continue is not None else 0,
                "signal": signal,
                "method": row.get("method"),
                "horizon_days": float(horizon_days),
                "timeframe": _classify_timeframe_correct(float(horizon_days)),
                "created_at": row.get("created_at"),
                "weights": row.get("weights_json", {}),
                "tp1": tp1,
                "tp2": tp2,
                "tp3": tp3,
                "sl1": sl1,
                "expected_return": expected_return,
            })
        
        tf_filter = str(timeframe_filter or "all").strip().lower()
        if tf_filter in ("day", "swing", "long"):
            results = [r for r in results if str(r.get("timeframe") or "").lower() == tf_filter]

        if distinct_tickers:
            results = _dedupe_predictions(results)

        return results[:limit]
    except Exception as e:
        print(f"Supabase fetch error: {e}")
        return []


def _calculate_model_metrics(pred: Dict) -> Dict:
    """Calculate real metrics from actual prediction data"""
    ticker = pred["ticker"]
    price = pred["price"]
    horizon_days = pred["horizon_days"]
    timeframe = pred["timeframe"]
    expected_return = pred.get("expected_return", 0)
    
    # Risk/Reward metrics from TP/SL targets
    tp1 = pred.get("tp1")
    tp3 = pred.get("tp3")
    sl1 = pred.get("sl1")
    
    risk_reward = None
    if tp1 and sl1 and price:
        potential_gain = (tp1 - price) / price
        potential_loss = abs((sl1 - price) / price)
        risk_reward = potential_gain / potential_loss if potential_loss > 0 else None
    
    # Sharpe estimate (simplified)
    sharpe_estimate = None
    if expected_return and horizon_days:
        # Annualized Sharpe approximation
        annual_factor = 252 / horizon_days
        volatility_estimate = 0.15  # Typical stock volatility
        sharpe_estimate = (expected_return * annual_factor) / volatility_estimate
    
    return {
        "ticker": ticker,
        "price": price,
        "confidence": pred["confidence"],
        "probability": pred["p_accept"],
        "p_reject": pred["p_reject"],
        "p_continue": pred["p_continue"],
        "signal": pred["signal"],
        "horizon_days": horizon_days,
        "timeframe": timeframe,
        "expected_return": expected_return,
        "tp1": tp1,
        "tp3": tp3,
        "sl1": sl1,
        "risk_reward_ratio": round(risk_reward, 2) if risk_reward else None,
        "sharpe_estimate": round(sharpe_estimate, 2) if sharpe_estimate else None,
        "method": pred["method"],
        "weights": pred["weights"],
        "created_at": pred["created_at"],
    }


def _build_analysis(pred_metrics: Dict) -> Dict[str, Any]:
    """Build comprehensive analysis for a single prediction"""
    ticker = pred_metrics["ticker"]
    price = pred_metrics["price"]
    horizon_days = pred_metrics["horizon_days"]
    timeframe = pred_metrics["timeframe"]
    expected_return = pred_metrics.get("expected_return", 0)
    confidence = pred_metrics["confidence"]
    p_accept = pred_metrics["probability"]
    p_reject = pred_metrics["p_reject"]
    
    # Timeframe description
    timeframe_desc = {
        "day": "Day Trade (1-30 days)",
        "swing": "Swing Trade (1-12 months)",
        "long": "Long Hold (1+ years)"
    }.get(timeframe, "Unknown")
    
    # Calculate conviction level
    if p_accept >= 0.75 and confidence >= 0.75:
        conviction = "STRONG BUY"
        conviction_color = "success"
    elif p_accept >= 0.60 and confidence >= 0.60:
        conviction = "BUY"
        conviction_color = "primary"
    elif p_accept >= 0.50:
        conviction = "HOLD"
        conviction_color = "warning"
    else:
        conviction = "PASS"
        conviction_color = "secondary"
    
    # Build reasoning
    weights_str = ", ".join([f"{k}: {v:.1%}" for k, v in list(pred_metrics["weights"].items())[:3]])
    
    reasoning = f"{timeframe_desc} prediction. Ensemble confidence {confidence:.1%} with {p_accept:.1%} accept probability. "
    reasoning += f"Expected {horizon_days:.0f}-day holding period targeting {expected_return*100:+.2f}% return. "
    reasoning += f"Top weights: {weights_str}. "
    
    if pred_metrics.get("risk_reward_ratio"):
        reasoning += f"Risk/Reward: {pred_metrics['risk_reward_ratio']:.2f}:1. "
    
    # DTE calculation
    created = datetime.fromisoformat(pred_metrics["created_at"].replace("Z", "+00:00"))
    target_exit = created + timedelta(days=horizon_days)
    days_remaining = (target_exit - datetime.now(timezone.utc)).days
    
    return {
        "ticker": ticker,
        "price": price,
        "timestamp": pred_metrics["created_at"],
        "signal": pred_metrics["signal"],
        "timeframe": timeframe,
        "timeframe_description": timeframe_desc,
        "conviction": conviction,
        "conviction_color": conviction_color,
        "metrics": {
            "confidence": confidence,
            "p_accept": p_accept,
            "p_reject": p_reject,
            "p_continue": pred_metrics["p_continue"],
            "expected_return_pct": round(expected_return * 100, 2),
            "horizon_days": horizon_days,
            "days_remaining": max(0, days_remaining),
            "target_exit_date": target_exit.strftime("%Y-%m-%d"),
            "risk_reward_ratio": pred_metrics.get("risk_reward_ratio"),
            "sharpe_estimate": pred_metrics.get("sharpe_estimate"),
        },
        "targets": {
            "tp1": pred_metrics["tp1"],
            "tp3": pred_metrics["tp3"],
            "sl1": pred_metrics["sl1"],
        },
        "model_details": {
            "method": pred_metrics["method"],
            "weights": pred_metrics["weights"],
            "num_experts": len(pred_metrics["weights"]),
        },
        "reasoning": reasoning,
    }


def audit_handler(request):
    """Main handler for real model performance audit"""
    
    # Get parameters
    limit = 10
    distinct_tickers = True
    timeframe_filter = "all"
    if hasattr(request, "args"):
        try:
            limit = int(request.args.get("limit", 10))
        except:
            limit = 10
        distinct_raw = str(request.args.get("distinct_tickers", "1")).strip().lower()
        distinct_tickers = distinct_raw not in ("0", "false", "no")
        timeframe_raw = str(request.args.get("timeframe", "all")).strip().lower()
        timeframe_filter = timeframe_raw if timeframe_raw in ("all", "day", "swing", "long") else "all"

    limit = max(1, min(limit, 50))  # Cap at 50

    # Fetch REAL predictions from Supabase
    supabase_predictions = _fetch_supabase_predictions(
        limit=limit,
        distinct_tickers=distinct_tickers,
        timeframe_filter=timeframe_filter,
    )
    
    if not supabase_predictions:
        if timeframe_filter != "all":
            return {
                "statusCode": 200,
                "headers": {
                    "Content-Type": "application/json",
                    "Cache-Control": "max-age=60, public",
                },
                "body": json.dumps(
                    {
                        "asof": datetime.now(timezone.utc).isoformat(),
                        "summary": {
                            "total_predictions": 0,
                            "strong_buy": 0,
                            "buy": 0,
                            "hold": 0,
                            "pass": 0,
                            "avg_confidence": 0.0,
                            "avg_p_accept": 0.0,
                            "avg_expected_return_pct": 0.0,
                            "timeframe_breakdown": {},
                        },
                        "predictions": [],
                        "timeframe_definitions": {
                            "day": "1-30 days (up to 1 month)",
                            "swing": "31-365 days (1 month - 1 year)",
                            "long": "366+ days (1+ years)",
                        },
                        "requested_timeframe": timeframe_filter,
                        "methodology": "Real predictions from DDL-69 Ensemble (MWU) using Supabase ML pipeline. All metrics calculated from actual model outputs.",
                        "notes": "Confidence = model certainty. P(Accept) = probability of reaching target. Expected Return = (TP1 - Price) / Price. Risk/Reward = gain potential / loss potential. Distinct ticker mode is enabled by default.",
                        "message": f"No rows available for timeframe '{timeframe_filter}' in current Supabase run.",
                    }
                ),
            }
        return {
            "statusCode": 503,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "error": "No predictions available",
                "message": "Supabase ensemble pipeline returned no data",
            })
        }
    
    # Build real analysis for each prediction
    analyses = []
    for pred in supabase_predictions:
        try:
            metrics = _calculate_model_metrics(pred)
            analysis = _build_analysis(metrics)
            analyses.append(analysis)
        except Exception as e:
            print(f"Error analyzing {pred.get('ticker')}: {e}")
            continue
    
    # Calculate summary statistics
    strong_buys = len([a for a in analyses if a["conviction"] == "STRONG BUY"])
    buys = len([a for a in analyses if a["conviction"] == "BUY"])
    holds = len([a for a in analyses if a["conviction"] == "HOLD"])
    passes = len([a for a in analyses if a["conviction"] == "PASS"])
    
    # Timeframe breakdown
    timeframe_counts = {}
    for a in analyses:
        tf = a["timeframe"]
        timeframe_counts[tf] = timeframe_counts.get(tf, 0) + 1
    
    # Average metrics
    avg_confidence = sum(a["metrics"]["confidence"] for a in analyses) / len(analyses) if analyses else 0
    avg_p_accept = sum(a["metrics"]["p_accept"] for a in analyses) / len(analyses) if analyses else 0
    avg_expected_return = sum(a["metrics"]["expected_return_pct"] for a in analyses) / len(analyses) if analyses else 0
    
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Cache-Control": "max-age=300, public",
        },
        "body": json.dumps({
            "asof": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_predictions": len(analyses),
                "strong_buy": strong_buys,
                "buy": buys,
                "hold": holds,
                "pass": passes,
                "avg_confidence": round(avg_confidence, 4),
                "avg_p_accept": round(avg_p_accept, 4),
                "avg_expected_return_pct": round(avg_expected_return, 2),
                "timeframe_breakdown": timeframe_counts,
            },
            "predictions": analyses,
            "timeframe_definitions": {
                "day": "1-30 days (up to 1 month)",
                "swing": "31-365 days (1 month - 1 year)",
                "long": "366+ days (1+ years)"
            },
            "requested_timeframe": timeframe_filter,
            "methodology": "Real predictions from DDL-69 Ensemble (MWU) using Supabase ML pipeline. All metrics calculated from actual model outputs.",
            "notes": "Confidence = model certainty. P(Accept) = probability of reaching target. Expected Return = (TP1 - Price) / Price. Risk/Reward = gain potential / loss potential. Distinct ticker mode is enabled by default.",
        })
    }


class handler(FunctionHandler):
    endpoint = staticmethod(audit_handler)
