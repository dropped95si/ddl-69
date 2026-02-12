"""
Model Audit & Performance Analysis
Real predictions from Supabase with proper timeframe classifications
Day: 1-30 days, Swing: 31-365 days, Long: 366+ days
"""

import json
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any

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


def _fetch_supabase_predictions(limit=10):
    """Fetch top predictions from our Supabase ensemble pipeline with REAL data"""
    try:
        from supabase import create_client
        
        url = os.getenv("SUPABASE_URL", "").strip()
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        
        if not url or not key:
            return []
        
        supa = create_client(url, key)
        
        # Get latest ensemble forecasts with ALL model predictions
        resp = (
            supa.table("v_latest_ensemble_forecasts")
            .select("event_id,method,probs_json,confidence,created_at,weights_json,explain_json,run_id")
            .order("confidence", desc=True)
            .limit(limit)
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
            
            # Parse horizon days CORRECTLY
            horizon_days = None
            if isinstance(horizon_json, dict):
                horizon_days = horizon_json.get("days") or horizon_json.get("horizon_days")
                if not horizon_days:
                    unit = horizon_json.get("unit", "")
                    if unit in ("d", "days", "day"):
                        horizon_days = horizon_json.get("value")
            elif isinstance(horizon_json, (int, float)):
                horizon_days = horizon_json
            
            if not horizon_days:
                horizon_days = 30  # Default to day boundary
            
            ticker = str(event.get("subject_id") or "").upper().strip()
            price = prices.get(ticker) if ticker else None

            # Targets are not in this view; keep null unless upstream provides them
            tp1 = None
            tp2 = None
            tp3 = None
            sl1 = None
            
            # Calculate expected return
            expected_return = None
            if tp1 and price and price > 0:
                expected_return = (tp1 - price) / price
            
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

            results.append({
                "ticker": ticker,
                "price": price,
                "confidence": confidence,
                "p_accept": float(p_accept),
                "p_reject": float(p_reject) if p_reject is not None else None,
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
        
        return results
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
    if hasattr(request, "args"):
        try:
            limit = int(request.args.get("limit", 10))
        except:
            limit = 10
    
    limit = max(1, min(limit, 50))  # Cap at 50
    
    # Fetch REAL predictions from Supabase
    supabase_predictions = _fetch_supabase_predictions(limit=limit)
    
    if not supabase_predictions:
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
            "methodology": "Real predictions from DDL-69 Ensemble (MWU) using Supabase ML pipeline. All metrics calculated from actual model outputs.",
            "notes": "Confidence = Model certainty. P(Accept) = Probability of reaching target. Expected Return = (TP1 - Price) / Price. Risk/Reward = Gain potential / Loss potential.",
        })
    }


class handler(FunctionHandler):
    endpoint = staticmethod(audit_handler)
