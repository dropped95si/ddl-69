"""
Model Audit & Comparison Endpoint
Compares DDL-69 Ensemble vs Qlib-LGB vs Chronos-T5 predictions
Returns probabilities, confidence, DTE projections, and reasoning
"""

import json
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any

try:
    from _http_adapter import FunctionHandler
except ModuleNotFoundError:
    from api._http_adapter import FunctionHandler


def _fetch_supabase_predictions(limit=10):
    """Fetch top predictions from our Supabase ensemble pipeline"""
    try:
        from supabase import create_client
        
        url = os.getenv("SUPABASE_URL", "").strip()
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        
        if not url or not key:
            return []
        
        supa = create_client(url, key)
        
        # Get latest ensemble forecasts
        resp = supa.table("v_latest_ensemble_forecasts")\
            .select("*")\
            .order("confidence", desc=True)\
            .limit(limit)\
            .execute()
        
        if not resp.data:
            return []
        
        # Get event horizons
        event_ids = list({r["event_id"] for r in resp.data if r.get("event_id")})
        events = {}
        if event_ids:
            ev_resp = supa.table("events")\
                .select("event_id,horizon_json,asof_ts")\
                .in_("event_id", event_ids)\
                .execute()
            events = {e["event_id"]: e for e in (ev_resp.data or [])}
        
        results = []
        for row in resp.data:
            event = events.get(row.get("event_id"), {})
            horizon_json = event.get("horizon_json", {})
            
            # Parse horizon days
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
                horizon_days = 7  # Default swing
            
            results.append({
                "ticker": row.get("ticker"),
                "price": row.get("price"),
                "confidence": row.get("confidence"),
                "p_accept": row.get("p_accept"),
                "p_reject": row.get("p_reject"),
                "signal": row.get("signal"),
                "method": row.get("method"),
                "horizon_days": horizon_days,
                "created_at": row.get("created_at"),
                "weights": row.get("weights_json", {}),
            })
        
        return results
    except Exception as e:
        print(f"Supabase fetch error: {e}")
        return []


def _calculate_qlib_comparison(ticker: str, price: float, horizon_days: int):
    """
    Simulate Qlib-LGB model comparison
    In production, this would call actual Qlib model
    """
    # Qlib uses different approach - pure quant factors
    # Simulating with adjusted confidence based on technical factors
    import random
    random.seed(hash(ticker) % 1000)
    
    base_confidence = 0.55 + random.random() * 0.25  # 0.55-0.80
    qlib_confidence = round(base_confidence, 4)
    
    # Qlib typically more conservative on swing trades
    qlib_dte = int(horizon_days * 0.85)  # Slightly shorter DTE
    
    # Calculate expected return (Qlib focuses on alpha)
    qlib_expected_return = round(random.uniform(0.02, 0.08), 4)
    
    return {
        "model": "Qlib-LGB",
        "confidence": qlib_confidence,
        "probability": qlib_confidence,
        "dte": qlib_dte,
        "expected_return": qlib_expected_return,
        "reasoning": f"Qlib quant factors suggest {qlib_dte}d holding period with {qlib_expected_return*100:.1f}% expected return. Conservative alpha-focused approach.",
    }


def _calculate_chronos_forecast(ticker: str, price: float, horizon_days: int):
    """
    Simulate Chronos-T5 time series forecast
    In production, this would call actual Chronos model
    """
    import random
    random.seed(hash(ticker + "chronos") % 1000)
    
    # Chronos uses transformer-based time series forecasting
    # More bullish on trends, wider uncertainty bands
    chronos_confidence = 0.70 + random.random() * 0.20  # 0.70-0.90
    chronos_confidence = round(chronos_confidence, 4)
    
    # Chronos predicts price path - longer horizon
    chronos_dte = int(horizon_days * 1.1)  # Slightly longer
    
    # Forecast with uncertainty
    forecast_low = price * (1 + random.uniform(0.01, 0.04))
    forecast_high = price * (1 + random.uniform(0.05, 0.12))
    forecast_median = (forecast_low + forecast_high) / 2
    
    expected_return = (forecast_median - price) / price
    
    return {
        "model": "Chronos-T5",
        "confidence": chronos_confidence,
        "probability": chronos_confidence,
        "dte": chronos_dte,
        "forecast_median": round(forecast_median, 2),
        "forecast_range": [round(forecast_low, 2), round(forecast_high, 2)],
        "expected_return": round(expected_return, 4),
        "reasoning": f"Time series forecast shows {chronos_dte}d path to ${forecast_median:.2f} (80% CI: ${forecast_low:.2f}-${forecast_high:.2f}). Transformer-based prediction.",
    }


def _build_comparison(supabase_pred: Dict) -> Dict[str, Any]:
    """Build comprehensive model comparison for a single ticker"""
    ticker = supabase_pred["ticker"]
    price = supabase_pred["price"]
    horizon_days = int(supabase_pred["horizon_days"])
    
    # Our ensemble model (DDL-69)
    our_model = {
        "model": "DDL-69 Ensemble",
        "confidence": supabase_pred["confidence"],
        "probability": supabase_pred["p_accept"],
        "p_reject": supabase_pred["p_reject"],
        "signal": supabase_pred["signal"],
        "dte": horizon_days,
        "method": supabase_pred["method"],
        "weights": supabase_pred["weights"],
        "reasoning": f"Hedge ensemble using MWU with {len(supabase_pred['weights'])} experts. Target {horizon_days}d swing trade based on multi-expert consensus.",
    }
    
    # Qlib comparison
    qlib_model = _calculate_qlib_comparison(ticker, price, horizon_days)
    
    # Chronos comparison
    chronos_model = _calculate_chronos_forecast(ticker, price, horizon_days)
    
    # Calculate consensus
    avg_confidence = (our_model["confidence"] + qlib_model["confidence"] + chronos_model["confidence"]) / 3
    avg_dte = int((our_model["dte"] + qlib_model["dte"] + chronos_model["dte"]) / 3)
    
    # Determine agreement
    confidences = [our_model["confidence"], qlib_model["confidence"], chronos_model["confidence"]]
    agreement_score = 1.0 - (max(confidences) - min(confidences))
    
    return {
        "ticker": ticker,
        "price": price,
        "timestamp": supabase_pred["created_at"],
        "models": {
            "ddl69": our_model,
            "qlib": qlib_model,
            "chronos": chronos_model,
        },
        "consensus": {
            "avg_confidence": round(avg_confidence, 4),
            "avg_dte": avg_dte,
            "agreement_score": round(agreement_score, 4),
            "recommendation": "STRONG BUY" if avg_confidence > 0.75 and agreement_score > 0.8 else
                            "BUY" if avg_confidence > 0.65 else
                            "HOLD" if avg_confidence > 0.55 else "PASS",
        },
    }


def audit_handler(request):
    """Main handler for model comparison audit"""
    
    # Get parameters
    limit = 10
    if hasattr(request, "args"):
        try:
            limit = int(request.args.get("limit", 10))
        except:
            limit = 10
    
    limit = max(1, min(limit, 50))  # Cap at 50
    
    # Fetch our predictions from Supabase
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
    
    # Build comparisons
    comparisons = []
    for pred in supabase_predictions:
        try:
            comp = _build_comparison(pred)
            comparisons.append(comp)
        except Exception as e:
            print(f"Error building comparison for {pred.get('ticker')}: {e}")
            continue
    
    # Calculate summary statistics
    strong_buys = len([c for c in comparisons if c["consensus"]["recommendation"] == "STRONG BUY"])
    buys = len([c for c in comparisons if c["consensus"]["recommendation"] == "BUY"])
    holds = len([c for c in comparisons if c["consensus"]["recommendation"] == "HOLD"])
    
    avg_agreement = sum(c["consensus"]["agreement_score"] for c in comparisons) / len(comparisons) if comparisons else 0
    
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Cache-Control": "max-age=300, public",
        },
        "body": json.dumps({
            "asof": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_comparisons": len(comparisons),
                "strong_buy": strong_buys,
                "buy": buys,
                "hold": holds,
                "avg_agreement": round(avg_agreement, 4),
            },
            "comparisons": comparisons,
            "methodology": {
                "ddl69": "Multiplicative Weights Update (MWU) ensemble with 6+ experts including TA, events, earnings, whale flow",
                "qlib": "LightGBM with quantitative alpha factors (simulated)",
                "chronos": "Amazon T5-based transformer for time series forecasting (simulated)",
            },
            "notes": "DTE = Days To Exit/Expiration. Confidence represents model certainty. Agreement score shows cross-model consensus (1.0 = perfect agreement).",
        })
    }


class handler(FunctionHandler):
    endpoint = staticmethod(audit_handler)
