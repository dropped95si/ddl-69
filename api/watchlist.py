"""Real watchlist endpoint - loads from Supabase + ML pipeline."""
import json
import os
from datetime import datetime, timezone

import requests

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")


def fetch_supabase_predictions(limit=100):
    """Fetch predictions from Supabase ledger."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return []
    
    try:
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
        }
        
        # Try to fetch from a predictions table/view
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/predictions?order=created_at.desc&limit={limit}",
            headers=headers,
            timeout=5,
        )
        
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 404:
            # Table doesn't exist, return empty
            return []
    except Exception as e:
        print(f"Supabase fetch error: {e}")
    
    return []


def calc_weights(row):
    """Extract weights from prediction or calculate defaults."""
    if isinstance(row.get("weights_json"), dict):
        return row["weights_json"]
    
    # Default weights based on signal
    signal = row.get("signal", "HOLD").upper()
    if signal == "BUY":
        return {"technical": 0.35, "sentiment": 0.25, "fundamental": 0.25, "ensemble": 0.15}
    elif signal == "SELL":
        return {"technical": 0.40, "sentiment": 0.30, "fundamental": 0.20, "ensemble": 0.10}
    else:
        return {"technical": 0.30, "sentiment": 0.25, "fundamental": 0.30, "ensemble": 0.15}


def handler(request):
    """Fetch watchlist - real data from Supabase or demo fallback."""
    try:
        # Try to get from Supabase first
        predictions = fetch_supabase_predictions(100)
        
        watchlist = []
        seen = set()
        
        # Add from Supabase predictions
        if predictions:
            for pred in predictions[:50]:
                ticker = pred.get("ticker") or pred.get("symbol") or ""
                if not ticker or ticker in seen:
                    continue
                seen.add(ticker)
                
                watchlist.append({
                    "ticker": ticker,
                    "symbol": ticker,
                    "label": pred.get("signal", "HOLD"),
                    "score": min(0.99, max(0.01, float(pred.get("probability", 0.5)))),
                    "p_accept": min(0.99, max(0.01, float(pred.get("probability", 0.5)))),
                    "signal": pred.get("signal", "HOLD"),
                    "weights": calc_weights(pred),
                    "weights_json": calc_weights(pred),
                    "created_at": pred.get("created_at"),
                })
        
        # Fallback to demo data if no Supabase data
        if not watchlist:
            watchlist = [
                {
                    "ticker": "NVDA",
                    "symbol": "NVDA",
                    "label": "BUY",
                    "score": 0.87,
                    "p_accept": 0.78,
                    "signal": "BUY",
                    "weights": {"technical": 0.35, "sentiment": 0.25, "fundamental": 0.25, "ensemble": 0.15},
                    "weights_json": {"technical": 0.35, "sentiment": 0.25, "fundamental": 0.25, "ensemble": 0.15},
                },
                {
                    "ticker": "TSLA",
                    "symbol": "TSLA",
                    "label": "HOLD",
                    "score": 0.62,
                    "p_accept": 0.58,
                    "signal": "HOLD",
                    "weights": {"technical": 0.30, "sentiment": 0.25, "fundamental": 0.30, "ensemble": 0.15},
                    "weights_json": {"technical": 0.30, "sentiment": 0.25, "fundamental": 0.30, "ensemble": 0.15},
                },
                {
                    "ticker": "SPY",
                    "symbol": "SPY",
                    "label": "HOLD",
                    "score": 0.55,
                    "p_accept": 0.52,
                    "signal": "HOLD",
                    "weights": {"technical": 0.30, "sentiment": 0.25, "fundamental": 0.30, "ensemble": 0.15},
                    "weights_json": {"technical": 0.30, "sentiment": 0.25, "fundamental": 0.30, "ensemble": 0.15},
                },
            ]
        
        # Sort by score
        watchlist = sorted(watchlist, key=lambda x: x.get("score", 0), reverse=True)
        
        # Calculate stats
        buy_count = len([w for w in watchlist if w.get("signal") == "BUY"])
        sell_count = len([w for w in watchlist if w.get("signal") == "SELL"])
        hold_count = len([w for w in watchlist if w.get("signal") == "HOLD"])
        
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json", "Cache-Control": "max-age=60"},
            "body": json.dumps({
                "asof": datetime.now(timezone.utc).isoformat(),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "source": "Supabase Ledger" if predictions else "Demo Data",
                "provider": "DDL-69 ML Pipeline",
                "count": len(watchlist),
                "ranked": watchlist,
                "tickers": [w.get("ticker") for w in watchlist],
                "stats": {
                    "total": len(watchlist),
                    "buy_count": buy_count,
                    "sell_count": sell_count,
                    "hold_count": hold_count,
                },
                "message": f"✅ Loaded {len(watchlist)} predictions",
            }),
        }
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e), "traceback": err_msg}),
        }
