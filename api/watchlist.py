import json
from datetime import datetime, timezone

def handler(request):
    try:
        watchlist = [
            {"ticker": "NVDA", "symbol": "NVDA", "label": "BUY", "score": 0.87, "p_accept": 0.78, "signal": "BUY", "weights": {"technical": 0.35, "sentiment": 0.25, "fundamental": 0.25, "ensemble": 0.15}, "weights_json": {"technical": 0.35, "sentiment": 0.25, "fundamental": 0.25, "ensemble": 0.15}},
            {"ticker": "TSLA", "symbol": "TSLA", "label": "HOLD", "score": 0.62, "p_accept": 0.58, "signal": "HOLD", "weights": {"technical": 0.30, "sentiment": 0.25, "fundamental": 0.30, "ensemble": 0.15}, "weights_json": {"technical": 0.30, "sentiment": 0.25, "fundamental": 0.30, "ensemble": 0.15}},
            {"ticker": "SPY", "symbol": "SPY", "label": "HOLD", "score": 0.55, "p_accept": 0.52, "signal": "HOLD", "weights": {"technical": 0.30, "sentiment": 0.25, "fundamental": 0.30, "ensemble": 0.15}, "weights_json": {"technical": 0.30, "sentiment": 0.25, "fundamental": 0.30, "ensemble": 0.15}},
        ]
        
        buy_count = len([w for w in watchlist if w.get("signal") == "BUY"])
        sell_count = len([w for w in watchlist if w.get("signal") == "SELL"])
        hold_count = len([w for w in watchlist if w.get("signal") == "HOLD"])
        
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json", "Cache-Control": "max-age=60"},
            "body": json.dumps({
                "asof": datetime.now(timezone.utc).isoformat(),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "source": "Demo Data",
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
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e), "traceback": traceback.format_exc()}),
        }
