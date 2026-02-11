"""Demo endpoint - Fast, always working, ready to click."""
import json
from datetime import datetime, timezone


def handler(request):
    """Return instant demo data for the UI."""
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json", "Cache-Control": "max-age=60"},
        "body": json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "watchlist": [
                    {
                        "symbol": "NVDA",
                        "name": "NVIDIA Corporation",
                        "price": 142.55,
                        "change": 2.34,
                        "score": 0.87,
                        "probability": 0.78,
                        "signal": "BUY",
                        "weights": {"technical": 0.35, "sentiment": 0.28, "fundamental": 0.22, "ensemble": 0.15}
                    },
                    {
                        "symbol": "TSLA",
                        "name": "Tesla Inc",
                        "price": 245.30,
                        "change": 1.82,
                        "score": 0.74,
                        "probability": 0.62,
                        "signal": "HOLD",
                        "weights": {"technical": 0.32, "sentiment": 0.30, "fundamental": 0.25, "ensemble": 0.13}
                    },
                    {
                        "symbol": "AAPL",
                        "name": "Apple Inc",
                        "price": 189.95,
                        "change": 0.45,
                        "score": 0.58,
                        "probability": 0.51,
                        "signal": "HOLD",
                        "weights": {"technical": 0.28, "sentiment": 0.25, "fundamental": 0.32, "ensemble": 0.15}
                    },
                    {
                        "symbol": "MSFT",
                        "name": "Microsoft Corp",
                        "price": 412.10,
                        "change": -0.32,
                        "score": 0.52,
                        "probability": 0.48,
                        "signal": "SELL",
                        "weights": {"technical": 0.25, "sentiment": 0.22, "fundamental": 0.28, "ensemble": 0.25}
                    },
                    {
                        "symbol": "GOOGL",
                        "name": "Alphabet Inc",
                        "price": 178.45,
                        "change": 1.12,
                        "score": 0.65,
                        "probability": 0.55,
                        "signal": "HOLD",
                        "weights": {"technical": 0.30, "sentiment": 0.26, "fundamental": 0.29, "ensemble": 0.15}
                    }
                ],
                "stats": {
                    "total_symbols": 5,
                    "avg_score": 0.672,
                    "buy_count": 1,
                    "hold_count": 3,
                    "sell_count": 1
                }
            },
            "message": "✅ Demo data loaded - ready to explore"
        }),
    }
