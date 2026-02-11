"""Demo endpoint - fast static payload for dashboard smoke tests."""

import json
from datetime import datetime, timezone

try:
    from _http_adapter import FunctionHandler
except ModuleNotFoundError:
    from api._http_adapter import FunctionHandler


def _handler_impl(request):
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json", "Cache-Control": "max-age=300, public"},
        "body": json.dumps(
            {
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
                            "weights": {
                                "technical": 0.35,
                                "sentiment": 0.28,
                                "fundamental": 0.22,
                                "ensemble": 0.15,
                            },
                        },
                        {
                            "symbol": "TSLA",
                            "name": "Tesla Inc",
                            "price": 245.30,
                            "change": 1.82,
                            "score": 0.74,
                            "probability": 0.62,
                            "signal": "HOLD",
                            "weights": {
                                "technical": 0.32,
                                "sentiment": 0.30,
                                "fundamental": 0.25,
                                "ensemble": 0.13,
                            },
                        },
                        {
                            "symbol": "AAPL",
                            "name": "Apple Inc",
                            "price": 189.95,
                            "change": 0.45,
                            "score": 0.58,
                            "probability": 0.51,
                            "signal": "HOLD",
                            "weights": {
                                "technical": 0.28,
                                "sentiment": 0.25,
                                "fundamental": 0.32,
                                "ensemble": 0.15,
                            },
                        },
                    ],
                    "stats": {
                        "total_symbols": 3,
                        "avg_score": 0.73,
                        "buy_count": 1,
                        "hold_count": 2,
                        "sell_count": 0,
                    },
                },
                "message": "Demo data loaded",
            }
        ),
    }


class handler(FunctionHandler):
    endpoint = staticmethod(_handler_impl)
