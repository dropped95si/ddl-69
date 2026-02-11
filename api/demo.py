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
                            "confidence": 0.92,
                            "probability": 0.78,
                            "signal": "BUY",
                            "reasoning": "Strong technical breakout above 20-day MA with bullish sentiment from institutional buying. RSI in overbought but showing momentum.",
                            "targets": {
                                "tp1": 148.50,
                                "tp2": 155.00,
                                "tp3": 165.25,
                            },
                            "stoploss": {
                                "sl1": 138.50,
                                "sl2": 132.00,
                                "sl3": 125.00,
                            },
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
                            "confidence": 0.68,
                            "probability": 0.62,
                            "signal": "HOLD",
                            "reasoning": "Consolidation phase near 20-week moving average. Mixed signals from sentiment metrics. Waiting for breakout confirmation above 250.",
                            "targets": {
                                "tp1": 255.00,
                                "tp2": 268.50,
                                "tp3": 285.00,
                            },
                            "stoploss": {
                                "sl1": 235.00,
                                "sl2": 220.00,
                                "sl3": 200.00,
                            },
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
                            "confidence": 0.55,
                            "probability": 0.51,
                            "signal": "HOLD",
                            "reasoning": "Neutral zone. Support holding at 185 but lacks conviction for upside. Fundamental strength offset by macro headwinds.",
                            "targets": {
                                "tp1": 195.00,
                                "tp2": 205.50,
                                "tp3": 218.00,
                            },
                            "stoploss": {
                                "sl1": 183.00,
                                "sl2": 175.00,
                                "sl3": 165.00,
                            },
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
                        "avg_confidence": 0.72,
                        "buy_count": 1,
                        "hold_count": 2,
                        "sell_count": 0,
                    },
                },
                "message": "Demo data with detailed targets and stoploss levels",
            }
        ),
    }


class handler(FunctionHandler):
    endpoint = staticmethod(_handler_impl)
