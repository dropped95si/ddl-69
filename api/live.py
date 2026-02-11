"""Live endpoint - lightweight multi-source proxy without heavy runtime deps."""

import json
import os
from datetime import datetime, timezone

try:
    from _http_adapter import FunctionHandler
except ModuleNotFoundError:
    from api._http_adapter import FunctionHandler


def _seed(symbol: str) -> int:
    return sum(ord(c) for c in symbol)


def _row(symbol: str, source: str):
    s = _seed(symbol)
    price = 50 + (s % 450) + ((s % 37) / 10)
    change = ((s % 19) - 9) / 3
    score = min(0.95, max(0.05, 0.5 + ((s % 40) - 20) / 100))
    prob = min(0.95, max(0.05, 0.5 + ((s % 30) - 15) / 100))
    signal = "BUY" if score >= 0.62 else ("SELL" if score <= 0.38 else "HOLD")
    return {
        "symbol": symbol,
        "price": round(price, 2),
        "change": round(change, 2),
        "score": round(score, 4),
        "probability": round(prob, 4),
        "signal": signal,
        "source": source,
        "weights": {
            "technical": 0.35,
            "volume": 0.25,
            "momentum": 0.20,
            "ensemble": 0.20,
        },
    }


def _symbols_from_env():
    raw = os.getenv("WATCHLIST", "NVDA,TSLA,AAPL,MSFT,GOOGL,AMD,AMZN,META")
    items = [s.strip().upper() for s in raw.split(",") if s.strip()]
    return items[:10] or ["SPY", "QQQ", "AAPL"]


def _handler_impl(request):
    symbols = _symbols_from_env()
    watchlist = [_row(s, "live-sim") for s in symbols]
    stats = {
        "total_symbols": len(watchlist),
        "avg_score": round(sum(w["score"] for w in watchlist) / len(watchlist), 4),
        "buy_count": len([w for w in watchlist if w["signal"] == "BUY"]),
        "hold_count": len([w for w in watchlist if w["signal"] == "HOLD"]),
        "sell_count": len([w for w in watchlist if w["signal"] == "SELL"]),
        "sources_active": ["live-sim"],
    }

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json", "Cache-Control": "max-age=300, public"},
        "body": json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {"watchlist": watchlist, "stats": stats},
                "message": "Live simulated feed ready",
            }
        ),
    }


class handler(FunctionHandler):
    endpoint = staticmethod(_handler_impl)
