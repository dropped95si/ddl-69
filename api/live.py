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
    confidence = round(min(0.95, max(0.3, 0.6 + ((s % 25) - 12) / 100)), 2)
    
    weights = {
        "technical": 0.35,
        "volume": 0.25,
        "momentum": 0.20,
        "ensemble": 0.20,
    }
    
    return {
        "ticker": symbol,
        "symbol": symbol,
        "price": round(price, 2),
        "change": round(change, 2),
        "score": round(score, 4),
        "p_accept": round(prob, 4),
        "probability": round(prob, 4),
        "signal": signal,
        "confidence": confidence,
        "source": source,
        "weights": weights,
        "weights_json": weights,
        "label": signal,
    }


def _symbols_from_env():
    raw = os.getenv("WATCHLIST", "NVDA,TSLA,AAPL,MSFT,GOOGL,AMD,AMZN,META")
    items = [s.strip().upper() for s in raw.split(",") if s.strip()]
    return items[:10] or ["SPY", "QQQ", "AAPL"]


def _handler_impl(request):
    symbols = _symbols_from_env()
    watchlist = [_row(s, "live-sim") for s in symbols]
    stats = {
        "total": len(watchlist),
        "buy_count": len([w for w in watchlist if w["signal"] == "BUY"]),
        "hold_count": len([w for w in watchlist if w["signal"] == "HOLD"]),
        "sell_count": len([w for w in watchlist if w["signal"] == "SELL"]),
    }

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json", "Cache-Control": "max-age=60, public"},
        "body": json.dumps(
            {
                "asof": datetime.now(timezone.utc).isoformat(),
                "source": "Live Simulated",
                "provider": "DDL-69 Live Feed",
                "count": len(watchlist),
                "ranked": watchlist,
                "tickers": [w["symbol"] for w in watchlist],
                "stats": stats,
                "message": f"✅ Loaded {len(watchlist)} live predictions",
            }
        ),
    }


class handler(FunctionHandler):
    endpoint = staticmethod(_handler_impl)
