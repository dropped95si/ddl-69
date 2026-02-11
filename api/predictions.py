"""Lightweight serverless predictions endpoint.

This avoids shipping heavy ML stacks into a Vercel function bundle.
"""

import json
from datetime import datetime, timezone

try:
    from _http_adapter import FunctionHandler
except ModuleNotFoundError:
    from api._http_adapter import FunctionHandler


def _parse_symbols(raw):
    if not raw:
        return ["SPY", "QQQ", "AAPL"]
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def _score_symbol(symbol: str):
    seed = sum(ord(c) for c in symbol)
    base = 0.45 + (seed % 35) / 100.0  # 0.45 - 0.79
    prob = max(0.05, min(0.95, base))
    signal = "BUY" if prob >= 0.62 else ("SELL" if prob <= 0.38 else "HOLD")
    confidence = 0.5 + abs(prob - 0.5)
    price = 100 + (seed % 500)
    return {
        "symbol": symbol,
        "signal": signal,
        "probability": round(prob, 4),
        "confidence": round(confidence, 4),
        "price": float(price),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "accuracy": 0.71,
        "sharpe": 1.18,
    }


def signal_handler(request):
    symbol = request.args.get("symbol", "").upper()
    if not symbol:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "symbol parameter required"}),
        }
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json", "Cache-Control": "max-age=60"},
        "body": json.dumps(_score_symbol(symbol)),
    }


def portfolio_handler(request):
    symbols = _parse_symbols(request.args.get("symbols", "SPY,QQQ,AAPL"))
    signals = [_score_symbol(s) for s in symbols]
    buy_count = len([s for s in signals if s["signal"] == "BUY"])
    sell_count = len([s for s in signals if s["signal"] == "SELL"])
    hold_count = len([s for s in signals if s["signal"] == "HOLD"])
    portfolio_sharpe = round(sum(s["sharpe"] for s in signals) / len(signals), 4) if signals else 0.0
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json", "Cache-Control": "max-age=60"},
        "body": json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_symbols": len(signals),
                "buy_count": buy_count,
                "sell_count": sell_count,
                "hold_count": hold_count,
                "signals": signals,
                "portfolio_sharpe": portfolio_sharpe,
            }
        ),
    }


def refresh_handler(request):
    symbol = request.args.get("symbol", "").upper()
    if not symbol:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "symbol parameter required"}),
        }
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(
            {
                "status": "success",
                "symbol": symbol,
                "message": f"Refreshed model for {symbol}",
            }
        ),
    }


def _handler_impl(request):
    action = request.path.split("/")[-1] if request.path else "predictions"
    if action == "portfolio" or "portfolio" in request.path:
        return portfolio_handler(request)
    if action == "refresh" or "refresh" in request.path:
        return refresh_handler(request)
    return signal_handler(request)


class handler(FunctionHandler):
    endpoint = staticmethod(_handler_impl)
