"""Predictions endpoint backed by real market TA rows."""

import json
from datetime import datetime, timezone

try:
    from _http_adapter import FunctionHandler
except ModuleNotFoundError:
    from api._http_adapter import FunctionHandler

try:
    from _real_market import build_rows_for_symbols
except ModuleNotFoundError:
    from api._real_market import build_rows_for_symbols


def _parse_symbols(raw):
    if not raw:
        return []
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def _to_signal_row(row):
    meta = row.get("meta") or {}
    return {
        "symbol": row.get("symbol"),
        "signal": row.get("signal"),
        "probability": row.get("p_accept"),
        "confidence": row.get("confidence"),
        "price": row.get("price"),
        "timestamp": row.get("created_at") or datetime.now(timezone.utc).isoformat(),
        "score": row.get("score"),
        "targets": {
            "tp1": meta.get("tp1"),
            "tp2": meta.get("tp2"),
            "tp3": meta.get("tp3"),
            "sl1": meta.get("sl1"),
            "sl2": meta.get("sl2"),
            "sl3": meta.get("sl3"),
        },
        "meta": meta,
    }


def signal_handler(request):
    symbol = (request.args.get("symbol", "").upper() if hasattr(request, "args") else "")
    if not symbol:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "symbol parameter required"}),
        }

    rows = build_rows_for_symbols([symbol], mode="swing")
    if not rows:
        return {
            "statusCode": 503,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": f"No live data available for {symbol}"}),
        }

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json", "Cache-Control": "max-age=60"},
        "body": json.dumps(_to_signal_row(rows[0])),
    }


def portfolio_handler(request):
    symbols = _parse_symbols(request.args.get("symbols", "SPY,QQQ,AAPL")) if hasattr(request, "args") else ["SPY", "QQQ", "AAPL"]
    rows = build_rows_for_symbols(symbols, mode="swing")
    signals = [_to_signal_row(r) for r in rows]

    buy_count = len([s for s in signals if s["signal"] == "BUY"])
    sell_count = len([s for s in signals if s["signal"] == "SELL"])
    hold_count = len([s for s in signals if s["signal"] == "HOLD"])
    avg_score = round(sum(float(s.get("score") or 0) for s in signals) / len(signals), 4) if signals else None

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
                "avg_score": avg_score,
                "source": "yahoo_screener_ta",
            }
        ),
    }


def refresh_handler(request):
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(
            {
                "status": "ok",
                "message": "Predictions refresh is computed on request from live market feed.",
            }
        ),
    }


def _handler_impl(request):
    path = request.path if hasattr(request, "path") else ""
    action = path.split("/")[-1] if path else "predictions"
    if action == "portfolio" or "portfolio" in path:
        return portfolio_handler(request)
    if action == "refresh" or "refresh" in path:
        return refresh_handler(request)
    return signal_handler(request)


class handler(FunctionHandler):
    endpoint = staticmethod(_handler_impl)

