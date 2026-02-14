"""Overlay endpoint with real technical overlays from market data."""

import json

try:
    from _http_adapter import FunctionHandler
except ModuleNotFoundError:
    from api._http_adapter import FunctionHandler

try:
    from _real_market import build_overlay_payload, build_symbol_universe
except ModuleNotFoundError:
    from api._real_market import build_overlay_payload, build_symbol_universe


def _parse_symbols(raw: str):
    if not raw:
        return []
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def _handler_impl(request):
    args = request.args if hasattr(request, "args") else {}
    mode = (args.get("mode") or "swing").lower()
    if mode not in ("day", "swing", "long"):
        mode = "swing"

    symbols = _parse_symbols(args.get("symbols") or "")
    if not symbols:
        limit_raw = args.get("count") or "30"
        try:
            limit = max(5, min(120, int(limit_raw)))
        except Exception:
            limit = 30
        symbols = build_symbol_universe(mode, limit)

    payload = build_overlay_payload(symbols=symbols, mode=mode, bars_limit=220)
    if not payload.get("symbols"):
        return {
            "statusCode": 503,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(
                {
                    "error": "No overlay data available from live market feed.",
                    "source": "yahoo_screener_ta",
                }
            ),
        }

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Cache-Control": "max-age=90, public",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(payload),
    }


class handler(FunctionHandler):
    endpoint = staticmethod(_handler_impl)

