"""Finviz-style watchlist endpoint backed by real Yahoo screener + TA data."""

import json
from datetime import datetime, timezone

try:
    from _http_adapter import FunctionHandler
except ModuleNotFoundError:
    from api._http_adapter import FunctionHandler

try:
    from _real_market import build_watchlist
except ModuleNotFoundError:
    from api._real_market import build_watchlist


def _handler_impl(request):
    mode = (request.args.get("mode") if hasattr(request, "args") else None) or "swing"
    mode = mode.lower().strip()
    if mode not in ("day", "swing", "long"):
        mode = "swing"

    count_raw = (request.args.get("count") if hasattr(request, "args") else None) or "100"
    try:
        count = max(1, min(300, int(count_raw)))
    except Exception:
        count = 100

    rows = build_watchlist(mode=mode, count=count)
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Cache-Control": "max-age=60, public",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(
            {
                "asof": datetime.now(timezone.utc).isoformat(),
                "source": f"finviz:{mode}:yahoo_screener_ta",
                "count": len(rows),
                "rows": rows,
            }
        ),
    }


class handler(FunctionHandler):
    endpoint = staticmethod(_handler_impl)

