"""Unified API router for Vercel.

Dispatches /api/* paths to existing endpoint modules so Vercel can build
one Python function instead of many separate ones.
"""

import json
import importlib

try:
    from _http_adapter import FunctionHandler
except ModuleNotFoundError:
    from api._http_adapter import FunctionHandler


# Public endpoint -> module name
MODULE_MAP = {
    "audit": "audit",
    "calibration": "calibration",
    "demo": "demo",
    "events": "events",
    "finviz": "finviz",
    "forecasts": "forecasts",
    "health": "health",
    "live": "live",
    "news": "news",
    "overlays": "overlays",
    "predictions": "predictions",
    "projection": "projection",
    "runs": "runs",
    "status": "status",
    "test": "test",
    "walkforward": "walkforward",
    "watchlist": "watchlist",
    # aliases
    "portfolio": "predictions",
}


def _extract_endpoint(path):
    raw = str(path or "").strip()
    if not raw:
        return ""
    if raw == "/api":
        return ""
    if raw.startswith("/api/"):
        raw = raw[len("/api/") :]
    raw = raw.strip("/")
    if raw.endswith(".py"):
        raw = raw[:-3]
    return raw.lower()


def _load_module(module_name):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        return importlib.import_module(f"api.{module_name}")


def _dispatch_to_module(module_name, request):
    mod = _load_module(module_name)
    endpoint_fn = getattr(mod, "_handler_impl", None)
    if callable(endpoint_fn):
        return endpoint_fn(request)

    handler_cls = getattr(mod, "handler", None)
    endpoint_attr = getattr(handler_cls, "endpoint", None) if handler_cls else None
    if callable(endpoint_attr):
        return endpoint_attr(request)

    raise RuntimeError(f"Endpoint module '{module_name}' has no callable handler")


def _handler_impl(request):
    endpoint = _extract_endpoint(getattr(request, "path", ""))
    if not endpoint:
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Cache-Control": "no-store",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps(
                {
                    "service": "ddl-69-api-router",
                    "message": "Use /api/{endpoint}",
                    "endpoints": sorted(MODULE_MAP.keys()),
                }
            ),
        }

    module_name = MODULE_MAP.get(endpoint)
    if not module_name:
        return {
            "statusCode": 404,
            "headers": {
                "Content-Type": "application/json",
                "Cache-Control": "no-store",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps(
                {
                    "error": "not_found",
                    "message": f"Unknown endpoint '/api/{endpoint}'.",
                }
            ),
        }

    try:
        return _dispatch_to_module(module_name, request)
    except Exception as exc:
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Cache-Control": "no-store",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps(
                {
                    "error": "router_dispatch_error",
                    "endpoint": endpoint,
                    "message": str(exc),
                }
            ),
        }


class handler(FunctionHandler):
    endpoint = staticmethod(_handler_impl)
