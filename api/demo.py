"""Demo endpoint disabled in strict mode."""

import json
from datetime import datetime, timezone

try:
    from _http_adapter import FunctionHandler
except ModuleNotFoundError:
    from api._http_adapter import FunctionHandler


def _handler_impl(request):
    return {
        "statusCode": 410,
        "headers": {
            "Content-Type": "application/json",
            "Cache-Control": "no-store",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(
            {
                "error": "endpoint_disabled",
                "message": "Demo endpoint is disabled in strict mode.",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        ),
    }


class handler(FunctionHandler):
    endpoint = staticmethod(_handler_impl)

