"""Legacy demo endpoint now returns live watchlist feed."""

try:
    from _http_adapter import FunctionHandler
except ModuleNotFoundError:
    from api._http_adapter import FunctionHandler

try:
    from live import _handler_impl as _live_handler
except ModuleNotFoundError:
    from api.live import _handler_impl as _live_handler


def _handler_impl(request):
    return _live_handler(request)


class handler(FunctionHandler):
    endpoint = staticmethod(_handler_impl)

