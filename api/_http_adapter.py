import json
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse


@dataclass
class Request:
    path: str
    args: dict
    method: str
    headers: dict
    body: bytes


class FunctionHandler(BaseHTTPRequestHandler):
    endpoint = None

    def _to_request(self) -> Request:
        parsed = urlparse(self.path or "/")
        args = {k: (v[-1] if v else "") for k, v in parse_qs(parsed.query).items()}
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
        except (ValueError, TypeError):
            length = 0
        body = self.rfile.read(length) if length > 0 else b""
        return Request(
            path=parsed.path,
            args=args,
            method=self.command,
            headers={k: v for k, v in self.headers.items()},
            body=body,
        )

    def _write(self, response: dict) -> None:
        status = int(response.get("statusCode", 200))
        headers = response.get("headers", {}) or {}
        body = response.get("body", "")

        if isinstance(body, (dict, list)):
            body = json.dumps(body)
            headers.setdefault("Content-Type", "application/json")
        elif not isinstance(body, (str, bytes)):
            body = str(body)

        if isinstance(body, str):
            body = body.encode("utf-8")

        self.send_response(status)
        for k, v in headers.items():
            self.send_header(str(k), str(v))
        if "Content-Type" not in headers:
            self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _dispatch(self) -> None:
        try:
            req = self._to_request()
            if not callable(self.endpoint):
                raise RuntimeError("Endpoint not configured")
            resp = self.endpoint(req)
        except Exception as exc:
            resp = {
                "statusCode": 500,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": str(exc)}),
            }
        self._write(resp)

    def do_GET(self):  # noqa: N802
        self._dispatch()

    def do_POST(self):  # noqa: N802
        self._dispatch()

    def do_OPTIONS(self):  # noqa: N802
        self._write(
            {
                "statusCode": 200,
                "headers": {
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
                    "Access-Control-Allow-Headers": "*",
                },
                "body": "",
            }
        )

    def log_message(self, format, *args):  # noqa: A003
        return
