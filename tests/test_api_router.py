import json
import unittest
from unittest.mock import patch

import api.index as router


class _Request:
    def __init__(self, path, args=None):
        self.path = path
        self.args = args or {}
        self.method = "GET"
        self.headers = {}
        self.body = b""


class ApiRouterTests(unittest.TestCase):
    def test_root_api_lists_endpoints(self):
        resp = router._handler_impl(_Request("/api"))
        self.assertEqual(resp["statusCode"], 200)
        body = json.loads(resp["body"])
        self.assertIn("endpoints", body)
        self.assertIn("live", body["endpoints"])

    def test_unknown_endpoint_404(self):
        resp = router._handler_impl(_Request("/api/nope"))
        self.assertEqual(resp["statusCode"], 404)
        body = json.loads(resp["body"])
        self.assertEqual(body["error"], "not_found")

    def test_dispatches_live_endpoint(self):
        req = _Request("/api/live", {"timeframe": "swing"})
        with patch("api.live._handler_impl", return_value={"statusCode": 200, "body": "{}"}) as live_mock:
            resp = router._handler_impl(req)
        self.assertEqual(resp["statusCode"], 200)
        live_mock.assert_called_once()
        passed_req = live_mock.call_args[0][0]
        self.assertEqual(passed_req.path, "/api/live")
        self.assertEqual(passed_req.args.get("timeframe"), "swing")

    def test_portfolio_alias_dispatches_predictions(self):
        req = _Request("/api/portfolio")
        with patch("api.predictions._handler_impl", return_value={"statusCode": 200, "body": "{}"}) as pred_mock:
            resp = router._handler_impl(req)
        self.assertEqual(resp["statusCode"], 200)
        pred_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
