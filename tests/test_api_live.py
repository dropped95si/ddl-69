import json
import unittest
from unittest.mock import patch

import api.live as live


class _Request:
    def __init__(self, args=None):
        self.args = args or {}


class LiveApiTests(unittest.TestCase):
    def test_handler_passes_run_scope_to_fetch(self) -> None:
        with patch.object(
            live,
            "_fetch_supabase",
            return_value=(None, {"error": "empty_watchlist", "active_run_id": "run-123"}),
        ) as fetch_mock:
            response = live._handler_impl(_Request({"timeframe": "swing", "run_id": "run-123"}))

        fetch_mock.assert_called_once_with(timeframe_filter="swing", run_id_filter="run-123")
        self.assertEqual(response["statusCode"], 200)
        body = json.loads(response["body"])
        self.assertEqual(body["run_id"], "run-123")
        self.assertEqual(body["count"], 0)
        self.assertIn("timeframe 'swing'", body["message"])

    def test_handler_success_includes_active_run_id(self) -> None:
        watchlist = [{"symbol": "AAPL", "signal": "BUY", "plan_type": "swing"}]
        with patch.object(
            live,
            "_fetch_supabase",
            return_value=(watchlist, {"active_run_id": "run-ok"}),
        ) as fetch_mock:
            response = live._handler_impl(_Request({"run_id": "run-ok"}))

        fetch_mock.assert_called_once_with(timeframe_filter="all", run_id_filter="run-ok")
        self.assertEqual(response["statusCode"], 200)
        body = json.loads(response["body"])
        self.assertEqual(body["run_id"], "run-ok")
        self.assertEqual(body["count"], 1)

    def test_event_market_cap_fallback_extracts_bucket(self) -> None:
        cap, bucket = live._event_market_cap_and_bucket({"context_json": {"market_cap": 1_500_000_000}})
        self.assertEqual(cap, 1_500_000_000)
        self.assertEqual(bucket, "small")

    def test_event_asset_type_fallback(self) -> None:
        asset = live._event_asset_type({"context_json": {"asset_type": "ETF"}})
        self.assertEqual(asset, "etf")


if __name__ == "__main__":
    unittest.main()
