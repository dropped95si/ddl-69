import json
import unittest
from unittest.mock import patch

import api.runs as runs


class _Request:
    def __init__(self, args=None):
        self.args = args or {}


class RunsApiTests(unittest.TestCase):
    def test_handler_returns_catalog_payload(self) -> None:
        payload = {
            "runs": [
                {
                    "run_id": "run-new",
                    "rows": 120,
                    "timeframe_counts": {"day": 20, "swing": 90, "long": 10},
                    "methods": ["hedge"],
                }
            ],
            "latest_run_id": "run-new",
            "run_count": 1,
            "sampled_forecast_rows": 120,
            "runs_table_total": 38,
        }
        with patch.object(runs, "_fetch_run_catalog", return_value=(payload, None)) as fetch_mock:
            response = runs._handler_impl(_Request({"limit_runs": "5", "lookback_rows": "900"}))
        self.assertEqual(response["statusCode"], 200)
        fetch_mock.assert_called_once_with(limit_runs=5, lookback_rows=900)
        body = json.loads(response["body"])
        self.assertEqual(body["latest_run_id"], "run-new")
        self.assertEqual(body["runs"][0]["run_id"], "run-new")

    def test_handler_returns_503_on_fetch_error(self) -> None:
        with patch.object(runs, "_fetch_run_catalog", return_value=(None, {"error": "missing_credentials"})):
            response = runs._handler_impl(_Request())
        self.assertEqual(response["statusCode"], 503)
        body = json.loads(response["body"])
        self.assertEqual(body["error"], "supabase_unavailable")


if __name__ == "__main__":
    unittest.main()
