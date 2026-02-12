import json
import unittest
from unittest.mock import patch

import api.walkforward as walkforward


class _Request:
    def __init__(self, args=None):
        self.args = args or {}


class WalkforwardApiTests(unittest.TestCase):
    def test_scoped_timeframe_uses_derived_payload(self) -> None:
        request = _Request({"timeframe": "day"})
        derived_payload = {"summary": {"run_id": "run-day"}}

        with (
            patch.object(walkforward, "_fetch_walkforward_artifact", return_value={"summary": {"run_id": "artifact"}}) as artifact_mock,
            patch.object(walkforward, "_derive_from_supabase_forecasts", return_value=derived_payload) as derive_mock,
        ):
            response = walkforward._handler_impl(request)

        self.assertEqual(response["statusCode"], 200)
        artifact_mock.assert_not_called()
        derive_mock.assert_called_once_with(timeframe_filter="day")
        body = json.loads(response["body"])
        self.assertEqual(body["summary"]["timeframe"], "day")

    def test_invalid_timeframe_falls_back_to_all(self) -> None:
        request = _Request({"timeframe": "bad_value"})
        artifact_payload = {"summary": {"run_id": "artifact"}}

        with (
            patch.object(walkforward, "_fetch_walkforward_artifact", return_value=artifact_payload) as artifact_mock,
            patch.object(walkforward, "_derive_from_supabase_forecasts", return_value=None) as derive_mock,
        ):
            response = walkforward._handler_impl(request)

        self.assertEqual(response["statusCode"], 200)
        artifact_mock.assert_called_once()
        derive_mock.assert_not_called()
        body = json.loads(response["body"])
        self.assertEqual(body["summary"]["timeframe"], "all")

    def test_scoped_failure_returns_timeframe_error(self) -> None:
        request = _Request({"timeframe": "long"})

        with (
            patch.object(walkforward, "_fetch_walkforward_artifact", return_value=None),
            patch.object(walkforward, "_derive_from_supabase_forecasts", return_value=None),
        ):
            response = walkforward._handler_impl(request)

        self.assertEqual(response["statusCode"], 503)
        body = json.loads(response["body"])
        self.assertEqual(body["error"], "supabase_unavailable")
        self.assertIn("timeframe 'long'", body["message"])


if __name__ == "__main__":
    unittest.main()
