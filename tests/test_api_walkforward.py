import json
import os
import unittest
from unittest.mock import patch

import api.walkforward as walkforward


class _Request:
    def __init__(self, args=None):
        self.args = args or {}


class WalkforwardApiTests(unittest.TestCase):
    def test_no_fallback_without_allow_derived(self) -> None:
        request = _Request({"timeframe": "day"})
        with (
            patch.dict(os.environ, {"WALKFORWARD_ALLOW_DERIVED": "0"}, clear=False),
            patch.object(walkforward, "_fetch_walkforward_artifact", return_value=None),
            patch.object(walkforward, "_derive_from_supabase_forecasts", return_value={"summary": {"run_id": "derived"}}) as derive_mock,
        ):
            response = walkforward._handler_impl(request)

        self.assertEqual(response["statusCode"], 503)
        derive_mock.assert_not_called()
        body = json.loads(response["body"])
        self.assertEqual(body["error"], "supabase_unavailable")
        self.assertIn("no fallback enabled", body["message"])

    def test_default_env_allows_derived_without_query_flag(self) -> None:
        request = _Request({"timeframe": "day"})
        derived_payload = {"summary": {"run_id": "run-day"}}
        with (
            patch.dict(os.environ, {}, clear=False),
            patch.object(walkforward, "_fetch_walkforward_artifact", return_value=None) as artifact_mock,
            patch.object(walkforward, "_derive_from_supabase_forecasts", return_value=derived_payload) as derive_mock,
        ):
            response = walkforward._handler_impl(request)

        self.assertEqual(response["statusCode"], 200)
        artifact_mock.assert_called_once()
        derive_mock.assert_called_once_with(timeframe_filter="day", run_id_filter="")

    def test_scoped_timeframe_uses_derived_payload_when_enabled(self) -> None:
        request = _Request({"timeframe": "day", "allow_derived": "1"})
        derived_payload = {"summary": {"run_id": "run-day"}}

        with (
            patch.object(walkforward, "_fetch_walkforward_artifact", return_value=None) as artifact_mock,
            patch.object(walkforward, "_derive_from_supabase_forecasts", return_value=derived_payload) as derive_mock,
        ):
            response = walkforward._handler_impl(request)

        self.assertEqual(response["statusCode"], 200)
        artifact_mock.assert_called_once()
        derive_mock.assert_called_once_with(timeframe_filter="day", run_id_filter="")
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

    def test_scoped_failure_returns_timeframe_error_when_derived_enabled(self) -> None:
        request = _Request({"timeframe": "long", "allow_derived": "true"})

        with (
            patch.object(walkforward, "_fetch_walkforward_artifact", return_value=None),
            patch.object(walkforward, "_derive_from_supabase_forecasts", return_value=None),
        ):
            response = walkforward._handler_impl(request)

        self.assertEqual(response["statusCode"], 503)
        body = json.loads(response["body"])
        self.assertEqual(body["error"], "supabase_unavailable")
        self.assertIn("timeframe 'long'", body["message"])

    def test_run_id_is_forwarded_to_derived_path(self) -> None:
        request = _Request({"timeframe": "swing", "run_id": "run-xyz", "allow_derived": "1"})
        derived_payload = {"summary": {"run_id": "run-xyz"}}

        with (
            patch.object(walkforward, "_fetch_walkforward_artifact", return_value=None),
            patch.object(walkforward, "_derive_from_supabase_forecasts", return_value=derived_payload) as derive_mock,
        ):
            response = walkforward._handler_impl(request)

        self.assertEqual(response["statusCode"], 200)
        derive_mock.assert_called_once_with(timeframe_filter="swing", run_id_filter="run-xyz")
        body = json.loads(response["body"])
        self.assertEqual(body["summary"]["requested_run_id"], "run-xyz")


if __name__ == "__main__":
    unittest.main()
