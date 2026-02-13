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

    def test_derived_summary_includes_open_source_diagnostics(self) -> None:
        class _Resp:
            def __init__(self, data):
                self.data = data

        class _Query:
            def __init__(self, table_name, tables):
                self.table_name = table_name
                self.tables = tables
                self._in_col = None
                self._in_vals = None

            def select(self, *_args, **_kwargs):
                return self

            def order(self, *_args, **_kwargs):
                return self

            def limit(self, *_args, **_kwargs):
                return self

            def in_(self, col, vals):
                self._in_col = col
                self._in_vals = set(vals or [])
                return self

            def execute(self):
                data = list(self.tables.get(self.table_name, []))
                if self._in_col is not None:
                    data = [r for r in data if r.get(self._in_col) in self._in_vals]
                return _Resp(data)

        class _FakeSupabase:
            def __init__(self, tables):
                self.tables = tables

            def table(self, name):
                return _Query(name, self.tables)

        tables = {
            "v_latest_ensemble_forecasts": [
                {
                    "event_id": "evt-1",
                    "weights_json": {"trend": 0.6, "rsi": 0.2, "momentum": 0.1},
                    "created_at": "2026-02-13T06:00:00+00:00",
                    "run_id": "run-a",
                    "probs_json": {"ACCEPT_CONTINUE": 0.71, "REJECT": 0.29, "BREAK_FAIL": 0.0},
                    "confidence": 0.73,
                    "method": "hedge",
                },
                {
                    "event_id": "evt-2",
                    "weights_json": {"trend": 0.4, "rsi": 0.15, "momentum": 0.05},
                    "created_at": "2026-02-13T05:59:00+00:00",
                    "run_id": "run-a",
                    "probs_json": {"ACCEPT_CONTINUE": 0.66, "REJECT": 0.34, "BREAK_FAIL": 0.0},
                    "confidence": 0.69,
                    "method": "hedge",
                },
            ],
            "events": [
                {
                    "event_id": "evt-1",
                    "horizon_json": {"type": "time", "value": 10, "unit": "d"},
                    "asof_ts": "2026-02-13T06:00:00+00:00",
                    "subject_id": "AAA",
                    "context_json": {"market_cap": 1_500_000_000},
                    "event_params_json": {},
                },
                {
                    "event_id": "evt-2",
                    "horizon_json": {"type": "time", "value": 10, "unit": "d"},
                    "asof_ts": "2026-02-13T05:59:00+00:00",
                    "subject_id": "BBB",
                    "context_json": {"market_cap": 15_000_000_000},
                    "event_params_json": {},
                },
            ],
        }

        with patch.object(walkforward, "_get_supabase_client", return_value=_FakeSupabase(tables)):
            payload = walkforward._derive_from_supabase_forecasts(timeframe_filter="day", run_id_filter="run-a")

        self.assertIsNotNone(payload)
        summary = payload.get("summary", {})
        diagnostics = summary.get("diagnostics", {})
        stats = summary.get("stats", {})
        self.assertIn("open_source_methods", diagnostics)
        self.assertIn("bootstrap_mean_ci", diagnostics.get("open_source_methods", []))
        self.assertIn("concentration", diagnostics)
        self.assertIn("probability", diagnostics)
        self.assertIn("benchmarks", diagnostics)
        self.assertIn("rolling_windows", diagnostics)
        self.assertIn("cap_bucket_stability", diagnostics)
        self.assertIsNotNone(stats.get("net_weight_ci_low"))
        self.assertIsNotNone(stats.get("net_weight_ci_high"))
        self.assertGreater(len(summary.get("weights_top", [])), 0)
        self.assertIn("ci_low", summary.get("weights_top", [])[0])


if __name__ == "__main__":
    unittest.main()
