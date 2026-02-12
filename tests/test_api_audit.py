import json
import unittest
from unittest.mock import patch

import api.audit as audit


class _Request:
    def __init__(self, args=None):
        self.args = args or {}


class AuditApiTests(unittest.TestCase):
    def test_dedupe_predictions_keeps_best_confidence(self) -> None:
        rows = [
            {"ticker": "XLP", "confidence": 0.81, "created_at": "2026-02-12T01:00:00+00:00", "p_accept": 0.8},
            {"ticker": "XLP", "confidence": 0.93, "created_at": "2026-02-12T00:00:00+00:00", "p_accept": 0.9},
            {"ticker": "SPY", "confidence": 0.75, "created_at": "2026-02-12T01:00:00+00:00", "p_accept": 0.7},
        ]
        deduped = audit._dedupe_predictions(rows)
        self.assertEqual(len(deduped), 2)
        xlp = next((r for r in deduped if r["ticker"] == "XLP"), None)
        self.assertIsNotNone(xlp)
        self.assertEqual(float(xlp["confidence"]), 0.93)

    def test_dedupe_predictions_breaks_tie_by_recency(self) -> None:
        rows = [
            {"ticker": "XLP", "confidence": 0.90, "created_at": "2026-02-12T00:00:00+00:00", "p_accept": 0.89},
            {"ticker": "XLP", "confidence": 0.90, "created_at": "2026-02-12T02:00:00+00:00", "p_accept": 0.88},
        ]
        deduped = audit._dedupe_predictions(rows)
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0]["created_at"], "2026-02-12T02:00:00+00:00")

    def test_handler_enables_distinct_tickers_by_default(self) -> None:
        with patch.object(audit, "_fetch_supabase_predictions", return_value=[]) as fetch_mock:
            response = audit.audit_handler(_Request())
        self.assertEqual(response["statusCode"], 503)
        fetch_mock.assert_called_once_with(limit=10, distinct_tickers=True, timeframe_filter="all")

    def test_handler_can_disable_distinct_tickers(self) -> None:
        with patch.object(audit, "_fetch_supabase_predictions", return_value=[]) as fetch_mock:
            response = audit.audit_handler(_Request({"distinct_tickers": "0"}))
        self.assertEqual(response["statusCode"], 503)
        fetch_mock.assert_called_once_with(limit=10, distinct_tickers=False, timeframe_filter="all")

    def test_handler_returns_empty_payload_for_scoped_no_rows(self) -> None:
        with patch.object(audit, "_fetch_supabase_predictions", return_value=[]):
            response = audit.audit_handler(_Request({"timeframe": "long"}))
        self.assertEqual(response["statusCode"], 200)
        body = json.loads(response["body"])
        self.assertEqual(body["requested_timeframe"], "long")
        self.assertEqual(body["summary"]["total_predictions"], 0)
        self.assertEqual(body["predictions"], [])
        self.assertIn("No rows available for timeframe 'long'", body["message"])

    def test_parse_horizon_days_supports_multiple_units(self) -> None:
        self.assertEqual(audit._parse_horizon_days({"value": 2, "unit": "weeks"}), 14.0)
        self.assertEqual(audit._parse_horizon_days({"value": 6, "unit": "months"}), 180.0)
        self.assertEqual(audit._parse_horizon_days({"value": 1, "unit": "year"}), 365.0)
        self.assertEqual(audit._parse_horizon_days("18mo"), 540.0)
        self.assertEqual(audit._classify_timeframe_correct(audit._parse_horizon_days("18mo")), "long")

    def test_handler_returns_predictions_payload(self) -> None:
        sample = [
            {
                "ticker": "XLP",
                "price": 88.4,
                "confidence": 0.9,
                "p_accept": 0.9,
                "p_reject": 0.1,
                "p_continue": 0.0,
                "signal": "BUY",
                "method": "hedge",
                "horizon_days": 5.0,
                "timeframe": "day",
                "created_at": "2026-02-12T00:00:00+00:00",
                "weights": {"rule_a": 0.6, "rule_b": 0.4},
                "tp1": 89.0,
                "tp2": 89.5,
                "tp3": 90.0,
                "sl1": 87.8,
                "expected_return": 0.009,
            }
        ]
        with patch.object(audit, "_fetch_supabase_predictions", return_value=sample):
            response = audit.audit_handler(_Request({"limit": "20"}))
        self.assertEqual(response["statusCode"], 200)
        body = json.loads(response["body"])
        self.assertEqual(body["summary"]["total_predictions"], 1)
        self.assertEqual(body["predictions"][0]["ticker"], "XLP")


if __name__ == "__main__":
    unittest.main()
