import unittest

import api.calibration as calibration


class _Resp:
    def __init__(self, data):
        self.data = data


class _Query:
    def __init__(self, data):
        self._data = data

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def order(self, *_args, **_kwargs):
        return self

    def limit(self, *_args, **_kwargs):
        return self

    def execute(self):
        return _Resp(self._data)


class _Supa:
    def __init__(self, mapping):
        self._mapping = mapping

    def table(self, name):
        return _Query(self._mapping.get(name, []))


class CalibrationApiTests(unittest.TestCase):
    def test_fetch_artifact_skips_reference_only_calibration_meta(self) -> None:
        rows = [
            {
                "meta_json": {"type": "calibration", "bars_url": "https://example/bars.json"},
                "created_at": "2026-02-12T00:00:00+00:00",
            },
            {
                "meta_json": {
                    "type": "calibration",
                    "var_95": -0.1,
                    "cvar_95": -0.2,
                    "max_drawdown": -0.3,
                    "sharpe_mean": 1.2,
                },
                "created_at": "2026-02-13T00:00:00+00:00",
            },
        ]
        supa = _Supa({"artifacts": rows})
        payload = calibration._fetch_calibration_artifact(supa)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["var_95"], -0.1)
        self.assertEqual(payload["artifact_created_at"], "2026-02-13T00:00:00+00:00")

    def test_derived_payload_includes_weight_sample_count(self) -> None:
        pred_rows = []
        for i in range(12):
            pred_rows.append(
                {
                    "weights_json": {
                        "rule_pos": 0.05 + (i * 0.001),
                        "rule_neg": -0.02 - (i * 0.001),
                        "rule_zero": 0.0,
                    },
                    "created_at": f"2026-02-13T00:{i:02d}:00+00:00",
                    "run_id": "run-abc",
                }
            )
        supa = _Supa({"v_latest_ensemble_forecasts": pred_rows})
        payload = calibration._derive_from_predictions(supa)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["n_simulations"], 12)
        self.assertGreaterEqual(payload["n_weight_samples"], 24)
        self.assertEqual(payload["source"], "supabase_predictions_derived")
        self.assertEqual(payload["run_id"], "run-abc")


if __name__ == "__main__":
    unittest.main()
