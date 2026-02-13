import unittest
from unittest.mock import patch

import api._prices as prices


class _Resp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


class PriceApiTests(unittest.TestCase):
    def setUp(self) -> None:
        prices._cache.clear()
        prices._quote_cache.clear()
        prices._profile_cache.clear()

    def tearDown(self) -> None:
        prices._cache.clear()
        prices._quote_cache.clear()
        prices._profile_cache.clear()

    def test_fetch_quote_snapshots_merges_polygon_profile(self) -> None:
        yahoo_payload = {
            "quoteResponse": {
                "result": [
                    {
                        "symbol": "AAPL",
                        "regularMarketPrice": 190.25,
                        "marketCap": None,
                        "quoteType": None,
                    }
                ]
            }
        }

        with patch.object(prices.requests, "get", return_value=_Resp(200, yahoo_payload)):
            with patch.object(
                prices,
                "_fetch_polygon_profiles",
                return_value={"AAPL": {"market_cap": 2_900_000_000_000, "quote_type": "equity"}},
            ) as polygon_mock:
                snapshots = prices.fetch_quote_snapshots(["AAPL"])

        polygon_mock.assert_called_once_with(["AAPL"])
        self.assertIn("AAPL", snapshots)
        self.assertEqual(snapshots["AAPL"]["price"], 190.25)
        self.assertEqual(snapshots["AAPL"]["market_cap"], 2_900_000_000_000)
        self.assertEqual(snapshots["AAPL"]["quote_type"], "equity")

    def test_fetch_polygon_profiles_falls_back_to_per_symbol_lookup(self) -> None:
        def _fake_get(url, params=None, headers=None, timeout=None):
            if url.endswith("/v3/reference/tickers"):
                return _Resp(200, {"results": []})
            if url.endswith("/v3/reference/tickers/AAPL"):
                return _Resp(
                    200,
                    {
                        "results": {
                            "ticker": "AAPL",
                            "type": "CS",
                            "market_cap": 2_800_000_000_000,
                        }
                    },
                )
            return _Resp(404, {})

        with patch.dict("os.environ", {"POLYGON_API_KEY": "test-key"}):
            with patch.object(prices.requests, "get", side_effect=_fake_get):
                profiles = prices._fetch_polygon_profiles(["AAPL"])

        self.assertIn("AAPL", profiles)
        self.assertEqual(profiles["AAPL"]["market_cap"], 2_800_000_000_000)
        self.assertEqual(profiles["AAPL"]["quote_type"], "equity")


if __name__ == "__main__":
    unittest.main()
