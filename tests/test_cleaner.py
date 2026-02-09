import unittest
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ddl69.data.cleaner import clean_bars, clean_news


class CleanerTests(unittest.TestCase):
    def test_clean_bars_basic(self) -> None:
        df = pd.DataFrame(
            {
                "timestamp": ["2024-01-01", "2024-01-02"],
                "ticker": ["spy", "spy"],
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [101.0, 102.0],
                "volume": [1000, 1100],
            }
        )
        cleaned, report = clean_bars(df, provider_id="test", timeframe="1d")
        self.assertEqual(report.rows_out, 2)
        self.assertIn("instrument_id", cleaned.columns)
        self.assertIn("ts", cleaned.columns)
        self.assertEqual(cleaned["instrument_id"].iloc[0], "SPY")

    def test_clean_news_tickers(self) -> None:
        df = pd.DataFrame(
            {
                "timestamp": ["2024-01-01T00:00:00Z"],
                "title": ["Headline"],
                "tickers": ["aapl, msft"],
            }
        )
        cleaned, report = clean_news(df, provider_id="news")
        self.assertEqual(report.rows_out, 1)
        self.assertIsInstance(cleaned["tickers"].iloc[0], list)
        self.assertEqual(cleaned["tickers"].iloc[0], ["AAPL", "MSFT"])


if __name__ == "__main__":
    unittest.main()
