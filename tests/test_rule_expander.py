import unittest
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ddl69.utils.rule_expander import add_sentiment_rules


class RuleExpanderTests(unittest.TestCase):
    def test_add_sentiment_rules_positive(self) -> None:
        bars = pd.DataFrame(
            {
                "instrument_id": ["TEST"] * 30,
                "ts": pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC"),
                "open": list(range(100, 130)),
                "high": list(range(101, 131)),
                "low": list(range(99, 129)),
                "close": list(range(100, 130)),
                "volume": [1000] * 30,
            }
        )
        sentiment = pd.DataFrame(
            {
                "instrument_id": ["TEST"],
                "sentiment": [0.5],
            }
        )
        out = add_sentiment_rules({"TEST": []}, bars, sentiment, horizon=1, prefix="NEWS")
        rules = [r.rule for r in out.get("TEST", [])]
        self.assertIn("NEWS_POS", rules)


if __name__ == "__main__":
    unittest.main()
