import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys

import typer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ddl69.cli import main as cli_main


class RefreshDailyTests(unittest.TestCase):
    def test_skips_training_when_inputs_missing(self) -> None:
        with patch.object(cli_main, "fetch_bars") as fetch_mock:
            cli_main.refresh_daily(
                symbols="AAPL",
                bars="__missing_bars__.csv",
                labels="__missing_labels__.csv",
                signals="__missing_signals__.zip",
                run_signals=True,
                require_training_inputs=False,
            )
            fetch_mock.assert_called_once()

    def test_can_require_training_inputs(self) -> None:
        with self.assertRaises(typer.BadParameter):
            cli_main.refresh_daily(
                symbols="AAPL",
                bars="__missing_bars__.csv",
                labels="__missing_labels__.csv",
                signals="__missing_signals__.zip",
                run_signals=False,
                require_training_inputs=True,
            )

    def test_runs_training_and_signals_when_inputs_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bars = Path(tmpdir) / "bars.csv"
            labels = Path(tmpdir) / "signals_rows.csv"
            registry = Path(tmpdir) / "signal_registry.zip"
            bars.write_text("timestamp,ticker,open,high,low,close,volume\n", encoding="utf-8")
            labels.write_text("ticker\nAAPL\n", encoding="utf-8")
            registry.write_text("registry", encoding="utf-8")

            with (
                patch.object(cli_main, "train_walkforward") as train_mock,
                patch.object(cli_main, "signals_run") as signals_mock,
                patch.object(cli_main, "fetch_bars") as fetch_mock,
            ):
                cli_main.refresh_daily(
                    symbols="AAPL",
                    bars=str(bars),
                    labels=str(labels),
                    signals=str(registry),
                    run_signals=True,
                    require_training_inputs=True,
                )

                train_mock.assert_called_once()
                signals_mock.assert_called_once()
                fetch_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
