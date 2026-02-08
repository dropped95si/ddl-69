from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd


@dataclass
class UniverseResult:
    asof: datetime
    tickers: List[str]
    source: str


def _load_start_end(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "ticker" not in df.columns:
        raise ValueError("start/end universe parquet missing 'ticker'")
    df = df.rename(columns={"start_date": "start", "end_date": "end"})
    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.upper()
    return df


def sp500_members_asof(
    *,
    asof: datetime,
    universe_dir: Optional[str] = None,
) -> UniverseResult:
    base = Path(universe_dir) if universe_dir else (Path("artifacts") / "universe")
    path = base / "sp500_ticker_start_end.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing universe file: {path}")
    df = _load_start_end(path)

    # start/end are tz-naive; compare to naive asof
    asof_naive = asof.replace(tzinfo=None)
    active = df[(df["start"] <= asof_naive) & ((df["end"].isna()) | (df["end"] >= asof_naive))]
    tickers = sorted(set(active["ticker"].tolist()))
    return UniverseResult(asof=asof, tickers=tickers, source=str(path))


def sp500_members_latest(
    *,
    universe_dir: Optional[str] = None,
) -> UniverseResult:
    return sp500_members_asof(asof=datetime.utcnow(), universe_dir=universe_dir)
