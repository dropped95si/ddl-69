from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd


@dataclass
class CleanReport:
    dataset: str
    rows_in: int
    rows_out: int
    dropped_missing_required: int
    dropped_duplicates: int
    dropped_invalid_ts: int
    columns_in: list[str]
    columns_out: list[str]


_SYNONYMS: Dict[str, list[str]] = {
    "timestamp": ["timestamp", "ts", "time", "datetime", "date", "asof_ts"],
    "instrument_id": ["instrument_id", "ticker", "symbol", "asset", "security", "instrument"],
    "provider_id": ["provider_id", "provider", "source", "vendor", "feed"],
    "timeframe": ["timeframe", "interval", "tf", "bar_size"],
    "open": ["open", "o"],
    "high": ["high", "h"],
    "low": ["low", "l"],
    "close": ["close", "c", "adj_close", "adjclose"],
    "volume": ["volume", "vol", "v"],
    "vwap": ["vwap"],
    "trades_count": ["trades_count", "trade_count", "count", "trades"],
    "price": ["price", "px", "trade_price"],
    "size": ["size", "qty", "quantity", "trade_size"],
    "side": ["side", "trade_side"],
    "bid": ["bid", "bid_px", "bid_price"],
    "ask": ["ask", "ask_px", "ask_price"],
    "bid_size": ["bid_size", "bid_sz", "bidqty"],
    "ask_size": ["ask_size", "ask_sz", "askqty"],
    "title": ["title", "headline"],
    "body": ["body", "content", "article", "text"],
    "url": ["url", "link"],
    "tickers": ["tickers", "symbols", "ticker_list"],
    "sentiment": ["sentiment", "sent", "score", "polarity"],
    "source": ["source", "provider", "origin", "platform"],
    "author": ["author", "user", "username", "handle"],
    "text_content": ["text_content", "text", "content", "body", "message", "comment"],
}


def load_dataframe(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".parquet"}:
        return pd.read_parquet(p)
    if suffix in {".csv"}:
        return pd.read_csv(p)
    if suffix in {".json"}:
        return pd.read_json(p)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(p, lines=True)
    raise ValueError(f"Unsupported file format: {suffix}")


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suffix = p.suffix.lower()
    if suffix in {".parquet"}:
        df.to_parquet(p, index=False)
        return
    if suffix in {".csv"}:
        df.to_csv(p, index=False)
        return
    if suffix in {".json"}:
        df.to_json(p, orient="records")
        return
    if suffix in {".jsonl", ".ndjson"}:
        df.to_json(p, orient="records", lines=True)
        return
    raise ValueError(f"Unsupported output format: {suffix}")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df = df.copy()
    df.columns = cols
    return df


def _apply_synonyms(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    colset = set(df.columns)
    rename: Dict[str, str] = {}
    for canonical, options in _SYNONYMS.items():
        if canonical in colset:
            continue
        for opt in options:
            if opt in colset:
                rename[opt] = canonical
                colset.remove(opt)
                colset.add(canonical)
                break
    if rename:
        df = df.rename(columns=rename)
    return df


def _parse_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _drop_missing_required(df: pd.DataFrame, required: Iterable[str]) -> tuple[pd.DataFrame, int]:
    required = [c for c in required if c in df.columns]
    before = len(df)
    df = df.dropna(subset=required)
    return df, before - len(df)


def _drop_duplicates(df: pd.DataFrame, subset: Iterable[str]) -> tuple[pd.DataFrame, int]:
    subset = [c for c in subset if c in df.columns]
    before = len(df)
    if subset:
        df = df.drop_duplicates(subset=subset, keep="last")
    return df, before - len(df)


def _finalize(
    df: pd.DataFrame,
    dataset: str,
    rows_in: int,
    dropped_missing_required: int,
    dropped_duplicates: int,
    dropped_invalid_ts: int,
) -> CleanReport:
    return CleanReport(
        dataset=dataset,
        rows_in=rows_in,
        rows_out=len(df),
        dropped_missing_required=dropped_missing_required,
        dropped_duplicates=dropped_duplicates,
        dropped_invalid_ts=dropped_invalid_ts,
        columns_in=[],
        columns_out=list(df.columns),
    )


def detect_dataset(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    if {"open", "high", "low", "close"}.issubset(cols):
        return "bars"
    if "price" in cols or "size" in cols:
        return "trades"
    if "bid" in cols or "ask" in cols:
        return "quotes"
    if "title" in cols or "body" in cols:
        return "news"
    if "text_content" in cols:
        return "social"
    return "generic"


def clean_bars(
    df: pd.DataFrame,
    *,
    provider_id: Optional[str] = None,
    timeframe: Optional[str] = None,
    upper_instrument: bool = True,
) -> tuple[pd.DataFrame, CleanReport]:
    rows_in = len(df)
    df = _apply_synonyms(_normalize_columns(df))
    columns_in = list(df.columns)

    if "timestamp" in df.columns and "ts" not in df.columns:
        df = df.rename(columns={"timestamp": "ts"})
    if "ts" in df.columns:
        df["ts"] = _parse_timestamp(df["ts"])

    dropped_invalid_ts = 0
    if "ts" in df.columns:
        before = len(df)
        df = df.dropna(subset=["ts"])
        dropped_invalid_ts = before - len(df)

    required = ["instrument_id", "ts", "open", "high", "low", "close"]
    df, dropped_missing_required = _drop_missing_required(df, required)

    if "provider_id" not in df.columns and provider_id:
        df["provider_id"] = provider_id
    if "timeframe" not in df.columns and timeframe:
        df["timeframe"] = timeframe

    if upper_instrument and "instrument_id" in df.columns:
        df["instrument_id"] = df["instrument_id"].astype(str).str.upper()

    _coerce_numeric(df, ["open", "high", "low", "close", "volume", "vwap", "trades_count"])

    df, dropped_duplicates = _drop_duplicates(
        df, ["instrument_id", "provider_id", "timeframe", "ts"]
    )
    df = df.sort_values("ts").reset_index(drop=True)

    report = _finalize(
        df,
        "bars",
        rows_in,
        dropped_missing_required,
        dropped_duplicates,
        dropped_invalid_ts,
    )
    report.columns_in = columns_in
    return df, report


def clean_trades(
    df: pd.DataFrame,
    *,
    provider_id: Optional[str] = None,
    upper_instrument: bool = True,
) -> tuple[pd.DataFrame, CleanReport]:
    rows_in = len(df)
    df = _apply_synonyms(_normalize_columns(df))
    columns_in = list(df.columns)

    if "timestamp" in df.columns and "ts" not in df.columns:
        df = df.rename(columns={"timestamp": "ts"})
    if "ts" in df.columns:
        df["ts"] = _parse_timestamp(df["ts"])

    dropped_invalid_ts = 0
    if "ts" in df.columns:
        before = len(df)
        df = df.dropna(subset=["ts"])
        dropped_invalid_ts = before - len(df)

    required = ["instrument_id", "ts", "price"]
    df, dropped_missing_required = _drop_missing_required(df, required)

    if "provider_id" not in df.columns and provider_id:
        df["provider_id"] = provider_id

    if upper_instrument and "instrument_id" in df.columns:
        df["instrument_id"] = df["instrument_id"].astype(str).str.upper()

    _coerce_numeric(df, ["price", "size"])

    df, dropped_duplicates = _drop_duplicates(df, ["instrument_id", "provider_id", "ts", "price"])
    df = df.sort_values("ts").reset_index(drop=True)

    report = _finalize(
        df,
        "trades",
        rows_in,
        dropped_missing_required,
        dropped_duplicates,
        dropped_invalid_ts,
    )
    report.columns_in = columns_in
    return df, report


def clean_quotes(
    df: pd.DataFrame,
    *,
    provider_id: Optional[str] = None,
    upper_instrument: bool = True,
) -> tuple[pd.DataFrame, CleanReport]:
    rows_in = len(df)
    df = _apply_synonyms(_normalize_columns(df))
    columns_in = list(df.columns)

    if "timestamp" in df.columns and "ts" not in df.columns:
        df = df.rename(columns={"timestamp": "ts"})
    if "ts" in df.columns:
        df["ts"] = _parse_timestamp(df["ts"])

    dropped_invalid_ts = 0
    if "ts" in df.columns:
        before = len(df)
        df = df.dropna(subset=["ts"])
        dropped_invalid_ts = before - len(df)

    required = ["instrument_id", "ts", "bid", "ask"]
    df, dropped_missing_required = _drop_missing_required(df, required)

    if "provider_id" not in df.columns and provider_id:
        df["provider_id"] = provider_id

    if upper_instrument and "instrument_id" in df.columns:
        df["instrument_id"] = df["instrument_id"].astype(str).str.upper()

    _coerce_numeric(df, ["bid", "ask", "bid_size", "ask_size"])

    df, dropped_duplicates = _drop_duplicates(df, ["instrument_id", "provider_id", "ts"])
    df = df.sort_values("ts").reset_index(drop=True)

    report = _finalize(
        df,
        "quotes",
        rows_in,
        dropped_missing_required,
        dropped_duplicates,
        dropped_invalid_ts,
    )
    report.columns_in = columns_in
    return df, report


def _normalize_tickers(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(v).strip().upper() for v in value if str(v).strip()]
    if isinstance(value, str):
        parts = [p.strip().upper() for p in value.replace(";", ",").split(",")]
        return [p for p in parts if p]
    return [str(value).strip().upper()]


def clean_news(
    df: pd.DataFrame,
    *,
    provider_id: Optional[str] = None,
) -> tuple[pd.DataFrame, CleanReport]:
    rows_in = len(df)
    df = _apply_synonyms(_normalize_columns(df))
    columns_in = list(df.columns)

    if "timestamp" in df.columns and "ts" not in df.columns:
        df = df.rename(columns={"timestamp": "ts"})
    if "ts" in df.columns:
        df["ts"] = _parse_timestamp(df["ts"])

    dropped_invalid_ts = 0
    if "ts" in df.columns:
        before = len(df)
        df = df.dropna(subset=["ts"])
        dropped_invalid_ts = before - len(df)

    required = ["ts"]
    df, dropped_missing_required = _drop_missing_required(df, required)

    if "provider_id" not in df.columns and provider_id:
        df["provider_id"] = provider_id

    if "tickers" in df.columns:
        df["tickers"] = df["tickers"].apply(_normalize_tickers)

    df, dropped_duplicates = _drop_duplicates(df, ["provider_id", "ts", "title", "url"])
    df = df.sort_values("ts").reset_index(drop=True)

    report = _finalize(
        df,
        "news",
        rows_in,
        dropped_missing_required,
        dropped_duplicates,
        dropped_invalid_ts,
    )
    report.columns_in = columns_in
    return df, report


def clean_social(
    df: pd.DataFrame,
    *,
    provider_id: Optional[str] = None,
) -> tuple[pd.DataFrame, CleanReport]:
    rows_in = len(df)
    df = _apply_synonyms(_normalize_columns(df))
    columns_in = list(df.columns)

    if "timestamp" in df.columns and "ts" not in df.columns:
        df = df.rename(columns={"timestamp": "ts"})
    if "ts" in df.columns:
        df["ts"] = _parse_timestamp(df["ts"])

    dropped_invalid_ts = 0
    if "ts" in df.columns:
        before = len(df)
        df = df.dropna(subset=["ts"])
        dropped_invalid_ts = before - len(df)

    required = ["ts"]
    df, dropped_missing_required = _drop_missing_required(df, required)

    if "provider_id" not in df.columns and provider_id:
        df["provider_id"] = provider_id

    if "tickers" in df.columns:
        df["tickers"] = df["tickers"].apply(_normalize_tickers)

    df, dropped_duplicates = _drop_duplicates(df, ["provider_id", "ts", "author", "text_content"])
    df = df.sort_values("ts").reset_index(drop=True)

    report = _finalize(
        df,
        "social",
        rows_in,
        dropped_missing_required,
        dropped_duplicates,
        dropped_invalid_ts,
    )
    report.columns_in = columns_in
    return df, report


def clean_generic(df: pd.DataFrame) -> tuple[pd.DataFrame, CleanReport]:
    rows_in = len(df)
    df = _apply_synonyms(_normalize_columns(df))
    columns_in = list(df.columns)

    dropped_invalid_ts = 0
    if "timestamp" in df.columns and "ts" not in df.columns:
        df = df.rename(columns={"timestamp": "ts"})
    if "ts" in df.columns:
        df["ts"] = _parse_timestamp(df["ts"])
        before = len(df)
        df = df.dropna(subset=["ts"])
        dropped_invalid_ts = before - len(df)

    df = df.drop_duplicates().reset_index(drop=True)
    report = _finalize(df, "generic", rows_in, 0, 0, dropped_invalid_ts)
    report.columns_in = columns_in
    return df, report


def clean_dataset(
    df: pd.DataFrame,
    *,
    dataset: str = "auto",
    provider_id: Optional[str] = None,
    timeframe: Optional[str] = None,
    upper_instrument: bool = True,
) -> tuple[pd.DataFrame, CleanReport]:
    dataset = dataset.lower()
    if dataset == "auto":
        dataset = detect_dataset(_apply_synonyms(_normalize_columns(df)))
    if dataset == "bars":
        return clean_bars(df, provider_id=provider_id, timeframe=timeframe, upper_instrument=upper_instrument)
    if dataset == "trades":
        return clean_trades(df, provider_id=provider_id, upper_instrument=upper_instrument)
    if dataset == "quotes":
        return clean_quotes(df, provider_id=provider_id, upper_instrument=upper_instrument)
    if dataset == "news":
        return clean_news(df, provider_id=provider_id)
    if dataset == "social":
        return clean_social(df, provider_id=provider_id)
    if dataset == "generic":
        return clean_generic(df)
    raise ValueError(f"Unknown dataset type: {dataset}")
