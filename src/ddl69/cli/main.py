from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any

import pandas as pd
import typer
import requests
from rich import print

from ddl69.core.settings import Settings
from ddl69.ledger.supabase_ledger import SupabaseLedger
from ddl69.data.parquet_store import ParquetStore
from ddl69.data.cleaner import clean_dataset, load_dataframe, save_dataframe
from ddl69.utils.signals import (
    aggregate_rule_weights,
    blend_probs,
    entropy,
    load_signals_rows,
    rule_to_probs,
    sanitize_json,
    weights_from_rules,
)
from ddl69.utils.rule_expander import expand_signals_rows
from ddl69.utils.universe import sp500_members_asof
import importlib.util

app = typer.Typer(add_completion=False)

@app.command()
def help() -> None:
    """Show quick usage."""
    print("ddl-69 CLI")
    print("  init_sql     ...prints SQL file paths (ledger + ingest)")
    print("  ingest_bars   ingest a CSV with OHLCV into Parquet + (optional) Supabase bars table")
    print("  clean_data    normalize/clean a dataset file to canonical columns")
    print("  signals_run   create run/events/forecasts from signals_rows.csv")
    print("  train_walkforward   build weights from signals_rows + registry")
    print("  fetch_bars    pull OHLCV (polygon/alpaca/yahoo/local) into Parquet/Supabase")
    print("  fetch_polygon_snapshot  pull Polygon snapshot JSON")
    print("  fetch_massive_s3  pull limited files from Massive S3 (safe caps)")
    print("  refresh_daily  train weights + fetch bars for watchlist")
    print("  build_sp500_watchlist  build S&P 500 universe watchlist")
    print("  demo_run      writes a demo run/event/forecast into Supabase")

@app.command()
def init_sql() -> None:
    """Print which SQL files to run in Supabase."""
    print("Run SQL in this order:")
    print("  1) sql/ledger_v1.sql")
    print("  2) sql/ledger_v2_patch.sql")
    print("  3) sql/ingest_v1.sql")

@app.command()
def ingest_bars(
    csv_path: str = typer.Argument(..., help="Path to CSV with columns: timestamp,ticker,open,high,low,close,volume"),
    provider: str = typer.Option("csv", help="Provider id (polygon/alpaca/yahoo/csv/etc)") ,
    to_supabase: bool = typer.Option(False, help="Also upsert into public.bars"),
) -> None:
    settings = Settings()
    store = ParquetStore(settings)
    df = pd.read_csv(csv_path)
    # minimal normalization
    required = {"timestamp","ticker","open","high","low","close","volume"}
    missing = required - set(df.columns)
    if missing:
        raise typer.BadParameter(f"CSV missing columns: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["provider"] = provider
    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()
    name = f"bars_{provider}_{ts_min.date().isoformat()}_{ts_max.date().isoformat()}"
    artifact = store.write_df(df, kind="bars", name=name)
    print(f"Parquet written: {artifact.uri} (rows={artifact.rows})")

    if to_supabase:
        ledger = SupabaseLedger(settings)
        # Upsert in batches (small helper; for huge loads, keep Parquet and only watermark)
        rows = df.copy()
        rows["ts"] = rows["timestamp"].dt.to_pydatetime()
        rows = rows.rename(columns={
            "ticker": "instrument_id",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        })
        rows["provider_id"] = provider
        rows["timeframe"] = "1d"

        tickers = sorted(set(rows["instrument_id"].astype(str).tolist()))
        for t in tickers:
            ledger.upsert_instrument(instrument_id=t)

        payload = []
        for r in rows.itertuples(index=False):
            payload.append({
                "instrument_id": str(r.instrument_id),
                "provider_id": str(r.provider_id),
                "timeframe": str(r.timeframe),
                "ts": pd.to_datetime(r.ts, utc=True).isoformat(),
                "open": float(r.open),
                "high": float(r.high),
                "low": float(r.low),
                "close": float(r.close),
                "volume": float(r.volume),
            })

        ledger.upsert_bars(payload)
        print("Upserted into Supabase public.bars")

@app.command()
def clean_data(
    input_path: str = typer.Argument(..., help="Path to CSV/Parquet/JSON/JSONL file"),
    output_path: Optional[str] = typer.Option(None, help="Where to write cleaned data (default: artifacts/cleaned/...)"),
    dataset: str = typer.Option("auto", help="Dataset type: auto/bars/trades/quotes/news/social/generic"),
    provider_id: Optional[str] = typer.Option(None, help="Provider id to fill if missing"),
    timeframe: Optional[str] = typer.Option(None, help="Timeframe to fill if missing (bars only)"),
    upper_instrument: bool = typer.Option(True, help="Uppercase instrument_id when cleaning"),
    dry_run: bool = typer.Option(False, help="Only print report; do not write output"),
) -> None:
    df = load_dataframe(input_path)
    cleaned, report = clean_dataset(
        df,
        dataset=dataset,
        provider_id=provider_id,
        timeframe=timeframe,
        upper_instrument=upper_instrument,
    )

    if output_path is None:
        in_path = Path(input_path)
        suffix = in_path.suffix or ".parquet"
        out_dir = Path("artifacts") / "cleaned"
        output_path = str(out_dir / f"{in_path.stem}_cleaned{suffix}")

    print(f"dataset={report.dataset} rows_in={report.rows_in} rows_out={report.rows_out}")
    print(
        "dropped_missing_required="
        f"{report.dropped_missing_required} "
        "dropped_invalid_ts="
        f"{report.dropped_invalid_ts} "
        "dropped_duplicates="
        f"{report.dropped_duplicates}"
    )
    print(f"columns_in={len(report.columns_in)} columns_out={len(report.columns_out)}")

    if dry_run:
        return
    save_dataframe(cleaned, output_path)
    print(f"Wrote cleaned data: {output_path}")


@app.command()
def signals_run(
    signals_path: str = typer.Option(
        "C:\\Users\\Stas\\Downloads\\signals_rows.csv",
        help="Path to signals_rows.csv",
    ),
    signal_doc_path: Optional[str] = typer.Option(
        "C:\\Users\\Stas\\Downloads\\SignalDoc.csv",
        help="Optional SignalDoc.csv path for artifact logging",
    ),
    mode: str = typer.Option("lean", help="Run mode"),
    method: str = typer.Option("hedge", help="Ensemble method name"),
    max_rows: Optional[int] = typer.Option(None, help="Limit number of rows"),
    chunk_size: int = typer.Option(25, help="Batch size for Supabase inserts"),
) -> None:
    settings = Settings()
    ledger = SupabaseLedger(settings)
    store = ParquetStore(settings)

    df = load_signals_rows(signals_path)
    if max_rows is not None:
        df = df.head(max_rows)

    now = datetime.now(timezone.utc)
    run_id = ledger.create_run(
        asof_ts=now, mode=mode, config_hash="signals_rows", code_version="0.1.0"
    )
    print(f"Created run_id={run_id}")

    # store raw signals_rows as artifact
    artifact = store.write_df(df, kind="signals", name=f"signals_rows_{now.date().isoformat()}")
    ledger.insert_artifact(
        run_id=run_id,
        kind="raw",
        uri=artifact.uri,
        sha256=artifact.sha256,
        row_count=artifact.rows,
        meta_json={"source": signals_path},
    )

    if signal_doc_path and Path(signal_doc_path).exists():
        signal_doc_df = load_dataframe(signal_doc_path)
        doc_art = store.write_df(
            signal_doc_df, kind="signals", name=f"signal_doc_{now.date().isoformat()}"
        )
        ledger.insert_artifact(
            run_id=run_id,
            kind="other",
            uri=doc_art.uri,
            sha256=doc_art.sha256,
            row_count=doc_art.rows,
            meta_json={"source": signal_doc_path},
        )

    batch_events: list[dict[str, Any]] = []
    batch_experts: list[dict[str, Any]] = []
    batch_ensembles: list[dict[str, Any]] = []

    def flush_batches() -> None:
        if batch_events:
            ledger.upsert_events(batch_events)
            batch_events.clear()
        if batch_experts:
            ledger.upsert_expert_forecasts(batch_experts)
            batch_experts.clear()
        if batch_ensembles:
            ledger.upsert_ensemble_forecasts(batch_ensembles)
            batch_ensembles.clear()

    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "")).upper()
        if not ticker:
            continue

        created_at = row.get("created_at")
        asof_ts = (
            pd.to_datetime(created_at, utc=True)
            if created_at is not None
            else now
        )
        event_id = str(row.get("id") or f"{ticker}|signal|{asof_ts.date().isoformat()}")

        event_params = {
            "plan_type": row.get("plan_type"),
            "label": row.get("label"),
            "entry": row.get("entry"),
            "stop": row.get("stop"),
            "targets": row.get("targets"),
        }
        context = {
            "rs_lookback": row.get("rs_lookback"),
            "rs_vs_spy": row.get("rs_vs_spy"),
            "vol_z": row.get("vol_z"),
            "fv": row.get("fv"),
            "pivots": row.get("pivots"),
            "fib": row.get("fib"),
            "learned_top_rules": row.get("learned_top_rules"),
        }
        event_params = sanitize_json(event_params)
        context = sanitize_json(context)

        batch_events.append(
            {
                "event_id": event_id,
                "subject_type": "ticker",
                "subject_id": ticker,
                "event_type": "state_event",
                "asof_ts": asof_ts.isoformat(),
                "horizon_json": {"type": "time", "value": 5, "unit": "d"},
                "event_params_json": event_params,
                "context_json": context,
            }
        )

        rules = row.get("learned_top_rules") or []
        weights = weights_from_rules(rules) if isinstance(rules, list) else {}

        weighted_probs = []
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            name = str(rule.get("rule") or "unknown_rule")
            probs = rule_to_probs(rule)
            conf = float(probs.get("ACCEPT_CONTINUE", 0.5))
            batch_experts.append(
                {
                    "run_id": run_id,
                    "event_id": event_id,
                    "expert_name": name,
                    "expert_version": "signals_rows",
                    "probs_json": probs,
                    "confidence": conf,
                    "uncertainty_json": {"entropy": entropy(probs)},
                    "loss_hint": "logloss",
                    "supports_calibration": True,
                    "calibration_group": str(row.get("plan_type") or "signals"),
                    "features_uri": None,
                    "artifact_uris": [],
                    "reasons_json": [],
                    "debug_json": sanitize_json({"rule": rule}),
                }
            )
            w = weights.get(name, 0.0)
            weighted_probs.append((probs, w))

        if weighted_probs:
            ensemble_probs = blend_probs(weighted_probs)
            batch_ensembles.append(
                {
                    "run_id": run_id,
                    "event_id": event_id,
                    "method": method,
                    "probs_json": ensemble_probs,
                    "confidence": float(ensemble_probs.get("ACCEPT_CONTINUE", 0.5)),
                    "uncertainty_json": {"entropy": entropy(ensemble_probs)},
                    "weights_json": weights,
                    "explain_json": {"source": "signals_rows"},
                    "artifact_uris": [],
                }
            )

        if len(batch_events) >= chunk_size or len(batch_experts) >= chunk_size or len(batch_ensembles) >= chunk_size:
            flush_batches()

    flush_batches()
    print("Signals run completed")


@app.command()
def train_walkforward(
    bars: str = typer.Option(
        "C:\\Users\\Stas\\Downloads\\HistoricalData_1769169971316.csv",
        help="Path to historical bars CSV",
    ),
    labels: str = typer.Option(
        "C:\\Users\\Stas\\Downloads\\signals_rows.csv",
        help="Path to signals_rows.csv",
    ),
    signals: str = typer.Option(
        "C:\\Users\\Stas\\Downloads\\signal_registry_top50_pack.zip",
        help="Path to signal registry pack (zip or dir)",
    ),
    expand_rules: bool = typer.Option(True, help="Expand learned_top_rules from historical bars"),
    horizon: int = typer.Option(5, help="Forward horizon in bars for rule scoring"),
    top_rules: int = typer.Option(8, help="Top rules per ticker to keep"),
    news_path: Optional[str] = typer.Option(None, help="Optional news dataset with sentiment"),
    social_path: Optional[str] = typer.Option(None, help="Optional social dataset with sentiment"),
    qlib_dir: Optional[str] = typer.Option(None, help="Optional Qlib data dir for factor rules"),
    mode: str = typer.Option("lean", help="Run mode"),
    method: str = typer.Option("hedge", help="Weight method"),
) -> None:
    settings = Settings()
    ledger = SupabaseLedger(settings)
    store = ParquetStore(settings)

    now = datetime.now(timezone.utc)
    run_id = ledger.create_run(
        asof_ts=now, mode=mode, config_hash="walkforward", code_version="0.1.0"
    )
    print(f"Created run_id={run_id}")

    # load labels/signals rows
    df = load_signals_rows(labels)
    if expand_rules:
        bars_df = load_dataframe(bars)
        bars_df = bars_df.rename(columns={"ticker": "instrument_id"}).copy()
        if "instrument_id" not in bars_df.columns and "symbol" in bars_df.columns:
            bars_df = bars_df.rename(columns={"symbol": "instrument_id"})
        # Handle common Date/Close/Last style CSVs (e.g., SPX history)
        if "instrument_id" not in bars_df.columns:
            if "Date" in bars_df.columns:
                bars_df = bars_df.rename(columns={"Date": "ts"})
            if "Close/Last" in bars_df.columns:
                bars_df = bars_df.rename(columns={"Close/Last": "close"})
            if "Open" in bars_df.columns:
                bars_df = bars_df.rename(columns={"Open": "open"})
            if "High" in bars_df.columns:
                bars_df = bars_df.rename(columns={"High": "high"})
            if "Low" in bars_df.columns:
                bars_df = bars_df.rename(columns={"Low": "low"})
            if "Volume" in bars_df.columns:
                bars_df = bars_df.rename(columns={"Volume": "volume"})
            if "instrument_id" not in bars_df.columns:
                bars_df["instrument_id"] = "SPX"
        if "ts" not in bars_df.columns and "timestamp" in bars_df.columns:
            bars_df = bars_df.rename(columns={"timestamp": "ts"})
        if "ts" in bars_df.columns:
            bars_df["ts"] = pd.to_datetime(bars_df["ts"], utc=True, errors="coerce")
        # Strip $ and commas from price-like columns
        for col in ["open", "high", "low", "close", "volume"]:
            if col in bars_df.columns:
                bars_df[col] = (
                    bars_df[col]
                    .astype(str)
                    .str.replace("$", "", regex=False)
                    .str.replace(",", "", regex=False)
                )
                bars_df[col] = pd.to_numeric(bars_df[col], errors="coerce")
        bars_df["instrument_id"] = bars_df["instrument_id"].astype(str).str.upper()
        news_df = load_dataframe(news_path) if news_path else None
        social_df = load_dataframe(social_path) if social_path else None
        df = expand_signals_rows(
            df,
            bars_df,
            horizon=horizon,
            top_n=top_rules,
            news_df=news_df,
            social_df=social_df,
            qlib_dir=qlib_dir,
        )
    weights, calib = aggregate_rule_weights(df.to_dict(orient="records"))

    # write artifacts
    weights_path = Path("artifacts") / "weights" / "latest.json"
    prev_weights = {}
    if weights_path.exists():
        try:
            prev_weights = json.loads(weights_path.read_text(encoding="utf-8"))
        except Exception:
            prev_weights = {}
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.write_text(json.dumps(weights, indent=2), encoding="utf-8")

    calib_path = Path("artifacts") / "calibration" / f"calibration_{now.date().isoformat()}.json"
    calib_path.parent.mkdir(parents=True, exist_ok=True)
    calib_path.write_text(json.dumps(calib, indent=2), encoding="utf-8")

    ledger.insert_artifact(
        run_id=run_id,
        kind="other",
        uri=str(weights_path),
        row_count=len(weights),
        meta_json={"type": "weights", "labels": labels, "signals": signals, "bars": bars},
    )
    ledger.insert_artifact(
        run_id=run_id,
        kind="other",
        uri=str(calib_path),
        row_count=calib.get("total_rules", 0),
        meta_json={"type": "calibration", "labels": labels, "signals": signals, "bars": bars},
    )

    # write weight update to ledger
    ledger._exec(
        "insert weight_updates",
        lambda: ledger.client.table("weight_updates").insert(
            {
                "asof_ts": now.isoformat(),
                "context_key": "signals_rows",
                "method": method,
                "weights_before_json": prev_weights or {},
                "weights_after_json": weights,
                "losses_json": {"source": "signals_rows"},
                "run_id": run_id,
            }
        ).execute(),
    )

    print(f"Wrote weights: {weights_path}")
    print(f"Wrote calibration: {calib_path}")


def _timeframe_from_polygon(timespan: str, multiplier: int) -> str:
    t = timespan.lower()
    if t in {"minute", "min", "minutes"}:
        return f"{multiplier}m"
    if t in {"hour", "hours"}:
        return f"{multiplier}h"
    if t in {"day", "days"}:
        return f"{multiplier}d"
    return f"{multiplier}{t}"


def _timeframe_from_alpaca(timeframe: str) -> str:
    tf = timeframe.lower()
    if tf.endswith("min"):
        return tf.replace("min", "m")
    if tf.endswith("hour"):
        return tf.replace("hour", "h")
    if tf.endswith("day"):
        return tf.replace("day", "d")
    return tf


def _fetch_polygon(
    settings: Settings,
    symbol: str,
    from_date: str,
    to_date: str,
    timespan: str,
    multiplier: int,
    adjusted: bool,
    limit: int,
) -> pd.DataFrame:
    if not settings.polygon_api_key:
        raise RuntimeError("POLYGON_API_KEY not set")
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/"
        f"{multiplier}/{timespan}/{from_date}/{to_date}"
    )
    params = {
        "adjusted": "true" if adjusted else "false",
        "sort": "asc",
        "limit": limit,
        "apiKey": settings.polygon_api_key,
    }
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Polygon error {resp.status_code}: {resp.text}")
    data = resp.json()
    results = data.get("results") or []
    if not results:
        raise RuntimeError("Polygon returned 0 bars")
    df = pd.DataFrame(results).rename(
        columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "t": "timestamp"}
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["ticker"] = symbol
    df["provider"] = "polygon"
    return df


def _fetch_alpaca(
    settings: Settings,
    symbol: str,
    from_date: str,
    to_date: str,
    timeframe: str,
    limit: int,
) -> pd.DataFrame:
    if not settings.alpaca_api_key or not settings.alpaca_secret_key:
        raise RuntimeError("ALPACA_API_KEY/ALPACA_SECRET_KEY not set")
    url = "https://data.alpaca.markets/v2/stocks/bars"
    headers = {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
    }
    params = {
        "symbols": symbol,
        "timeframe": timeframe,
        "start": from_date,
        "end": to_date,
        "limit": limit,
        "adjustment": "all",
    }
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Alpaca error {resp.status_code}: {resp.text}")
    data = resp.json()
    bars = (data.get("bars") or {}).get(symbol) or []
    if not bars:
        raise RuntimeError("Alpaca returned 0 bars")
    df = pd.DataFrame(bars).rename(
        columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "t": "timestamp"}
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["ticker"] = symbol
    df["provider"] = "alpaca"
    return df


def _fetch_yahoo(
    symbol: str,
    from_date: str,
    to_date: str,
    interval: str,
) -> pd.DataFrame:
    try:
        import yfinance as yf
    except Exception as exc:
        raise RuntimeError("yfinance not installed; pip install yfinance") from exc
    df = yf.download(symbol, start=from_date, end=to_date, interval=interval, auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError("Yahoo returned 0 bars")
    df = df.reset_index().rename(
        columns={
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["ticker"] = symbol
    df["provider"] = "yahoo"
    return df


@app.command()
def fetch_bars(
    symbol: str = typer.Option("AAPL", help="Ticker symbol"),
    from_date: Optional[str] = typer.Option(None, help="YYYY-MM-DD (default: 30 days ago)"),
    to_date: Optional[str] = typer.Option(None, help="YYYY-MM-DD (default: today)"),
    source: str = typer.Option("auto", help="auto/polygon/alpaca/yahoo/local"),
    timespan: str = typer.Option("day", help="Polygon timespan: minute/hour/day"),
    multiplier: int = typer.Option(1, help="Polygon timespan multiplier"),
    alpaca_timeframe: str = typer.Option("1Day", help="Alpaca timeframe"),
    yahoo_interval: str = typer.Option("1d", help="Yahoo interval"),
    limit: int = typer.Option(5000, help="Max bars"),
    local_path: Optional[str] = typer.Option(None, help="Local CSV/Parquet fallback"),
    to_supabase: bool = typer.Option(True, help="Upsert into public.bars"),
    upload_storage: bool = typer.Option(False, help="Upload Parquet to Supabase Storage"),
) -> None:
    settings = Settings()
    now = datetime.now(timezone.utc)
    if to_date is None:
        to_date = now.date().isoformat()
    if from_date is None:
        from_date = (now.date() - pd.Timedelta(days=30)).isoformat()

    symbol = symbol.upper()
    src = source.lower()
    df = None
    errors: list[str] = []

    def _try(fn, label: str) -> Optional[pd.DataFrame]:
        try:
            return fn()
        except Exception as exc:
            errors.append(f"{label}: {exc}")
            return None

    if src in {"auto", "polygon"}:
        df = _try(
            lambda: _fetch_polygon(settings, symbol, from_date, to_date, timespan, multiplier, True, limit),
            "polygon",
        )
        if src == "polygon" and df is None:
            raise RuntimeError("; ".join(errors))

    if df is None and src in {"auto", "alpaca"}:
        df = _try(
            lambda: _fetch_alpaca(settings, symbol, from_date, to_date, alpaca_timeframe, limit),
            "alpaca",
        )
        if src == "alpaca" and df is None:
            raise RuntimeError("; ".join(errors))

    if df is None and src in {"auto", "yahoo"}:
        df = _try(lambda: _fetch_yahoo(symbol, from_date, to_date, yahoo_interval), "yahoo")
        if src == "yahoo" and df is None:
            raise RuntimeError("; ".join(errors))

    if df is None and local_path:
        df = load_dataframe(local_path)
        df = df.rename(columns={"date": "timestamp"}).copy()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["ticker"] = df.get("ticker", symbol)
        df["provider"] = "local"

    if df is None:
        raise RuntimeError("No data fetched. " + "; ".join(errors))

    store = ParquetStore(settings)
    name = f"bars_{df['provider'].iloc[0]}_{symbol}_{from_date}_{to_date}"
    artifact = store.write_df(df, kind="bars", name=name)
    print(f"Parquet written: {artifact.uri} (rows={artifact.rows})")

    if to_supabase or upload_storage:
        ledger = SupabaseLedger(settings)
        provider_id = str(df["provider"].iloc[0])
        if upload_storage:
            dest_path = f"bars/{Path(artifact.uri).name}"
            storage_uri = ledger.upload_storage(
                bucket=settings.supabase_storage_bucket,
                local_path=artifact.uri,
                dest_path=dest_path,
                upsert=True,
            )
            print(f"Uploaded to Supabase Storage: {storage_uri}")

        if to_supabase:
            ledger.upsert_instrument(instrument_id=symbol)
            if provider_id == "polygon":
                timeframe = _timeframe_from_polygon(timespan, multiplier)
            elif provider_id == "alpaca":
                timeframe = _timeframe_from_alpaca(alpaca_timeframe)
            else:
                timeframe = "1d"

            payload = []
            for r in df.itertuples(index=False):
                payload.append(
                    {
                        "instrument_id": symbol,
                        "provider_id": provider_id,
                        "timeframe": timeframe,
                        "ts": pd.to_datetime(r.timestamp, utc=True).isoformat(),
                        "open": float(r.open),
                        "high": float(r.high),
                        "low": float(r.low),
                        "close": float(r.close),
                        "volume": float(r.volume),
                    }
                )
            ledger.upsert_bars(payload)
            print("Upserted into Supabase public.bars")


@app.command()
def fetch_polygon_snapshot(
    tickers: str = typer.Option("AAPL,SPY,QQQ", help="Comma-separated tickers"),
    upload_storage: bool = typer.Option(False, help="Upload snapshot JSON to Supabase Storage"),
) -> None:
    settings = Settings()
    if not settings.polygon_api_key:
        raise typer.BadParameter("POLYGON_API_KEY not set in environment")

    symbols = ",".join([t.strip().upper() for t in tickers.split(",") if t.strip()])
    url = "https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers"
    params = {"tickers": symbols, "apiKey": settings.polygon_api_key}
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Polygon error {resp.status_code}: {resp.text}")
    data = resp.json()

    out_dir = Path("artifacts") / "snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"polygon_snapshot_{datetime.now(timezone.utc).date().isoformat()}.json"
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote snapshot: {out_path}")

    if upload_storage:
        ledger = SupabaseLedger(settings)
        storage_uri = ledger.upload_storage(
            bucket=settings.supabase_storage_bucket,
            local_path=str(out_path),
            dest_path=f"snapshots/{out_path.name}",
            upsert=True,
        )
        print(f"Uploaded to Supabase Storage: {storage_uri}")


@app.command()
def fetch_massive_s3(
    prefix: str = typer.Option("", help="S3 prefix to pull"),
    max_keys: int = typer.Option(5, help="Max objects to fetch"),
    dest_dir: Optional[str] = typer.Option(None, help="Destination directory (default: artifacts/massive)"),
    dry_run: bool = typer.Option(False, help="Only list objects; do not download"),
) -> None:
    settings = Settings()
    if not settings.massive_s3_endpoint or not settings.massive_s3_bucket:
        raise typer.BadParameter("MASSIVE_S3_ENDPOINT and MASSIVE_S3_BUCKET must be set")
    if not settings.massive_access_key or not settings.massive_secret_key:
        raise typer.BadParameter("MASSIVE_ACCESS_KEY and MASSIVE_SECRET_KEY must be set")

    try:
        import boto3
        from botocore.config import Config
        from botocore.exceptions import ClientError
    except Exception as exc:
        raise RuntimeError("boto3 required for Massive S3 pulls") from exc

    dest = Path(dest_dir) if dest_dir else (Path("artifacts") / "massive")
    dest.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client(
        "s3",
        endpoint_url=settings.massive_s3_endpoint,
        aws_access_key_id=settings.massive_access_key,
        aws_secret_access_key=settings.massive_secret_key,
        region_name=settings.massive_region,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )
    resp = s3.list_objects_v2(
        Bucket=settings.massive_s3_bucket,
        Prefix=prefix,
        MaxKeys=max(1, max_keys),
    )
    contents = resp.get("Contents") or []
    if not contents:
        print("No objects found for prefix.")
        return
    print(f"Found {len(contents)} objects (max_keys={max_keys})")
    for obj in contents:
        key = obj["Key"]
        size = obj.get("Size", 0)
        print(f"- {key} ({size} bytes)")
        if dry_run:
            continue
        out_path = dest / Path(key).name
        try:
            s3.download_file(settings.massive_s3_bucket, key, str(out_path))
            print(f"  downloaded -> {out_path}")
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "unknown")
            msg = exc.response.get("Error", {}).get("Message", "download failed")
            print(f"  download failed ({code}): {msg}")
            break


@app.command()
def refresh_daily(
    symbols: Optional[str] = typer.Option(None, help="Comma-separated watchlist"),
    universe: Optional[str] = typer.Option(
        None, help="Optional universe source (e.g., sp500) when symbols not provided"
    ),
    max_symbols: int = typer.Option(50, help="Max symbols to fetch when universe is used"),
    source: str = typer.Option("auto", help="auto/polygon/alpaca/yahoo/local"),
    upload_storage: bool = typer.Option(True, help="Upload Parquet to Supabase Storage"),
) -> None:
    settings = Settings()
    if symbols:
        tickers = [t.strip().upper() for t in symbols.split(",") if t.strip()]
    elif settings.watchlist:
        tickers = [t.strip().upper() for t in settings.watchlist.split(",") if t.strip()]
    elif universe and universe.lower() == "sp500":
        members = sp500_members_asof(asof=datetime.now(timezone.utc))
        tickers = members.tickers[: max_symbols]
        print(f"Universe sp500 members={len(members.tickers)} using={len(tickers)}")
    else:
        tickers = ["AAPL", "SPY", "QQQ"]
    if not tickers:
        raise typer.BadParameter("No symbols provided")

    train_walkforward(
        bars=r"C:\Users\Stas\Downloads\HistoricalData_1769169971316.csv",
        labels=r"C:\Users\Stas\Downloads\signals_rows.csv",
        signals=r"C:\Users\Stas\Downloads\signal_registry_top50_pack.zip",
        expand_rules=True,
        horizon=5,
        top_rules=8,
        news_path=None,
        social_path=None,
        qlib_dir=None,
        mode="lean",
        method="hedge",
    )
    for t in tickers:
        fetch_bars(
            symbol=t,
            from_date=None,
            to_date=None,
            source=source,
            timespan="day",
            multiplier=1,
            alpaca_timeframe="1Day",
            yahoo_interval="1d",
            limit=5000,
            local_path=None,
            to_supabase=True,
            upload_storage=upload_storage,
        )


@app.command()
def build_sp500_watchlist(
    asof: Optional[str] = typer.Option(None, help="YYYY-MM-DD (default: today UTC)"),
    universe_dir: Optional[str] = typer.Option(None, help="Universe parquet directory"),
    max_symbols: Optional[int] = typer.Option(500, help="Max symbols to keep"),
    output_path: Optional[str] = typer.Option(None, help="Output JSON path"),
) -> None:
    dt = datetime.now(timezone.utc)
    if asof:
        dt = pd.to_datetime(asof, utc=True)
    result = sp500_members_asof(asof=dt, universe_dir=universe_dir)
    tickers = result.tickers[: max_symbols] if max_symbols else result.tickers

    out_dir = Path("artifacts") / "watchlist"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(output_path) if output_path else out_dir / f"sp500_watchlist_{dt.date().isoformat()}.json"
    out = {
        "asof": dt.isoformat(),
        "count": len(tickers),
        "source": result.source,
        "tickers": tickers,
    }
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote watchlist: {out_path} (count={len(tickers)})")

@app.command()
def demo_run(mode: str = typer.Option("lean"), subject_id: str = typer.Option("AAPL")) -> None:
    """Writes a minimal demo run + event + one expert + one ensemble forecast."""
    settings = Settings()
    ledger = SupabaseLedger(settings)

    now = datetime.now(timezone.utc)
    run_id = ledger.create_run(asof_ts=now, mode=mode, config_hash="demo", code_version="0.1.0")
    print(f"Created run_id={run_id}")

    event_id = f"{subject_id}|state_event|demo|{now.date().isoformat()}"
    ledger.upsert_event(
        event_id=event_id,
        subject_type="ticker",
        subject_id=subject_id,
        event_type="state_event",
        asof_ts=now,
        horizon_json={"type":"time","value":5,"unit":"d"},
        event_params_json={"zone":{"low":100.0,"high":105.0,"method":"demo","zone_id":"demo"}},
        context_json={"news_state":{"intensity":0.2,"sentiment":0.0},"regime":{"id":"demo","probs":{"demo":1.0}}},
    )
    print(f"Upserted event_id={event_id}")

    ledger.insert_expert_forecast(
        run_id=run_id,
        event_id=event_id,
        expert_name="demo_expert",
        expert_version="0.1",
        probs_json={"REJECT":0.25,"BREAK_FAIL":0.15,"ACCEPT_CONTINUE":0.60},
        confidence=0.55,
        uncertainty_json={"entropy":0.95,"margin":0.35},
        calibration_group="demo",
    )

    ledger.insert_ensemble_forecast(
        run_id=run_id,
        event_id=event_id,
        method="hedge",
        probs_json={"REJECT":0.22,"BREAK_FAIL":0.18,"ACCEPT_CONTINUE":0.60},
        confidence=0.58,
        uncertainty_json={"entropy":0.92,"margin":0.38},
        weights_json={"demo_expert":1.0},
        explain_json={"attribution":[{"expert":"demo_expert","delta":0.0}]},
    )

    print("Wrote expert_forecasts + ensemble_forecasts")

@app.command()
def tools_status() -> None:
    """Check availability of optional integrations."""
    checks = {
        "discord.py": "discord",
        "transformers": "transformers",
        "finrl": "finrl",
        "qlib": "qlib",
        "finviz": "finviz",
    }
    for label, module in checks.items():
        available = importlib.util.find_spec(module) is not None
        status = "OK" if available else "missing"
        print(f"{label}: {status}")


@app.command()
def discord_pull(
    token: str = typer.Option(..., help="Discord bot token"),
    channels: str = typer.Option(..., help="Comma-separated channel IDs"),
    limit: int = typer.Option(200, help="Messages per channel"),
    after: Optional[str] = typer.Option(None, help="ISO timestamp (optional)"),
    before: Optional[str] = typer.Option(None, help="ISO timestamp (optional)"),
    output_path: Optional[str] = typer.Option(None, help="Output JSON path"),
) -> None:
    """Pull recent Discord messages into a JSON artifact."""
    from ddl69.integrations.discord_ingest import parse_channel_ids, pull_messages

    after_dt = pd.to_datetime(after, utc=True) if after else None
    before_dt = pd.to_datetime(before, utc=True) if before else None
    channel_ids = parse_channel_ids(channels)
    data = pull_messages(token=token, channel_ids=channel_ids, limit=limit, after=after_dt, before=before_dt)

    out_dir = Path("artifacts") / "social"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(output_path) if output_path else out_dir / f"discord_messages_{datetime.now(timezone.utc).date().isoformat()}.json"
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote Discord messages: {out_path}")


@app.command()
def fingpt_sentiment(
    text: Optional[str] = typer.Option(None, help="Single text input"),
    input_path: Optional[str] = typer.Option(None, help="CSV/JSON/JSONL file with text column"),
    text_column: Optional[str] = typer.Option(None, help="Column name for text (auto-detect if omitted)"),
    model: Optional[str] = typer.Option(None, help="HuggingFace model id (required)"),
    max_rows: int = typer.Option(200, help="Max rows to score from file"),
    output_path: Optional[str] = typer.Option(None, help="Output JSON path for file mode"),
    device: int = typer.Option(-1, help="Device id for transformers pipeline (-1 CPU)"),
) -> None:
    """Run sentiment scoring via a HuggingFace model (FinGPT-compatible)."""
    if not model:
        raise typer.BadParameter("model is required (e.g., a FinGPT or other HF sentiment model id)")
    try:
        from transformers import pipeline
    except Exception as exc:
        raise RuntimeError("transformers not installed; pip install transformers") from exc

    pipe = pipeline("text-classification", model=model, tokenizer=model, device=device)

    if text:
        result = pipe(text)
        print(result)
        return

    if not input_path:
        raise typer.BadParameter("Provide --text or --input-path")

    df = load_dataframe(input_path)
    if text_column and text_column in df.columns:
        col = text_column
    else:
        candidates = ["text_content", "text", "content", "body", "title", "headline", "message"]
        col = next((c for c in candidates if c in df.columns), None)
    if not col:
        raise typer.BadParameter("No text column found; pass --text-column")

    texts = df[col].dropna().astype(str).tolist()[:max_rows]
    scores = pipe(texts)
    out = [{"text": t, "score": s} for t, s in zip(texts, scores)]

    out_dir = Path("artifacts") / "nlp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(output_path) if output_path else out_dir / f"fingpt_sentiment_{datetime.now(timezone.utc).date().isoformat()}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote sentiment scores: {out_path}")


def _pick_text_column(df: pd.DataFrame, text_column: Optional[str]) -> str:
    if text_column and text_column in df.columns:
        return text_column
    candidates = ["text_content", "text", "content", "body", "title", "headline", "message"]
    col = next((c for c in candidates if c in df.columns), None)
    if not col:
        raise typer.BadParameter("No text column found; pass --text-column")
    return col


def _score_to_sentiment(score: dict[str, Any]) -> float:
    label = str(score.get("label", "")).lower()
    value = float(score.get("score", 0.0))
    if "neg" in label:
        return -value
    if "pos" in label:
        return value
    return 0.0


@app.command()
def fingpt_score_dataset(
    input_path: str = typer.Option(..., help="CSV/JSON/JSONL input with text column"),
    text_column: Optional[str] = typer.Option(None, help="Column name for text (auto-detect if omitted)"),
    model: Optional[str] = typer.Option(None, help="HuggingFace model id (required)"),
    max_rows: int = typer.Option(500, help="Max rows to score"),
    output_path: Optional[str] = typer.Option(None, help="Output file with sentiment column"),
    device: int = typer.Option(-1, help="Device id for transformers pipeline (-1 CPU)"),
) -> None:
    """Score a dataset and append a sentiment column (FinGPT-compatible)."""
    if not model:
        raise typer.BadParameter("model is required (e.g., a FinGPT or other HF sentiment model id)")
    try:
        from transformers import pipeline
    except Exception as exc:
        raise RuntimeError("transformers not installed; pip install transformers") from exc

    df = load_dataframe(input_path)
    col = _pick_text_column(df, text_column)
    texts = df[col].fillna("").astype(str).tolist()[:max_rows]

    pipe = pipeline("text-classification", model=model, tokenizer=model, device=device)
    scores = pipe(texts)
    sentiments = [_score_to_sentiment(s) for s in scores]

    out_df = df.copy()
    out_df = out_df.head(len(sentiments)).copy()
    out_df["sentiment"] = sentiments

    out_dir = Path("artifacts") / "nlp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(output_path) if output_path else out_dir / f"{Path(input_path).stem}_sentiment.json"
    save_dataframe(out_df, out_path)
    print(f"Wrote sentiment dataset: {out_path}")


@app.command()
def finrl_check() -> None:
    """Verify FinRL import."""
    try:
        import finrl  # type: ignore
    except Exception as exc:
        raise RuntimeError("finrl not installed; install from the FinRL GitHub repo") from exc
    print(f"FinRL import OK: {finrl.__name__}")


@app.command()
def finrl_download(
    tickers: str = typer.Option("AAPL,SPY,QQQ", help="Comma-separated tickers"),
    start: str = typer.Option("2019-01-01", help="Start date YYYY-MM-DD"),
    end: str = typer.Option(None, help="End date YYYY-MM-DD (default: today)"),
    output_path: Optional[str] = typer.Option(None, help="Output CSV path"),
) -> None:
    """Download OHLCV via FinRL's YahooDownloader."""
    try:
        from finrl.meta.preprocessor.yahoodownloader import YahooDownloader  # type: ignore
    except Exception as exc:
        raise RuntimeError("finrl not installed; install from the FinRL GitHub repo") from exc

    if not end:
        end = datetime.now(timezone.utc).date().isoformat()
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not ticker_list:
        raise typer.BadParameter("No tickers provided")

    df = YahooDownloader(start_date=start, end_date=end, ticker_list=ticker_list).fetch_data()
    out_dir = Path("artifacts") / "finrl"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(output_path) if output_path else out_dir / f"finrl_yahoo_{start}_{end}.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote FinRL Yahoo data: {out_path}")


@app.command()
def qlib_check(qlib_dir: str = typer.Option(..., help="Qlib data directory")) -> None:
    """Verify Qlib init with a local data directory."""
    try:
        import qlib  # type: ignore
    except Exception as exc:
        raise RuntimeError("qlib not installed; install from the Qlib GitHub repo") from exc
    qlib.init(provider_uri=qlib_dir, region="us")
    print("Qlib init OK")


def _extract_tar_gz_strip_first(src_path: Path, dest_dir: Path) -> None:
    with tarfile.open(src_path, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            parts = member.name.split("/")
            if len(parts) <= 1:
                continue
            member.name = "/".join(parts[1:])
            if not member.name:
                continue
            tar.extract(member, path=dest_dir)


@app.command()
def qlib_download(
    target_dir: str = typer.Option(..., help="Target directory for Qlib data"),
    region: str = typer.Option("us", help="Qlib region (us/cn)"),
    interval: Optional[str] = typer.Option(None, help="Interval (e.g., 1d, 1min)"),
    use_community: bool = typer.Option(
        False,
        help="Download community dataset tarball (open-source mirror) instead of Qlib CLI downloader",
    ),
) -> None:
    """Download Qlib data into a local directory."""
    dest = Path(target_dir)
    dest.mkdir(parents=True, exist_ok=True)

    if use_community:
        url = "https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz"
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "qlib_bin.tar.gz"
            urllib.request.urlretrieve(url, archive_path)
            _extract_tar_gz_strip_first(archive_path, dest)
        print(f"Downloaded community Qlib data to {dest}")
        return

    if importlib.util.find_spec("qlib") is None:
        raise RuntimeError("qlib not installed; install from the Qlib GitHub repo")

    cmd = [sys.executable, "-m", "qlib.cli.data", "qlib_data", "--target_dir", str(dest), "--region", region]
    if interval:
        cmd.extend(["--interval", interval])
    env = os.environ.copy()
    result = subprocess.run(cmd, check=False, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Qlib download failed (exit code {result.returncode})")
    print(f"Downloaded Qlib data to {dest}")


@app.command()
def finviz_check() -> None:
    """Verify finviz library import (unofficial)."""
    if importlib.util.find_spec("finviz") is None:
        raise RuntimeError("finviz package not installed (unofficial scraper)")
    print("finviz import OK")

if __name__ == "__main__":
    app()
