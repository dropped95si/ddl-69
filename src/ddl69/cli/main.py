from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
import asyncio
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
import requests
from rich import print

from ddl69.core.settings import Settings
from ddl69.ledger.supabase_ledger import SupabaseLedger
from ddl69.data.parquet_store import ParquetStore
from ddl69.experts.finbert import FinBertExpert
from ddl69.experts.qlib_adapter import QlibAdapter
from ddl69.experts.qlib_baseline import QlibBaseline
from ddl69.core.direction_engine import compute_direction
from ddl69.core.event_engine import compute_touch_zone_prob
from ddl69.core.execution_engine import compute_execution
from ddl69.core.probability_stack import Evidence, combine_probabilities
from ddl69.core.scope import get_scope
from ddl69.utils.ta_features import TAFeatures
from ddl69.data.cleaner import clean_dataset, load_dataframe, save_dataframe
from ddl69.utils.signals import (
    aggregate_rule_weights,
    blend_probs,
    ensemble_probs_from_rules,
    ensemble_probs_with_weights,
    entropy,
    load_signals_rows,
    rule_to_probs,
    sanitize_json,
    weights_from_rules,
)

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
    print("  fetch_options_proxy  build options-flow proxy from Yahoo")
    print("  rank_watchlist  rank tickers by highest predicted probability")
    print("  fetch_news_polygon  pull news for tickers from Polygon")
    print("  watchlist_report  build ranked watchlist + news summary")
    print("  refresh_daily  train weights + fetch bars for watchlist")
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
    calibration_path: Optional[str] = typer.Option(None, help="Calibration JSON path"),
    apply_calibration: bool = typer.Option(True, help="Apply calibration if available"),
) -> None:
    settings = Settings()
    ledger = SupabaseLedger(settings)
    store = ParquetStore(settings)

    df = load_signals_rows(signals_path)
    if max_rows is not None:
        df = df.head(max_rows)

    # Load calibration mapping if available
    calib = None
    if calibration_path:
        cpath = Path(calibration_path)
    else:
        cdir = Path("artifacts") / "calibration"
        cpath = None
        if cdir.exists():
            candidates = sorted(cdir.glob("calibration_*.json"))
            if candidates:
                cpath = candidates[-1]
    if cpath and cpath.exists():
        try:
            calib = json.loads(cpath.read_text(encoding="utf-8"))
        except Exception:
            calib = None

    calibrator = None
    if apply_calibration and calib and calib.get("walkforward_folds"):
        # build isotonic mapping from stored calibration pairs
        pairs = []
        for fold in calib.get("walkforward_folds", []):
            pairs.extend(fold.get("calibration", []))
        if pairs:
            try:
                from sklearn.isotonic import IsotonicRegression
                xs = [p[0] for p in pairs]
                ys = [p[1] for p in pairs]
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(xs, ys)
                calibrator = iso
            except Exception:
                calibrator = None

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
        # regime gating if available
        if calib and "weights_by_regime" in calib and "regime" in row and row["regime"] in calib["weights_by_regime"]:
            weights = calib["weights_by_regime"][row["regime"]]

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
            if calib and "weights_by_regime" in calib and "regime" in row and row["regime"] in calib["weights_by_regime"]:
                ensemble_probs = ensemble_probs_with_weights(rules, weights)
            else:
                ensemble_probs = blend_probs(weighted_probs)
            # apply calibration to ACCEPT_CONTINUE if available
            if calibrator is not None:
                p_acc = float(ensemble_probs.get("ACCEPT_CONTINUE", 0.5))
                p_acc_c = float(calibrator.predict([p_acc])[0])
                other = max(1e-9, 1.0 - p_acc)
                other_c = max(0.0, 1.0 - p_acc_c)
                scale = other_c / other
                ensemble_probs["ACCEPT_CONTINUE"] = p_acc_c
                ensemble_probs["REJECT"] = float(ensemble_probs.get("REJECT", 0.0)) * scale
                ensemble_probs["BREAK_FAIL"] = float(ensemble_probs.get("BREAK_FAIL", 0.0)) * scale
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
    mode: str = typer.Option("lean", help="Run mode"),
    method: str = typer.Option("hedge", help="Weight method"),
    walkforward_window_days: int = typer.Option(120, help="Train window size in days"),
    test_window_days: int = typer.Option(30, help="Test window size in days"),
    purge_days: int = typer.Option(5, help="Purge window around test"),
    embargo_days: int = typer.Option(2, help="Embargo window after test"),
    acceptance_rule: str = typer.Option("N_bars", help="Acceptance rule: N_bars/atr/both"),
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
    if "created_at" in df.columns:
        df = df.dropna(subset=["created_at"]).sort_values("created_at").reset_index(drop=True)
    def _compute_regime_map() -> Optional[pd.DataFrame]:
        try:
            bars_df = pd.read_csv(bars)
        except Exception:
            return None
        # normalize
        bars_df = bars_df.rename(
            columns={
                "Date": "timestamp",
                "date": "timestamp",
                "Close/Last": "close",
                "Close": "close",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Volume": "volume",
            }
        )
        if "timestamp" not in bars_df.columns or "close" not in bars_df.columns:
            return None
        # strip $ if present
        bars_df["close"] = (
            bars_df["close"]
            .astype(str)
            .str.replace("$", "", regex=False)
        )
        bars_df["close"] = pd.to_numeric(bars_df["close"], errors="coerce")
        bars_df["timestamp"] = pd.to_datetime(bars_df["timestamp"], utc=True, errors="coerce")
        bars_df = bars_df.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
        if bars_df.empty:
            return None
        bars_df["ret"] = bars_df["close"].pct_change()
        roll = 20
        bars_df["roll_ret"] = bars_df["close"].pct_change(roll)
        bars_df["roll_vol"] = bars_df["ret"].rolling(roll).std()
        def _regime(row: pd.Series) -> str:
            if pd.isna(row["roll_ret"]) or pd.isna(row["roll_vol"]):
                return "unknown"
            if row["roll_ret"] > row["roll_vol"]:
                return "bull"
            if row["roll_ret"] < -row["roll_vol"]:
                return "bear"
            return "range"
        bars_df["regime"] = bars_df.apply(_regime, axis=1)
        return bars_df[["timestamp", "regime"]]

    regime_map = _compute_regime_map()
    if regime_map is not None:
        df = df.copy()
        df["regime"] = pd.merge_asof(
            df.sort_values("created_at"),
            regime_map.sort_values("timestamp"),
            left_on="created_at",
            right_on="timestamp",
            direction="backward",
        )["regime"].values

    # walk-forward splits with purge/embargo
    def _walkforward_splits(dates: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        if dates.empty:
            return []
        start = dates.min()
        end = dates.max()
        splits = []
        cursor = start
        while True:
            train_end = cursor + pd.Timedelta(days=walkforward_window_days)
            test_start = train_end + pd.Timedelta(days=purge_days)
            test_end = test_start + pd.Timedelta(days=test_window_days)
            if test_start > end:
                break
            splits.append((test_start, test_end))
            cursor = test_end + pd.Timedelta(days=embargo_days)
            if cursor > end:
                break
        return splits

    splits = _walkforward_splits(df["created_at"])
    fold_calib = []
    fold_weights = []
    for test_start, test_end in splits:
        # purge train window around test
        train_end = test_start - pd.Timedelta(days=purge_days)
        train_start = train_end - pd.Timedelta(days=walkforward_window_days)
        train_mask = (df["created_at"] >= train_start) & (df["created_at"] <= train_end)
        test_mask = (df["created_at"] >= test_start) & (df["created_at"] <= test_end)
        train_rows = df[train_mask]
        test_rows = df[test_mask]
        if train_rows.empty or test_rows.empty:
            continue
        weights, calib = aggregate_rule_weights(train_rows.to_dict(orient="records"))
        fold_weights.append(
            {
                "train_start": str(train_start.date()),
                "train_end": str(train_end.date()),
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "weights": weights,
            }
        )
        # optional calibration metrics if realized_label exists
        if "realized_label" in test_rows.columns:
            y = (test_rows["realized_label"].astype(str) == "ACCEPT_CONTINUE").astype(int).values
            preds = []
            for r in test_rows.to_dict(orient="records"):
                rules = r.get("learned_top_rules") or []
                probs, _w = ensemble_probs_from_rules(rules)
                preds.append(float(probs.get("ACCEPT_CONTINUE", 0.5)))
            try:
                from sklearn.isotonic import IsotonicRegression
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(preds, y)
                cal_preds = iso.predict(preds)
                brier = float(((cal_preds - y) ** 2).mean())
                fold_calib.append(
                    {
                        "test_start": str(test_start.date()),
                        "test_end": str(test_end.date()),
                        "brier": brier,
                        "calibration": list(zip(preds[:200], cal_preds[:200])),
                    }
                )
            except Exception:
                pass

    # global weights from all data
    weights, calib = aggregate_rule_weights(df.to_dict(orient="records"))
    calib["acceptance_rule"] = acceptance_rule
    calib["walkforward_folds"] = fold_calib
    calib["weights_by_fold"] = fold_weights
    if "regime" in df.columns:
        calib["weights_by_regime"] = {
            r: aggregate_rule_weights(df[df["regime"] == r].to_dict(orient="records"))[0]
            for r in sorted(df["regime"].dropna().unique())
        }

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
def refresh_daily(
    symbols: Optional[str] = typer.Option(None, help="Comma-separated watchlist"),
    source: str = typer.Option("auto", help="auto/polygon/alpaca/yahoo/local"),
    upload_storage: bool = typer.Option(True, help="Upload Parquet to Supabase Storage"),
) -> None:
    settings = Settings()
    watchlist = symbols or settings.watchlist or "AAPL,SPY,QQQ"
    tickers = [t.strip().upper() for t in watchlist.split(",") if t.strip()]
    if not tickers:
        raise typer.BadParameter("No symbols provided")

    train_walkforward()
    for t in tickers:
        fetch_bars(
            symbol=t,
            source=source,
            to_supabase=True,
            upload_storage=upload_storage,
        )


@app.command()
def fetch_options_proxy(
    symbol: str = typer.Option("AAPL", help="Ticker symbol"),
    expirations: Optional[int] = typer.Option(1, help="Number of expirations to fetch"),
    upload_storage: bool = typer.Option(False, help="Upload CSV to Supabase Storage"),
) -> None:
    try:
        import yfinance as yf
    except Exception as exc:
        raise RuntimeError("yfinance not installed; pip install yfinance") from exc

    settings = Settings()
    ticker = yf.Ticker(symbol.upper())
    exps = ticker.options or []
    if not exps:
        raise RuntimeError("No options expirations available from Yahoo")
    exps = exps[: max(1, expirations)]

    rows = []
    for exp in exps:
        chain = ticker.option_chain(exp)
        for side, df in (("call", chain.calls), ("put", chain.puts)):
            if df is None or df.empty:
                continue
            df = df.copy()
            df["expiration"] = exp
            df["side"] = side
            rows.append(df)

    if not rows:
        raise RuntimeError("No option chain data returned")

    df = pd.concat(rows, ignore_index=True)
    # flow proxy: volume / openInterest and notional
    df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0.0)
    df["openInterest"] = pd.to_numeric(df.get("openInterest", 0), errors="coerce").fillna(0.0)
    df["lastPrice"] = pd.to_numeric(df.get("lastPrice", 0), errors="coerce").fillna(0.0)
    df["flow_proxy"] = df["volume"] / df["openInterest"].replace(0, pd.NA)
    df["flow_proxy"] = df["flow_proxy"].fillna(0.0)
    df["notional"] = df["lastPrice"] * df["volume"] * 100.0
    df["symbol"] = symbol.upper()

    out_dir = Path("artifacts") / "options"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"options_proxy_{symbol.upper()}_{datetime.now(timezone.utc).date().isoformat()}.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote options proxy: {out_path}")

    if upload_storage:
        ledger = SupabaseLedger(settings)
        storage_uri = ledger.upload_storage(
            bucket=settings.supabase_storage_bucket,
            local_path=str(out_path),
            dest_path=f"options/{out_path.name}",
            upsert=True,
        )
        print(f"Uploaded to Supabase Storage: {storage_uri}")


def _load_latest_calibration() -> Optional[dict]:
    cdir = Path("artifacts") / "calibration"
    if not cdir.exists():
        return None
    candidates = sorted(cdir.glob("calibration_*.json"))
    if not candidates:
        return None
    try:
        return json.loads(candidates[-1].read_text(encoding="utf-8"))
    except Exception:
        return None


def _build_calibrator(calib: Optional[dict]):
    if not calib or not calib.get("walkforward_folds"):
        return None
    pairs = []
    for fold in calib.get("walkforward_folds", []):
        pairs.extend(fold.get("calibration", []))
    if not pairs:
        return None
    try:
        from sklearn.isotonic import IsotonicRegression
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(xs, ys)
        return iso
    except Exception:
        return None


def _load_latest_bars_parquet() -> Optional[pd.DataFrame]:
    bars_dir = Path("artifacts") / "bars"
    if not bars_dir.exists():
        return None
    candidates = sorted(bars_dir.glob("*.parquet"))
    if not candidates:
        return None
    frames = []
    for path in candidates[-50:]:
        try:
            df = pd.read_parquet(path)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        df = df.rename(
            columns={
                "instrument_id": "ticker",
                "symbol": "ticker",
                "timestamp": "timestamp",
                "time": "timestamp",
            }
        )
        if "ticker" not in df.columns:
            continue
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _compute_ta_signals(bars_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    if bars_df is None or bars_df.empty:
        return {}
    required = {"open", "high", "low", "close"}
    if not required.issubset(set(bars_df.columns)):
        return {}
    out: dict[str, dict[str, float]] = {}
    ta = TAFeatures()
    try:
        import talib  # type: ignore
    except Exception:
        talib = None
    for ticker, grp in bars_df.groupby("ticker"):
        g = grp.sort_values("timestamp").tail(260)
        if g.shape[0] < 80:
            continue
        feats = ta.compute(g)
        latest = feats.iloc[-1]
        rsi = float(latest.get("rsi", 50.0))
        atr = float(latest.get("atr", 0.0))
        macd_hist = float(latest.get("macd_hist", 0.0))
        ema_fast = float(latest.get("ema_fast", np.nan))
        ema_slow = float(latest.get("ema_slow", np.nan))
        close = float(latest.get("close", np.nan))
        vol = float(latest.get("volume", np.nan)) if "volume" in latest else np.nan

        # TA signal weights (soft, signed)
        weights: dict[str, float] = {}
        if rsi <= 30:
            weights["RSI_OVERSOLD"] = 0.18
        elif rsi >= 70:
            weights["RSI_OVERBOUGHT"] = -0.14

        if macd_hist > 0:
            weights["MACD_POSITIVE"] = 0.14
        elif macd_hist < 0:
            weights["MACD_NEGATIVE"] = -0.12

        if not np.isnan(ema_fast) and not np.isnan(ema_slow):
            if ema_fast > ema_slow:
                weights["EMA_CROSS_POS"] = 0.14
            else:
                weights["EMA_CROSS_NEG"] = -0.12

        # ATR expansion proxy
        atr_mean = feats["atr"].rolling(20).mean().iloc[-1]
        if not np.isnan(atr_mean) and atr_mean > 0 and atr / atr_mean > 1.2:
            weights["ATR_EXPANSION"] = 0.10

        # Volume spike proxy
        if "volume" in feats.columns and not np.isnan(vol):
            vol_mean = feats["volume"].rolling(20).mean().iloc[-1]
            if not np.isnan(vol_mean) and vol_mean > 0 and vol / vol_mean > 1.5:
                weights["VOL_SPIKE"] = 0.10

        # Range break proxy
        if not np.isnan(close):
            high20 = feats["high"].rolling(20).max().iloc[-1] if "high" in feats.columns else np.nan
            low20 = feats["low"].rolling(20).min().iloc[-1] if "low" in feats.columns else np.nan
            if not np.isnan(high20) and close >= high20:
                weights["BREAKOUT_20D"] = 0.12
            if not np.isnan(low20) and close <= low20:
                weights["BREAKDOWN_20D"] = -0.10

        # Optional TA-Lib enrichments
        if talib is not None:
            try:
                close_arr = g["close"].astype(float).to_numpy()
                high_arr = g["high"].astype(float).to_numpy()
                low_arr = g["low"].astype(float).to_numpy()
                vol_arr = g["volume"].astype(float).to_numpy() if "volume" in g.columns else None

                bb_upper, bb_mid, bb_lower = talib.BBANDS(close_arr, timeperiod=20)
                if close_arr[-1] > bb_upper[-1]:
                    weights["BB_UPPER_BREAK"] = 0.08
                elif close_arr[-1] < bb_lower[-1]:
                    weights["BB_LOWER_BREAK"] = -0.08

                slowk, slowd = talib.STOCH(high_arr, low_arr, close_arr)
                if slowk[-1] < 20 and slowd[-1] < 20:
                    weights["STOCH_OVERSOLD"] = 0.08
                elif slowk[-1] > 80 and slowd[-1] > 80:
                    weights["STOCH_OVERBOUGHT"] = -0.08

                adx = talib.ADX(high_arr, low_arr, close_arr, timeperiod=14)
                if adx[-1] > 25:
                    weights["ADX_TRENDING"] = 0.06

                cci = talib.CCI(high_arr, low_arr, close_arr, timeperiod=20)
                if cci[-1] > 100:
                    weights["CCI_STRONG"] = 0.06
                elif cci[-1] < -100:
                    weights["CCI_WEAK"] = -0.06

                if vol_arr is not None:
                    mfi = talib.MFI(high_arr, low_arr, close_arr, vol_arr, timeperiod=14)
                    if mfi[-1] < 20:
                        weights["MFI_OVERSOLD"] = 0.06
                    elif mfi[-1] > 80:
                        weights["MFI_OVERBOUGHT"] = -0.06
            except Exception:
                pass

        if weights:
            out[str(ticker).upper()] = weights
    return out


def _ta_weights_to_prob(weights: dict[str, float]) -> float:
    if not weights:
        return 0.5
    score = float(sum(weights.values()))
    # map ~[-1..1] to [0.05..0.95]
    score = max(-1.0, min(1.0, score))
    p = 0.5 + 0.45 * score
    return float(max(0.05, min(0.95, p)))


def _load_latest_sentiment_probs() -> dict[str, float]:
    sent_dir = Path("artifacts") / "sentiment"
    if not sent_dir.exists():
        return {}
    candidates = sorted(sent_dir.glob("news_sentiment_*.json"))
    if not candidates:
        return {}
    try:
        data = json.loads(candidates[-1].read_text(encoding="utf-8"))
    except Exception:
        return {}
    rows = data.get("rows") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return {}
    per: dict[str, list[float]] = {}
    for row in rows:
        t = str(row.get("ticker") or "").upper()
        probs = row.get("probs") or {}
        if not t or not isinstance(probs, dict):
            continue
        p_pos = float(probs.get("positive", 0.0))
        p_neg = float(probs.get("negative", 0.0))
        p = 0.5 + 0.5 * (p_pos - p_neg)
        per.setdefault(t, []).append(max(0.05, min(0.95, p)))
    return {t: float(np.mean(vals)) for t, vals in per.items()}


def _load_latest_social_probs() -> dict[str, float]:
    social_dir = Path("artifacts") / "social"
    if not social_dir.exists():
        return {}
    candidates = sorted(social_dir.glob("social_sentiment_*.json"))
    if not candidates:
        return {}
    try:
        data = json.loads(candidates[-1].read_text(encoding="utf-8"))
    except Exception:
        return {}
    rows = data.get("rows") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return {}
    per: dict[str, list[float]] = {}
    for row in rows:
        t = str(row.get("ticker") or "").upper()
        probs = row.get("probs") or {}
        if not t or not isinstance(probs, dict):
            continue
        p_pos = float(probs.get("positive", 0.0))
        p_neg = float(probs.get("negative", 0.0))
        p = 0.5 + 0.5 * (p_pos - p_neg)
        per.setdefault(t, []).append(max(0.05, min(0.95, p)))
    return {t: float(np.mean(vals)) for t, vals in per.items()}


def _load_qlib_probs() -> dict[str, float]:
    data_dir = os.getenv("QLIB_DATA_DIR", "").strip()
    if not data_dir:
        return {}
    model_path = Path("artifacts") / "qlib" / "linear_baseline.joblib"
    if not model_path.exists():
        return {}
    try:
        from joblib import load
        baseline = QlibBaseline(data_dir=data_dir)
        df = baseline.load_features(market="sp500")
        x, _ = baseline._build_dataset(df)
        if x.empty or not isinstance(x.index, pd.MultiIndex):
            return {}
        model = load(model_path)
        preds = pd.Series(model.predict(x), index=x.index)
        # latest date per instrument
        preds = preds.reset_index().rename(columns={"level_0": "dt", "level_1": "ticker", 0: "score"})
        preds["dt"] = pd.to_datetime(preds["dt"], errors="coerce")
        latest = preds.sort_values("dt").groupby("ticker").tail(1)
        scores = latest.set_index("ticker")["score"]
        probs_df = QlibBaseline.scores_to_probs(scores)
        return {str(k).upper(): float(v) for k, v in probs_df["p_accept"].to_dict().items()}
    except Exception:
        return {}


def _load_latest_regime_prob() -> Optional[float]:
    reg_dir = Path("artifacts") / "regime"
    if not reg_dir.exists():
        return None
    candidates = sorted(reg_dir.glob("regime_probs*.json"))
    if not candidates:
        return None
    try:
        data = json.loads(candidates[-1].read_text(encoding="utf-8"))
    except Exception:
        return None
    probs = data.get("probs")
    if not isinstance(probs, list) or not probs:
        return None
    last = probs[-1]
    if not isinstance(last, list) or not last:
        return None
    bull_state = data.get("bull_state")
    bear_state = data.get("bear_state")
    if bull_state is None or bear_state is None:
        return float(max(last))
    try:
        p_bull = float(last[int(bull_state)])
        p_bear = float(last[int(bear_state)])
    except Exception:
        return float(max(last))
    p = 0.5 + 0.5 * (p_bull - p_bear)
    return max(0.05, min(0.95, p))


def _load_latest_monte_carlo_probs() -> dict[str, float]:
    mc_dir = Path("artifacts") / "monte_carlo"
    if not mc_dir.exists():
        return {}
    candidates = sorted(mc_dir.glob("mc_paths*.json"))
    if not candidates:
        return {}
    per: dict[str, float] = {}
    for path in candidates[-5:]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        symbol = str(data.get("symbol") or "").upper()
        paths = data.get("paths_tail") or []
        s0 = data.get("s0")
        if not symbol or not isinstance(paths, list) or s0 is None:
            continue
        try:
            s0 = float(s0)
            vals = np.array(paths, dtype=float)
        except Exception:
            continue
        if vals.size == 0:
            continue
        p_up = float(np.mean(vals > s0))
        per[symbol] = max(0.05, min(0.95, p_up))
    return per


def _load_latest_options_proxy() -> dict[str, float]:
    opt_dir = Path("artifacts") / "options"
    if not opt_dir.exists():
        return {}
    candidates = sorted(opt_dir.glob("options_proxy_*.csv"))
    if not candidates:
        return {}
    per: dict[str, float] = {}
    for path in candidates[-10:]:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        df["notional"] = pd.to_numeric(df.get("notional", 0), errors="coerce").fillna(0.0)
        if "symbol" not in df.columns or "side" not in df.columns:
            continue
        for sym, g in df.groupby("symbol"):
            calls = g[g["side"] == "call"]["notional"].sum()
            puts = g[g["side"] == "put"]["notional"].sum()
            total = calls + puts
            if total <= 0:
                continue
            p = 0.5 + 0.5 * ((calls - puts) / total)
            per[str(sym).upper()] = max(0.05, min(0.95, p))
    return per


def _load_latest_finviz_probs() -> dict[str, float]:
    if os.getenv("DISABLE_FINVIZ", "").strip().lower() in {"1", "true", "yes"}:
        return {}
    fin_dir = Path("artifacts") / "finviz"
    if not fin_dir.exists():
        return {}
    candidates = sorted(fin_dir.glob("finviz_*.json"))
    if not candidates:
        return {}
    try:
        rows = json.loads(candidates[-1].read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(rows, list) or not rows:
        return {}
    # Rank by relative volume or volume; fallback to list order
    def _score(row: dict) -> float:
        for key in ("Relative Volume", "Rel Volume", "RelVolume", "Volume", "Avg Volume"):
            if key in row:
                try:
                    return float(str(row[key]).replace(",", ""))
                except Exception:
                    continue
        return 0.0
    rows = [r for r in rows if isinstance(r, dict)]
    rows_sorted = sorted(rows, key=_score, reverse=True)
    n = len(rows_sorted)
    per: dict[str, float] = {}
    for i, r in enumerate(rows_sorted):
        t = str(r.get("Ticker") or r.get("ticker") or "").upper()
        if not t:
            continue
        # map rank to probability
        p = 0.9 - 0.6 * (i / max(1, n - 1))
        per[t] = max(0.05, min(0.95, p))
    return per


def _compute_mc_probs_from_bars(bars_df: pd.DataFrame, horizon_days: int = 5) -> dict[str, float]:
    if bars_df is None or bars_df.empty:
        return {}
    per: dict[str, float] = {}
    for ticker, grp in bars_df.groupby("ticker"):
        g = grp.sort_values("timestamp").tail(120)
        if g.shape[0] < 60:
            continue
        close = g["close"].astype(float)
        rets = np.log(close).diff().dropna()
        if rets.empty:
            continue
        mu = float(rets.mean())
        sigma = float(rets.std())
        h = max(1, horizon_days)
        # Normal approx probability of positive return over horizon
        mean = mu * h
        std = sigma * np.sqrt(h) if sigma > 0 else 1e-6
        z = (0 - mean) / std
        p_up = 0.5 * (1.0 - math.erf(z / np.sqrt(2)))
        per[str(ticker).upper()] = max(0.05, min(0.95, float(p_up)))
    return per


def _compute_lopez_barrier_probs(bars_df: pd.DataFrame, sims: int = 200, horizon_days: int = 10) -> dict[str, float]:
    if bars_df is None or bars_df.empty:
        return {}
    per: dict[str, float] = {}
    rng = np.random.default_rng(7)
    for ticker, grp in bars_df.groupby("ticker"):
        g = grp.sort_values("timestamp").tail(180)
        if g.shape[0] < 90:
            continue
        close = g["close"].astype(float)
        rets = np.log(close).diff().dropna()
        if rets.empty:
            continue
        mu = float(rets.mean())
        sigma = float(rets.std())
        last = float(close.iloc[-1])
        atr = None
        try:
            feats = TAFeatures().compute(g)
            atr = float(feats["atr"].iloc[-1])
        except Exception:
            atr = None
        if atr is None or atr <= 0:
            atr = float(close.pct_change().rolling(20).std().iloc[-1] * last)
        upper = last + 2.0 * atr
        lower = max(0.01, last - 2.0 * atr)
        # simulate paths
        rand = rng.normal(size=(sims, horizon_days))
        paths = last * np.exp(np.cumsum((mu - 0.5 * sigma ** 2) + sigma * rand, axis=1))
        hit_upper = (paths >= upper).any(axis=1)
        hit_lower = (paths <= lower).any(axis=1)
        both = hit_upper & hit_lower
        p_up = float(np.mean(hit_upper & ~hit_lower) + 0.5 * np.mean(both))
        per[str(ticker).upper()] = max(0.05, min(0.95, p_up))
    return per


def _compute_direction_probs(bars_df: pd.DataFrame) -> dict[str, float]:
    if bars_df is None or bars_df.empty:
        return {}
    per: dict[str, float] = {}
    for ticker, grp in bars_df.groupby("ticker"):
        g = grp.sort_values("timestamp").tail(120)
        if g.shape[0] < 60:
            continue
        try:
            direction = compute_direction(g)
        except Exception:
            continue
        if direction.bias == "UP":
            p = 0.5 + 0.5 * float(direction.confidence)
        elif direction.bias == "DOWN":
            p = 0.5 - 0.5 * float(direction.confidence)
        else:
            p = 0.5
        per[str(ticker).upper()] = max(0.05, min(0.95, p))
    return per


@app.command()
def rank_watchlist(
    labels: str = typer.Option(
        "C:\\Users\\Stas\\Downloads\\signals_rows.csv",
        help="Path to signals_rows.csv",
    ),
    top_n: int = typer.Option(30, help="How many tickers to keep"),
    output_path: Optional[str] = typer.Option(None, help="Output JSON path"),
    use_calibration: bool = typer.Option(True, help="Apply calibration if available"),
    upload_storage: bool = typer.Option(False, help="Upload watchlist JSON to Supabase Storage"),
    to_supabase: bool = typer.Option(False, help="Insert into public.watchlist_rankings"),
) -> None:
    df = load_signals_rows(labels)
    if df.empty:
        raise RuntimeError("No labels/signals data loaded")

    bars_df = _load_latest_bars_parquet()
    ta_weights = _compute_ta_signals(bars_df) if bars_df is not None else {}
    news_probs = _load_latest_sentiment_probs()
    social_probs = _load_latest_social_probs()
    qlib_probs = _load_qlib_probs()
    regime_prob = _load_latest_regime_prob()
    mc_probs = _load_latest_monte_carlo_probs()
    mc_bar_probs = _compute_mc_probs_from_bars(bars_df) if bars_df is not None else {}
    lopez_probs = _compute_lopez_barrier_probs(bars_df) if bars_df is not None else {}
    direction_probs = _compute_direction_probs(bars_df) if bars_df is not None else {}
    options_probs = _load_latest_options_proxy()
    finviz_probs = _load_latest_finviz_probs() if use_finviz else {}

    calib = _load_latest_calibration()
    calibrator = _build_calibrator(calib) if use_calibration else None
    now = datetime.now(timezone.utc)

    rows = []
    for _, row in df.iterrows():
        rules = row.get("learned_top_rules") or []
        probs, weights = ensemble_probs_from_rules(rules) if isinstance(rules, list) else (
            {"REJECT": 0.34, "BREAK_FAIL": 0.33, "ACCEPT_CONTINUE": 0.33},
            {},
        )
        ticker = str(row.get("ticker", "")).upper()
        p_acc = float(probs.get("ACCEPT_CONTINUE", 0.5))
        if calibrator is not None:
            p_acc_c = float(calibrator.predict([p_acc])[0])
        else:
            p_acc_c = p_acc
        # Avoid hard 0/1 scores from tiny samples or perfect win rates.
        p_acc_c = max(0.01, min(0.99, p_acc_c))
        # Blend additional evidence sources (TA / News / Qlib / HMM / MC / Options / Finviz / Social)
        evidence = {
            "signals_rows": Evidence("signals_rows", p_acc_c, 0.42),
        }
        if ticker in ta_weights:
            evidence["ta"] = Evidence("ta", _ta_weights_to_prob(ta_weights[ticker]), 0.12)
        if ticker in news_probs:
            evidence["news"] = Evidence("news", news_probs[ticker], 0.10)
        if ticker in social_probs:
            evidence["social"] = Evidence("social", social_probs[ticker], 0.08)
        if ticker in qlib_probs:
            evidence["qlib"] = Evidence("qlib", qlib_probs[ticker], 0.10)
        if ticker in mc_probs:
            evidence["mc"] = Evidence("mc", mc_probs[ticker], 0.06)
        if ticker in mc_bar_probs:
            evidence["mc_bar"] = Evidence("mc_bar", mc_bar_probs[ticker], 0.06)
        if ticker in lopez_probs:
            evidence["lopez"] = Evidence("lopez", lopez_probs[ticker], 0.06)
        if ticker in options_probs:
            evidence["options"] = Evidence("options", options_probs[ticker], 0.06)
        if ticker in finviz_probs:
            evidence["finviz"] = Evidence("finviz", finviz_probs[ticker], 0.05)
        if ticker in direction_probs:
            evidence["direction"] = Evidence("direction", direction_probs[ticker], 0.05)
        if regime_prob is not None:
            evidence["regime"] = Evidence("regime", regime_prob, 0.05)

        p_final = combine_probabilities(evidence, bias=0.0)
        p_final = max(0.01, min(0.99, float(p_final)))
        score = p_final - 0.1 * entropy(probs)

        # Build weights for UI (rule weights scaled + extra sources)
        weights = weights or {}
        rule_sum = sum(float(v) for v in weights.values()) or 1.0
        scaled_rules = {k: float(v) / rule_sum * 0.40 for k, v in weights.items()}
        if ticker in ta_weights:
            for k, v in ta_weights[ticker].items():
                scaled_rules[k] = float(v)
        if ticker in news_probs:
            scaled_rules["NEWS_SENTIMENT"] = 0.10
        if ticker in social_probs:
            scaled_rules["SOCIAL_SENTIMENT"] = 0.08
        if ticker in qlib_probs:
            scaled_rules["QLIB_SIGNAL"] = 0.10
        if ticker in mc_probs:
            scaled_rules["MC_PROB"] = 0.06
        if ticker in mc_bar_probs:
            scaled_rules["MC_RETURN_PROB"] = 0.06
        if ticker in lopez_probs:
            scaled_rules["LOPEZ_BARRIER"] = 0.06
        if ticker in options_probs:
            scaled_rules["OPTIONS_FLOW"] = 0.06
        if ticker in finviz_probs:
            scaled_rules["FINVIZ_POPULAR"] = 0.05
        if ticker in direction_probs:
            scaled_rules["DIRECTION_ENGINE"] = 0.05
        if regime_prob is not None:
            scaled_rules["REGIME_HMM"] = 0.05

        rows.append(
            {
                "ticker": ticker,
                "label": row.get("label"),
                "plan_type": row.get("plan_type"),
                "created_at": str(row.get("created_at")) if row.get("created_at") is not None else None,
                "p_accept": p_final,
                "score": score,
                "weights": scaled_rules,
                "meta": {
                    "p_base": p_acc_c,
                    "p_ta": _ta_weights_to_prob(ta_weights[ticker]) if ticker in ta_weights else None,
                    "p_news": news_probs.get(ticker),
                    "p_social": social_probs.get(ticker),
                    "p_qlib": qlib_probs.get(ticker),
                    "p_mc": mc_probs.get(ticker),
                    "p_mc_bar": mc_bar_probs.get(ticker),
                    "p_lopez": lopez_probs.get(ticker),
                    "p_options": options_probs.get(ticker),
                    "p_finviz": finviz_probs.get(ticker),
                    "p_direction": direction_probs.get(ticker),
                    "p_regime": regime_prob,
                },
            }
        )

    # de-duplicate by ticker first (keep best score per ticker), then rank
    dedup: dict[str, dict[str, Any]] = {}
    for r in rows:
        t = r["ticker"]
        if not t:
            continue
        if t not in dedup or r["score"] > dedup[t]["score"]:
            dedup[t] = r
    ranked = sorted(dedup.values(), key=lambda r: r["score"], reverse=True)[:top_n]
    out = {
        "asof": now.isoformat(),
        "top_n": top_n,
        "ranked": ranked,
    }
    if output_path is None:
        out_dir = Path("artifacts") / "watchlist"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"watchlist_{now.date().isoformat()}.json")
    Path(output_path).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote watchlist: {output_path}")

    if upload_storage or to_supabase:
        ledger = SupabaseLedger(Settings())
        if upload_storage:
            try:
                storage_uri = ledger.upload_storage(
                    bucket=Settings().supabase_storage_bucket,
                    local_path=str(output_path),
                    dest_path=f"watchlist/{Path(output_path).name}",
                    upsert=True,
                )
                public_url = ledger.public_storage_url(
                    bucket=Settings().supabase_storage_bucket,
                    dest_path=f"watchlist/{Path(output_path).name}",
                )
                print(f"Uploaded to Supabase Storage: {storage_uri}")
                print(f"Public URL: {public_url}")
            except Exception as exc:
                print(f"Storage upload failed (create bucket in Supabase): {exc}")

        if to_supabase:
            payload = []
            for r in ranked:
                payload.append(
                    {
                        "asof_ts": now.isoformat(),
                        "ticker": r["ticker"],
                        "score": float(r["score"]),
                        "p_accept": float(r["p_accept"]),
                        "label": r.get("label"),
                        "plan_type": r.get("plan_type"),
                        "weights_json": r.get("weights") or {},
                        "meta_json": {
                            "created_at": r.get("created_at"),
                        },
                    }
                )
            try:
                ledger.client.table("watchlist_rankings").upsert(
                    payload, on_conflict="asof_ts,ticker"
                ).execute()
                print("Inserted into Supabase public.watchlist_rankings")
            except Exception as exc:
                print(f"Supabase watchlist insert failed: {exc}")
    return out


@app.command()
def fetch_news_polygon(
    tickers: str = typer.Option("AAPL,SPY,QQQ", help="Comma-separated tickers"),
    limit: int = typer.Option(50, help="Max items per ticker"),
    upload_storage: bool = typer.Option(False, help="Upload JSON to Supabase Storage"),
    to_supabase: bool = typer.Option(False, help="Insert into public.news_items if table exists"),
) -> None:
    settings = Settings()
    if not settings.polygon_api_key:
        raise typer.BadParameter("POLYGON_API_KEY not set in environment")

    symbols = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    all_items = []
    for sym in symbols:
        url = "https://api.polygon.io/v2/reference/news"
        params = {"ticker": sym, "limit": limit, "apiKey": settings.polygon_api_key}
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            print(f"Polygon news error for {sym}: {resp.status_code}")
            continue
        data = resp.json()
        items = data.get("results") or []
        for item in items:
            item["tickers"] = item.get("tickers") or [sym]
        all_items.extend(items)

    if not all_items:
        print("No news returned")
        return

    out_dir = Path("artifacts") / "news"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"polygon_news_{datetime.now(timezone.utc).date().isoformat()}.json"
    out_path.write_text(json.dumps(all_items, indent=2), encoding="utf-8")
    print(f"Wrote news: {out_path}")

    if upload_storage:
        ledger = SupabaseLedger(settings)
        try:
            storage_uri = ledger.upload_storage(
                bucket=settings.supabase_storage_bucket,
                local_path=str(out_path),
                dest_path=f"news/{out_path.name}",
                upsert=True,
            )
            public_url = ledger.public_storage_url(
                bucket=settings.supabase_storage_bucket,
                dest_path=f"news/{out_path.name}",
            )
            print(f"Uploaded to Supabase Storage: {storage_uri}")
            print(f"Public URL: {public_url}")
        except Exception as exc:
            print(f"Storage upload failed (create bucket in Supabase): {exc}")

    if to_supabase:
        ledger = SupabaseLedger(settings)
        payload = []
        for item in all_items:
            payload.append(
                {
                    "provider_id": "polygon",
                    "ts": item.get("published_utc"),
                    "title": item.get("title"),
                    "body": item.get("description") or item.get("article_url"),
                    "url": item.get("article_url"),
                    "tickers": item.get("tickers") or [],
                    "meta_json": item,
                }
            )
        try:
            ledger.client.table("news_items").insert(payload).execute()
            print("Inserted into Supabase public.news_items")
        except Exception as exc:
            print(f"Supabase news insert failed: {exc}")


@app.command()
def news_sentiment(
    news_path: Optional[str] = typer.Option(
        None, help="Path to Polygon news JSON (defaults to latest in artifacts/news)"
    ),
    max_items: int = typer.Option(200, help="Max items to score"),
    output_path: Optional[str] = typer.Option(None, help="Output JSON path"),
    upload_storage: bool = typer.Option(False, help="Upload JSON to Supabase Storage"),
) -> None:
    settings = Settings()
    if news_path is None:
        news_dir = Path("artifacts") / "news"
        if not news_dir.exists():
            raise typer.BadParameter("No artifacts/news folder found. Run fetch_news_polygon first.")
        candidates = sorted(news_dir.glob("polygon_news_*.json"))
        if not candidates:
            raise typer.BadParameter("No polygon_news_*.json found. Run fetch_news_polygon first.")
        news_path = str(candidates[-1])

    items = json.loads(Path(news_path).read_text(encoding="utf-8"))
    if not isinstance(items, list) or not items:
        raise typer.BadParameter("News file is empty or invalid JSON.")

    expert = FinBertExpert()
    rows = []
    for item in items[:max_items]:
        title = item.get("title") or ""
        desc = item.get("description") or ""
        text = (title + " " + desc).strip()
        if not text:
            continue
        scores = expert.predict([text])[0]
        probs = expert.to_probs(scores)
        rows.append(
            {
                "ticker": (item.get("tickers") or [""])[0],
                "title": title,
                "published_utc": item.get("published_utc"),
                "scores": scores,
                "probs": probs,
                "url": item.get("article_url") or item.get("url"),
            }
        )

    out = {
        "asof": datetime.now(timezone.utc).isoformat(),
        "source": news_path,
        "rows": rows,
    }
    if output_path is None:
        out_dir = Path("artifacts") / "sentiment"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"news_sentiment_{datetime.now(timezone.utc).date().isoformat()}.json")
    Path(output_path).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote sentiment: {output_path}")

    if upload_storage:
        ledger = SupabaseLedger(settings)
        try:
            storage_uri = ledger.upload_storage(
                bucket=settings.supabase_storage_bucket,
                local_path=str(output_path),
                dest_path=f"sentiment/{Path(output_path).name}",
                upsert=True,
            )
            public_url = ledger.public_storage_url(
                bucket=settings.supabase_storage_bucket,
                dest_path=f"sentiment/{Path(output_path).name}",
            )
            print(f"Uploaded to Supabase Storage: {storage_uri}")
            print(f"Public URL: {public_url}")
        except Exception as exc:
            print(f"Storage upload failed (create bucket in Supabase): {exc}")


@app.command()
def qlib_download_data(
    target_dir: str = typer.Option(..., help="Destination folder for Qlib data"),
    region: str = typer.Option("us", help="Qlib region (us or cn)"),
    interval: Optional[str] = typer.Option(None, help="Interval (e.g., 1d or 1min)"),
    repo_dir: str = typer.Option(".cache/qlib_repo", help="Local clone of Qlib repo"),
) -> None:
    repo_path = Path(repo_dir)
    if not (repo_path / ".git").exists():
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cloning Qlib into {repo_path} ...")
        subprocess.run(
            ["git", "clone", "https://github.com/microsoft/qlib.git", str(repo_path)],
            check=True,
        )

    script = repo_path / "scripts" / "get_data.py"
    if not script.exists():
        raise RuntimeError(f"Qlib get_data.py not found at {script}")

    dest = str(Path(target_dir).expanduser().resolve())
    cmd = [sys.executable, str(script), "qlib_data", "--target_dir", dest, "--region", region]
    if interval:
        cmd += ["--interval", interval]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Qlib data ready at: {dest}")


@app.command()
def massive_ws_capture(
    ws_url: str = typer.Option(..., help="Massive WebSocket URL"),
    auth_json: str = typer.Option(..., help="Auth JSON or @path/to/file.json"),
    subscribe_json: str = typer.Option(..., help="Subscribe JSON or @path/to/file.json"),
    seconds: int = typer.Option(30, help="How long to capture"),
    max_messages: int = typer.Option(1000, help="Max messages to capture"),
    output_path: Optional[str] = typer.Option(None, help="Output JSONL path"),
) -> None:
    try:
        import websockets
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("websockets package required") from exc

    def _load_payload(val: str) -> str:
        if val.startswith("@"):
            return Path(val[1:]).read_text(encoding="utf-8")
        return val

    auth_payload = _load_payload(auth_json)
    sub_payload = _load_payload(subscribe_json)

    if output_path is None:
        out_dir = Path("artifacts") / "massive"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"massive_ws_{datetime.now(timezone.utc).date().isoformat()}.jsonl")

    async def _run() -> None:
        start = time.time()
        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            async with websockets.connect(ws_url) as ws:
                await ws.send(auth_payload)
                await ws.send(sub_payload)
                while True:
                    if time.time() - start >= seconds:
                        break
                    if count >= max_messages:
                        break
                    msg = await ws.recv()
                    f.write(msg if msg.endswith("\n") else msg + "\n")
                    count += 1
        print(f"Wrote {count} messages to {output_path}")

    asyncio.run(_run())


@app.command()
def finviz_screener(
    signal: str = typer.Option("ta_topgainers", help="Finviz preset signal"),
    output_path: Optional[str] = typer.Option(None, help="Output JSON path"),
) -> None:
    try:
        from finvizfinance.screener.overview import Overview
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("finvizfinance required. Install requirements-finviz.txt") from exc

    screener = Overview()
    try:
        df = screener.screener_view(signal=signal)
    except TypeError:
        # finvizfinance versions vary; fall back to default view
        df = screener.screener_view()
    out = df.to_dict(orient="records")

    if output_path is None:
        out_dir = Path("artifacts") / "finviz"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"finviz_{signal}.json")
    Path(output_path).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote: {output_path}")


@app.command()
def qlib_probe(
    data_dir: str = typer.Option(..., help="Qlib data directory (provider_uri)"),
    market: str = typer.Option("csi300", help="Market name for Qlib instruments"),
    start: Optional[str] = typer.Option(None, help="Start date (YYYY-MM-DD)"),
    end: Optional[str] = typer.Option(None, help="End date (YYYY-MM-DD)"),
    output_path: Optional[str] = typer.Option(None, help="Output JSON path"),
) -> None:
    adapter = QlibAdapter(data_dir=data_dir)
    result = adapter.probe_market(market=market, start=start, end=end)
    out = {
        "asof": datetime.now(timezone.utc).isoformat(),
        "data_dir": data_dir,
        "result": result,
    }
    if output_path is None:
        out_dir = Path("artifacts") / "qlib"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"qlib_probe_{datetime.now(timezone.utc).date().isoformat()}.json")
    Path(output_path).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote Qlib probe: {output_path}")


@app.command()
def qlib_train_baseline(
    data_dir: str = typer.Option(..., help="Qlib data directory"),
    market: str = typer.Option("sp500", help="Market name (sp500, nasdaq100, all)"),
    start: Optional[str] = typer.Option(None, help="Start date (YYYY-MM-DD)"),
    end: Optional[str] = typer.Option(None, help="End date (YYYY-MM-DD)"),
) -> None:
    baseline = QlibBaseline(data_dir=data_dir)
    df = baseline.load_features(market=market, start=start, end=end)
    result = baseline.train_linear(df)

    out_dir = Path("artifacts") / "qlib"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "linear_baseline.joblib"
    meta_path = out_dir / f"linear_baseline_{datetime.now(timezone.utc).date().isoformat()}.json"

    from joblib import dump

    dump(result["model"], model_path)
    meta = {
        "asof": datetime.now(timezone.utc).isoformat(),
        "market": market,
        "rows_train": result["rows_train"],
        "rows_test": result["rows_test"],
        "mse": result["mse"],
        "model_path": str(model_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote Qlib baseline model: {model_path}")
    print(f"Wrote Qlib baseline metadata: {meta_path}")


@app.command()
def fetch_sp500_shiller(
    output_path: Optional[str] = typer.Option(None, help="Output CSV path"),
    to_parquet: bool = typer.Option(False, help="Also write Parquet version"),
) -> None:
    """
    Download S&P 500 monthly dataset from datasets/s-and-p-500 (Shiller).
    Note: This is monthly index data (price/dividends/earnings), not OHLCV.
    """
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500/master/data/data.csv"
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Download failed: {resp.status_code}")

    out_dir = Path("artifacts") / "macro"
    out_dir.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = str(out_dir / "sp500_shiller_monthly.csv")
    Path(output_path).write_text(resp.text, encoding="utf-8")
    print(f"Wrote: {output_path}")

    if to_parquet:
        df = pd.read_csv(output_path)
        pq_path = str(out_dir / "sp500_shiller_monthly.parquet")
        df.to_parquet(pq_path, index=False)
        print(f"Wrote: {pq_path}")


@app.command()
def ta_features(
    csv_path: str = typer.Argument(..., help="CSV with columns: timestamp,open,high,low,close,volume"),
    output_path: Optional[str] = typer.Option(None, help="Output CSV path"),
    to_parquet: bool = typer.Option(True, help="Also write Parquet"),
) -> None:
    df = pd.read_csv(csv_path)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise typer.BadParameter(f"Missing columns: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    feats = TAFeatures().compute(df)

    out_dir = Path("artifacts") / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = str(out_dir / "ta_features.csv")
    feats.to_csv(output_path, index=False)
    print(f"Wrote: {output_path}")
    if to_parquet:
        pq_path = str(out_dir / "ta_features.parquet")
        feats.to_parquet(pq_path, index=False)
        print(f"Wrote: {pq_path}")


@app.command()
def regime_hmm(
    csv_path: str = typer.Argument(..., help="CSV with columns: timestamp,close"),
    n_states: int = typer.Option(3, help="Number of regimes"),
    output_path: Optional[str] = typer.Option(None, help="Output JSON path"),
) -> None:
    try:
        from hmmlearn.hmm import GaussianHMM
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("hmmlearn is required. Install requirements-hmm.txt") from exc

    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns or "close" not in df.columns:
        raise typer.BadParameter("CSV must include timestamp and close")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    returns = np.log(df["close"]).diff().dropna()
    vol = returns.rolling(20).std().dropna()
    aligned = pd.concat([returns, vol], axis=1).dropna()
    aligned.columns = ["ret", "vol"]

    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=200)
    model.fit(aligned.values)
    probs = model.predict_proba(aligned.values)
    means = model.means_.tolist() if hasattr(model, "means_") else []
    bull_state = None
    bear_state = None
    if means:
        # means[state] = [ret, vol]
        ret_means = [m[0] for m in means]
        bull_state = int(np.argmax(ret_means))
        bear_state = int(np.argmin(ret_means))
    out = {
        "asof": datetime.now(timezone.utc).isoformat(),
        "states": n_states,
        "timestamps": aligned.index.astype(str).tolist(),
        "probs": probs.tolist(),
        "state_means": means,
        "bull_state": bull_state,
        "bear_state": bear_state,
    }
    if output_path is None:
        out_dir = Path("artifacts") / "regime"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / "regime_probs.json")
    Path(output_path).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote: {output_path}")


@app.command()
def monte_carlo(
    csv_path: str = typer.Argument(..., help="CSV with columns: timestamp,close"),
    symbol: Optional[str] = typer.Option(None, help="Optional ticker symbol"),
    horizon_days: int = typer.Option(10, help="Simulation horizon"),
    sims: int = typer.Option(1000, help="Number of simulations"),
    output_path: Optional[str] = typer.Option(None, help="Output JSON path"),
) -> None:
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns or "close" not in df.columns:
        raise typer.BadParameter("CSV must include timestamp and close")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    close = df["close"].astype(float)
    returns = np.log(close).diff().dropna()
    mu = returns.mean()
    sigma = returns.std()
    s0 = close.iloc[-1]

    dt = 1.0
    rand = np.random.normal(size=(sims, horizon_days))
    paths = s0 * np.exp(np.cumsum((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rand, axis=1))

    out = {
        "asof": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "s0": float(s0),
        "mu": float(mu),
        "sigma": float(sigma),
        "horizon_days": horizon_days,
        "sims": sims,
        "paths_tail": paths[:, -1].tolist(),
    }
    if output_path is None:
        out_dir = Path("artifacts") / "monte_carlo"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / "mc_paths.json")
    Path(output_path).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote: {output_path}")


@app.command()
def direction_event_exec(
    csv_path: str = typer.Argument(..., help="CSV with columns: timestamp,open,high,low,close,volume"),
    zone_low: float = typer.Option(..., help="Zone low"),
    zone_high: float = typer.Option(..., help="Zone high"),
    horizon_bars: int = typer.Option(-1, help="Event horizon in bars (<=0 uses scope default)"),
    scope: str = typer.Option("day", help="Scope: day, swing, long"),
    output_path: Optional[str] = typer.Option(None, help="Output JSON path"),
) -> None:
    df = pd.read_csv(csv_path)
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise typer.BadParameter(f"Missing columns: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")

    scope_cfg = get_scope(scope)
    # override horizon if not explicitly set
    if horizon_bars <= 0:
        horizon_bars = scope_cfg.horizon_bars
    direction = compute_direction(df)
    event = compute_touch_zone_prob(df, zone_low, zone_high, horizon_bars=horizon_bars)
    execution = compute_execution(zone_low, zone_high, direction.bias)

    evidence = {
        "event_prior": Evidence("event_prior", event.p_event, 1.0),
        "direction_bias": Evidence(
            "direction_bias",
            0.6 if direction.bias == "UP" else 0.4,
            0.5,
        ),
    }
    p_final = combine_probabilities(evidence, bias=0.0)

    out = {
        "asof": datetime.now(timezone.utc).isoformat(),
        "scope": scope_cfg.__dict__,
        "direction": direction.__dict__,
        "event": event.__dict__,
        "execution": execution.__dict__,
        "p_final": p_final,
        "label": "EARLY_WATCH_ONLY" if p_final < 0.5 else "PLAN_READY",
    }

    if output_path is None:
        out_dir = Path("artifacts") / "decisions"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / "direction_event_exec.json")
    Path(output_path).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote: {output_path}")


@app.command()
def watchlist_report(
    labels: str = typer.Option(
        "C:\\Users\\Stas\\Downloads\\signals_rows.csv",
        help="Path to signals_rows.csv",
    ),
    tickers: Optional[str] = typer.Option(None, help="Comma-separated tickers"),
    top_n: int = typer.Option(25, help="How many tickers to keep"),
    upload_storage: bool = typer.Option(True, help="Upload outputs to Supabase Storage"),
    to_supabase: bool = typer.Option(True, help="Insert watchlist into Supabase"),
    fetch_polygon_news: bool = typer.Option(False, help="Fetch Polygon news (rate-limited)"),
    use_finviz: bool = typer.Option(False, help="Use Finviz screener in watchlist weights"),
) -> None:
    symbols: str
    if tickers:
        symbols = tickers
    else:
        # Initial rank to derive tickers list
        rank_watchlist(
            labels=labels,
            top_n=top_n,
            output_path=None,
            use_calibration=True,
            upload_storage=upload_storage,
            to_supabase=to_supabase,
        )
        wdir = Path("artifacts") / "watchlist"
        latest_files = sorted(wdir.glob("watchlist_*.json"))
        if not latest_files:
            raise RuntimeError("No watchlist files found. Run rank_watchlist or pass --tickers.")
        latest = latest_files[-1]
        data = json.loads(latest.read_text(encoding="utf-8"))
        symbols = ",".join([r["ticker"] for r in data.get("ranked", []) if r.get("ticker")])
    # Fetch news for those tickers (optional; rate-limited)
    if fetch_polygon_news and os.getenv("DISABLE_POLYGON_NEWS", "").strip().lower() not in {"1", "true", "yes"}:
        fetch_news_polygon(
            tickers=symbols,
            limit=20,
            upload_storage=upload_storage,
            to_supabase=False,
        )
    # Build sentiment file for rank_watchlist enrichment (if any news exists)
    try:
        news_sentiment(
            news_path=None,
            max_items=200,
            output_path=None,
            upload_storage=upload_storage,
        )
    except Exception as exc:
        print(f"News sentiment skipped: {exc}")
    # Re-rank to include news/qlib/ta weights if available
    if not tickers:
        rank_watchlist(
            labels=labels,
            top_n=top_n,
            output_path=None,
            use_calibration=True,
            upload_storage=upload_storage,
            to_supabase=to_supabase,
        )

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

if __name__ == "__main__":
    app()
