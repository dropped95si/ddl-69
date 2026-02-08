from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich import print

from ddl69.core.settings import Settings
from ddl69.ledger.supabase_ledger import SupabaseLedger
from ddl69.data.parquet_store import ParquetStore
from ddl69.data.cleaner import clean_dataset, load_dataframe, save_dataframe
from ddl69.utils.signals import (
    blend_probs,
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
