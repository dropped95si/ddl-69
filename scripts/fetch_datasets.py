#!/usr/bin/env python3
"""
Download training datasets from Supabase Storage for Walk-Forward retraining.

Required env:
  SUPABASE_URL
  SUPABASE_SERVICE_ROLE_KEY
  SUPABASE_STORAGE_BUCKET

Object keys (set as envs so bucket structure is configurable):
  DATASET_BARS_OBJECT          default: latest/bars.parquet
  DATASET_SIGNALS_OBJECT       default: latest/signals_rows.parquet
  DATASET_REGISTRY_OBJECT      default: latest/signal_registry.json

Gate:
  REQUIRE_WALK_FORWARD=true -> fail hard if any required dataset is missing.
  
HARDENING:
  - Enforces schema invariants (symbol, ts, ohlcv)
  - Merges bar files if multiple are found
  - Sorts by symbol/timestamp (dedupes overlapping)
  - Checks minimum row counts
"""
from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import pandas as pd
except ImportError:
    print("❌ Failed to import pandas. Ensure it is installed.")
    sys.exit(1)

try:
    from supabase import create_client
except ImportError:
    print("❌ Failed to import supabase. Ensure requirements.txt is installed.")
    sys.exit(1)


def truthy(v: Optional[str]) -> bool:
    if v is None:
        return False
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def write_github_output(name: str, value: str) -> None:
    gh_out = os.getenv("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a", encoding="utf-8") as f:
            f.write(f"{name}={value}\n")
    # Also print for local debugging
    print(f"[fetch_datasets] output {name}={value}")


def download_object(sb, bucket: str, object_key: str, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        resp = sb.storage.from_(bucket).download(object_key)
        # supabase-py returns bytes for download()
        with open(out_path, 'wb') as f:
            f.write(resp)
        print(f"[fetch_datasets] downloaded {object_key} -> {out_path}")
        return True
    except Exception as e:
        print(f"[fetch_datasets] WARN: failed to download {bucket}/{object_key}: {e}", file=sys.stderr)
        return False
        
def fetch_and_merge_bars(sb, bucket: str, local_dir: Path) -> Optional[Path]:
    """
    Scans 'bars/' folder in bucket, downloads all parquet files, merges, sorts, validates.
    Returns path to merged parquet if successful, None otherwise.
    """
    bars_staging = local_dir / "bars_staging"
    bars_staging.mkdir(parents=True, exist_ok=True)
    
    dfs = []
    
    try:
        # List files in 'bars/' prefix
        res = sb.storage.from_(bucket).list("bars")
        # Ensure res is a list of dicts or objects we can iterate clearly
        files = [x for x in res if x['name'].endswith('.parquet')]
        
        if not files:
            print(f"[fetch_datasets] WARN: No parquet files found in {bucket}/bars/")
            return None
            
        print(f"[fetch_datasets] Found {len(files)} bar files in {bucket}/bars/")
        
        for item in files:
            remote_path = f"bars/{item['name']}"
            local_file = bars_staging / item['name']
            
            print(f"[fetch_datasets] Downloading {remote_path}...")
            file_bytes = sb.storage.from_(bucket).download(remote_path)
            with open(local_file, 'wb+') as f:
                f.write(file_bytes)
            
            try:
                df = pd.read_parquet(local_file)
                # Normalize symbol column (some use 'ticker', some 'symbol')
                if 'ticker' in df.columns:
                    df.rename(columns={'ticker': 'symbol'}, inplace=True)
                
                # Normalize timestamp
                if 'ts' in df.columns:
                    df.rename(columns={'ts': 'timestamp'}, inplace=True)
                    
                # Schema validation (Invariant A)
                required = {'symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume'}
                missing = required - set(df.columns)
                if missing:
                     print(f"⚠️ Skipping {item['name']}: Missing columns {missing}")
                     continue
                     
                dfs.append(df)
            except Exception as e:
                print(f"[fetch_datasets] WARN: Failed to read {local_file}: {e}")
        
    except Exception as e:
        print(f"[fetch_datasets] ERROR listing/downloading bars: {e}", file=sys.stderr)
        return None

    if not dfs:
        return None
        
    # Merge
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Invariant B: Deduplicate (prefer latest)
    before_dedupe = len(full_df)
    full_df.drop_duplicates(subset=['symbol', 'timestamp'], keep='last', inplace=True)
    deduped = before_dedupe - len(full_df)
    if deduped > 0:
        print(f"[fetch_datasets] Dropped {deduped} duplicate rows")

    # Invariant C: Sort
    full_df['timestamp'] = pd.to_datetime(full_df['timestamp'])
    full_df.sort_values(by=['symbol', 'timestamp'], inplace=True)
    
    # Invariant D: Minimum Data Logic
    MIN_ROWS = 200
    counts = full_df['symbol'].value_counts()
    valid_symbols = counts[counts >= MIN_ROWS].index
    dropped_symbols = counts[counts < MIN_ROWS].index

    if len(dropped_symbols) > 0:
        print(f"[fetch_datasets] Dropping {len(dropped_symbols)} symbols with < {MIN_ROWS} rows: {list(dropped_symbols)}")
        full_df = full_df[full_df['symbol'].isin(valid_symbols)]
        
    unique_symbols = full_df['symbol'].nunique()
    total_rows = len(full_df)
    print(f"[fetch_datasets] Aggregated: {total_rows} rows across {unique_symbols} symbols")
    
    if unique_symbols < 1:
        print("[fetch_datasets] ERROR: No symbols found in data", file=sys.stderr)
        return None

    merged_path = local_dir / "bars.parquet"
    full_df.to_parquet(merged_path)
    return merged_path


def main() -> int:
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    bucket = os.getenv("SUPABASE_STORAGE_BUCKET") or "artifacts"

    require = truthy(os.getenv("REQUIRE_WALK_FORWARD"))

    if not supabase_url or not supabase_key:
        msg = "[fetch_datasets] ERROR: missing SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY"
        print(msg, file=sys.stderr)
        return 1 if require else 0

    sb = create_client(supabase_url, supabase_key)
    
    # Where to put data on the runner
    data_dir = Path(os.getenv("DATA_DIR", "data")).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fetch Bars (with invariants)
    bars_path = fetch_and_merge_bars(sb, bucket, data_dir)
    
    # 2. Fetch Signals (Pre-computed features)
    signals_key = os.getenv("DATASET_SIGNALS_OBJECT", "latest/signals_rows.parquet")
    signals_path = data_dir / "signals_rows.parquet"
    ok_signals = download_object(sb, bucket, signals_key, signals_path)
    
    # 3. Registry
    registry_key = os.getenv("DATASET_REGISTRY_OBJECT", "latest/signal_registry.json")
    registry_path = data_dir / "signal_registry.json"
    ok_registry = download_object(sb, bucket, registry_key, registry_path)

    # Outputs
    if bars_path and bars_path.exists():
        write_github_output("BARS_PATH", str(bars_path))
    if ok_signals and signals_path.exists():
        write_github_output("SIGNALS_PATH", str(signals_path))
    if ok_registry and registry_path.exists():
        write_github_output("SIGNAL_REGISTRY_PATH", str(registry_path))

    # Gate
    missing = []
    if not bars_path:
        missing.append("bars")
    # if not ok_signals: missing.append("signals")
    # if not ok_registry: missing.append("registry")

    if missing:
        msg = f"[fetch_datasets] {'ERROR' if require else 'WARN'}: missing datasets: {', '.join(missing)}"
        print(msg, file=sys.stderr)
        return 1 if require else 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
