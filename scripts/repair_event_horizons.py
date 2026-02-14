#!/usr/bin/env python
"""Repair event horizon_json using learned rule horizons from context_json.

Usage:
  python scripts/repair_event_horizons.py --dry-run
  python scripts/repair_event_horizons.py --apply
  python scripts/repair_event_horizons.py --apply --run-id <uuid>
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from supabase import create_client

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ddl69.utils.signals import infer_horizon_days_from_rules, parse_horizon_days


def _load_env_file(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            txt = line.strip()
            if not txt or txt.startswith("#") or "=" not in txt:
                continue
            key, value = txt.split("=", 1)
            os.environ.setdefault(key, value)


def _chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _fmt_horizon_json(days: int) -> dict[str, Any]:
    return {"type": "time", "unit": "d", "value": int(days)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="", help="Specific run_id to repair (default: latest)")
    parser.add_argument("--apply", action="store_true", help="Apply updates to Supabase")
    parser.add_argument("--batch-size", type=int, default=200)
    args = parser.parse_args()

    _load_env_file(".env")
    _load_env_file(".env.local")

    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not url or not key:
        print("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")
        return 1

    client = create_client(url, key)

    run_id = args.run_id.strip()
    if not run_id:
        latest = (
            client.table("v_latest_ensemble_forecasts")
            .select("run_id,created_at")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
            .data
            or []
        )
        if not latest or not latest[0].get("run_id"):
            print("No latest run_id found from v_latest_ensemble_forecasts")
            return 1
        run_id = str(latest[0]["run_id"])

    print(f"Target run_id: {run_id}")

    forecast_rows = (
        client.table("v_latest_ensemble_forecasts")
        .select("event_id")
        .eq("run_id", run_id)
        .limit(5000)
        .execute()
        .data
        or []
    )
    event_ids = list({r.get("event_id") for r in forecast_rows if r.get("event_id")})
    if not event_ids:
        print("No event_ids found for this run.")
        return 1

    events: list[dict[str, Any]] = []
    for chunk in _chunked(event_ids, max(1, int(args.batch_size))):
        resp = (
            client.table("events")
            .select("event_id,horizon_json,context_json,event_params_json,subject_id")
            .in_("event_id", chunk)
            .execute()
        )
        events.extend(resp.data or [])

    print(f"Loaded {len(events)} events")

    updates: list[tuple[str, dict[str, Any], dict[str, Any], int]] = []
    old_tf = Counter()
    new_tf = Counter()

    for event in events:
        horizon_json = event.get("horizon_json")
        old_days = parse_horizon_days(horizon_json)
        if old_days is None:
            old_days = 5.0
        old_tf["day" if old_days <= 30 else ("swing" if old_days <= 365 else "long")] += 1

        context = event.get("context_json") or {}
        rules = context.get("learned_top_rules") if isinstance(context, dict) else None
        inferred = infer_horizon_days_from_rules(rules)
        if inferred is None or inferred <= 0:
            inferred = old_days

        new_days = max(1, int(round(float(inferred))))
        new_tf["day" if new_days <= 30 else ("swing" if new_days <= 365 else "long")] += 1

        old_days_int = max(1, int(round(float(old_days))))
        if new_days != old_days_int:
            updates.append(
                (
                    str(event["event_id"]),
                    horizon_json if isinstance(horizon_json, dict) else {"value": old_days_int, "unit": "d"},
                    _fmt_horizon_json(new_days),
                    new_days,
                )
            )

    print(f"Old timeframe mix: {dict(old_tf)}")
    print(f"New timeframe mix: {dict(new_tf)}")
    print(f"Events to update: {len(updates)}")

    if updates:
        print("Sample updates:")
        for event_id, old_h, new_h, days in updates[:8]:
            print(f"  {event_id}: {old_h} -> {new_h} (days={days})")

    if not args.apply:
        print("Dry run only. Re-run with --apply to persist updates.")
        return 0

    for event_id, _old_h, new_h, _days in updates:
        client.table("events").update({"horizon_json": new_h}).eq("event_id", event_id).execute()

    print(f"Applied {len(updates)} horizon updates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
