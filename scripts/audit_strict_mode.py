#!/usr/bin/env python3
"""Fail CI if strict-mode shortcuts/fallbacks are reintroduced."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


CHECKS = [
    ("vercel.json", ["/api/demo"], "Demo API route must stay removed."),
    (
        "api/walkforward.py",
        ["market_ta_proxy", "build_watchlist("],
        "Walkforward must be Supabase-only with no proxy fallback.",
    ),
    (
        "api/forecasts.py",
        ["def _market_rows", 'source = "market_ta"'],
        "Forecasts must not fallback to market TA.",
    ),
    (
        "ui/dashboard.html",
        ["loadDemoData", "/api/demo"],
        "Legacy dashboard must not call demo endpoint.",
    ),
    (
        "ui/app.js",
        ["demo fallback", 'row.source === "demo"'],
        "Watchlist UI must not include demo source logic.",
    ),
]


def main() -> int:
    failures: list[str] = []
    for rel_path, banned_tokens, reason in CHECKS:
        path = ROOT / rel_path
        if not path.exists():
            failures.append(f"[MISSING] {rel_path} ({reason})")
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in banned_tokens:
            if token in text:
                failures.append(f"[BLOCKED] {rel_path}: found token '{token}' ({reason})")

    if failures:
        print("Strict mode audit failed:\n")
        for item in failures:
            print(f"- {item}")
        return 1

    print("Strict mode audit passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

