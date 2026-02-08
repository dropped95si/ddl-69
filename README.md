# ddl-69 — Institutional Probability Ensemble (Clean Repo)

**What this is:** a **probability engine** (not a trading bot).

It answers questions like:
- Stocks: “P(REJECT/BREAK_FAIL/ACCEPT_CONTINUE) at this zone within 5D?”
- Stocks: “P(UPPER/LOWER/NONE) within 20 bars?”
- Weather: “P(temp ≥ 70°F today)?”

**Core constraints (locked):**
- No single tool rules. Everything is an **expert** with a **weight**.
- Weights are **non-zero priors** and must **earn forward** via realized outcomes.
- Probabilities are **calibrated** (70% means 70%).
- Evaluation is leakage-safe (purge/embargo; optional CPCV).

## Two modes

### Mode 1 — Lean Engine
Runs on a schedule, writes forecasts to Supabase, saves heavy artifacts to Parquet.

### Mode 2 — Research Runner
Uses Qlib as experiment harness and writes results back to the same Supabase ledger.

## Storage rule
- **Supabase** stores ledger facts: runs/events/forecasts/outcomes/weights.
- **Parquet** stores heavy matrices + fold outputs. Supabase stores pointers/checksums.

## Quickstart
1) Create a new Supabase project.
2) Run SQL in this order:
   - `sql/ledger_v1.sql`
   - `sql/ledger_v2_patch.sql`
   - `sql/ingest_v1.sql`
3) Set environment:
   - copy `.env.example` → `.env`
4) Install and run:
   - `pip install -r requirements.txt`
   - `python -m ddl69.cli.main help`

## Key files
- Supabase ledger: `sql/ledger_v1.sql`, `sql/ledger_v2_patch.sql`
- Normalized ingest tables: `sql/ingest_v1.sql`
- Expert interface: `src/ddl69/core/contracts.py`

