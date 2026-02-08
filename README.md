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

## Vercel UI (static)
Static UI is in `public/`. Deploy the repo to Vercel as a static site.
Paste the public Supabase Storage URLs (watchlist + news JSON) into the UI.

## Optional NLP expert (FinBERT)
Install `requirements-nlp.txt` and run:
`python -m ddl69.cli.main news-sentiment --upload-storage`

## Optional Qlib adapter
Install `requirements-qlib.txt` (uses `pyqlib`), prepare a Qlib dataset, then:
`python -m ddl69.cli.main qlib-probe --data-dir /path/to/qlib_data --market csi300`

To download Qlib data:
`python -m ddl69.cli.main qlib-download-data --target-dir ./data/qlib --region us`

Baseline training:
`python -m ddl69.cli.main qlib-train-baseline --data-dir ./data/qlib --market sp500`

## SP500 monthly (Shiller) dataset
`python -m ddl69.cli.main fetch-sp500-shiller --to-parquet`

## TA + Regime + Monte Carlo
TA features:
`python -m ddl69.cli.main ta-features ./ohlcv_60days.csv`

Regime HMM (requires `requirements-hmm.txt`):
`python -m ddl69.cli.main regime-hmm ./ohlcv_60days.csv`

Monte Carlo:
`python -m ddl69.cli.main monte-carlo ./ohlcv_60days.csv`

## Direction/Event/Execution (industry standard stack)
`python -m ddl69.cli.main direction-event-exec ./ohlcv_60days.csv --zone-low 100 --zone-high 105 --horizon-bars 5`

## Finviz (optional)
`pip install -r requirements-finviz.txt`
`python -m ddl69.cli.main finviz-screener --signal ta_topgainers`

## Massive WebSocket capture
Capture raw messages from Massive WebSocket (auth + subscribe JSON required):
`python -m ddl69.cli.main massive-ws-capture --ws-url wss://... --auth-json @auth.json --subscribe-json @sub.json`
