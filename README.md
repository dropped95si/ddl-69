# ddl-69 -- Institutional Probability Ensemble (Clean Repo)

**What this is:** a **probability engine** (not a trading bot).

It answers questions like:
- Stocks: "P(REJECT/BREAK_FAIL/ACCEPT_CONTINUE) at this zone within 5D?"
- Stocks: "P(UPPER/LOWER/NONE) within 20 bars?"
- Weather: "P(temp >= 70F today)?"

**Core constraints (locked):**
- No single tool rules. Everything is an **expert** with a **weight**.
- Weights are **non-zero priors** and must **earn forward** via realized outcomes.
- Probabilities are **calibrated** (70% means 70%).
- Evaluation is leakage-safe (purge/embargo; optional CPCV).

## Two modes

### Mode 1 -- Lean Engine
Runs on a schedule, writes forecasts to Supabase, saves heavy artifacts to Parquet.

### Mode 2 -- Research Runner
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
   - copy `.env.example` -> `.env`
4) Install and run:
   - `pip install -r requirements.txt`
   - `python -m ddl69.cli.main help`

## Open-source tools (optional)
This repo can integrate with open-source research stacks, but they are optional and heavy.

- Check what is installed:
  - `python -m ddl69.cli.main tools_status`
- Download Qlib data (open-source community mirror):
  - `python -m ddl69.cli.main qlib_download --target-dir .qlib/us_data --region us --use-community`
- Download Qlib data via Qlib CLI (requires `qlib` installed):
  - `python -m ddl69.cli.main qlib_download --target-dir .qlib/us_data --region us`
- Verify Qlib init:
  - `python -m ddl69.cli.main qlib_check --qlib-dir .qlib/us_data`
- FinGPT-style sentiment (requires `transformers` + model):
  - `python -m ddl69.cli.main fingpt_sentiment --text "NVDA beat earnings" --model <HF_MODEL_ID>`
- FinGPT dataset scoring (adds `sentiment` column):
  - `python -m ddl69.cli.main fingpt_score_dataset --input-path <file> --model <HF_MODEL_ID>`
- FinRL import check (requires FinRL installed):
  - `python -m ddl69.cli.main finrl_check`
- FinRL Yahoo download (open-source data):
  - `python -m ddl69.cli.main finrl_download --tickers AAPL,SPY --start 2020-01-01`

### Open-source repos
- FinRL: https://github.com/AI4Finance-Foundation/FinRL
- FinGPT: https://github.com/AI4Finance-Foundation/FinGPT
- Qlib: https://github.com/microsoft/qlib
- scikit-learn: https://github.com/scikit-learn/scikit-learn

## Key files
- Supabase ledger: `sql/ledger_v1.sql`, `sql/ledger_v2_patch.sql`
- Normalized ingest tables: `sql/ingest_v1.sql`
- Expert interface: `src/ddl69/core/contracts.py`

