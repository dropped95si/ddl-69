# DDL-69 Strategy v6.2

## Core Layers (industry standard)

### 1) Direction Engine (HTF bias)
- **Goal:** determine dominant directional intent (UP / DOWN / NEUTRAL).
- **Timeframes:** Daily + 4H only.
- **Inputs:** value location (VWAP/VP), acceptance vs value high/low, volatility regime.
- **Output:** direction_bias, confidence, invalidation rule.

### 2) Event Engine (probability over horizon)
- **Goal:** probability of a specific event within a horizon, not buy/sell.
- **Timeframes:** 4H / 1H / 30m.
- **Examples:**
  - P(TOUCH_ZONE in 5 bars)
  - P(REJECT_ZONE in 10 bars)
  - P(BREAK_CONTINUE in 20 bars)
  - P(VOL_SPIKE in 10 bars)

### 3) Execution Engine (risk placement)
- **Goal:** express bias with minimal risk, no direction logic.
- **Timeframes:** 15m / 5m / 3m.

## Probability Stack (sources)
- **TA rules:** RSI, SMA/EMA crosses, ATR expansion, vol spike.
- **Qlib factors:** optional factor momentum / regime features.
- **News / Social:** sentiment-weighted signals.
- **Historical calibration:** walk-forward weights + calibration maps.

## Aggregation (evidence-based)
- Convert each expert probability to log-odds.
- Weighted sum of evidence.
- Convert back to probability.

## Outputs
- **watchlist_rankings** with:
  - p_accept
  - score
  - label (EARLY_WATCH_ONLY / WATCH / PLAN_READY)
  - weights_json

## Notes
- Below 50% is a valid state: watch-only / value zone.
- All outcomes are event-specific and horizon-specific.
Appendix — Data Coverage

- News data loaded: artifacts/news/polygon_news_2026-02-08.json
- Sentiment field not present in Polygon news payload; NEWS_* rules not generated.
- Qlib data dir used: C:\Users\Stas\Downloads\ddl-69_v6_1_clean_20260208_054601\ddl-69\data\qlib
- QLIB_MOM_20 rule appears only if qlib instruments match tickers.
