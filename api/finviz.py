import json
import os
from datetime import datetime, timedelta, timezone

import requests
from bs4 import BeautifulSoup  # type: ignore


# Simple Finviz scraper to build a watchlist response for the UI.
# Modes map to Finviz "signal" presets:
# - day   -> top gainers (intraday momentum)
# - swing -> unusual volume (swing candidates)
# - long  -> above 200 SMA (long bias)
SIGNALS = {
    "day": "ta_topgainers",
    "swing": "ta_unusualvolume",
    "long": "ta_sma200_pa",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch_finviz(mode: str, count: int):
    signal = SIGNALS.get(mode, SIGNALS["day"])
    url = f"https://finviz.com/screener.ashx?v=111&s={signal}"
    tickers = []

    page = 1
    while len(tickers) < count:
        paged_url = url + (f"&r={(page-1)*20+1}" if page > 1 else "")
        resp = requests.get(paged_url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.select("a.screener-link-primary"):
            t = (a.text or "").strip().upper()
            if t and t not in tickers:
                tickers.append(t)
            if len(tickers) >= count:
                break
        # stop if no pagination results
        if not soup.select("a.screener-link-primary"):
            break
        page += 1

    return tickers[:count]


def to_watchlist(tickers, mode):
    # default heuristic targets based on timeframe
    if mode == "day":
        horizon_days = 2
        tps_pct = [0.015, 0.03, 0.05]
        sls_pct = [-0.01, -0.02, -0.03]
        p_up = 0.58
    elif mode == "long":
        horizon_days = 45
        tps_pct = [0.10, 0.20, 0.35]
        sls_pct = [-0.05, -0.08, -0.12]
        p_up = 0.64
    else:  # swing
        horizon_days = 10
        tps_pct = [0.04, 0.08, 0.12]
        sls_pct = [-0.02, -0.04, -0.06]
        p_up = 0.60

    eta = (datetime.now(timezone.utc) + timedelta(days=horizon_days)).isoformat()
    base_price = 100.0  # placeholder reference to express TP/SL as dollars

    rows = []
    for t in tickers:
        tp_vals = [round(base_price * (1 + p), 2) for p in tps_pct]
        sl_vals = [round(base_price * (1 + p), 2) for p in sls_pct]
        rows.append(
            {
                "ticker": t,
                "score": p_up,
                "p_accept": p_up,
                "p_reject": 1 - p_up,
                "p_continue": 0,
                "plan_type": mode,
                "label": f"{mode.upper()}_AUTO",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "weights": {},
                "meta": {
                    "source": "finviz",
                    "mode": mode,
                    "p_up": p_up,
                    "p_down": 1 - p_up,
                    "eta": eta,
                    "eta_up": eta,
                    "eta_down": eta,
                    "tp1": tp_vals[0],
                    "tp2": tp_vals[1],
                    "tp3": tp_vals[2],
                    "sl1": sl_vals[0],
                    "sl2": sl_vals[1],
                    "sl3": sl_vals[2],
                    "reason": f"Finviz {mode} screen: momentum/volume filter with heuristic TP/SL bands and horizon {horizon_days}d",
                },
            }
        )
    return {
        "asof": datetime.now(timezone.utc).isoformat(),
        "source": f"finviz:{mode}",
        "count": len(rows),
        "rows": rows,
    }


def handler(request):
    mode = (request.args.get("mode") if hasattr(request, "args") else None) or "swing"
    count_raw = (request.args.get("count") if hasattr(request, "args") else None) or os.getenv("FINVIZ_COUNT", "100")
    try:
        count = max(1, min(200, int(count_raw)))
    except Exception:
        count = 100

    try:
        tickers = fetch_finviz(mode, count)
        if not tickers:
            raise RuntimeError("Empty list from Finviz")
        body = to_watchlist(tickers, mode)
    except Exception:
        # fallback minimal list if Finviz blocks us
        fallback = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"][:count]
        body = to_watchlist(fallback, mode)

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json", "Cache-Control": "no-cache"},
        "body": json.dumps(body),
    }
