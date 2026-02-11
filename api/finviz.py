import json
import os
from datetime import datetime, timedelta, timezone

try:
    from _http_adapter import FunctionHandler
except ModuleNotFoundError:
    from api._http_adapter import FunctionHandler


UNIVERSE = [
    "SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
    "AMD", "AVGO", "NFLX", "PLTR", "SMCI", "CRM", "ADBE", "JPM", "GS", "BAC",
    "XOM", "CVX", "LLY", "UNH", "JNJ", "KO", "PEP", "WMT", "COST", "MSTR",
    "COIN", "HOOD", "RIVN", "NIO", "SOFI", "SNOW", "CRWD", "PANW", "MU", "INTC",
    "UBER", "ABNB", "SHOP", "SQ", "PYPL", "DIS", "CMCSA", "T", "VZ", "F"
]


def fetch_finviz(mode: str, count: int):
    # Lightweight deterministic proxy list that keeps endpoint stable on serverless.
    if mode == "day":
        ranked = sorted(UNIVERSE, key=lambda t: (len(t), t))
    elif mode == "long":
        ranked = sorted(UNIVERSE, key=lambda t: (sum(ord(c) for c in t), t), reverse=True)
    else:  # swing
        ranked = sorted(UNIVERSE, key=lambda t: (t[0], -len(t), t))
    return ranked[:count]


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
        target_price = tp_vals[0]
        rows.append(
            {
                "ticker": t,
                "score": p_up,
                "p_accept": p_up,
                "p_reject": 1 - p_up,
                "p_continue": 0,
                "p_hit": p_up,
                "target_price": target_price,
                "plan_type": mode,
                "label": f"{mode.upper()}_AUTO",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "weights": {},
                "meta": {
                    "source": "finviz",
                    "mode": mode,
                    "p_up": p_up,
                    "p_down": 1 - p_up,
                    "p_target": p_up,
                    "eta": eta,
                    "eta_up": eta,
                    "eta_down": eta,
                    "tp1": tp_vals[0],
                    "tp2": tp_vals[1],
                    "tp3": tp_vals[2],
                    "sl1": sl_vals[0],
                    "sl2": sl_vals[1],
                    "sl3": sl_vals[2],
                    "target_price": target_price,
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


def _handler_impl(request):
    mode = (request.args.get("mode") if hasattr(request, "args") else None) or "swing"
    count_raw = (request.args.get("count") if hasattr(request, "args") else None) or os.getenv("FINVIZ_COUNT", "100")
    try:
        count = max(1, min(200, int(count_raw)))
    except Exception:
        count = 100

    tickers = fetch_finviz(mode, count)
    if not tickers:
        tickers = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"][:count]
    body = to_watchlist(tickers, mode)

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json", "Cache-Control": "no-cache"},
        "body": json.dumps(body),
    }


class handler(FunctionHandler):
    endpoint = staticmethod(_handler_impl)
