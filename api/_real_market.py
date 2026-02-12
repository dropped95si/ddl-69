"""Real market data + technical analysis helpers for serverless APIs.

All outputs here are derived from Yahoo Finance screeners/spark history,
never from deterministic/sample generators.
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timedelta, timezone

import requests

YAHOO_HEADERS = {"User-Agent": "Mozilla/5.0"}
_CACHE_TTL_SEC = 120
_SCREEN_CACHE = {}
_HISTORY_CACHE = {}
_CORE_UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA", "VTI", "XLF", "XLK", "XLE", "XLV", "XLI",
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD", "NFLX",
    "PLTR", "SMCI", "CRM", "ADBE", "ORCL", "INTC", "MU", "QCOM", "IBM", "CSCO",
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "V", "MA", "PYPL",
    "XOM", "CVX", "COP", "SLB", "OXY", "HAL", "MPC", "VLO", "PSX", "KMI",
    "UNH", "LLY", "JNJ", "PFE", "MRK", "ABBV", "TMO", "DHR", "ABT", "BMY",
    "WMT", "COST", "TGT", "HD", "LOW", "NKE", "MCD", "SBUX", "DIS", "CMCSA",
    "KO", "PEP", "PG", "CL", "MDLZ", "PM", "MO", "KHC", "GIS", "HSY",
    "UBER", "ABNB", "SHOP", "SQ", "ROKU", "SNOW", "CRWD", "PANW", "ZS", "DDOG",
    "COIN", "HOOD", "RIVN", "NIO", "SOFI", "LCID", "DKNG", "PLUG", "F", "GM",
    "BA", "GE", "CAT", "DE", "HON", "RTX", "LMT", "NOC", "MMM", "ETN",
    "T", "VZ", "TMUS", "CHTR", "PARA", "WBD", "EA", "TTWO", "SONY", "U",
]


def _now_ts() -> float:
    return time.time()


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _safe_float(value, default=None):
    try:
        if value is None:
            return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def _chunked(values, size: int):
    for i in range(0, len(values), size):
        yield values[i : i + size]


def _valid_symbol(symbol: str) -> bool:
    if not symbol:
        return False
    s = symbol.strip().upper()
    if len(s) > 8:
        return False
    if not all(ch.isalnum() or ch in ".-" for ch in s):
        return False
    return True


def fetch_screener_symbols(scr_id: str, count: int = 50) -> list[str]:
    """Fetch symbols from Yahoo predefined screeners."""
    count = int(_clamp(count, 1, 300))
    cache_key = (scr_id, count)
    now = _now_ts()
    cached = _SCREEN_CACHE.get(cache_key)
    if cached and (now - cached["ts"]) < _CACHE_TTL_SEC:
        return list(cached["symbols"])

    url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
    symbols = []

    seen = set()
    page_size = min(25, count)
    start = 0
    while len(symbols) < count and start < 500:
        params = {"scrIds": scr_id, "count": page_size, "start": start}
        try:
            resp = requests.get(url, params=params, headers=YAHOO_HEADERS, timeout=10)
            if resp.status_code != 200:
                break
            payload = resp.json()
            quotes = (
                payload.get("finance", {})
                .get("result", [{}])[0]
                .get("quotes", [])
            )
            if not quotes:
                break
            for quote in quotes:
                sym = str(quote.get("symbol") or "").upper().strip()
                if not _valid_symbol(sym):
                    continue
                if sym in seen:
                    continue
                seen.add(sym)
                symbols.append(sym)
                if len(symbols) >= count:
                    break
            if len(quotes) < page_size:
                break
            start += page_size
        except Exception:
            break

    _SCREEN_CACHE[cache_key] = {"ts": now, "symbols": symbols}
    return symbols


def build_symbol_universe(mode: str, count: int = 100) -> list[str]:
    """Build a real ticker universe by strategy mode."""
    count = int(_clamp(count, 5, 300))
    mode = (mode or "swing").lower()
    if mode not in ("day", "swing", "long"):
        mode = "swing"

    if mode == "day":
        plan = [("day_gainers", count), ("most_actives", count)]
    elif mode == "long":
        plan = [("most_actives", count), ("day_gainers", max(25, count // 2))]
    else:
        plan = [
            ("most_actives", count),
            ("day_gainers", max(20, count // 2)),
            ("day_losers", max(20, count // 2)),
        ]

    universe = []
    seen = set()

    # Seed with liquid core names first to guarantee broad, stable coverage.
    seed_count = max(20, count // 2)
    for sym in _CORE_UNIVERSE[:seed_count]:
        if sym in seen:
            continue
        seen.add(sym)
        universe.append(sym)
        if len(universe) >= count:
            return universe[:count]

    for scr_id, scr_count in plan:
        for sym in fetch_screener_symbols(scr_id, scr_count):
            if sym in seen:
                continue
            seen.add(sym)
            universe.append(sym)
            if len(universe) >= count:
                return universe

    for sym in _CORE_UNIVERSE:
        if sym in seen:
            continue
        seen.add(sym)
        universe.append(sym)
        if len(universe) >= count:
            break

    return universe[:count]


def fetch_histories(symbols: list[str], range_: str = "6mo", interval: str = "1d") -> dict:
    """Fetch OHLCV histories with Yahoo spark endpoint in batches."""
    out = {}
    now = _now_ts()
    symbols = [s.strip().upper() for s in symbols if _valid_symbol(s)]
    symbols = list(dict.fromkeys(symbols))

    missing = []
    for sym in symbols:
        cache_key = (sym, range_, interval)
        cached = _HISTORY_CACHE.get(cache_key)
        if cached and (now - cached["ts"]) < _CACHE_TTL_SEC:
            out[sym] = list(cached["bars"])
        else:
            missing.append(sym)

    for chunk in _chunked(missing, 20):
        try:
            resp = requests.get(
                "https://query1.finance.yahoo.com/v7/finance/spark",
                params={
                    "symbols": ",".join(chunk),
                    "range": range_,
                    "interval": interval,
                },
                headers=YAHOO_HEADERS,
                timeout=12,
            )
            if resp.status_code != 200:
                continue
            payload = resp.json()
            results = payload.get("spark", {}).get("result", [])
        except Exception:
            results = []

        for item in results:
            sym = str(item.get("symbol") or "").upper()
            response = (item.get("response") or [{}])[0]
            timestamps = response.get("timestamp", []) or []
            quote = response.get("indicators", {}).get("quote", [{}])[0]
            opens = quote.get("open", []) or []
            highs = quote.get("high", []) or []
            lows = quote.get("low", []) or []
            closes = quote.get("close", []) or []
            volumes = quote.get("volume", []) or []

            bars = []
            n = len(timestamps)
            for i in range(n):
                ts = _safe_float(timestamps[i], None)
                close = _safe_float(closes[i] if i < len(closes) else None, None)
                if ts is None or close is None:
                    continue
                open_ = _safe_float(opens[i] if i < len(opens) else None, close)
                high = _safe_float(highs[i] if i < len(highs) else None, max(open_, close))
                low = _safe_float(lows[i] if i < len(lows) else None, min(open_, close))
                vol = _safe_float(volumes[i] if i < len(volumes) else None, 0.0)
                bars.append(
                    {
                        "time": int(ts),
                        "open": float(open_),
                        "high": float(high),
                        "low": float(low),
                        "close": float(close),
                        "volume": float(vol or 0.0),
                    }
                )

            if bars:
                bars.sort(key=lambda b: b["time"])
                out[sym] = bars
                _HISTORY_CACHE[(sym, range_, interval)] = {"ts": now, "bars": bars}

    return out


def _ema_series(values: list[float], period: int) -> list[float]:
    if not values:
        return []
    alpha = 2.0 / (period + 1.0)
    ema = values[0]
    out = [ema]
    for v in values[1:]:
        ema = alpha * v + (1.0 - alpha) * ema
        out.append(ema)
    return out


def _sma_last(values: list[float], period: int):
    if not values:
        return None
    if len(values) < period:
        return sum(values) / len(values)
    return sum(values[-period:]) / period


def _rsi_last(values: list[float], period: int = 14) -> float:
    if len(values) < period + 1:
        return 50.0
    gains = []
    losses = []
    for i in range(1, period + 1):
        delta = values[i] - values[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    for i in range(period + 1, len(values)):
        delta = values[i] - values[i - 1]
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _atr_last(highs: list[float], lows: list[float], closes: list[float], period: int = 14):
    if len(closes) < period + 1:
        return None
    trs = []
    for i in range(1, len(closes)):
        high = highs[i]
        low = lows[i]
        prev_close = closes[i - 1]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    if not trs:
        return None
    if len(trs) < period:
        return sum(trs) / len(trs)
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = ((atr * (period - 1)) + tr) / period
    return atr


def _macd_last(values: list[float]):
    if len(values) < 35:
        return 0.0, 0.0, 0.0
    ema12 = _ema_series(values, 12)
    ema26 = _ema_series(values, 26)
    macd = [a - b for a, b in zip(ema12, ema26)]
    signal = _ema_series(macd, 9)
    macd_last = macd[-1]
    signal_last = signal[-1] if signal else 0.0
    return macd_last, signal_last, macd_last - signal_last


def compute_metrics_from_bars(bars: list[dict]):
    if not bars or len(bars) < 22:
        return None

    closes = [b["close"] for b in bars if _safe_float(b.get("close"), None) is not None]
    highs = [b["high"] for b in bars if _safe_float(b.get("high"), None) is not None]
    lows = [b["low"] for b in bars if _safe_float(b.get("low"), None) is not None]
    vols = [max(0.0, _safe_float(b.get("volume"), 0.0)) for b in bars]
    if len(closes) < 22 or len(highs) != len(closes) or len(lows) != len(closes):
        return None

    price = closes[-1]
    prev = closes[-2] if len(closes) > 1 else price
    c5 = closes[-6] if len(closes) > 6 else closes[0]
    c20 = closes[-21] if len(closes) > 21 else closes[0]

    change_1d = (price / prev - 1.0) if prev else 0.0
    change_5d = (price / c5 - 1.0) if c5 else 0.0
    change_20d = (price / c20 - 1.0) if c20 else 0.0

    sma20 = _sma_last(closes, 20) or price
    sma50 = _sma_last(closes, 50) or sma20
    ema12 = _ema_series(closes, 12)[-1]
    ema26 = _ema_series(closes, 26)[-1]
    macd_line, macd_signal, macd_hist = _macd_last(closes)
    rsi = _rsi_last(closes, 14)
    atr = _atr_last(highs, lows, closes, 14) or (price * 0.02)
    atr_pct = max(0.003, atr / price) if price else 0.02

    vol_last = vols[-1] if vols else 0.0
    vol_avg20 = (sum(vols[-20:]) / min(len(vols), 20)) if vols else 0.0
    vol_ratio = (vol_last / vol_avg20) if vol_avg20 > 0 else 1.0

    window = bars[-30:]
    support = min(b["low"] for b in window) if window else price * 0.98
    resistance = max(b["high"] for b in window) if window else price * 1.02

    return {
        "price": float(price),
        "change_1d": float(change_1d),
        "change_5d": float(change_5d),
        "change_20d": float(change_20d),
        "sma20": float(sma20),
        "sma50": float(sma50),
        "ema12": float(ema12),
        "ema26": float(ema26),
        "macd_line": float(macd_line),
        "macd_signal": float(macd_signal),
        "macd_hist": float(macd_hist),
        "rsi": float(rsi),
        "atr": float(atr),
        "atr_pct": float(atr_pct),
        "vol_ratio": float(vol_ratio),
        "support": float(support),
        "resistance": float(resistance),
    }


def score_metrics(metrics: dict, mode: str = "swing") -> dict:
    mode = (mode or "swing").lower()
    if mode not in ("day", "swing", "long"):
        mode = "swing"

    if mode == "day":
        weights = {"trend": 0.2, "momentum": 0.35, "rsi": 0.2, "volume": 0.25}
    elif mode == "long":
        weights = {"trend": 0.45, "momentum": 0.25, "rsi": 0.15, "volume": 0.15}
    else:
        weights = {"trend": 0.35, "momentum": 0.3, "rsi": 0.2, "volume": 0.15}

    price = metrics["price"]
    sma20 = metrics["sma20"] or price
    sma50 = metrics["sma50"] or sma20
    trend_price = _clamp((price / sma20 - 1.0) / 0.08, -1.0, 1.0)
    trend_sma = _clamp((sma20 / sma50 - 1.0) / 0.06, -1.0, 1.0)
    trend_mom = _clamp(metrics["change_20d"] / 0.18, -1.0, 1.0)
    trend = _clamp((0.35 * trend_price) + (0.35 * trend_sma) + (0.30 * trend_mom), -1.0, 1.0)

    momentum = _clamp(
        metrics["macd_hist"] / max(metrics["atr"] * 1.2, price * 0.006),
        -1.0,
        1.0,
    )
    rsi_component = _clamp((metrics["rsi"] - 50.0) / 30.0, -1.0, 1.0)
    volume_component = _clamp((metrics["vol_ratio"] - 1.0) / 2.0, -1.0, 1.0)

    c_trend = trend * weights["trend"]
    c_momentum = momentum * weights["momentum"]
    c_rsi = rsi_component * weights["rsi"]
    c_volume = volume_component * weights["volume"]
    raw = c_trend + c_momentum + c_rsi + c_volume

    p_up = _clamp(_sigmoid(raw * 1.7), 0.06, 0.94)
    p_down = 1.0 - p_up
    score = max(p_up, p_down)

    if p_up >= 0.58:
        signal = "BUY"
    elif p_up <= 0.42:
        signal = "SELL"
    else:
        signal = "HOLD"

    confidence = _clamp(
        0.45 + abs(raw) * 0.45 + max(0.0, volume_component) * 0.08,
        0.35,
        0.95,
    )

    return {
        "signal": signal,
        "score": float(score),
        "p_up": float(p_up),
        "p_down": float(p_down),
        "confidence": float(confidence),
        "weights": weights,
        "contrib": {
            "trend": float(c_trend),
            "momentum": float(c_momentum),
            "rsi": float(c_rsi),
            "volume": float(c_volume),
        },
    }


def target_profile(mode: str, atr_pct: float):
    mode = (mode or "swing").lower()
    if mode not in ("day", "swing", "long"):
        mode = "swing"

    if mode == "day":
        horizon_days = 2
        tp_mult = [0.8, 1.4, 2.0]
        sl_mult = [0.7, 1.2, 1.8]
        tp_bounds = [(0.007, 0.03), (0.012, 0.05), (0.018, 0.08)]
        sl_bounds = [(0.006, 0.025), (0.01, 0.04), (0.015, 0.06)]
    elif mode == "long":
        horizon_days = 45
        tp_mult = [3.0, 5.0, 8.0]
        sl_mult = [1.8, 2.8, 3.8]
        tp_bounds = [(0.05, 0.20), (0.10, 0.30), (0.18, 0.45)]
        sl_bounds = [(0.03, 0.12), (0.05, 0.18), (0.08, 0.25)]
    else:
        horizon_days = 10
        tp_mult = [1.5, 2.5, 4.0]
        sl_mult = [1.0, 1.6, 2.2]
        tp_bounds = [(0.02, 0.08), (0.04, 0.12), (0.06, 0.18)]
        sl_bounds = [(0.015, 0.06), (0.03, 0.09), (0.04, 0.12)]

    tp_pct = [
        _clamp(atr_pct * tp_mult[i], tp_bounds[i][0], tp_bounds[i][1]) for i in range(3)
    ]
    sl_pct = [
        -_clamp(atr_pct * sl_mult[i], sl_bounds[i][0], sl_bounds[i][1]) for i in range(3)
    ]
    return {"horizon_days": horizon_days, "tp_pct": tp_pct, "sl_pct": sl_pct}


def build_watchlist(mode: str = "swing", count: int = 100) -> list[dict]:
    """Build real TA-based ranked watchlist rows."""
    mode = (mode or "swing").lower()
    if mode not in ("day", "swing", "long"):
        mode = "swing"
    count = int(_clamp(count, 1, 300))

    universe = build_symbol_universe(mode, max(30, count * 3))
    histories = fetch_histories(universe, range_="6mo", interval="1d")
    now = datetime.now(timezone.utc)
    rows = []

    for symbol in universe:
        bars = histories.get(symbol) or []
        metrics = compute_metrics_from_bars(bars)
        if not metrics:
            continue

        scored = score_metrics(metrics, mode=mode)
        profile = target_profile(mode, metrics["atr_pct"])
        eta = (now + timedelta(days=profile["horizon_days"])).isoformat()
        price = metrics["price"]

        tp_vals = [round(price * (1 + p), 2) for p in profile["tp_pct"]]
        sl_vals = [round(price * (1 + p), 2) for p in profile["sl_pct"]]

        reason = (
            f"TA blend: RSI {metrics['rsi']:.1f}, MACD hist {metrics['macd_hist']:.4f}, "
            f"vol x{metrics['vol_ratio']:.2f}, trend20 {metrics['change_20d'] * 100:.1f}%"
        )

        rows.append(
            {
                "ticker": symbol,
                "symbol": symbol,
                "price": round(price, 2),
                "change": round(metrics["change_1d"] * 100, 2),
                "score": round(scored["score"], 4),
                "p_accept": round(scored["p_up"], 4),
                "p_reject": round(scored["p_down"], 4),
                "p_continue": 0.0,
                "probability": round(scored["p_up"], 4),
                "signal": scored["signal"],
                "confidence": round(scored["confidence"], 4),
                "plan_type": mode,
                "label": scored["signal"],
                "source": "market_ta",
                "method": "yahoo_screener_ta",
                "target_price": tp_vals[0],
                "p_hit": round(scored["p_up"], 4),
                "created_at": now.isoformat(),
                "weights": {k: round(v, 4) for k, v in scored["contrib"].items()},
                "weights_json": {k: round(v, 4) for k, v in scored["contrib"].items()},
                "meta": {
                    "source": "yahoo_screener_ta",
                    "mode": mode,
                    "p_up": round(scored["p_up"], 4),
                    "p_down": round(scored["p_down"], 4),
                    "p_target": round(scored["p_up"], 4),
                    "horizon": f"{profile['horizon_days']}d",
                    "eta": eta,
                    "eta_up": eta,
                    "eta_down": eta,
                    "last_price": round(price, 2),
                    "target_price": tp_vals[0],
                    "tp1": tp_vals[0],
                    "tp2": tp_vals[1],
                    "tp3": tp_vals[2],
                    "sl1": sl_vals[0],
                    "sl2": sl_vals[1],
                    "sl3": sl_vals[2],
                    "tp1_pct": round(profile["tp_pct"][0], 6),
                    "tp2_pct": round(profile["tp_pct"][1], 6),
                    "tp3_pct": round(profile["tp_pct"][2], 6),
                    "sl1_pct": round(profile["sl_pct"][0], 6),
                    "sl2_pct": round(profile["sl_pct"][1], 6),
                    "sl3_pct": round(profile["sl_pct"][2], 6),
                    "rsi_14": round(metrics["rsi"], 3),
                    "macd_hist": round(metrics["macd_hist"], 6),
                    "atr_14": round(metrics["atr"], 6),
                    "atr_pct": round(metrics["atr_pct"], 6),
                    "sma_20": round(metrics["sma20"], 4),
                    "sma_50": round(metrics["sma50"], 4),
                    "support": round(metrics["support"], 4),
                    "resistance": round(metrics["resistance"], 4),
                    "volume_ratio": round(metrics["vol_ratio"], 4),
                    "reason": reason,
                    "weights": scored["weights"],
                },
            }
        )

    rows.sort(key=lambda r: r.get("score", 0), reverse=True)
    return rows[:count]


def build_rows_for_symbols(symbols: list[str], mode: str = "swing") -> list[dict]:
    """Build TA rows for an explicit symbol set."""
    symbols = [s.strip().upper() for s in symbols if _valid_symbol(s)]
    symbols = list(dict.fromkeys(symbols))
    if not symbols:
        return []

    mode = (mode or "swing").lower()
    if mode not in ("day", "swing", "long"):
        mode = "swing"

    histories = fetch_histories(symbols, range_="6mo", interval="1d")
    now = datetime.now(timezone.utc)
    rows = []

    for symbol in symbols:
        bars = histories.get(symbol) or []
        metrics = compute_metrics_from_bars(bars)
        if not metrics:
            continue
        scored = score_metrics(metrics, mode=mode)
        profile = target_profile(mode, metrics["atr_pct"])
        eta = (now + timedelta(days=profile["horizon_days"])).isoformat()
        price = metrics["price"]
        tp_vals = [round(price * (1 + p), 2) for p in profile["tp_pct"]]
        sl_vals = [round(price * (1 + p), 2) for p in profile["sl_pct"]]

        rows.append(
            {
                "ticker": symbol,
                "symbol": symbol,
                "price": round(price, 2),
                "score": round(scored["score"], 4),
                "p_accept": round(scored["p_up"], 4),
                "p_reject": round(scored["p_down"], 4),
                "signal": scored["signal"],
                "confidence": round(scored["confidence"], 4),
                "plan_type": mode,
                "label": scored["signal"],
                "source": "market_ta",
                "method": "yahoo_screener_ta",
                "weights": {k: round(v, 4) for k, v in scored["contrib"].items()},
                "weights_json": {k: round(v, 4) for k, v in scored["contrib"].items()},
                "meta": {
                    "source": "yahoo_screener_ta",
                    "mode": mode,
                    "p_up": round(scored["p_up"], 4),
                    "p_down": round(scored["p_down"], 4),
                    "p_target": round(scored["p_up"], 4),
                    "horizon": f"{profile['horizon_days']}d",
                    "eta": eta,
                    "eta_up": eta,
                    "eta_down": eta,
                    "last_price": round(price, 2),
                    "target_price": tp_vals[0],
                    "tp1": tp_vals[0],
                    "tp2": tp_vals[1],
                    "tp3": tp_vals[2],
                    "sl1": sl_vals[0],
                    "sl2": sl_vals[1],
                    "sl3": sl_vals[2],
                    "rsi_14": round(metrics["rsi"], 3),
                    "macd_hist": round(metrics["macd_hist"], 6),
                    "atr_14": round(metrics["atr"], 6),
                    "atr_pct": round(metrics["atr_pct"], 6),
                    "volume_ratio": round(metrics["vol_ratio"], 4),
                },
            }
        )

    rows.sort(key=lambda r: r.get("score", 0), reverse=True)
    return rows


def build_overlay_payload(symbols: list[str], mode: str = "swing", bars_limit: int = 180) -> dict:
    """Build overlay payload from real OHLCV + TA levels."""
    symbols = [s.strip().upper() for s in symbols if _valid_symbol(s)]
    symbols = list(dict.fromkeys(symbols))
    if not symbols:
        symbols = build_symbol_universe(mode, 20)

    histories = fetch_histories(symbols, range_="6mo", interval="1d")
    asof = datetime.now(timezone.utc).isoformat()
    payload = {"asof": asof, "symbols": {}}

    for sym in symbols:
        bars = histories.get(sym) or []
        if len(bars) < 40:
            continue
        metrics = compute_metrics_from_bars(bars)
        if not metrics:
            continue
        scored = score_metrics(metrics, mode=mode)
        profile = target_profile(mode, metrics["atr_pct"])

        trimmed = bars[-bars_limit:]
        series = [
            {
                "time": datetime.fromtimestamp(b["time"], tz=timezone.utc).isoformat(),
                "value": round(b["close"], 2),
            }
            for b in trimmed
        ]

        markers = []
        if series:
            markers.append(
                {
                    "time": series[-1]["time"],
                    "position": "aboveBar",
                    "shape": "arrowUp" if scored["signal"] == "BUY" else "arrowDown",
                    "color": "#10b981" if scored["signal"] == "BUY" else "#ef4444",
                    "text": scored["signal"],
                }
            )

        payload["symbols"][sym] = {
            "symbol": sym,
            "asof": asof,
            "series": series,
            "zones": [
                {
                    "from": round(metrics["support"], 2),
                    "to": round(metrics["support"] * 1.01, 2),
                    "label": "Support",
                    "color": "rgba(0,229,160,0.12)",
                },
                {
                    "from": round(metrics["resistance"] * 0.99, 2),
                    "to": round(metrics["resistance"], 2),
                    "label": "Resistance",
                    "color": "rgba(255,107,107,0.10)",
                },
            ],
            "lines": [
                {
                    "value": round(metrics["sma20"], 2),
                    "label": "SMA20",
                    "color": "#12d6ff",
                    "style": "solid",
                },
                {
                    "value": round(metrics["sma50"], 2),
                    "label": "SMA50",
                    "color": "#ffd479",
                    "style": "dashed",
                },
                {
                    "value": round(metrics["price"] * (1 + profile["tp_pct"][0]), 2),
                    "label": "TP1",
                    "color": "#10b981",
                    "style": "dotted",
                },
                {
                    "value": round(metrics["price"] * (1 + profile["sl_pct"][0]), 2),
                    "label": "SL1",
                    "color": "#ef4444",
                    "style": "dotted",
                },
            ],
            "levels": [
                {
                    "value": round(metrics["support"], 2),
                    "label": "Support",
                    "color": "#00e5a0",
                    "style": "dashed",
                },
                {
                    "value": round(metrics["resistance"], 2),
                    "label": "Resistance",
                    "color": "#ff8fa3",
                    "style": "dashed",
                },
            ],
            "percent_levels": [
                {"percent": round(profile["tp_pct"][0], 5), "label": f"+{profile['tp_pct'][0] * 100:.1f}%"},
                {"percent": round(profile["sl_pct"][0], 5), "label": f"{profile['sl_pct'][0] * 100:.1f}%"},
            ],
            "markers": markers,
            "meta": {
                "signal": scored["signal"],
                "p_up": round(scored["p_up"], 4),
                "rsi_14": round(metrics["rsi"], 3),
                "macd_hist": round(metrics["macd_hist"], 6),
                "volume_ratio": round(metrics["vol_ratio"], 4),
            },
        }

    return payload
