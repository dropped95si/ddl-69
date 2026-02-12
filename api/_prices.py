"""Batch stock price fetcher using Yahoo Finance v8 (no API key needed)."""

import json
import time

_cache = {}
_CACHE_TTL = 300  # 5 minutes


def fetch_prices(tickers):
    """Return {TICKER: price} for a list of ticker symbols.

    Uses Yahoo Finance v8 quote endpoint (free, no auth).
    Returns only tickers that resolved successfully.
    Caches results for 5 minutes to avoid hammering Yahoo.
    """
    import requests

    now = time.time()
    result = {}
    need = []

    for t in tickers:
        t = t.upper()
        if t in _cache and (now - _cache[t][1]) < _CACHE_TTL:
            result[t] = _cache[t][0]
        else:
            need.append(t)

    if not need:
        return result

    # Yahoo v8 supports batch quotes (comma-separated)
    try:
        symbols = ",".join(need[:80])  # Yahoo limit ~100
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{need[0]}"
        # For batch, use v7 spark endpoint or individual calls
        # v8 chart only does one at a time, so use v7 quote
        url = "https://query1.finance.yahoo.com/v7/finance/quote"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(
            url,
            params={"symbols": symbols, "fields": "regularMarketPrice,symbol"},
            headers=headers,
            timeout=8,
        )
        if resp.status_code == 200:
            data = resp.json()
            for q in data.get("quoteResponse", {}).get("result", []):
                sym = q.get("symbol", "").upper()
                price = q.get("regularMarketPrice")
                if sym and price and float(price) > 0:
                    result[sym] = float(price)
                    _cache[sym] = (float(price), now)
            return result
    except Exception:
        pass

    # Fallback: try individual v8 chart calls for remaining tickers
    for t in need:
        if t in result:
            continue
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{t}"
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, params={"range": "1d", "interval": "1d"}, headers=headers, timeout=5)
            if resp.status_code == 200:
                chart = resp.json().get("chart", {}).get("result", [{}])[0]
                meta = chart.get("meta", {})
                price = meta.get("regularMarketPrice") or meta.get("previousClose")
                if price and float(price) > 0:
                    result[t] = float(price)
                    _cache[t] = (float(price), now)
        except Exception:
            continue

    return result
