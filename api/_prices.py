"""Batch stock price fetcher using Yahoo Finance v8 (no API key needed)."""

import json
import time

_cache = {}
_quote_cache = {}
_CACHE_TTL = 300  # 5 minutes


def _normalize_symbol(raw):
    txt = str(raw or "").upper().strip()
    return txt


def _parse_quote_type(quote):
    raw = quote.get("quoteType") or quote.get("typeDisp")
    txt = str(raw or "").strip().upper()
    if not txt:
        return None
    if txt in ("ETF", "ETN", "MUTUALFUND", "MUTUAL FUND", "FUND"):
        return "etf"
    if txt in ("EQUITY", "STOCK", "COMMONSTOCK", "COMMON STOCK"):
        return "equity"
    return txt.lower()


def _to_float(value):
    try:
        n = float(value)
    except Exception:
        return None
    if n <= 0:
        return None
    return n


def _to_int(value):
    try:
        n = int(float(value))
    except Exception:
        return None
    if n <= 0:
        return None
    return n


def fetch_quote_snapshots(tickers):
    """Return {TICKER: {price, market_cap, quote_type}} for ticker symbols.

    Uses Yahoo Finance v8 quote endpoint (free, no auth).
    Returns only tickers that resolved successfully.
    Caches results for 5 minutes to avoid hammering Yahoo.
    """
    import requests

    now = time.time()
    result = {}
    need = []

    for t in tickers:
        t = _normalize_symbol(t)
        if not t:
            continue
        if t in _quote_cache and (now - _quote_cache[t][1]) < _CACHE_TTL:
            result[t] = dict(_quote_cache[t][0])
        else:
            need.append(t)

    if not need:
        return result

    # Yahoo quote endpoint supports batch quotes (comma-separated symbols).
    try:
        symbols = ",".join(need[:80])  # Yahoo limit ~100
        url = "https://query1.finance.yahoo.com/v7/finance/quote"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(
            url,
            params={"symbols": symbols, "fields": "regularMarketPrice,marketCap,quoteType,symbol"},
            headers=headers,
            timeout=8,
        )
        if resp.status_code == 200:
            data = resp.json()
            for q in data.get("quoteResponse", {}).get("result", []):
                sym = _normalize_symbol(q.get("symbol"))
                if not sym:
                    continue
                snapshot = {
                    "price": _to_float(q.get("regularMarketPrice")),
                    "market_cap": _to_int(q.get("marketCap")),
                    "quote_type": _parse_quote_type(q),
                }
                if snapshot["price"] is None and snapshot["market_cap"] is None and snapshot["quote_type"] is None:
                    continue
                result[sym] = snapshot
                _quote_cache[sym] = (dict(snapshot), now)
                if snapshot["price"] is not None:
                    _cache[sym] = (snapshot["price"], now)
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
                price = _to_float(meta.get("regularMarketPrice") or meta.get("previousClose"))
                snapshot = {
                    "price": price,
                    "market_cap": _to_int(meta.get("marketCap")),
                    "quote_type": _parse_quote_type(
                        {"quoteType": meta.get("instrumentType") or meta.get("quoteType")}
                    ),
                }
                if snapshot["price"] is None and snapshot["market_cap"] is None and snapshot["quote_type"] is None:
                    continue
                result[t] = snapshot
                _quote_cache[t] = (dict(snapshot), now)
                if snapshot["price"] is not None:
                    _cache[t] = (snapshot["price"], now)
        except Exception:
            continue

    return result


def fetch_prices(tickers):
    """Return {TICKER: price} for a list of ticker symbols."""
    snapshots = fetch_quote_snapshots(tickers)
    prices = {}
    for sym, snap in snapshots.items():
        price = snap.get("price")
        if price is not None:
            prices[sym] = price
    return prices
