"""Quote snapshot helpers with Yahoo primary + Polygon profile enrichment."""

import os
import time

import requests

_cache = {}
_quote_cache = {}
_profile_cache = {}
_CACHE_TTL = 300  # 5 minutes
_PROFILE_TTL = 3600  # 1 hour
_USER_AGENT = {"User-Agent": "Mozilla/5.0"}


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


def _polygon_api_key():
    for key_name in ("POLYGON_API_KEY", "POLYGON_KEY"):
        key = os.getenv(key_name, "").strip()
        if key:
            return key
    return ""


def _chunked(values, size):
    for i in range(0, len(values), size):
        yield values[i : i + size]


def _parse_polygon_quote_type(raw):
    txt = str(raw or "").strip().upper()
    if not txt:
        return None
    if txt in ("ETF", "ETN", "FUND"):
        return "etf"
    if txt in ("CS", "COMMON_STOCK", "COMMON STOCK", "EQUITY", "STOCK", "ADR", "ADRC"):
        return "equity"
    return txt.lower()


def _normalize_snapshot(snapshot):
    if not isinstance(snapshot, dict):
        return None
    normalized = {
        "price": _to_float(snapshot.get("price")),
        "market_cap": _to_int(snapshot.get("market_cap")),
        "quote_type": _asset_or_none(snapshot.get("quote_type")),
    }
    if (
        normalized["price"] is None
        and normalized["market_cap"] is None
        and normalized["quote_type"] is None
    ):
        return None
    return normalized


def _asset_or_none(raw):
    txt = str(raw or "").strip().lower()
    if not txt:
        return None
    return txt


def _fetch_polygon_profiles(tickers):
    """Return {TICKER: {market_cap, quote_type}} using Polygon reference data."""
    key = _polygon_api_key()
    if not key:
        return {}

    now = time.time()
    out = {}
    unresolved = []

    symbols = []
    seen = set()
    for raw in tickers:
        sym = _normalize_symbol(raw)
        if not sym or sym in seen:
            continue
        seen.add(sym)
        symbols.append(sym)

    for sym in symbols:
        cached = _profile_cache.get(sym)
        if cached and (now - cached[1]) < _PROFILE_TTL:
            cached_profile = dict(cached[0])
            if cached_profile:
                out[sym] = cached_profile
        else:
            unresolved.append(sym)

    if not unresolved:
        return out

    def _record_profile(sym, profile):
        normalized = _normalize_snapshot({"market_cap": profile.get("market_cap"), "quote_type": profile.get("quote_type")}) if profile else None
        if normalized is None:
            _profile_cache[sym] = ({}, now)
            return
        compact = {"market_cap": normalized.get("market_cap"), "quote_type": normalized.get("quote_type")}
        _profile_cache[sym] = (compact, now)
        out[sym] = dict(compact)

    # Batch query first (faster when supported by Polygon plan).
    for chunk in _chunked(unresolved, 40):
        try:
            resp = requests.get(
                "https://api.polygon.io/v3/reference/tickers",
                params={
                    "ticker.any_of": ",".join(chunk),
                    "active": "true",
                    "limit": 1000,
                    "apiKey": key,
                },
                headers=_USER_AGENT,
                timeout=5,
            )
        except Exception:
            resp = None
        if not resp or resp.status_code != 200:
            continue
        payload = resp.json() if callable(getattr(resp, "json", None)) else {}
        results = payload.get("results") if isinstance(payload, dict) else []
        if not isinstance(results, list):
            results = []
        for item in results:
            if not isinstance(item, dict):
                continue
            sym = _normalize_symbol(item.get("ticker"))
            if not sym or sym not in chunk:
                continue
            _record_profile(
                sym,
                {
                    "market_cap": item.get("market_cap"),
                    "quote_type": _parse_polygon_quote_type(item.get("type")),
                },
            )

    # Per-symbol fallback for any still unresolved.
    for sym in unresolved:
        if sym in out:
            continue
        try:
            resp = requests.get(
                f"https://api.polygon.io/v3/reference/tickers/{sym}",
                params={"apiKey": key},
                headers=_USER_AGENT,
                timeout=5,
            )
        except Exception:
            _profile_cache[sym] = ({}, now)
            continue
        if resp.status_code in (401, 403):
            break
        if resp.status_code != 200:
            _profile_cache[sym] = ({}, now)
            continue
        payload = resp.json() if callable(getattr(resp, "json", None)) else {}
        item = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(item, dict):
            _profile_cache[sym] = ({}, now)
            continue
        _record_profile(
            sym,
            {
                "market_cap": item.get("market_cap"),
                "quote_type": _parse_polygon_quote_type(item.get("type")),
            },
        )

    return out


def fetch_quote_snapshots(tickers):
    """Return {TICKER: {price, market_cap, quote_type}} for ticker symbols.

    Uses Yahoo Finance v8 quote endpoint (free, no auth).
    Returns only tickers that resolved successfully.
    Caches results for 5 minutes to avoid hammering Yahoo.
    """
    now = time.time()
    result = {}
    need = []
    requested = []
    seen = set()

    for t in tickers:
        t = _normalize_symbol(t)
        if not t or t in seen:
            continue
        seen.add(t)
        requested.append(t)
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
        resp = requests.get(
            url,
            params={"symbols": symbols, "fields": "regularMarketPrice,marketCap,quoteType,symbol"},
            headers=_USER_AGENT,
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
    except Exception:
        pass

    # Fallback: try individual v8 chart calls for remaining tickers
    for t in need:
        if t in result:
            continue
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{t}"
            resp = requests.get(url, params={"range": "1d", "interval": "1d"}, headers=_USER_AGENT, timeout=5)
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

    # Enrich missing market_cap / quote_type from Polygon reference metadata (optional).
    need_profile = []
    for sym in requested:
        snap = result.get(sym) if isinstance(result.get(sym), dict) else {}
        if snap.get("market_cap") is None or snap.get("quote_type") is None:
            need_profile.append(sym)
    if need_profile:
        profiles = _fetch_polygon_profiles(need_profile)
        for sym in need_profile:
            profile = profiles.get(sym)
            if not isinstance(profile, dict):
                continue
            snapshot = result.get(sym)
            if not isinstance(snapshot, dict):
                snapshot = {"price": None, "market_cap": None, "quote_type": None}
            if snapshot.get("market_cap") is None and profile.get("market_cap") is not None:
                snapshot["market_cap"] = _to_int(profile.get("market_cap"))
            if snapshot.get("quote_type") is None and profile.get("quote_type") is not None:
                snapshot["quote_type"] = _asset_or_none(profile.get("quote_type"))
            normalized = _normalize_snapshot(snapshot)
            if normalized is None:
                continue
            result[sym] = normalized
            _quote_cache[sym] = (dict(normalized), now)
            if normalized["price"] is not None:
                _cache[sym] = (normalized["price"], now)

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
