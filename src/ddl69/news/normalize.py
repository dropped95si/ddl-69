from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional
import pandas as pd
from datetime import datetime, timezone


def to_utc_iso(ts: Any) -> str:
    if ts is None:
        return datetime.now(timezone.utc).isoformat()
    if isinstance(ts, (int, float)):
        if ts > 10_000_000_000:
            dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        else:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.isoformat()
    if isinstance(ts, str):
        try:
            dt = pd.to_datetime(ts, utc=True)
            return dt.to_pydatetime().isoformat()
        except Exception:
            return datetime.now(timezone.utc).isoformat()
    return datetime.now(timezone.utc).isoformat()


@dataclass
class NewsItem:
    id: str
    published_at: str
    tickers: List[str]
    title: str
    body: str
    source: str
    url: Optional[str] = None


def normalize_polygon_news(raw: Any) -> List[NewsItem]:
    items: list[dict] = []

    if isinstance(raw, dict):
        if isinstance(raw.get("results"), list):
            items = raw["results"]
        elif isinstance(raw.get("data"), list):
            items = raw["data"]
        elif isinstance(raw.get("items"), list):
            items = raw["items"]
        else:
            items = [raw]
    elif isinstance(raw, list):
        items = raw
    else:
        return []

    out: List[NewsItem] = []
    for i, x in enumerate(items):
        if not isinstance(x, dict):
            continue

        _id = str(x.get("id") or x.get("uuid") or x.get("article_id") or f"item_{i}")
        published_at = to_utc_iso(
            x.get("published_utc")
            or x.get("published_at")
            or x.get("timestamp")
            or x.get("time")
            or x.get("created_at")
        )

        tickers = x.get("tickers") or x.get("ticker") or x.get("symbols") or []
        if isinstance(tickers, str):
            tickers = [tickers]
        tickers = [t.upper().strip() for t in tickers if isinstance(t, str) and t.strip()]

        title = str(x.get("title") or x.get("headline") or "").strip()
        body = str(x.get("description") or x.get("summary") or x.get("content") or x.get("article") or "").strip()

        publisher = x.get("publisher")
        if isinstance(publisher, dict):
            source = str(publisher.get("name") or "unknown").strip()
        else:
            source = str(x.get("source") or x.get("publisher") or "unknown").strip()

        url = x.get("article_url") or x.get("url") or x.get("link")

        if not title and not body:
            continue

        out.append(
            NewsItem(
                id=_id,
                published_at=published_at,
                tickers=tickers,
                title=title,
                body=body,
                source=source,
                url=url,
            )
        )
    return out
