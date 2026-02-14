"""News endpoint - real news only (Supabase table/storage)."""

import json
import logging
import os
from datetime import datetime, timedelta, timezone

import requests

logger = logging.getLogger(__name__)

try:
    from _http_adapter import FunctionHandler
except ModuleNotFoundError:
    from api._http_adapter import FunctionHandler


STORAGE_BASE = "https://iyqzrzesrbfltoryfzet.supabase.co/storage/v1/object/public/artifacts/news"


def _fetch_from_storage():
    for days_ago in range(0, 14):
        date = (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        url = f"{STORAGE_BASE}/polygon_news_{date}.json"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                continue
            data = resp.json()
            if not isinstance(data, list) or not data:
                continue

            news = []
            for item in data[:120]:
                tickers = item.get("tickers") or []
                publisher = item.get("publisher") or {}
                source = publisher.get("name") if isinstance(publisher, dict) else None
                news.append(
                    {
                        "ticker": tickers[0] if tickers else None,
                        "tickers": tickers,
                        "title": item.get("title"),
                        "url": item.get("article_url"),
                        "published": item.get("published_utc"),
                        "sentiment": None,
                        "source": source,
                        "description": item.get("description"),
                    }
                )
            return news, f"polygon_storage:{date}"
        except Exception as exc:
            logger.warning("news storage fetch failed for %s: %s", date, exc)
            continue
    return None, None


def _fetch_from_table():
    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not supabase_url or not service_key:
        return None, None

    try:
        from supabase import create_client
    except Exception as exc:
        logger.warning("supabase import failed: %s", exc)
        return None, None

    try:
        supa = create_client(supabase_url, service_key)
        resp = (
            supa.table("news")
            .select("ticker,title,url,published_utc,sentiment,publisher_name")
            .order("published_utc", desc=True)
            .limit(200)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            return None, None
        news = []
        for r in rows:
            news.append(
                {
                    "ticker": r.get("ticker"),
                    "title": r.get("title"),
                    "url": r.get("url"),
                    "published": r.get("published_utc"),
                    "sentiment": r.get("sentiment"),
                    "source": r.get("publisher_name"),
                }
            )
        return news, "supabase_table"
    except Exception as exc:
        logger.warning("news table fetch failed: %s", exc)
        return None, None


def _handler_impl(request):
    news, source = _fetch_from_table()
    if not news:
        news, source = _fetch_from_storage()

    if not news:
        return {
            "statusCode": 503,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(
                {
                    "error": "No live news artifacts available.",
                    "source": "none",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
            ),
        }

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Cache-Control": "max-age=180, public",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(
            {
                "news": news,
                "total": len(news),
                "source": source,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        ),
    }


class handler(FunctionHandler):
    endpoint = staticmethod(_handler_impl)

