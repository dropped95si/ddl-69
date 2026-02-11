import json
from datetime import datetime, timedelta, timezone

try:
    from _http_adapter import FunctionHandler
except ModuleNotFoundError:
    from api._http_adapter import FunctionHandler

DEFAULT_SUPABASE_URL = ""
DEFAULT_SUPABASE_SERVICE_ROLE_KEY = ""


def _fallback():
    """Return sample news data when Supabase is not configured."""
    news = []
    for i in range(10):
        date = (datetime.now(timezone.utc) - timedelta(days=i)).strftime('%Y-%m-%d')
        news.append({
            "ticker": ["AAPL", "NVDA", "TSLA", "SPY"][i % 4],
            "title": f"Sample news headline {i + 1}",
            "url": f"https://example.com/news/{i}",
            "published": date,
            "sentiment": round((i % 3 - 1) * 0.3, 2),
            "source": "Sample Source"
        })

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json", "Cache-Control": "max-age=300"},
        "body": json.dumps({
            "news": news,
            "total": len(news),
            "generated_at": datetime.utcnow().isoformat()
        })
    }


def _handler_impl(request):
    """Return latest news from Supabase (falls back to sample data)."""
    import os

    supabase_url = os.getenv("SUPABASE_URL", DEFAULT_SUPABASE_URL).strip()
    supabase_service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", DEFAULT_SUPABASE_SERVICE_ROLE_KEY).strip()
    if not supabase_url or not supabase_service_role_key:
        return _fallback()

    try:
        from supabase import create_client

        supa = create_client(supabase_url, supabase_service_role_key)
        # Query news table
        resp = supa.table("news")\
            .select("ticker,title,url,published_utc,sentiment,publisher_name")\
            .order("published_utc", desc=True)\
            .limit(100)\
            .execute()
        rows = resp.data or []

        news = []
        for r in rows:
            news.append({
                "ticker": r.get("ticker"),
                "title": r.get("title"),
                "url": r.get("url"),
                "published": r.get("published_utc"),
                "sentiment": r.get("sentiment"),
                "source": r.get("publisher_name")
            })

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json", "Cache-Control": "max-age=120"},
            "body": json.dumps({
                "news": news,
                "total": len(news),
                "generated_at": datetime.utcnow().isoformat()
            })
        }
    except Exception as exc:
        # fallback so API keeps working even if Supabase errors
        return _fallback()


class handler(FunctionHandler):
    endpoint = staticmethod(_handler_impl)
