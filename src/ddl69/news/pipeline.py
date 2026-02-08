from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import requests

from ddl69.news.normalize import normalize_polygon_news
from ddl69.news.summarize import summarize
from ddl69.news.event_features import detect_events
from ddl69.news.score import score_with_weights
from ddl69.news.aggregate import aggregate_scores


def fetch_json(url: str, timeout: int = 30) -> Any:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def load_json(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def run_news_pipeline(
    news_url: str,
    weights_path: Optional[str],
    calibration_path: Optional[str],
    outdir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw = fetch_json(news_url)
    items = normalize_polygon_news(raw)

    weights = load_json(weights_path) or {"bias": 0.0, "weights": {}}
    calibration = load_json(calibration_path)

    rows: list[dict] = []
    for it in items:
        summ = summarize(it.title, it.body)
        feats = detect_events(it.title, it.body)
        score = score_with_weights(feats, weights=weights, calibration=calibration)

        rows.append(
            {
                "id": it.id,
                "published_at": it.published_at,
                "tickers": it.tickers,
                "source": it.source,
                "title": it.title,
                "summary": summ,
                "events": {k: v for k, v in feats.items() if k.startswith("EV_") and v > 0},
                **score,
                "url": it.url,
            }
        )

    df_items = pd.DataFrame(rows)
    df_agg = aggregate_scores(df_items)

    od = Path(outdir)
    od.mkdir(parents=True, exist_ok=True)

    df_items.to_json(od / "scored_news_items.json", orient="records", indent=2)
    df_items.to_csv(od / "scored_news_items.csv", index=False)

    df_agg.to_json(od / "scored_news_agg.json", orient="records", indent=2)
    df_agg.to_csv(od / "scored_news_agg.csv", index=False)

    return df_items, df_agg
