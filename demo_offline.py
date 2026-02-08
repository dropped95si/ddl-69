from ddl69.news.pipeline import run_news_pipeline
from pathlib import Path
import json

# Offline demo: load sample file via file:// emulation by directly reading json
# (the CLI normally fetches from URL; this is just for local validation)

p = Path("data/sample_news.json")
raw = json.loads(p.read_text(encoding="utf-8"))

# Reuse internals without requests:
from ddl69.news.normalize import normalize_polygon_news
from ddl69.news.summarize import summarize
from ddl69.news.event_features import detect_events
from ddl69.news.score import score_with_weights
from ddl69.news.aggregate import aggregate_scores
import pandas as pd

weights = json.loads(Path("artifacts/weights/latest.json").read_text(encoding="utf-8"))
calib = json.loads(Path("artifacts/calibration/calibration_latest.json").read_text(encoding="utf-8"))

items = normalize_polygon_news(raw)
rows = []
for it in items:
    feats = detect_events(it.title, it.body)
    score = score_with_weights(feats, weights=weights, calibration=calib)
    rows.append({
        "id": it.id,
        "published_at": it.published_at,
        "tickers": it.tickers,
        "title": it.title,
        "p_up": score["p_up"],
        "confidence": score["confidence"],
    })

df = pd.DataFrame(rows)
agg = aggregate_scores(pd.DataFrame([{
    "id": r["id"],
    "published_at": r["published_at"],
    "tickers": r["tickers"],
    "confidence": r["confidence"],
    "p_up": r["p_up"],
} for r in rows]))
print(df)
print("\nAGG\n", agg)
