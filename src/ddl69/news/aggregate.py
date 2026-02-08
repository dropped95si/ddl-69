from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_scores(df_items: pd.DataFrame) -> pd.DataFrame:
    if df_items.empty:
        return pd.DataFrame()

    df = df_items.copy()
    df["date"] = pd.to_datetime(df["published_at"], utc=True).dt.date.astype(str)

    e = df.explode("tickers").rename(columns={"tickers": "ticker"})
    e = e[e["ticker"].notna() & (e["ticker"] != "")]

    e["w"] = np.clip(e["confidence"].astype(float), 0.05, 1.0)

    g = (
        e.groupby(["date", "ticker"], as_index=False)
        .apply(
            lambda x: pd.Series(
                {
                    "news_count": int(len(x)),
                    "p_up": float(np.average(x["p_up"], weights=x["w"])),
                    "confidence": float(np.average(x["confidence"], weights=x["w"])),
                }
            )
        )
        .reset_index(drop=True)
    )
    g["p_down"] = 1.0 - g["p_up"]
    return g.sort_values(["date", "ticker"]).reset_index(drop=True)
