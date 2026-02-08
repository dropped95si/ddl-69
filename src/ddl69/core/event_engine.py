from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class EventResult:
    name: str
    horizon_bars: int
    zone_low: float
    zone_high: float
    p_event: float
    support: int
    meta: Dict[str, float]


def label_touch_zone(df: pd.DataFrame, zone_low: float, zone_high: float, horizon: int) -> pd.Series:
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    n = len(df)
    labels = np.zeros(n, dtype=int)
    for i in range(n):
        j = min(n, i + horizon)
        h = highs[i:j]
        l = lows[i:j]
        if np.any((l <= zone_high) & (h >= zone_low)):
            labels[i] = 1
    return pd.Series(labels, index=df.index)


def compute_touch_zone_prob(
    df: pd.DataFrame,
    zone_low: float,
    zone_high: float,
    horizon_bars: int = 5,
    lookback: int = 252,
) -> EventResult:
    labels = label_touch_zone(df, zone_low, zone_high, horizon_bars)
    if len(labels) > lookback:
        labels = labels.iloc[-lookback:]
    p = float(labels.mean()) if len(labels) else 0.0
    return EventResult(
        name="TOUCH_ZONE",
        horizon_bars=horizon_bars,
        zone_low=zone_low,
        zone_high=zone_high,
        p_event=p,
        support=int(len(labels)),
        meta={"lookback": float(lookback)},
    )


__all__ = ["EventResult", "compute_touch_zone_prob"]
