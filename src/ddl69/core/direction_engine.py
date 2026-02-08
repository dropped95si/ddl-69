from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class DirectionResult:
    bias: str
    confidence: float
    state: str
    invalidation: str
    meta: Dict[str, float]


def _vwap(series_close: pd.Series, series_vol: pd.Series, window: int = 20) -> pd.Series:
    pv = series_close * series_vol
    return pv.rolling(window).sum() / series_vol.rolling(window).sum()


def compute_direction(df: pd.DataFrame) -> DirectionResult:
    close = df["close"].astype(float)
    vol = df.get("volume", pd.Series(1.0, index=df.index)).astype(float)
    vwap20 = _vwap(close, vol, window=20)
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    atr = (df["high"] - df["low"]).rolling(14).mean()

    last = close.iloc[-1]
    vwap_last = vwap20.iloc[-1]
    sma20_last = sma20.iloc[-1]
    sma50_last = sma50.iloc[-1]

    bias = "NEUTRAL"
    if last > vwap_last and sma20_last > sma50_last:
        bias = "UP"
    elif last < vwap_last and sma20_last < sma50_last:
        bias = "DOWN"

    # confidence from distance to vwap + slope alignment
    dist = abs(last - vwap_last) / (atr.iloc[-1] + 1e-9)
    conf = float(np.clip(0.4 + 0.1 * dist, 0.0, 0.95))
    state = "CONTINUATION" if bias != "NEUTRAL" else "BALANCE"
    invalidation = (
        f"close < vwap20 - 1.0*atr" if bias == "UP" else f"close > vwap20 + 1.0*atr"
    )

    return DirectionResult(
        bias=bias,
        confidence=conf,
        state=state,
        invalidation=invalidation,
        meta={
            "last": float(last),
            "vwap20": float(vwap_last),
            "sma20": float(sma20_last),
            "sma50": float(sma50_last),
        },
    )


__all__ = ["DirectionResult", "compute_direction"]
