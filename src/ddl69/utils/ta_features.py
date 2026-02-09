from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class TAFeatures:
    rsi_period: int = 14
    atr_period: int = 14
    ema_fast: int = 12
    ema_slow: int = 26
    macd_signal: int = 9
    sma_fast: int = 20
    sma_mid: int = 50
    sma_slow: int = 200
    roc_fast: int = 10
    roc_slow: int = 20
    vol_window: int = 20

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float) if "volume" in df.columns else None

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(self.rsi_period).mean()
        avg_loss = loss.rolling(self.rsi_period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        tr1 = (high - low).abs()
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_period).mean()

        ema_fast = close.ewm(span=self.ema_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.ema_slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=self.macd_signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        sma_fast = close.rolling(self.sma_fast).mean()
        sma_mid = close.rolling(self.sma_mid).mean()
        sma_slow = close.rolling(self.sma_slow).mean()
        roc_fast = close.pct_change(self.roc_fast)
        roc_slow = close.pct_change(self.roc_slow)
        vol = close.pct_change().rolling(self.vol_window).std()

        out = df.copy()
        out["rsi"] = rsi
        out["atr"] = atr
        out["ema_fast"] = ema_fast
        out["ema_slow"] = ema_slow
        out["macd"] = macd
        out["macd_signal"] = macd_signal
        out["macd_hist"] = macd_hist
        out["sma_fast"] = sma_fast
        out["sma_mid"] = sma_mid
        out["sma_slow"] = sma_slow
        out["roc_fast"] = roc_fast
        out["roc_slow"] = roc_slow
        out["volatility"] = vol
        if volume is not None:
            out["volume"] = volume
            out["volume_z"] = (volume - volume.rolling(self.vol_window).mean()) / volume.rolling(self.vol_window).std()
        return out


__all__ = ["TAFeatures"]
