from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass
class RuleStats:
    rule: str
    win_rate: float
    avg_return: float
    samples: int

    def as_rule_dict(self) -> Dict[str, Any]:
        # Match existing rule_to_probs expectations (h60 bucket)
        return {
            "rule": self.rule,
            "h60": {
                "win_rate": float(self.win_rate),
                "avg_return": float(self.avg_return),
                "samples": int(self.samples),
            },
        }


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def build_ta_features(bars: pd.DataFrame) -> pd.DataFrame:
    df = bars.copy()
    df = df.sort_values(["instrument_id", "ts"]).reset_index(drop=True)
    df["ret_1"] = df.groupby("instrument_id")["close"].pct_change()
    df["sma_5"] = df.groupby("instrument_id")["close"].transform(lambda s: s.rolling(5).mean())
    df["sma_20"] = df.groupby("instrument_id")["close"].transform(lambda s: s.rolling(20).mean())
    df["ema_12"] = df.groupby("instrument_id")["close"].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    df["ema_26"] = df.groupby("instrument_id")["close"].transform(lambda s: s.ewm(span=26, adjust=False).mean())
    df["rsi_14"] = df.groupby("instrument_id")["close"].transform(_rsi)
    df["atr_14"] = df.groupby("instrument_id").apply(
        lambda g: _atr(g["high"], g["low"], g["close"])
    ).reset_index(level=0, drop=True)
    df["vol_20"] = df.groupby("instrument_id")["ret_1"].transform(lambda s: s.rolling(20).std())
    return df


def compute_forward_return(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    return (
        df.groupby("instrument_id")["close"]
        .pct_change(periods=horizon)
        .shift(-horizon)
    )


def _score_rule(mask: pd.Series, fwd_ret: pd.Series) -> Optional[RuleStats]:
    sel = fwd_ret[mask.fillna(False)]
    if sel.empty:
        return None
    wins = (sel > 0).sum()
    samples = int(sel.shape[0])
    win_rate = float(wins / samples) if samples > 0 else 0.0
    avg_ret = float(sel.mean()) if samples > 0 else 0.0
    return RuleStats(rule="", win_rate=win_rate, avg_return=avg_ret, samples=samples)


def generate_rule_stats(df: pd.DataFrame, horizon: int = 5) -> Dict[str, List[RuleStats]]:
    df = build_ta_features(df)
    df["fwd_ret"] = compute_forward_return(df, horizon=horizon)

    rule_map: Dict[str, List[RuleStats]] = {}
    for sym, g in df.groupby("instrument_id"):
        stats: List[RuleStats] = []

        candidates = {
            "SMA_CROSS_5_20": g["sma_5"] > g["sma_20"],
            "EMA_CROSS_12_26": g["ema_12"] > g["ema_26"],
            "RSI_OVERBOUGHT": g["rsi_14"] > 70,
            "RSI_OVERSOLD": g["rsi_14"] < 30,
            "VOL_SPIKE": g["vol_20"] > g["vol_20"].rolling(50).mean(),
            "ATR_EXPANSION": g["atr_14"] > g["atr_14"].rolling(50).mean(),
        }

        for name, mask in candidates.items():
            r = _score_rule(mask, g["fwd_ret"])
            if r:
                r.rule = name
                stats.append(r)

        rule_map[sym] = stats
    return rule_map


def expand_signals_rows(
    signals_df: pd.DataFrame,
    bars_df: pd.DataFrame,
    horizon: int = 5,
    top_n: int = 8,
) -> pd.DataFrame:
    rule_map = generate_rule_stats(bars_df, horizon=horizon)
    out = signals_df.copy()

    def _rules_for_ticker(ticker: str) -> List[Dict[str, Any]]:
        stats = rule_map.get(ticker, [])
        stats = sorted(stats, key=lambda s: (s.win_rate, s.avg_return), reverse=True)
        return [s.as_rule_dict() for s in stats[:top_n]]

    out["learned_top_rules"] = out["ticker"].astype(str).str.upper().apply(_rules_for_ticker)
    return out
