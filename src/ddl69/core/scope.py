from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class ScopeConfig:
    name: str
    direction_timeframes: list[str]
    event_timeframes: list[str]
    execution_timeframes: list[str]
    horizon_bars: int
    expert_weights: Dict[str, float]


SCOPE_MAP: Dict[str, ScopeConfig] = {
    "day": ScopeConfig(
        name="day",
        direction_timeframes=["1h", "30m"],
        event_timeframes=["30m", "15m"],
        execution_timeframes=["15m", "5m", "3m"],
        horizon_bars=20,
        expert_weights={"TA": 0.45, "NEWS": 0.2, "EARN": 0.15, "REGIME": 0.1, "QLIB": 0.1},
    ),
    "swing": ScopeConfig(
        name="swing",
        direction_timeframes=["1d", "1w"],
        event_timeframes=["4h", "1d"],
        execution_timeframes=["30m", "15m"],
        horizon_bars=60,
        expert_weights={"TA": 0.25, "NEWS": 0.2, "EARN": 0.2, "REGIME": 0.2, "QLIB": 0.15},
    ),
    "long": ScopeConfig(
        name="long",
        direction_timeframes=["1w", "1m"],
        event_timeframes=["1d", "1w"],
        execution_timeframes=["1d", "4h"],
        horizon_bars=120,
        expert_weights={"TA": 0.15, "NEWS": 0.15, "EARN": 0.2, "REGIME": 0.25, "QLIB": 0.25},
    ),
}


def get_scope(name: str) -> ScopeConfig:
    key = name.lower().strip()
    if key not in SCOPE_MAP:
        raise ValueError(f"Unknown scope: {name}")
    return SCOPE_MAP[key]


__all__ = ["ScopeConfig", "SCOPE_MAP", "get_scope"]
