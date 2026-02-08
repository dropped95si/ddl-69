from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class ExecutionPlan:
    entry: float
    stop: float
    target: float
    risk_to_reward: float
    notes: Dict[str, str]


def compute_execution(zone_low: float, zone_high: float, bias: str) -> ExecutionPlan:
    if bias == "UP":
        entry = zone_low
        stop = zone_low * 0.98
        target = zone_high * 1.02
    elif bias == "DOWN":
        entry = zone_high
        stop = zone_high * 1.02
        target = zone_low * 0.98
    else:
        entry = (zone_low + zone_high) / 2
        stop = zone_low * 0.98
        target = zone_high * 1.02

    rr = abs(target - entry) / max(abs(entry - stop), 1e-6)
    return ExecutionPlan(
        entry=float(entry),
        stop=float(stop),
        target=float(target),
        risk_to_reward=float(rr),
        notes={"bias": bias},
    )


__all__ = ["ExecutionPlan", "compute_execution"]
