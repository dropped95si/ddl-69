from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


@dataclass
class Evidence:
    name: str
    p: float
    weight: float


def combine_probabilities(evidence: Dict[str, Evidence], bias: float = 0.0) -> float:
    total = bias
    for ev in evidence.values():
        total += ev.weight * logit(ev.p)
    return float(sigmoid(total))


__all__ = ["Evidence", "combine_probabilities"]
