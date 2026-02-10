from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Literal

EventType = Literal["state_event", "barrier_event"]
Mode = Literal["lean", "qlib"]

STATE_OUTCOMES = ["REJECT", "BREAK_FAIL", "ACCEPT_CONTINUE"]
BARRIER_OUTCOMES = ["UPPER", "LOWER", "NONE"]

@dataclass
class Horizon:
    type: Literal["time", "bars"]
    value: int
    unit: str  # e.g. 'm','h','d','bars'

@dataclass
class Zone:
    low: float
    high: float
    method: str  # vp|vwap|pivot|manual
    zone_id: str

@dataclass
class Barriers:
    up: float
    down: float
    type: str  # pct|atr|abs

@dataclass
class ForecastRequest:
    run_id: str
    subject_type: str
    subject_id: str
    event_type: EventType
    event_id: str
    asof_ts: datetime
    horizon: Horizon
    zone: Optional[Zone] = None
    barriers: Optional[Barriers] = None
    context: Optional[Dict[str, Any]] = None
    features_uri: Optional[str] = None
    raw_data_uri: Optional[str] = None

@dataclass
class ForecastResult:
    expert_name: str
    version: str
    event_id: str
    asof_ts: datetime
    probs: Dict[str, float]
    confidence: float
    uncertainty: Dict[str, Any]
    loss_hint: Literal["logloss", "brier"] = "logloss"
    supports_calibration: bool = True
    calibration_group: Optional[str] = None
    reasons: Optional[List[Dict[str, Any]]] = None
    debug: Optional[Dict[str, Any]] = None
    artifact_uris: Optional[List[str]] = None

    def validate(self, event_type: EventType) -> None:
        total = sum(float(v) for v in self.probs.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"probs must sum to 1.0, got {total}")
        outcomes = STATE_OUTCOMES if event_type == "state_event" else BARRIER_OUTCOMES
        missing = [o for o in outcomes if o not in self.probs]
        if missing:
            raise ValueError(f"missing outcomes: {missing}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0,1]")
