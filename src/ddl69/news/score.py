from __future__ import annotations

import math
from typing import Any, Dict, Optional


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def score_with_weights(
    features: Dict[str, float],
    weights: Dict[str, Any],
    calibration: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    if "weights" in weights:
        w_map = weights["weights"]
        bias = float(weights.get("bias", 0.0))
    else:
        w_map = weights
        bias = 0.0

    logit = bias
    for k, v in features.items():
        logit += float(w_map.get(k, 0.0)) * float(v)

    if calibration:
        temp = float(calibration.get("temperature", 1.0))
        cal_bias = float(calibration.get("bias", 0.0))
        logit = (logit / max(1e-6, temp)) + cal_bias

    p_up = sigmoid(logit)
    p_down = 1.0 - p_up
    confidence = abs(p_up - 0.5) * 2.0
    return {"p_up": p_up, "p_down": p_down, "confidence": confidence, "logit": logit}
