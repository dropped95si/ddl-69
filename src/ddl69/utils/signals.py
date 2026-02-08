from __future__ import annotations

import json
import math
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def parse_json_field(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return value
    return value


def sanitize_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): sanitize_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_json(v) for v in value]
    return value


def load_signals_rows(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    json_cols = ["entry", "stop", "targets", "fv", "pivots", "fib", "learned_top_rules"]
    for c in json_cols:
        if c in df.columns:
            df[c] = df[c].apply(parse_json_field)
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    return df


def _extract_rule_stats(rule: Dict[str, Any]) -> tuple[Optional[float], Optional[float], Optional[int]]:
    for horizon_key in ("h60", "h90", "h120"):
        if horizon_key in rule and isinstance(rule[horizon_key], dict):
            h = rule[horizon_key]
            return (
                _safe_float(h.get("win_rate")),
                _safe_float(h.get("avg_return")),
                _safe_int(h.get("samples")),
            )
    return None, None, None


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return float(v)
    except Exception:
        return None


def _safe_int(v: Any) -> Optional[int]:
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return int(v)
    except Exception:
        return None


def rule_to_probs(rule: Dict[str, Any]) -> Dict[str, float]:
    win_rate, _avg_return, _samples = _extract_rule_stats(rule)
    if win_rate is None:
        win_rate = 0.5
    win_rate = max(0.0, min(1.0, win_rate))
    p_accept = win_rate
    p_break_fail = (1.0 - p_accept) * 0.6
    p_reject = max(0.0, 1.0 - p_accept - p_break_fail)
    total = p_accept + p_break_fail + p_reject
    if total <= 0:
        return {"REJECT": 0.34, "BREAK_FAIL": 0.33, "ACCEPT_CONTINUE": 0.33}
    return {
        "REJECT": p_reject / total,
        "BREAK_FAIL": p_break_fail / total,
        "ACCEPT_CONTINUE": p_accept / total,
    }


def weights_from_rules(rules: List[Dict[str, Any]]) -> Dict[str, float]:
    scores = []
    names = []
    for r in rules:
        name = r.get("rule") or r.get("name") or "unknown_rule"
        score = _safe_float(r.get("score"))
        if score is None:
            win_rate, avg_return, samples = _extract_rule_stats(r)
            if win_rate is None:
                score = 0.0
            else:
                score = win_rate
            if avg_return is not None:
                score += avg_return
            if samples:
                score *= max(1.0, math.log1p(samples))
        scores.append(score)
        names.append(str(name))

    if not scores:
        return {}

    arr = np.array(scores, dtype=float)
    if np.all(arr <= 0):
        weights = np.ones_like(arr) / len(arr)
    else:
        arr = np.maximum(arr, 0.0)
        s = arr.sum()
        weights = arr / s if s > 0 else np.ones_like(arr) / len(arr)
    return {names[i]: float(weights[i]) for i in range(len(names))}


def blend_probs(weighted: Iterable[tuple[Dict[str, float], float]]) -> Dict[str, float]:
    acc = {"REJECT": 0.0, "BREAK_FAIL": 0.0, "ACCEPT_CONTINUE": 0.0}
    total_w = 0.0
    for probs, w in weighted:
        if w is None:
            continue
        total_w += w
        for k in acc:
            acc[k] += float(probs.get(k, 0.0)) * w
    if total_w <= 0:
        return {"REJECT": 0.34, "BREAK_FAIL": 0.33, "ACCEPT_CONTINUE": 0.33}
    return {k: v / total_w for k, v in acc.items()}


def entropy(probs: Dict[str, float]) -> float:
    e = 0.0
    for p in probs.values():
        if p > 0:
            e -= p * math.log(p)
    return float(e)
