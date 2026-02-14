from __future__ import annotations

import json
import math
import re
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
    json_cols = [
        "entry",
        "stop",
        "targets",
        "fv",
        "pivots",
        "fib",
        "learned_top_rules",
        "horizon_json",
    ]
    for c in json_cols:
        if c in df.columns:
            df[c] = df[c].apply(parse_json_field)
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    return df


def parse_horizon_days(value: Any) -> Optional[float]:
    def _as_float(v: Any) -> Optional[float]:
        try:
            if v is None:
                return None
            return float(v)
        except Exception:
            return None

    if value is None:
        return None

    if isinstance(value, (int, float)):
        return _as_float(value)

    if isinstance(value, dict):
        direct = _as_float(value.get("days"))
        if direct is None:
            direct = _as_float(value.get("horizon_days"))
        if direct is not None:
            return direct

        unit = str(value.get("unit") or "").strip().lower()
        raw = _as_float(value.get("value"))
        if raw is None:
            return None
        if unit in ("", "d", "day", "days"):
            return raw
        if unit in ("w", "wk", "week", "weeks"):
            return raw * 7.0
        if unit in ("mo", "mon", "month", "months", "m"):
            return raw * 30.0
        if unit in ("y", "yr", "year", "years"):
            return raw * 365.0
        return None

    if isinstance(value, str):
        txt = value.strip().lower()
        if not txt:
            return None
        match = re.match(
            r"^([0-9]+(?:\.[0-9]+)?)\s*(d|day|days|w|wk|week|weeks|mo|mon|month|months|m|y|yr|year|years)?$",
            txt,
        )
        if not match:
            return None
        raw = _as_float(match.group(1))
        unit = (match.group(2) or "d").strip()
        if raw is None:
            return None
        if unit in ("d", "day", "days"):
            return raw
        if unit in ("w", "wk", "week", "weeks"):
            return raw * 7.0
        if unit in ("mo", "mon", "month", "months", "m"):
            return raw * 30.0
        if unit in ("y", "yr", "year", "years"):
            return raw * 365.0
        return None

    return None


def infer_horizon_days_from_rules(rules: Any) -> Optional[float]:
    if not isinstance(rules, list):
        return None
    horizons: list[int] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        for key, value in rule.items():
            if value is None:
                continue
            m = re.fullmatch(r"h(\d+)", str(key).strip().lower())
            if not m:
                continue
            h = int(m.group(1))
            if h > 0:
                horizons.append(h)
    if not horizons:
        return None
    # Use the longest proven rule horizon available for this row.
    return float(max(horizons))


def infer_row_horizon_days(
    row: Dict[str, Any],
    default_horizon_days: int = 5,
) -> int:
    horizon_candidates = [
        row.get("horizon_days"),
        row.get("holding_days"),
        row.get("target_horizon_days"),
        row.get("horizon"),
        row.get("horizon_json"),
    ]
    for candidate in horizon_candidates:
        parsed = parse_horizon_days(candidate)
        if parsed is not None and parsed > 0:
            return max(1, int(round(parsed)))

    rules = row.get("learned_top_rules")
    inferred_from_rules = infer_horizon_days_from_rules(rules)
    if inferred_from_rules is not None and inferred_from_rules > 0:
        return max(1, int(round(inferred_from_rules)))

    bucket = str(
        row.get("timeframe")
        or row.get("scope")
        or row.get("mode")
        or row.get("horizon_bucket")
        or ""
    ).strip().lower()
    if bucket == "day":
        return 10
    if bucket == "swing":
        return 180
    if bucket == "long":
        return 400

    return max(1, int(default_horizon_days))


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


def _clip_prob(p: float, eps: float = 0.01) -> float:
    return max(eps, min(1.0 - eps, p))


def rule_to_probs(rule: Dict[str, Any]) -> Dict[str, float]:
    win_rate, _avg_return, samples = _extract_rule_stats(rule)
    if win_rate is None:
        win_rate = 0.5
    win_rate = max(0.0, min(1.0, win_rate))
    if samples is not None and samples > 0:
        # Beta prior smoothing to avoid hard 0/1 probabilities.
        alpha = 2.0
        beta = 2.0
        wins = win_rate * float(samples)
        win_rate = (wins + alpha) / (float(samples) + alpha + beta)
    p_accept = _clip_prob(win_rate)
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


def aggregate_rule_weights(rows: Iterable[Dict[str, Any]]) -> tuple[Dict[str, float], Dict[str, Any]]:
    stats: Dict[str, Dict[str, float]] = {}

    for row in rows:
        rules = row.get("learned_top_rules") or []
        if not isinstance(rules, list):
            continue
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            name = str(rule.get("rule") or "unknown_rule")
            win_rate, avg_return, samples = _extract_rule_stats(rule)
            score = _safe_float(rule.get("score"))
            sample_weight = float(samples) if samples and samples > 0 else 1.0

            if name not in stats:
                stats[name] = {
                    "samples": 0.0,
                    "win_rate_sum": 0.0,
                    "avg_return_sum": 0.0,
                    "score_sum": 0.0,
                    "count": 0.0,
                }
            s = stats[name]
            s["samples"] += sample_weight
            if win_rate is not None:
                s["win_rate_sum"] += win_rate * sample_weight
            if avg_return is not None:
                s["avg_return_sum"] += avg_return * sample_weight
            if score is not None:
                s["score_sum"] += score * sample_weight
            s["count"] += 1.0

    weights: Dict[str, float] = {}
    for name, s in stats.items():
        denom = s["samples"] if s["samples"] > 0 else s["count"]
        if denom <= 0:
            continue
        avg_win = s["win_rate_sum"] / denom if s["win_rate_sum"] > 0 else 0.0
        avg_ret = s["avg_return_sum"] / denom if s["avg_return_sum"] != 0 else 0.0
        avg_score = s["score_sum"] / denom if s["score_sum"] != 0 else 0.0
        weight_score = max(0.0, avg_win) + max(0.0, avg_ret) + max(0.0, avg_score)
        weights[name] = float(weight_score)

    if not weights:
        return {}, {"rules": {}, "total_rules": 0}

    arr = np.array(list(weights.values()), dtype=float)
    if np.all(arr <= 0):
        arr = np.ones_like(arr)
    arr = arr / arr.sum()
    weights = {k: float(arr[i]) for i, k in enumerate(weights.keys())}

    calib = {
        "total_rules": len(stats),
        "rules": {
            k: {
                "samples": s["samples"],
                "avg_win_rate": (s["win_rate_sum"] / s["samples"]) if s["samples"] > 0 else None,
                "avg_return": (s["avg_return_sum"] / s["samples"]) if s["samples"] > 0 else None,
                "avg_score": (s["score_sum"] / s["samples"]) if s["samples"] > 0 else None,
            }
            for k, s in stats.items()
        },
    }
    return weights, calib
