"""Walk-forward endpoint.

Primary source:
- Supabase walkforward artifact

Fallback source:
- Derived summary from latest Supabase ensemble forecasts + events
  (still real Supabase data; no synthetic/sample rows)
"""

import json
import os
from datetime import datetime, timezone
import re
import math
import random

try:
    from _http_adapter import FunctionHandler
except ModuleNotFoundError:
    from api._http_adapter import FunctionHandler


def _safe_float(value, default=None):
    try:
        if value is None:
            return default
        out = float(value)
        if out != out:  # NaN
            return default
        return out
    except Exception:
        return default


def _round_or_none(value, digits=6):
    out = _safe_float(value, None)
    if out is None:
        return None
    return round(out, digits)


def _quantile(sorted_values, q):
    values = list(sorted_values or [])
    n = len(values)
    if n == 0:
        return None
    if n == 1:
        return float(values[0])
    q = max(0.0, min(1.0, float(q)))
    idx = (n - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(values[lo])
    frac = idx - lo
    return float(values[lo] * (1.0 - frac) + values[hi] * frac)


def _trimmed_mean(values, trim_ratio=0.1):
    vals = []
    for v in values:
        x = _safe_float(v, None)
        if x is not None:
            vals.append(x)
    vals.sort()
    n = len(vals)
    if n == 0:
        return None
    k = int(n * max(0.0, min(0.45, float(trim_ratio))))
    kept = vals[k : (n - k)] if (n - 2 * k) > 0 else vals
    if not kept:
        return None
    return float(sum(kept) / len(kept))


def _mean_and_std(values):
    vals = [_safe_float(v, None) for v in values]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    mean_v = float(sum(vals) / len(vals))
    if len(vals) <= 1:
        return mean_v, 0.0
    var = sum((v - mean_v) ** 2 for v in vals) / len(vals)
    return mean_v, float(math.sqrt(max(0.0, var)))


def _bootstrap_mean_ci(values, sample_count=400, alpha=0.05, seed=69):
    vals = [_safe_float(v, None) for v in values]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        only = float(vals[0])
        return only, only
    rng = random.Random(seed)
    means = []
    n = len(vals)
    for _ in range(max(100, int(sample_count))):
        sample = [vals[rng.randrange(0, n)] for __ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = _quantile(means, alpha / 2.0)
    hi = _quantile(means, 1.0 - alpha / 2.0)
    return lo, hi


def _entropy_of_probs(pa, pr, pc):
    probs = [max(0.0, float(pa or 0.0)), max(0.0, float(pr or 0.0)), max(0.0, float(pc or 0.0))]
    total = sum(probs)
    if total <= 0:
        return None
    probs = [p / total for p in probs]
    return -sum(p * math.log(max(1e-12, p)) for p in probs if p > 0)


def _parse_iso_ts(value):
    txt = str(value or "").strip()
    if not txt:
        return None
    try:
        if txt.endswith("Z"):
            txt = txt[:-1] + "+00:00"
        return datetime.fromisoformat(txt)
    except Exception:
        return None


def _cap_bucket_from_market_cap(market_cap):
    mc = _safe_float(market_cap, None)
    if mc is None or mc <= 0:
        return "unknown"
    if mc < 2_000_000_000:
        return "small"
    if mc < 10_000_000_000:
        return "mid"
    return "large"


def _extract_cap_bucket(event_row):
    if not isinstance(event_row, dict):
        return "unknown"
    context = event_row.get("context_json") if isinstance(event_row.get("context_json"), dict) else {}
    params = event_row.get("event_params_json") if isinstance(event_row.get("event_params_json"), dict) else {}
    for source in (context, params):
        bucket = str(source.get("cap_bucket") or "").strip().lower()
        if bucket in ("small", "mid", "large"):
            return bucket
    for source in (context, params):
        bucket = _cap_bucket_from_market_cap(source.get("market_cap"))
        if bucket != "unknown":
            return bucket
    return "unknown"


def _build_rolling_windows(rows):
    if not rows:
        return []
    with_ts = []
    for row in rows:
        ts = _parse_iso_ts(row.get("created_at")) or _parse_iso_ts(row.get("_wf_asof_ts"))
        if ts is None:
            continue
        with_ts.append((ts, row))
    if not with_ts:
        return []
    with_ts.sort(key=lambda t: t[0])  # oldest -> newest
    n = len(with_ts)
    if n >= 24:
        window_count = 4
    elif n >= 12:
        window_count = 3
    elif n >= 6:
        window_count = 2
    else:
        window_count = 1
    chunk = max(1, n // window_count)
    windows = []
    for i in range(window_count):
        start = i * chunk
        end = n if i == (window_count - 1) else min(n, (i + 1) * chunk)
        seg = with_ts[start:end]
        if not seg:
            continue
        rows_seg = [r for _, r in seg]
        conf_vals = []
        pa_vals = []
        ent_vals = []
        net_vals = []
        for row in rows_seg:
            conf = _safe_float(row.get("confidence"), None)
            if conf is not None:
                conf_vals.append(conf)
            probs = row.get("probs_json") if isinstance(row.get("probs_json"), dict) else {}
            pa = _safe_float(probs.get("ACCEPT_CONTINUE") or probs.get("accept_continue"), None)
            pr = _safe_float(probs.get("REJECT") or probs.get("reject"), None)
            pc = _safe_float(probs.get("BREAK_FAIL") or probs.get("break_fail"), None)
            if pa is not None:
                pa_vals.append(pa)
            ent = _entropy_of_probs(pa, pr, pc)
            if ent is not None:
                ent_vals.append(ent)
            weights = row.get("weights_json") if isinstance(row.get("weights_json"), dict) else {}
            net = sum([_safe_float(v, 0.0) or 0.0 for v in weights.values()]) if weights else None
            if net is not None:
                net_vals.append(net)
        avg_conf, _ = _mean_and_std(conf_vals)
        avg_pa, _ = _mean_and_std(pa_vals)
        avg_ent, _ = _mean_and_std(ent_vals)
        avg_net, _ = _mean_and_std(net_vals)
        windows.append(
            {
                "window": i + 1,
                "rows": len(rows_seg),
                "start": seg[0][0].isoformat(),
                "end": seg[-1][0].isoformat(),
                "avg_confidence": _round_or_none(avg_conf, 6),
                "avg_p_accept": _round_or_none(avg_pa, 6),
                "avg_entropy": _round_or_none(avg_ent, 6),
                "avg_net_weight": _round_or_none(avg_net, 6),
            }
        )
    return windows


def _parse_horizon_days(raw_horizon):
    try:
        if isinstance(raw_horizon, dict):
            days = raw_horizon.get("days") or raw_horizon.get("horizon_days")
            if days is None:
                unit = str(raw_horizon.get("unit") or "").lower().strip()
                value = raw_horizon.get("value")
                if value is not None:
                    value = float(value)
                    if unit in ("", "d", "day", "days"):
                        days = value
                    elif unit in ("w", "wk", "week", "weeks"):
                        days = value * 7.0
                    elif unit in ("mo", "mon", "month", "months", "m"):
                        days = value * 30.0
                    elif unit in ("y", "yr", "year", "years"):
                        days = value * 365.0
            if days is not None:
                return max(1, int(float(days)))
        elif isinstance(raw_horizon, (int, float)):
            return max(1, int(float(raw_horizon)))
        elif isinstance(raw_horizon, str):
            txt = raw_horizon.strip().lower()
            match = re.match(
                r"^([0-9]+(?:\.[0-9]+)?)\s*(d|day|days|w|wk|week|weeks|mo|mon|month|months|m|y|yr|year|years)?$",
                txt,
            )
            if match:
                value = float(match.group(1))
                unit = (match.group(2) or "d").strip()
                if unit in ("d", "day", "days"):
                    days = value
                elif unit in ("w", "wk", "week", "weeks"):
                    days = value * 7.0
                elif unit in ("mo", "mon", "month", "months", "m"):
                    days = value * 30.0
                elif unit in ("y", "yr", "year", "years"):
                    days = value * 365.0
                else:
                    days = value
                return max(1, int(round(days)))
    except Exception:
        return None
    return None


def _classify_timeframe(horizon_days):
    if horizon_days is None:
        return "swing"
    if horizon_days <= 30:
        return "day"
    if horizon_days <= 365:
        return "swing"
    return "long"


def _default_horizon_for_timeframe(timeframe):
    if timeframe == "day":
        return 10
    if timeframe == "long":
        return 400
    return 180


def _get_supabase_client():
    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not supabase_url or not service_key:
        return None
    try:
        from supabase import create_client

        return create_client(supabase_url, service_key)
    except Exception:
        return None


def _fetch_walkforward_artifact():
    supa = _get_supabase_client()
    if supa is None:
        return None

    try:
        resp = (
            supa.table("artifacts")
            .select("meta_json,created_at")
            .eq("kind", "other")
            .order("created_at", desc=True)
            .limit(30)
            .execute()
        )
        for row in resp.data or []:
            meta = row.get("meta_json") or {}
            if isinstance(meta, dict) and meta.get("type") == "walkforward_summary":
                meta.setdefault("artifact_created_at", row.get("created_at"))
                return meta
        return None
    except Exception:
        return None


def _derive_from_supabase_forecasts(timeframe_filter="all", run_id_filter=""):
    supa = _get_supabase_client()
    if supa is None:
        return None

    try:
        pred_resp = (
            supa.table("v_latest_ensemble_forecasts")
            .select("event_id,weights_json,created_at,run_id,probs_json,confidence,method")
            .order("created_at", desc=True)
            .limit(600)
            .execute()
        )
        pred_rows = pred_resp.data or []
        if not pred_rows:
            return None

        event_ids = [r.get("event_id") for r in pred_rows if r.get("event_id")]
        events_map = {}
        if event_ids:
            ev_resp = (
                supa.table("events")
                .select("event_id,horizon_json,asof_ts,subject_id,context_json,event_params_json")
                .in_("event_id", event_ids)
                .execute()
            )
            for ev in ev_resp.data or []:
                eid = ev.get("event_id")
                if eid:
                    events_map[eid] = ev

        run_ids = [r.get("run_id") for r in pred_rows if r.get("run_id")]
        latest_run_id = run_ids[0] if run_ids else None
        requested_run_id = str(run_id_filter or "").strip()
        active_run_id = requested_run_id or latest_run_id
        if active_run_id:
            rows = [r for r in pred_rows if r.get("run_id") == active_run_id]
        else:
            rows = list(pred_rows)

        if not rows and requested_run_id:
            fallback_asof = pred_rows[0].get("created_at") or datetime.now(timezone.utc).isoformat()
            return {
                "summary": {
                    "run_id": requested_run_id,
                    "asof": fallback_asof,
                    "horizon": _default_horizon_for_timeframe(timeframe_filter),
                    "top_rules": 0,
                    "signals_rows": 0,
                    "weights": {},
                    "weights_top": [],
                    "stats": {
                        "total_rules": 0,
                        "pos_count": 0,
                        "neg_count": 0,
                        "net_weight": 0.0,
                        "avg_win_rate": None,
                        "avg_return": None,
                    },
                    "source": "supabase_forecasts_derived",
                    "timeframe": timeframe_filter,
                    "timeframe_counts": {"day": 0, "swing": 0, "long": 0},
                    "available_runs": list(dict.fromkeys(run_ids))[:20],
                    "note": f"No rows found for run_id '{requested_run_id}'.",
                }
            }

        if not rows and not requested_run_id:
            rows = list(pred_rows)

        tf_counts = {"day": 0, "swing": 0, "long": 0}
        scoped_rows = []
        for row in rows:
            ev = events_map.get(row.get("event_id"), {})
            horizon_days = _parse_horizon_days(ev.get("horizon_json"))
            tf = _classify_timeframe(horizon_days)
            tf_counts[tf] = tf_counts.get(tf, 0) + 1
            if timeframe_filter != "all" and tf != timeframe_filter:
                continue
            row_copy = dict(row)
            row_copy["_wf_horizon_days"] = horizon_days
            row_copy["_wf_asof_ts"] = ev.get("asof_ts")
            row_copy["_wf_subject_id"] = ev.get("subject_id")
            row_copy["_wf_cap_bucket"] = _extract_cap_bucket(ev)
            scoped_rows.append(row_copy)

        rows = scoped_rows
        if not rows:
            if timeframe_filter != "all":
                fallback_asof = pred_rows[0].get("created_at") or datetime.now(timezone.utc).isoformat()
                return {
                    "summary": {
                        "run_id": active_run_id or "derived_supabase",
                        "asof": fallback_asof,
                        "horizon": _default_horizon_for_timeframe(timeframe_filter),
                        "top_rules": 0,
                        "signals_rows": 0,
                        "weights": {},
                        "weights_top": [],
                        "stats": {
                            "total_rules": 0,
                            "pos_count": 0,
                            "neg_count": 0,
                            "net_weight": 0.0,
                            "avg_win_rate": None,
                            "avg_return": None,
                        },
                        "source": "supabase_forecasts_derived",
                        "timeframe": timeframe_filter,
                        "timeframe_counts": tf_counts,
                        "note": f"No rows available for timeframe '{timeframe_filter}' in latest Supabase run.",
                    }
                }
            return None

        rule_values = {}
        horizon_days_values = []
        asof_candidates = []
        created_candidates = []
        method_counts = {}
        row_net_weights = []
        confidence_values = []
        p_accept_values = []
        p_reject_values = []
        p_continue_values = []
        entropy_values = []
        cap_bucket_rows = {}
        cap_bucket_conf = {}
        symbol_set = set()

        for row in rows:
            created_at = row.get("created_at")
            if created_at:
                created_candidates.append(created_at)

            asof_ts = row.get("_wf_asof_ts")
            if asof_ts:
                asof_candidates.append(asof_ts)
            days = row.get("_wf_horizon_days")
            if days is not None:
                horizon_days_values.append(days)

            method = str(row.get("method") or "").strip().lower() or "unknown"
            method_counts[method] = method_counts.get(method, 0) + 1

            symbol = str(row.get("_wf_subject_id") or "").strip().upper()
            if symbol:
                symbol_set.add(symbol)
            cap_bucket = str(row.get("_wf_cap_bucket") or "unknown").strip().lower() or "unknown"
            if cap_bucket not in cap_bucket_rows:
                cap_bucket_rows[cap_bucket] = []
            if cap_bucket not in cap_bucket_conf:
                cap_bucket_conf[cap_bucket] = []

            confidence = _safe_float(row.get("confidence"), None)
            if confidence is not None:
                confidence_values.append(confidence)
                cap_bucket_conf[cap_bucket].append(confidence)

            probs = row.get("probs_json") if isinstance(row.get("probs_json"), dict) else {}
            pa = _safe_float(probs.get("ACCEPT_CONTINUE") or probs.get("accept_continue"), None)
            pr = _safe_float(probs.get("REJECT") or probs.get("reject"), None)
            pc = _safe_float(probs.get("BREAK_FAIL") or probs.get("break_fail"), None)
            if pa is not None:
                p_accept_values.append(pa)
            if pr is not None:
                p_reject_values.append(pr)
            if pc is not None:
                p_continue_values.append(pc)
            ent = _entropy_of_probs(pa, pr, pc)
            if ent is not None:
                entropy_values.append(ent)

            weights = row.get("weights_json") or {}
            if not isinstance(weights, dict):
                continue
            row_net = 0.0
            has_row_weight = False
            for rule, value in weights.items():
                w = _safe_float(value, None)
                if w is None:
                    continue
                has_row_weight = True
                row_net += w
                if rule not in rule_values:
                    rule_values[rule] = []
                rule_values[rule].append(w)
            if has_row_weight:
                row_net_weights.append(row_net)
                cap_bucket_rows[cap_bucket].append(row_net)

        if not rule_values:
            return None

        # Robust rule aggregation: trimmed mean + bootstrap CI + shrinkage to reduce single-row noise.
        mean_weights = {}
        rule_diagnostics = []
        for rule, values in rule_values.items():
            clean_values = [v for v in values if _safe_float(v, None) is not None]
            if not clean_values:
                continue
            mean_v, std_v = _mean_and_std(clean_values)
            median_v = _quantile(sorted(clean_values), 0.5)
            trimmed_v = _trimmed_mean(clean_values, trim_ratio=0.1)
            ci_low, ci_high = _bootstrap_mean_ci(clean_values, sample_count=300, alpha=0.05, seed=(69 + len(rule)))
            n = len(clean_values)
            pos_count = len([v for v in clean_values if v > 0])
            neg_count = len([v for v in clean_values if v < 0])
            sign_agreement = max(pos_count, neg_count) / n if n else 0.0
            # Simple empirical-Bayes style shrinkage to 0.0 (conservative) for thin samples.
            shrunk_weight = ((mean_v or 0.0) * n) / (n + 5.0)
            stability = 1.0 - ((std_v or 0.0) / (abs(mean_v or 0.0) + 1e-6))
            stability = max(0.0, min(1.0, stability))
            mean_weights[rule] = shrunk_weight
            rule_diagnostics.append(
                {
                    "rule": rule,
                    "weight": round(shrunk_weight, 6),
                    "mean": _round_or_none(mean_v, 6),
                    "median": _round_or_none(median_v, 6),
                    "trimmed_mean": _round_or_none(trimmed_v, 6),
                    "std": _round_or_none(std_v, 6),
                    "ci_low": _round_or_none(ci_low, 6),
                    "ci_high": _round_or_none(ci_high, 6),
                    "count": n,
                    "sign_agreement": _round_or_none(sign_agreement, 4),
                    "stability": _round_or_none(stability, 4),
                }
            )

        if not mean_weights:
            return None

        top = sorted(
            rule_diagnostics,
            key=lambda x: abs(x["weight"]),
            reverse=True,
        )

        horizon_days = (
            int(round(sum(horizon_days_values) / len(horizon_days_values)))
            if horizon_days_values
            else (_default_horizon_for_timeframe(timeframe_filter) if timeframe_filter != "all" else None)
        )
        asof = (
            asof_candidates[0]
            if asof_candidates
            else (
                created_candidates[0]
                if created_candidates
                else datetime.now(timezone.utc).isoformat()
            )
        )

        net_weight_total = float(sum(mean_weights.values()))
        net_mean, net_std = _mean_and_std(row_net_weights)
        net_ci_low, net_ci_high = _bootstrap_mean_ci(row_net_weights, sample_count=300, alpha=0.05, seed=420)
        avg_confidence, _ = _mean_and_std(confidence_values)
        avg_entropy, _ = _mean_and_std(entropy_values)
        avg_p_accept, _ = _mean_and_std(p_accept_values)
        avg_p_reject, _ = _mean_and_std(p_reject_values)
        avg_p_continue, _ = _mean_and_std(p_continue_values)

        abs_weights = [abs(v) for v in mean_weights.values() if _safe_float(v, None) is not None]
        abs_total = sum(abs_weights)
        if abs_total > 0:
            shares = [w / abs_total for w in abs_weights]
            hhi = float(sum(s * s for s in shares))
            effective_rules = float(1.0 / hhi) if hhi > 0 else None
            top_share = float(max(shares))
        else:
            hhi = None
            effective_rules = None
            top_share = None

        temporal_drift = None
        if len(row_net_weights) >= 6:
            window = max(2, len(row_net_weights) // 3)
            newest = row_net_weights[:window]
            oldest = row_net_weights[-window:]
            newest_mean = sum(newest) / len(newest)
            oldest_mean = sum(oldest) / len(oldest)
            temporal_drift = {
                "newest_mean": _round_or_none(newest_mean, 6),
                "oldest_mean": _round_or_none(oldest_mean, 6),
                "delta": _round_or_none(newest_mean - oldest_mean, 6),
            }

        rolling_windows = _build_rolling_windows(rows)
        oos_delta = None
        if len(rolling_windows) >= 2:
            latest_w = rolling_windows[-1]
            earliest_w = rolling_windows[0]
            latest_net = _safe_float(latest_w.get("avg_net_weight"), None)
            earliest_net = _safe_float(earliest_w.get("avg_net_weight"), None)
            latest_conf = _safe_float(latest_w.get("avg_confidence"), None)
            earliest_conf = _safe_float(earliest_w.get("avg_confidence"), None)
            oos_delta = {
                "net_weight_delta": _round_or_none(
                    (latest_net - earliest_net) if (latest_net is not None and earliest_net is not None) else None, 6
                ),
                "confidence_delta": _round_or_none(
                    (latest_conf - earliest_conf) if (latest_conf is not None and earliest_conf is not None) else None, 6
                ),
                "latest_window": latest_w.get("window"),
                "earliest_window": earliest_w.get("window"),
            }

        directional_edge = None
        if p_accept_values and p_reject_values:
            pairs = zip(p_accept_values, p_reject_values)
            deltas = [abs(a - b) for a, b in pairs]
            if deltas:
                directional_edge = sum(deltas) / len(deltas)
        baseline_entropy = math.log(3.0)
        entropy_edge = None if avg_entropy is None else (baseline_entropy - avg_entropy)
        confidence_edge = None if avg_confidence is None else (avg_confidence - (1.0 / 3.0))
        benchmark_block = {
            "neutral_entropy_baseline": _round_or_none(baseline_entropy, 6),
            "entropy_edge": _round_or_none(entropy_edge, 6),
            "confidence_edge_vs_neutral": _round_or_none(confidence_edge, 6),
            "directional_edge_abs_pdiff": _round_or_none(directional_edge, 6),
        }

        cap_bucket_stability = []
        for bucket, values in cap_bucket_rows.items():
            mean_net, std_net = _mean_and_std(values)
            avg_conf_bucket, _ = _mean_and_std(cap_bucket_conf.get(bucket, []))
            cap_bucket_stability.append(
                {
                    "bucket": bucket,
                    "rows": len(values),
                    "avg_net_weight": _round_or_none(mean_net, 6),
                    "std_net_weight": _round_or_none(std_net, 6),
                    "avg_confidence": _round_or_none(avg_conf_bucket, 6),
                }
            )
        cap_bucket_stability.sort(key=lambda x: int(x.get("rows") or 0), reverse=True)

        return {
            "summary": {
                "run_id": active_run_id or "derived_supabase",
                "asof": asof,
                "horizon": horizon_days,
                "top_rules": min(8, len(top)),
                "signals_rows": len(rows),
                "weights": {k: round(v, 6) for k, v in mean_weights.items()},
                "weights_top": top[:8],
                "stats": {
                    "total_rules": len(mean_weights),
                    "pos_count": len([w for w in mean_weights.values() if w > 0]),
                    "neg_count": len([w for w in mean_weights.values() if w < 0]),
                    "net_weight": round(net_weight_total, 6),
                    "net_weight_ci_low": _round_or_none(net_ci_low, 6),
                    "net_weight_ci_high": _round_or_none(net_ci_high, 6),
                    "net_weight_row_mean": _round_or_none(net_mean, 6),
                    "net_weight_row_std": _round_or_none(net_std, 6),
                    "avg_confidence": _round_or_none(avg_confidence, 6),
                    "avg_entropy": _round_or_none(avg_entropy, 6),
                    "avg_p_accept": _round_or_none(avg_p_accept, 6),
                    "avg_p_reject": _round_or_none(avg_p_reject, 6),
                    "avg_p_continue": _round_or_none(avg_p_continue, 6),
                    "avg_win_rate": None,
                    "avg_return": None,
                },
                "diagnostics": {
                    "open_source_methods": [
                        "bootstrap_mean_ci",
                        "trimmed_mean",
                        "shannon_entropy",
                        "hhi_concentration",
                        "empirical_bayes_shrinkage",
                    ],
                    "coverage": {
                        "rows": len(rows),
                        "unique_symbols": len(symbol_set),
                        "unique_events": len(set([r.get("event_id") for r in rows if r.get("event_id")])),
                        "unique_methods": len(method_counts),
                        "method_counts": method_counts,
                    },
                    "probability": {
                        "avg_p_accept": _round_or_none(avg_p_accept, 6),
                        "avg_p_reject": _round_or_none(avg_p_reject, 6),
                        "avg_p_continue": _round_or_none(avg_p_continue, 6),
                        "avg_entropy": _round_or_none(avg_entropy, 6),
                        "avg_confidence": _round_or_none(avg_confidence, 6),
                    },
                    "concentration": {
                        "hhi": _round_or_none(hhi, 6),
                        "effective_rules": _round_or_none(effective_rules, 3),
                        "top_rule_share": _round_or_none(top_share, 6),
                    },
                    "temporal_drift": temporal_drift,
                    "rolling_windows": rolling_windows,
                    "oos_delta": oos_delta,
                    "benchmarks": benchmark_block,
                    "cap_bucket_stability": cap_bucket_stability[:6],
                    "rule_stability_top": top[:12],
                },
                "source": "supabase_forecasts_derived",
                "timeframe": timeframe_filter,
                "timeframe_counts": tf_counts,
                "available_runs": list(dict.fromkeys(run_ids))[:20],
                "note": (
                    "Walk-forward artifact unavailable; derived from latest Supabase ensemble weights with bootstrap diagnostics."
                    if timeframe_filter == "all"
                    else f"Walk-forward artifact unavailable; derived from Supabase ensemble weights ({timeframe_filter}) with bootstrap diagnostics."
                ),
            }
        }
    except Exception:
        return None


def _handler_impl(request):
    args = request.args if hasattr(request, "args") else {}
    timeframe = str((args.get("timeframe") if args else "") or "all").strip().lower()
    if timeframe not in ("all", "day", "swing", "long"):
        timeframe = "all"
    run_id = str((args.get("run_id") if args else "") or "").strip()

    allow_derived_arg = str((args.get("allow_derived") if args else "") or "").strip().lower()
    allow_derived = allow_derived_arg in ("1", "true", "yes", "on")
    if not allow_derived:
        allow_derived = str(os.getenv("WALKFORWARD_ALLOW_DERIVED", "1")).strip().lower() in ("1", "true", "yes", "on")

    payload = None
    # Run-specific requests cannot be satisfied by global artifact payloads.
    if run_id and allow_derived:
        payload = _derive_from_supabase_forecasts(timeframe_filter=timeframe, run_id_filter=run_id)
    if payload is None:
        payload = _fetch_walkforward_artifact()
    if payload is None and allow_derived:
        payload = _derive_from_supabase_forecasts(timeframe_filter=timeframe, run_id_filter=run_id)
    if payload is None:
        return {
            "statusCode": 503,
            "headers": {
                "Content-Type": "application/json",
                "Cache-Control": "no-store",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps(
                {
                    "error": "supabase_unavailable",
                    "message": (
                        (
                            f"Supabase walk-forward artifact required; run_id '{run_id}' needs allow_derived=1."
                            if run_id and not allow_derived
                            else "Supabase walk-forward artifact required; no fallback enabled."
                        )
                        if not allow_derived
                        else (
                            "Supabase walk-forward artifact and derived forecast aggregates are unavailable."
                            if timeframe == "all"
                            else (
                                f"Supabase walk-forward artifact and derived forecast aggregates are unavailable for timeframe '{timeframe}'."
                                if not run_id
                                else f"Supabase walk-forward artifact and derived forecast aggregates are unavailable for timeframe '{timeframe}' in run '{run_id}'."
                            )
                        )
                    ),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
            ),
        }

    summary = payload.get("summary") if isinstance(payload, dict) else None
    if isinstance(summary, dict):
        summary.setdefault("requested_run_id", run_id or None)
        if allow_derived:
            summary.setdefault("timeframe", timeframe)
            if run_id:
                summary.setdefault("run_id", run_id)
        else:
            # Artifact is run-level and not guaranteed to be timeframe-scoped.
            summary.setdefault("timeframe", "all")
            if timeframe != "all":
                note = str(summary.get("note") or "").strip()
                scope_note = f"Requested timeframe '{timeframe}' uses run-level artifact (global scope)."
                summary["note"] = f"{note} {scope_note}".strip() if note else scope_note
            if run_id:
                note = str(summary.get("note") or "").strip()
                run_note = f"Requested run_id '{run_id}' is ignored for artifact payload (global scope)."
                summary["note"] = f"{note} {run_note}".strip() if note else run_note

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Cache-Control": "max-age=120, public",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(payload),
    }


class handler(FunctionHandler):
    endpoint = staticmethod(_handler_impl)
