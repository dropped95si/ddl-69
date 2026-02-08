from __future__ import annotations
import numpy as np
import pandas as pd

def estimate_vol(close: pd.Series, span: int = 20) -> pd.Series:
    r = np.log(close).diff()
    return r.ewm(span=span, adjust=False).std()

def triple_barrier_labels(
    bars: pd.DataFrame,
    horizon_bars: int,
    k: float = 1.0,
    vol_span: int = 20,
) -> pd.DataFrame:
    """
    bars: columns [ts, symbol, open, high, low, close] sorted by ts per symbol.
    Returns df with: event_id, symbol, asof_ts, horizon_bars, upper, lower, label, label_source="tb"
    label ∈ {"UP","DOWN","TIMEOUT"}
    """
    out = []
    for sym, g in bars.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        vol = estimate_vol(g["close"], span=vol_span)
        for i in range(len(g) - horizon_bars):
            asof_ts = g.loc[i, "ts"]
            p0 = float(g.loc[i, "close"])
            sigma = float(vol.loc[i]) if pd.notna(vol.loc[i]) else np.nan
            if not np.isfinite(sigma) or sigma <= 0:
                continue
            upper = p0 * (1.0 + k * sigma)
            lower = p0 * (1.0 - k * sigma)

            window = g.loc[i+1:i+horizon_bars]
            hit_up = (window["high"] >= upper)
            hit_dn = (window["low"] <= lower)

            label = "TIMEOUT"
            # first-touch logic
            for j, row in window.iterrows():
                if row["high"] >= upper:
                    label = "UP"
                    break
                if row["low"] <= lower:
                    label = "DOWN"
                    break

            event_id = f"{sym}|tb|h{horizon_bars}|{pd.Timestamp(asof_ts).isoformat()}"
            out.append({
                "event_id": event_id,
                "symbol": sym,
                "asof_ts": pd.Timestamp(asof_ts).to_pydatetime(),
                "horizon_bars": horizon_bars,
                "upper": upper,
                "lower": lower,
                "label": label,
                "label_source": "tb",
            })
    return pd.DataFrame(out)
