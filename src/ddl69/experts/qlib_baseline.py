from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class QlibBaseline:
    data_dir: str
    region: str = "us"

    def _init(self) -> None:
        try:
            import qlib
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("pyqlib is required. Install requirements-qlib.txt") from exc
        qlib.init(provider_uri=self.data_dir, region=self.region)

    def load_features(
        self,
        market: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        self._init()
        from qlib.data import D

        fields = ["$close", "$open", "$high", "$low", "$volume"]
        df = D.features(
            D.instruments(market),
            fields,
            start_time=start,
            end_time=end,
        )
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise RuntimeError("No features returned from Qlib")
        df.columns = [c.replace("$", "") for c in df.columns]
        return df

    @staticmethod
    def _build_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        # df index: datetime, instrument
        close = df["close"].copy()
        # next-day return as label
        y = close.groupby(level=1).shift(-1) / close - 1.0
        x = df.copy()
        # simple feature engineering
        x["ret_1"] = close.groupby(level=1).pct_change(1, fill_method=None)
        x["ret_5"] = close.groupby(level=1).pct_change(5, fill_method=None)
        x["range"] = (df["high"] - df["low"]) / df["open"].replace(0, np.nan)
        x = x.drop(columns=["close", "open", "high", "low", "volume"], errors="ignore")
        x = x.replace([np.inf, -np.inf], np.nan).dropna()
        y = y.loc[x.index].replace([np.inf, -np.inf], np.nan).dropna()
        x = x.loc[y.index]
        return x, y

    def train_linear(self, df: pd.DataFrame) -> Dict[str, Any]:
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error

        x, y = self._build_dataset(df)
        if x.empty:
            raise RuntimeError("No training data after feature engineering")
        # simple time split
        if isinstance(x.index, pd.MultiIndex):
            lvl0 = x.index.get_level_values(0)
            lvl1 = x.index.get_level_values(1)
        else:
            lvl0 = x.index
            lvl1 = None

        dates = None
        try:
            d0 = pd.to_datetime(lvl0, errors="coerce")
            if d0.notna().mean() > 0.5:
                dates = d0
        except Exception:
            dates = None

        if dates is None and lvl1 is not None:
            d1 = pd.to_datetime(lvl1, errors="coerce")
            if d1.notna().mean() > 0.5:
                dates = d1

        if dates is None:
            raise RuntimeError("Could not locate datetime index in Qlib features")

        dates_i = dates.astype("int64")
        cutoff = np.quantile(dates_i, 0.8)
        train_mask = dates_i <= cutoff
        x_train, y_train = x[train_mask], y[train_mask]
        x_test, y_test = x[~train_mask], y[~train_mask]

        model = LinearRegression()
        model.fit(x_train, y_train)
        preds = model.predict(x_test) if not x_test.empty else np.array([])
        mse = float(mean_squared_error(y_test, preds)) if preds.size else None

        return {
            "model": model,
            "mse": mse,
            "rows_train": int(len(x_train)),
            "rows_test": int(len(x_test)),
        }

    @staticmethod
    def scores_to_probs(scores: pd.Series) -> pd.DataFrame:
        # percentile rank -> p_accept
        ranks = scores.rank(pct=True)
        p_accept = ranks.clip(0.01, 0.99)
        p_break_fail = (1.0 - p_accept) * 0.6
        p_reject = (1.0 - p_accept) - p_break_fail
        return pd.DataFrame(
            {
                "p_accept": p_accept,
                "p_break_fail": p_break_fail,
                "p_reject": p_reject,
            }
        )


__all__ = ["QlibBaseline"]
