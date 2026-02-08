from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class QlibAdapter:
    data_dir: str
    region: str = "us"

    def _init(self) -> None:
        try:
            import qlib
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "qlib is required. Install with requirements-qlib.txt"
            ) from exc
        qlib.init(provider_uri=self.data_dir, region=self.region)

    def probe_market(
        self,
        market: str = "csi300",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._init()
        from qlib.data import D

        instruments = D.instruments(market)
        features = D.features(
            instruments,
            ["$close", "$volume"],
            start_time=start,
            end_time=end,
        )
        return {
            "market": market,
            "start": start,
            "end": end,
            "rows": int(features.shape[0]) if hasattr(features, "shape") else None,
            "columns": list(getattr(features, "columns", [])),
        }


__all__ = ["QlibAdapter"]
