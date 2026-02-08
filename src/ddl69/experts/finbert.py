from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List


@dataclass
class FinBertExpert:
    model_name: str = "ProsusAI/finbert"
    device: int = -1

    def _pipeline(self):
        try:
            from transformers import pipeline
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "transformers is required for FinBERT. Install with requirements-nlp.txt"
            ) from exc
        return pipeline(
            "text-classification",
            model=self.model_name,
            tokenizer=self.model_name,
            return_all_scores=True,
            device=self.device,
        )

    def predict(self, texts: Iterable[str]) -> List[Dict[str, float]]:
        pipe = self._pipeline()
        outputs = pipe(list(texts))
        rows: List[Dict[str, float]] = []
        for item in outputs:
            scores = {d["label"].lower(): float(d["score"]) for d in item}
            # Normalize for safety
            total = sum(scores.values()) or 1.0
            scores = {k: v / total for k, v in scores.items()}
            rows.append(scores)
        return rows

    @staticmethod
    def to_probs(sent_scores: Dict[str, float]) -> Dict[str, float]:
        # Map sentiment to accept/break/reject proxy
        pos = float(sent_scores.get("positive", 0.0))
        neg = float(sent_scores.get("negative", 0.0))
        neu = float(sent_scores.get("neutral", 0.0))
        p_accept = pos
        p_break_fail = neu * 0.6 + neg * 0.2
        p_reject = max(0.0, 1.0 - p_accept - p_break_fail)
        total = p_accept + p_break_fail + p_reject
        if total <= 0:
            return {"REJECT": 0.34, "BREAK_FAIL": 0.33, "ACCEPT_CONTINUE": 0.33}
        return {
            "REJECT": p_reject / total,
            "BREAK_FAIL": p_break_fail / total,
            "ACCEPT_CONTINUE": p_accept / total,
        }


__all__ = ["FinBertExpert"]
