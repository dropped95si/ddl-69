from __future__ import annotations

import re
from typing import Dict
import numpy as np

EVENT_RULES: Dict[str, list[str]] = {
    "EARNINGS": [r"\bearnings\b", r"\bEPS\b", r"\brevenue\b", r"\bguidance\b", r"\bQ[1-4]\b"],
    "GUIDANCE": [r"\bguidance\b", r"\bforecast\b", r"\boutlook\b", r"\braises?\s+guidance\b", r"\bcuts?\s+guidance\b"],
    "ANALYST": [r"\bdowngrade\b", r"\bupgrade\b", r"\bprice target\b", r"\binitiated\b", r"\bcoverage\b"],
    "MNA": [r"\bacquire\b", r"\bacquisition\b", r"\bmerger\b", r"\bbuyout\b", r"\btakeover\b"],
    "LAWSUIT": [r"\blawsuit\b", r"\bSEC\b", r"\binvestigation\b", r"\bsettlement\b", r"\bsubpoena\b"],
    "MACRO": [r"\bCPI\b", r"\bfed\b", r"\brates\b", r"\bjobs report\b", r"\bGDP\b"],
    "PRODUCT": [r"\blaunch\b", r"\bpartnership\b", r"\bcontract\b", r"\bAI\b", r"\bdatacenter\b"],
}

POS_HINTS = [r"\bbeats?\b", r"\braises?\b", r"\bstrong\b", r"\bgain\b", r"\brecord\b", r"\bup\b"]
NEG_HINTS = [r"\bmiss(es|ed)?\b", r"\bcuts?\b", r"\bweak\b", r"\bloss\b", r"\bplunge\b", r"\bdown\b"]


def detect_events(title: str, body: str) -> Dict[str, float]:
    text = f"{title} {body}".lower()
    feats: Dict[str, float] = {}

    for event, pats in EVENT_RULES.items():
        feats[f"EV_{event}"] = 1.0 if any(re.search(p, text) for p in pats) else 0.0

    pos = sum(1 for p in POS_HINTS if re.search(p, text))
    neg = sum(1 for p in NEG_HINTS if re.search(p, text))

    feats["HINT_POS"] = float(min(1.0, pos / 3.0))
    feats["HINT_NEG"] = float(min(1.0, neg / 3.0))
    feats["LEAN"] = float(np.clip(feats["HINT_POS"] - feats["HINT_NEG"], -1.0, 1.0))
    return feats
