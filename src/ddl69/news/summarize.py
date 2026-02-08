from __future__ import annotations

import os
import re
from typing import Literal

SummaryMode = Literal["extractive", "llm"]


def summarize_extractive(text: str, max_sentences: int = 3) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    sents = re.split(r"(?<=[.!?])\s+", text)
    sents = [s.strip() for s in sents if s.strip()]
    return " ".join(sents[:max_sentences])


def summarize(title: str, body: str) -> str:
    mode = os.getenv("DDL69_SUMMARY", "extractive").lower()
    if mode != "llm":
        return summarize_extractive(f"{title}. {body}", max_sentences=3)

    # Hook point: implement your own LLM summary here.
    # Keep this function signature stable so the CLI doesn't change.
    return summarize_extractive(f"{title}. {body}", max_sentences=3)
