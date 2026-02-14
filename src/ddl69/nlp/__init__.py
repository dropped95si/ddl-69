"""NLP tools: FinGPT sentiment and forecasting."""

from __future__ import annotations

__all__ = [
    "FinGPTAnalyzer",
    "FinGPTForecaster",
    "analyze_financial_text",
]

try:
    from .fingpt import FinGPTAnalyzer, FinGPTForecaster, analyze_financial_text
except ImportError:
    FinGPTAnalyzer = None
    FinGPTForecaster = None
    analyze_financial_text = None
