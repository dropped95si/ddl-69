"""
FinGPT integration for financial NLP tasks.

Provides:
- Sentiment analysis
- Named entity recognition (NER)
- Financial event extraction
- News summarization
- Market forecasting from text
"""
from __future__ import annotations

import warnings
from typing import Optional, Any, Literal

import pandas as pd
import numpy as np

# Try transformers
try:
    from transformers import (
        pipeline,
        AutoTokenizer,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("transformers not available. Install via: pip install transformers torch")


class FinGPTAnalyzer:
    """
    FinGPT-based financial text analyzer.

    Supports multiple FinGPT models for various NLP tasks.
    """

    DEFAULT_MODELS = {
        "sentiment": "ProsusAI/finbert",  # FinBERT for sentiment
        "sentiment_fingpt": "FinGPT/fingpt-sentiment-cls",  # FinGPT sentiment
        "ner": "nlpaueb/sec-bert-num",  # Financial NER
        "forecasting": "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora",  # FinGPT forecaster
    }

    def __init__(
        self,
        task: Literal["sentiment", "ner", "forecasting"] = "sentiment",
        model_name: Optional[str] = None,
        device: int = -1,
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers not installed")

        self.task = task
        self.device = device
        self.model_name = model_name or self.DEFAULT_MODELS.get(task)
        self.pipeline = None
        self.tokenizer = None
        self.model = None

        if self.model_name:
            self._load_model()

    def _load_model(self) -> None:
        """Load model and tokenizer."""
        try:
            if self.task in ["sentiment", "forecasting"]:
                self.pipeline = pipeline(
                    "text-classification",
                    model=self.model_name,
                    tokenizer=self.model_name,
                    device=self.device,
                    truncation=True,
                    max_length=512,
                )
            elif self.task == "ner":
                self.pipeline = pipeline(
                    "ner",
                    model=self.model_name,
                    tokenizer=self.model_name,
                    device=self.device,
                    aggregation_strategy="simple",
                )
            print(f"Loaded {self.model_name}")
        except Exception as e:
            warnings.warn(f"Failed to load model {self.model_name}: {e}")
            self.pipeline = None

    def sentiment(
        self,
        texts: list[str] | str,
        batch_size: int = 8,
    ) -> list[dict[str, Any]]:
        """
        Analyze sentiment of financial text.

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for inference

        Returns:
            List of sentiment results
        """
        if isinstance(texts, str):
            texts = [texts]

        if self.pipeline is None:
            raise RuntimeError("Model not loaded")

        if self.task != "sentiment":
            warnings.warn(f"Task is {self.task}, not sentiment. Results may be unexpected.")

        results = self.pipeline(texts, batch_size=batch_size)

        # Normalize results
        normalized = []
        for result in results:
            label = result["label"].lower()
            score = result["score"]

            # Map to standard sentiment
            if "pos" in label or label == "positive":
                sentiment_score = score
                sentiment_label = "positive"
            elif "neg" in label or label == "negative":
                sentiment_score = -score
                sentiment_label = "negative"
            else:
                sentiment_score = 0.0
                sentiment_label = "neutral"

            normalized.append({
                "label": sentiment_label,
                "score": sentiment_score,
                "confidence": score,
                "raw_label": result["label"],
            })

        return normalized

    def sentiment_aggregate(
        self,
        texts: list[str],
        method: Literal["mean", "weighted", "majority"] = "mean",
    ) -> dict[str, Any]:
        """
        Aggregate sentiment across multiple texts.

        Args:
            texts: List of texts
            method: Aggregation method

        Returns:
            Aggregated sentiment
        """
        results = self.sentiment(texts)

        if method == "mean":
            avg_score = np.mean([r["score"] for r in results])
            if avg_score > 0.1:
                agg_label = "positive"
            elif avg_score < -0.1:
                agg_label = "negative"
            else:
                agg_label = "neutral"

            return {
                "label": agg_label,
                "score": float(avg_score),
                "count": len(results),
                "distribution": {
                    "positive": sum(1 for r in results if r["label"] == "positive"),
                    "negative": sum(1 for r in results if r["label"] == "negative"),
                    "neutral": sum(1 for r in results if r["label"] == "neutral"),
                },
            }

        elif method == "weighted":
            # Weight by confidence
            weighted_sum = sum(r["score"] * r["confidence"] for r in results)
            total_weight = sum(r["confidence"] for r in results)
            avg_score = weighted_sum / total_weight if total_weight > 0 else 0.0

            return {
                "label": "positive" if avg_score > 0.1 else ("negative" if avg_score < -0.1 else "neutral"),
                "score": float(avg_score),
                "count": len(results),
            }

        elif method == "majority":
            # Majority vote
            labels = [r["label"] for r in results]
            from collections import Counter
            counts = Counter(labels)
            majority_label = counts.most_common(1)[0][0]

            return {
                "label": majority_label,
                "count": len(results),
                "votes": dict(counts),
            }

        else:
            raise ValueError(f"Unknown method: {method}")

    def ner(
        self,
        texts: list[str] | str,
        batch_size: int = 8,
    ) -> list[list[dict[str, Any]]]:
        """
        Extract named entities from financial text.

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for inference

        Returns:
            List of entity lists
        """
        if isinstance(texts, str):
            texts = [texts]

        if self.pipeline is None:
            raise RuntimeError("Model not loaded")

        if self.task != "ner":
            warnings.warn(f"Task is {self.task}, not ner. Results may be unexpected.")

        results = self.pipeline(texts, batch_size=batch_size)

        return results

    def score_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        output_column: str = "sentiment",
        batch_size: int = 16,
    ) -> pd.DataFrame:
        """
        Score sentiment for a DataFrame column.

        Args:
            df: Input DataFrame
            text_column: Column containing text
            output_column: Column name for sentiment scores
            batch_size: Batch size for inference

        Returns:
            DataFrame with sentiment column added
        """
        texts = df[text_column].fillna("").astype(str).tolist()
        results = self.sentiment(texts, batch_size=batch_size)

        df = df.copy()
        df[output_column] = [r["score"] for r in results]
        df[f"{output_column}_label"] = [r["label"] for r in results]
        df[f"{output_column}_confidence"] = [r["confidence"] for r in results]

        return df


class FinGPTForecaster:
    """
    FinGPT forecaster for market predictions from news/text.

    Uses instruction-tuned FinGPT models for forecasting.
    """

    def __init__(
        self,
        model_name: str = "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora",
        device: int = -1,
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers not installed")

        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None

        # Note: FinGPT forecaster requires special handling (instruction format)
        warnings.warn(
            "FinGPT forecaster requires instruction-tuned format. "
            "This is a simplified wrapper. For full functionality, use FinGPT repo directly."
        )

    def forecast(
        self,
        ticker: str,
        news_list: list[str],
        instruction: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Generate forecast from news.

        Args:
            ticker: Stock ticker
            news_list: List of news headlines/articles
            instruction: Optional custom instruction

        Returns:
            Forecast result
        """
        if instruction is None:
            instruction = (
                f"Instruction: What is the movement of {ticker} stock price in the next week?\n"
                f"Answer with: up (increase), down (decrease), or same (no significant change).\n\n"
                f"Recent news:\n"
            )

        # Concatenate news
        news_text = "\n".join([f"- {news}" for news in news_list[:10]])  # Limit to 10 news items
        prompt = instruction + news_text

        # Simplified sentiment-based forecast (placeholder for full FinGPT)
        # In practice, you'd use the actual FinGPT forecaster model
        analyzer = FinGPTAnalyzer(task="sentiment")
        sentiment_agg = analyzer.sentiment_aggregate(news_list, method="weighted")

        # Map sentiment to forecast
        if sentiment_agg["score"] > 0.2:
            forecast = "up"
            confidence = abs(sentiment_agg["score"])
        elif sentiment_agg["score"] < -0.2:
            forecast = "down"
            confidence = abs(sentiment_agg["score"])
        else:
            forecast = "same"
            confidence = 1.0 - abs(sentiment_agg["score"])

        return {
            "ticker": ticker,
            "forecast": forecast,
            "confidence": float(confidence),
            "sentiment": sentiment_agg,
            "news_count": len(news_list),
        }


def analyze_financial_text(
    texts: list[str],
    task: str = "sentiment",
    model_name: Optional[str] = None,
    device: int = -1,
) -> list[dict[str, Any]]:
    """
    Convenience function for financial text analysis.

    Args:
        texts: List of texts to analyze
        task: "sentiment" or "ner"
        model_name: Optional model name
        device: Device ID (-1 for CPU)

    Returns:
        Analysis results
    """
    analyzer = FinGPTAnalyzer(task=task, model_name=model_name, device=device)

    if task == "sentiment":
        return analyzer.sentiment(texts)
    elif task == "ner":
        return analyzer.ner(texts)
    else:
        raise ValueError(f"Unknown task: {task}")
