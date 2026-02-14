"""
ML-based expert implementations.

Wraps ML models (FinRL, Qlib, Sklearn) as Expert instances.
"""
from __future__ import annotations

from typing import Optional, Any
import warnings

import pandas as pd
import numpy as np

from ddl69.core.contracts import Expert, ExpertResult


class SklearnExpert(Expert):
    """
    Expert using sklearn ensemble.

    Wraps sklearn models (RF, XGBoost, LightGBM) for probability forecasting.
    """

    def __init__(
        self,
        name: str = "sklearn_ensemble",
        version: str = "1.0",
        models: Optional[list[str]] = None,
    ):
        self.name = name
        self.version = version
        self.models = models or ["rf", "xgb", "lgb"]
        self.ensemble = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **kwargs: Any,
    ) -> None:
        """Train the sklearn ensemble."""
        from ddl69.agents import SklearnEnsemble

        self.ensemble = SklearnEnsemble(
            task="classification",
            models=self.models,
        )
        self.ensemble.fit(X_train, y_train, **kwargs)

    def predict(
        self,
        X: pd.DataFrame,
        event: Any,
    ) -> ExpertResult:
        """Generate forecast."""
        if self.ensemble is None:
            raise RuntimeError("Model not trained")

        # Get probabilities
        proba = self.ensemble.predict_proba(X)

        # Assuming 3-class: REJECT, BREAK_FAIL, ACCEPT_CONTINUE
        if proba.shape[1] == 3:
            probs = {
                "REJECT": float(proba[0, 0]),
                "BREAK_FAIL": float(proba[0, 1]),
                "ACCEPT_CONTINUE": float(proba[0, 2]),
            }
        elif proba.shape[1] == 2:
            # Binary classification
            probs = {
                "REJECT": float(proba[0, 0]),
                "ACCEPT_CONTINUE": float(proba[0, 1]),
                "BREAK_FAIL": 0.0,
            }
        else:
            # Fallback
            probs = {"ACCEPT_CONTINUE": float(proba[0, -1])}

        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}

        # Calculate confidence (max probability)
        confidence = max(probs.values())

        # Calculate entropy
        entropy = -sum(p * np.log2(p + 1e-10) for p in probs.values())

        return ExpertResult(
            expert_name=self.name,
            expert_version=self.version,
            probs=probs,
            confidence=confidence,
            uncertainty={"entropy": entropy},
            calibration_group="ml_sklearn",
        )


class FinRLExpert(Expert):
    """
    Expert using FinRL RL agent.

    Wraps FinRL agents (PPO, A2C, SAC) for trading decisions.
    """

    def __init__(
        self,
        name: str = "finrl_agent",
        version: str = "1.0",
        algorithm: str = "ppo",
    ):
        self.name = name
        self.version = version
        self.algorithm = algorithm
        self.trader = None

    def train(
        self,
        df: pd.DataFrame,
        total_timesteps: int = 100_000,
        **kwargs: Any,
    ) -> None:
        """Train the FinRL agent."""
        from ddl69.agents import train_finrl_agent

        self.trader, _ = train_finrl_agent(
            df=df,
            algorithm=self.algorithm,
            total_timesteps=total_timesteps,
            **kwargs,
        )

    def predict(
        self,
        df: pd.DataFrame,
        event: Any,
    ) -> ExpertResult:
        """Generate forecast from RL agent."""
        if self.trader is None:
            raise RuntimeError("Agent not trained")

        # Get prediction (simplified - use actions as signals)
        account_values, metrics = self.trader.predict(df)

        # Map return to probabilities
        total_return = metrics["total_return"] / 100.0  # Convert percentage

        # Simple mapping: positive return -> ACCEPT, negative -> REJECT
        if total_return > 0.05:
            probs = {"ACCEPT_CONTINUE": 0.7, "BREAK_FAIL": 0.2, "REJECT": 0.1}
        elif total_return < -0.05:
            probs = {"REJECT": 0.7, "BREAK_FAIL": 0.2, "ACCEPT_CONTINUE": 0.1}
        else:
            probs = {"ACCEPT_CONTINUE": 0.5, "BREAK_FAIL": 0.3, "REJECT": 0.2}

        confidence = max(probs.values())
        entropy = -sum(p * np.log2(p + 1e-10) for p in probs.values())

        return ExpertResult(
            expert_name=self.name,
            expert_version=self.version,
            probs=probs,
            confidence=confidence,
            uncertainty={"entropy": entropy, "sharpe": metrics["sharpe_ratio"]},
            calibration_group="ml_finrl",
        )


class QlibExpert(Expert):
    """
    Expert using Qlib factor model.

    Wraps Qlib strategies for quantitative forecasting.
    """

    def __init__(
        self,
        name: str = "qlib_strategy",
        version: str = "1.0",
        qlib_dir: Optional[str] = None,
    ):
        self.name = name
        self.version = version
        self.qlib_dir = qlib_dir
        self.strategy = None

    def train(
        self,
        instruments: str = "csi300",
        start_time: str = "2018-01-01",
        end_time: str = "2023-12-31",
        **kwargs: Any,
    ) -> None:
        """Train the Qlib strategy."""
        if self.qlib_dir is None:
            raise RuntimeError("qlib_dir not provided")

        from ddl69.strategies import train_qlib_model

        self.strategy, _ = train_qlib_model(
            qlib_dir=self.qlib_dir,
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            **kwargs,
        )

    def predict(
        self,
        signal: float,  # Qlib signal score
        event: Any,
    ) -> ExpertResult:
        """Generate forecast from Qlib signal."""
        if self.strategy is None:
            raise RuntimeError("Strategy not trained")

        # Map signal to probabilities
        # Assuming signal is normalized around 0
        if signal > 0.5:
            probs = {"ACCEPT_CONTINUE": 0.7, "BREAK_FAIL": 0.2, "REJECT": 0.1}
        elif signal < -0.5:
            probs = {"REJECT": 0.7, "BREAK_FAIL": 0.2, "ACCEPT_CONTINUE": 0.1}
        else:
            probs = {"ACCEPT_CONTINUE": 0.4, "BREAK_FAIL": 0.3, "REJECT": 0.3}

        confidence = max(probs.values())
        entropy = -sum(p * np.log2(p + 1e-10) for p in probs.values())

        return ExpertResult(
            expert_name=self.name,
            expert_version=self.version,
            probs=probs,
            confidence=confidence,
            uncertainty={"entropy": entropy, "qlib_signal": signal},
            calibration_group="ml_qlib",
        )


class SentimentExpert(Expert):
    """
    Expert using FinGPT sentiment analysis.

    Analyzes news sentiment to forecast market movement.
    """

    def __init__(
        self,
        name: str = "fingpt_sentiment",
        version: str = "1.0",
        model_name: Optional[str] = None,
    ):
        self.name = name
        self.version = version
        self.model_name = model_name
        self.analyzer = None

    def load(self) -> None:
        """Load FinGPT analyzer."""
        from ddl69.nlp import FinGPTAnalyzer

        self.analyzer = FinGPTAnalyzer(
            task="sentiment",
            model_name=self.model_name,
        )

    def predict(
        self,
        news_texts: list[str],
        event: Any,
    ) -> ExpertResult:
        """Generate forecast from news sentiment."""
        if self.analyzer is None:
            self.load()

        # Aggregate sentiment
        sentiment_agg = self.analyzer.sentiment_aggregate(news_texts, method="weighted")

        score = sentiment_agg["score"]

        # Map sentiment to probabilities
        if score > 0.2:
            probs = {"ACCEPT_CONTINUE": 0.65, "BREAK_FAIL": 0.25, "REJECT": 0.1}
        elif score < -0.2:
            probs = {"REJECT": 0.65, "BREAK_FAIL": 0.25, "ACCEPT_CONTINUE": 0.1}
        else:
            probs = {"ACCEPT_CONTINUE": 0.4, "BREAK_FAIL": 0.4, "REJECT": 0.2}

        confidence = max(probs.values())
        entropy = -sum(p * np.log2(p + 1e-10) for p in probs.values())

        return ExpertResult(
            expert_name=self.name,
            expert_version=self.version,
            probs=probs,
            confidence=confidence,
            uncertainty={"entropy": entropy, "sentiment_score": score},
            calibration_group="nlp_sentiment",
        )
