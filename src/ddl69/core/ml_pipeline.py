"""
Unified ML pipeline integrating all tools.

Combines:
- TA-Lib indicators
- Lopez de Prado labeling
- Monte Carlo simulations
- FinRL RL agents
- Qlib strategies
- Sklearn ensembles
- FinGPT sentiment
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Any, Literal
import warnings

import pandas as pd
import numpy as np

# Import our tools
from ddl69.indicators.talib_wrapper import TALibWrapper, compute_all_indicators
from ddl69.labeling.lopez_prado import frac_diff_ffd, meta_labeling, get_sample_weights, PurgedKFold
from ddl69.labeling.triple_barrier import estimate_vol
from ddl69.simulations.monte_carlo import (
    monte_carlo_returns,
    monte_carlo_portfolio,
    sharpe_ratio_distribution,
    value_at_risk_mc,
)


class MLPipeline:
    """
    Unified machine learning pipeline for trading.

    Orchestrates the full workflow from data to predictions.
    """

    def __init__(
        self,
        use_talib: bool = True,
        use_lopez: bool = True,
        use_finrl: bool = False,
        use_qlib: bool = False,
        use_sklearn: bool = True,
        use_fingpt: bool = False,
    ):
        self.use_talib = use_talib
        self.use_lopez = use_lopez
        self.use_finrl = use_finrl
        self.use_qlib = use_qlib
        self.use_sklearn = use_sklearn
        self.use_fingpt = use_fingpt

        # Initialize components
        self.talib_wrapper = TALibWrapper() if use_talib else None
        self.sklearn_ensemble = None
        self.finrl_trader = None
        self.qlib_strategy = None
        self.fingpt_analyzer = None

    def prepare_features(
        self,
        df: pd.DataFrame,
        add_indicators: bool = True,
        add_ffd: bool = True,
        ffd_threshold: float = 0.01,
    ) -> pd.DataFrame:
        """
        Prepare features from OHLCV data.

        Args:
            df: DataFrame with columns: open, high, low, close, volume
            add_indicators: Add TA-Lib indicators
            add_ffd: Add fractionally differentiated features
            ffd_threshold: FFD threshold

        Returns:
            DataFrame with features
        """
        df = df.copy()

        # Add technical indicators
        if add_indicators and self.talib_wrapper:
            df = compute_all_indicators(df, use_talib=True)

        # Add fractionally differentiated features
        if add_ffd and self.use_lopez:
            for col in ["close", "volume"]:
                if col in df.columns:
                    df[f"{col}_ffd"] = frac_diff_ffd(df[col], d=0.5, thres=ffd_threshold)

        # Drop NaN rows
        df = df.dropna()

        return df

    def create_labels(
        self,
        df: pd.DataFrame,
        method: Literal["triple_barrier", "returns"] = "triple_barrier",
        ptsl: Optional[tuple[float, float]] = None,
        horizon: int = 5,
    ) -> pd.DataFrame:
        """
        Create labels for supervised learning.

        Args:
            df: DataFrame with OHLCV data
            method: Labeling method
            ptsl: Profit-taking / stop-loss levels (e.g., (0.02, 0.02))
            horizon: Forward horizon in bars

        Returns:
            DataFrame with labels column
        """
        df = df.copy()

        if method == "triple_barrier":
            if not self.use_lopez:
                raise RuntimeError("Lopez de Prado methods not enabled")

            # Estimate volatility
            vol = estimate_vol(df["close"], span=20)

            # Use triple_barrier_labels from our labeling module
            from ddl69.labeling.triple_barrier import triple_barrier_labels

            # Prepare bars DataFrame in expected format
            bars = df[["open", "high", "low", "close"]].copy()
            bars["ts"] = df.index if df.index.dtype.kind == "M" else range(len(df))
            bars["symbol"] = "SYM"

            tb_labels = triple_barrier_labels(bars, horizon_bars=horizon, k=2.0, vol_span=20)

            # Map triple barrier labels to numeric
            label_map = {"UP": 1, "DOWN": -1, "TIMEOUT": 0}
            df["label"] = 0
            if not tb_labels.empty:
                for _, row in tb_labels.iterrows():
                    idx = df.index[df.index >= row["asof_ts"]]
                    if len(idx) > 0:
                        df.loc[idx[0], "label"] = label_map.get(row["label"], 0)

        elif method == "returns":
            # Simple forward returns
            df["label"] = df["close"].pct_change(horizon).shift(-horizon)
            df = df.dropna()

        else:
            raise ValueError(f"Unknown method: {method}")

        return df

    def train_sklearn(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        models: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Train sklearn ensemble.

        Args:
            X_train: Training features
            y_train: Training labels
            models: List of model names
            **kwargs: Hyperparameters

        Returns:
            Training metrics
        """
        if not self.use_sklearn:
            raise RuntimeError("Sklearn not enabled")

        from ddl69.agents import SklearnEnsemble

        self.sklearn_ensemble = SklearnEnsemble(
            task="classification" if y_train.nunique() <= 10 else "regression",
            models=models or ["rf", "xgb", "lgb"],
        )

        self.sklearn_ensemble.fit(X_train, y_train, **kwargs)

        # Return feature importance
        try:
            importance = self.sklearn_ensemble.feature_importance(top_n=20)
            return {"feature_importance": importance.to_dict("records")}
        except Exception:
            return {}

    def train_finrl(
        self,
        df: pd.DataFrame,
        algorithm: str = "ppo",
        total_timesteps: int = 100_000,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Train FinRL agent.

        Args:
            df: DataFrame with OHLCV data (columns: date, tic, open, high, low, close, volume)
            algorithm: RL algorithm
            total_timesteps: Training timesteps
            **kwargs: Additional parameters

        Returns:
            Training metrics
        """
        if not self.use_finrl:
            raise RuntimeError("FinRL not enabled")

        from ddl69.agents import train_finrl_agent

        self.finrl_trader, metrics = train_finrl_agent(
            df=df,
            algorithm=algorithm,
            total_timesteps=total_timesteps,
            **kwargs,
        )

        return metrics

    def train_qlib(
        self,
        qlib_dir: str,
        instruments: str = "csi300",
        start_time: str = "2018-01-01",
        end_time: str = "2023-12-31",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Train Qlib strategy.

        Args:
            qlib_dir: Qlib data directory
            instruments: Universe
            start_time: Start date
            end_time: End date
            **kwargs: Additional parameters

        Returns:
            Training metrics
        """
        if not self.use_qlib:
            raise RuntimeError("Qlib not enabled")

        from ddl69.strategies import train_qlib_model

        self.qlib_strategy, metrics = train_qlib_model(
            qlib_dir=qlib_dir,
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            **kwargs,
        )

        return metrics

    def predict_sklearn(self, X: pd.DataFrame) -> np.ndarray:
        """Predict with sklearn ensemble."""
        if self.sklearn_ensemble is None:
            raise RuntimeError("Sklearn ensemble not trained")
        return self.sklearn_ensemble.predict(X)

    def predict_finrl(self, df: pd.DataFrame) -> dict[str, Any]:
        """Predict with FinRL agent."""
        if self.finrl_trader is None:
            raise RuntimeError("FinRL trader not trained")
        account_values, metrics = self.finrl_trader.predict(df)
        return {"account_values": account_values, "metrics": metrics}

    def predict_qlib(self, segment: str = "test") -> pd.DataFrame:
        """Predict with Qlib strategy."""
        if self.qlib_strategy is None:
            raise RuntimeError("Qlib strategy not trained")
        # Note: requires dataset from training
        raise NotImplementedError("Qlib prediction requires dataset context")

    def analyze_sentiment(
        self,
        texts: list[str],
        model_name: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Analyze sentiment with FinGPT."""
        if not self.use_fingpt:
            raise RuntimeError("FinGPT not enabled")

        from ddl69.nlp import analyze_financial_text

        return analyze_financial_text(texts, task="sentiment", model_name=model_name)

    def monte_carlo_analysis(
        self,
        returns: pd.Series,
        n_simulations: int = 1000,
        n_periods: int = 252,
    ) -> dict[str, Any]:
        """
        Run Monte Carlo analysis.

        Args:
            returns: Historical returns
            n_simulations: Number of simulations
            n_periods: Forward periods

        Returns:
            Monte Carlo metrics
        """
        # Monte Carlo returns
        simulated = monte_carlo_returns(
            returns=returns,
            n_simulations=n_simulations,
            n_periods=n_periods,
            method="bootstrap",
        )

        # Sharpe distribution
        sharpe_dist = sharpe_ratio_distribution(returns, n_simulations=n_simulations)

        # VaR
        var_95 = value_at_risk_mc(returns, confidence_level=0.95, n_simulations=n_simulations)
        var_99 = value_at_risk_mc(returns, confidence_level=0.99, n_simulations=n_simulations)

        return {
            "simulated_returns": simulated,
            "sharpe_mean": float(np.mean(sharpe_dist)),
            "sharpe_std": float(np.std(sharpe_dist)),
            "var_95": float(var_95),
            "var_99": float(var_99),
        }


def create_full_pipeline(
    df: pd.DataFrame,
    enable_all: bool = False,
) -> tuple[MLPipeline, pd.DataFrame]:
    """
    Create and prepare full ML pipeline.

    Args:
        df: Input OHLCV DataFrame
        enable_all: Enable all optional tools (FinRL, Qlib, FinGPT)

    Returns:
        (pipeline, prepared_df)
    """
    pipeline = MLPipeline(
        use_talib=True,
        use_lopez=True,
        use_sklearn=True,
        use_finrl=enable_all,
        use_qlib=enable_all,
        use_fingpt=enable_all,
    )

    # Prepare features
    df_prepared = pipeline.prepare_features(df)

    # Create labels
    df_prepared = pipeline.create_labels(df_prepared, method="triple_barrier")

    return pipeline, df_prepared
