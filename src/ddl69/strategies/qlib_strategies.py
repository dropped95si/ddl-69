"""
Qlib strategies and factor library integration.

Provides Qlib factor extraction, model training, and backtesting.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Any, Literal

import pandas as pd
import numpy as np

# Try to import Qlib
try:
    import qlib
    from qlib.data import D
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.contrib.strategy import TopkDropoutStrategy
    from qlib.contrib.evaluate import backtest as qlib_backtest, risk_analysis
    from qlib.workflow import R
    from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False
    warnings.warn("Qlib not available. Install from: https://github.com/microsoft/qlib")


class QlibStrategy:
    """
    Qlib-based quantitative trading strategy.

    Supports:
    - Factor extraction from Qlib data
    - Model training (LightGBM by default)
    - Backtesting with TopkDropout strategy
    """

    def __init__(
        self,
        qlib_dir: str,
        market: str = "csi300",
        region: str = "cn",
    ):
        if not QLIB_AVAILABLE:
            raise RuntimeError("Qlib is not installed")

        self.qlib_dir = Path(qlib_dir)
        self.market = market
        self.region = region
        self.model = None
        self.pred = None

        # Initialize Qlib
        qlib.init(provider_uri=str(self.qlib_dir), region=region)
        print(f"Qlib initialized with {qlib_dir}")

    def prepare_data(
        self,
        instruments: str = "csi300",
        start_time: str = "2018-01-01",
        end_time: str = "2023-12-31",
        train_split: str = "2021-01-01",
        valid_split: str = "2022-01-01",
        factors: Optional[list[str]] = None,
    ) -> DatasetH:
        """
        Prepare dataset with Qlib factors.

        Args:
            instruments: Universe (e.g., "csi300", "csi500", or custom list)
            start_time: Start date
            end_time: End date
            train_split: Train/valid split date
            valid_split: Valid/test split date
            factors: List of factor expressions

        Returns:
            DatasetH object
        """
        if factors is None:
            # Default alpha158 factors
            factors = [
                "($close-Ref($close,1))/Ref($close,1)",  # return
                "Mean($close,5)/Mean($close,10)-1",  # MA trend
                "Mean($close,5)/Mean($close,30)-1",
                "Std($close,5)/Mean($close,5)",  # volatility
                "Std($close,20)/Mean($close,20)",
                "($high-$low)/$open",  # range
                "Mean(($high-$low)/$open,5)",
                "$volume/Mean($volume,5)-1",  # volume
                "Corr($close,$volume,5)",  # price-volume corr
                "Corr($close,$volume,10)",
                "Rank($close)",  # cross-sectional rank
                "Rank($volume)",
                "(Ref($close,1)-Ref($close,5))/Ref($close,5)",  # momentum
                "(Ref($close,1)-Ref($close,10))/Ref($close,10)",
                "(Ref($close,1)-Ref($close,20))/Ref($close,20)",
            ]

        data_handler_config = {
            "start_time": start_time,
            "end_time": end_time,
            "fit_start_time": start_time,
            "fit_end_time": train_split,
            "instruments": instruments,
            "infer_processors": [
                {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
            ],
            "learn_processors": [
                {"class": "DropnaLabel"},
                {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
            ],
            "label": ["Ref($close, -2)/Ref($close, -1) - 1"],  # 1-day return
        }

        # Add factors
        data_handler_config["feature"] = factors

        segments = {
            "train": (start_time, train_split),
            "valid": (train_split, valid_split),
            "test": (valid_split, end_time),
        }

        dataset = DatasetH(
            handler=DataHandlerLP(**data_handler_config),
            segments=segments,
        )

        return dataset

    def train(
        self,
        dataset: DatasetH,
        model_type: str = "lgb",
        **kwargs: Any,
    ) -> dict[str, float]:
        """
        Train model on dataset.

        Args:
            dataset: Qlib DatasetH
            model_type: Model type (currently supports: lgb)
            **kwargs: Model hyperparameters

        Returns:
            Validation metrics
        """
        if model_type == "lgb":
            model_config = {
                "loss": "mse",
                "colsample_bytree": kwargs.get("colsample_bytree", 0.8),
                "learning_rate": kwargs.get("learning_rate", 0.1),
                "subsample": kwargs.get("subsample", 0.8),
                "lambda_l1": kwargs.get("lambda_l1", 0.0),
                "lambda_l2": kwargs.get("lambda_l2", 0.0),
                "max_depth": kwargs.get("max_depth", 8),
                "num_leaves": kwargs.get("num_leaves", 128),
                "num_threads": kwargs.get("num_threads", -1),
                "n_estimators": kwargs.get("n_estimators", 100),
                "early_stopping_rounds": kwargs.get("early_stopping_rounds", 50),
            }
            self.model = LGBModel(**model_config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Train
        print("Training model...")
        self.model.fit(dataset)

        # Predict on validation
        self.pred = self.model.predict(dataset)

        # Calculate IC (Information Coefficient)
        val_pred = self.pred.loc["valid"]
        val_label = dataset.prepare("valid", col_set="label")

        # Align and calculate IC
        merged = pd.concat([val_pred, val_label], axis=1, join="inner")
        merged.columns = ["pred", "label"]

        ic = merged.groupby(level="datetime").apply(
            lambda df: df["pred"].corr(df["label"], method="spearman")
        )

        metrics = {
            "ic_mean": float(ic.mean()),
            "ic_std": float(ic.std()),
            "icir": float(ic.mean() / ic.std()) if ic.std() > 0 else 0.0,
            "ic_positive_ratio": float((ic > 0).sum() / len(ic)),
        }

        print(f"Validation IC: {metrics['ic_mean']:.4f} ± {metrics['ic_std']:.4f}")
        print(f"ICIR: {metrics['icir']:.4f}")

        return metrics

    def backtest(
        self,
        dataset: DatasetH,
        strategy: str = "topk_dropout",
        topk: int = 50,
        buffer_margin: float = 0.1,
        initial_capital: float = 1_000_000,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        """
        Run backtest.

        Args:
            dataset: Qlib DatasetH
            strategy: Strategy name (topk_dropout)
            topk: Top K stocks to hold
            buffer_margin: Buffer for TopkDropout
            initial_capital: Initial capital
            **kwargs: Additional strategy parameters

        Returns:
            (backtest_report, metrics)
        """
        if self.pred is None:
            raise RuntimeError("Model not trained. Call train() first.")

        # Prepare strategy
        if strategy == "topk_dropout":
            strategy_config = {
                "model": self.model,
                "dataset": dataset,
                "topk": topk,
                "buffer_margin": buffer_margin,
                "signal": self.pred,
            }
            strat = TopkDropoutStrategy(**strategy_config)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Prepare portfolio config
        portfolio_config = {
            "executor": {
                "class": "SimulatorExecutor",
                "module_path": "qlib.backtest.executor",
                "kwargs": {
                    "time_per_step": "day",
                    "generate_portfolio_metrics": True,
                },
            },
            "strategy": strategy_config,
        }

        # Run backtest
        print("Running backtest...")
        report, positions = qlib_backtest(
            pred=self.pred,
            strategy=strat,
            executor=portfolio_config["executor"],
            benchmark="SH000300",  # CSI300 benchmark
        )

        # Calculate metrics
        analysis = risk_analysis(report["return"])

        metrics = {
            "total_return": float(analysis["cum_return"].iloc[-1]),
            "annual_return": float(analysis.get("annual_return", 0)),
            "max_drawdown": float(analysis.get("max_drawdown", 0)),
            "sharpe_ratio": float(analysis.get("sharpe", 0)),
            "information_ratio": float(analysis.get("information_ratio", 0)),
            "win_rate": float(analysis.get("win_rate", 0)),
        }

        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")

        return report, metrics

    def predict(
        self,
        dataset: DatasetH,
        segment: str = "test",
    ) -> pd.DataFrame:
        """
        Generate predictions for a dataset segment.

        Args:
            dataset: Qlib DatasetH
            segment: Segment name (train/valid/test)

        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            raise RuntimeError("Model not trained")

        pred = self.model.predict(dataset, segment=segment)
        return pred


class QlibFactorLibrary:
    """
    Qlib factor expression library.

    Common alpha factors from academic literature and industry practice.
    """

    @staticmethod
    def get_alpha101() -> list[str]:
        """WorldQuant Alpha101 factors (simplified versions)."""
        return [
            "Corr($close,$volume,10)",
            "Rank(Delta($close,1))",
            "Rank(Corr($high,$volume,5))",
            "Sum(($close>Ref($close,1)?1:0),10)/10",
            "Mean(($high-$low)/$close,10)",
        ]

    @staticmethod
    def get_alpha158() -> list[str]:
        """Alpha158 factors (Qlib default)."""
        # Momentum
        factors = [
            f"Ref($close,{d})/Ref($close,{d+1})-1"
            for d in [1, 2, 3, 4, 5, 10, 20, 30, 60]
        ]

        # Moving averages
        for period in [5, 10, 20, 30, 60]:
            factors.append(f"Mean($close,{period})/$close-1")

        # Volatility
        for period in [5, 10, 20, 30, 60]:
            factors.append(f"Std($close,{period})")

        # Volume
        factors.extend([
            "$volume/Mean($volume,5)",
            "$volume/Mean($volume,10)",
            "$volume/Mean($volume,20)",
            "Corr($close,$volume,5)",
            "Corr($close,$volume,10)",
        ])

        # High-low
        factors.extend([
            "($high-$low)/$close",
            "Mean(($high-$low)/$close,5)",
            "Std(($high-$low)/$close,5)",
        ])

        return factors

    @staticmethod
    def get_technical() -> list[str]:
        """Common technical indicators."""
        return [
            # RSI
            "RSI($close,14)",
            # Bollinger Bands
            "($close-Mean($close,20))/Std($close,20)",
            # MACD
            "EMA($close,12)-EMA($close,26)",
            # ATR
            "Mean(Max(Max($high-$low,Abs($high-Ref($close,1))),Abs($low-Ref($close,1))),14)",
            # Williams %R
            "($high-$close)/(Max($high,20)-Min($low,20))",
        ]


def train_qlib_model(
    qlib_dir: str,
    instruments: str = "csi300",
    start_time: str = "2018-01-01",
    end_time: str = "2023-12-31",
    train_split: str = "2021-01-01",
    valid_split: str = "2022-01-01",
    model_type: str = "lgb",
    **kwargs: Any,
) -> tuple[QlibStrategy, dict[str, Any]]:
    """
    Convenience function to train Qlib model.

    Args:
        qlib_dir: Path to Qlib data directory
        instruments: Universe
        start_time: Start date
        end_time: End date
        train_split: Train/valid split
        valid_split: Valid/test split
        model_type: Model type
        **kwargs: Model hyperparameters

    Returns:
        (trained_strategy, test_metrics)
    """
    strategy = QlibStrategy(qlib_dir=qlib_dir)

    # Prepare data
    dataset = strategy.prepare_data(
        instruments=instruments,
        start_time=start_time,
        end_time=end_time,
        train_split=train_split,
        valid_split=valid_split,
    )

    # Train
    val_metrics = strategy.train(dataset, model_type=model_type, **kwargs)

    # Backtest on test set
    test_report, test_metrics = strategy.backtest(dataset)

    return strategy, {"validation": val_metrics, "backtest": test_metrics}
