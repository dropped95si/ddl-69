"""Walk-forward backtester: compare strategies over time with purged cross-validation"""

from __future__ import annotations

import logging
from typing import Optional
import pandas as pd
import numpy as np

from ddl69.labeling.lopez_prado import PurgedKFold, get_sample_weights
from ddl69.core.real_pipeline import RealDataPipeline
from ddl69.agents.sklearn_ensemble import SklearnEnsemble
from ddl69.data.loaders import DataLoader

logger = logging.getLogger(__name__)


class WalkForwardBacktester:
    """Walk-forward backtester with purged k-fold cross-validation"""

    def __init__(
        self,
        symbol: str,
        artifact_root: Optional[str] = None,
        n_splits: int = 5,
    ):
        self.symbol = symbol
        self.artifact_root = artifact_root
        self.n_splits = n_splits
        self.loader = DataLoader(artifact_root)
        self.results = []

    def run(self) -> dict:
        """Execute walk-forward backtest

        Returns:
          {
            symbol: str,
            total_bars: int,
            n_splits: int,
            splits: [{period, train_bars, test_bars, accuracy, sharpe, max_dd, trades}],
            aggregate: {avg_accuracy, avg_sharpe, total_trades, win_rate, ...}
          }
        """
        try:
            # Load and prepare data
            df = self.loader.load(self.symbol)
            pipeline = RealDataPipeline(artifact_root=self.artifact_root)
            df = pipeline._preprocess(df)
            df = self._add_indicators_to_df(df, pipeline)
            df = pipeline._create_labels(df)

            # Get feature columns
            feature_cols = [
                c for c in df.columns
                if c not in [
                    "timestamp", "open", "high", "low", "close", "volume",
                    "returns", "label", "forward_return"
                ]
                and df[c].dtype in ["float64", "int64"]
            ]

            # Purged K-fold split
            pkf = PurgedKFold(n_splits=self.n_splits)
            splits_results = []

            for split_idx, (train_indices, test_indices) in enumerate(
                pkf.split(
                    X=df[feature_cols].fillna(0).values,
                    y=df["label"].values,
                    t1=df["timestamp"].values if "timestamp" in df.columns else None,
                )
            ):
                try:
                    train_df = df.iloc[train_indices]
                    test_df = df.iloc[test_indices]

                    # Train model
                    X_train = train_df[feature_cols].fillna(0)
                    y_train = (train_df["label"] > 0).astype(int)

                    X_test = test_df[feature_cols].fillna(0)
                    y_test = (test_df["label"] > 0).astype(int)

                    model = SklearnEnsemble(
                        task="classification",
                        models=["rf", "xgb", "lgb"],
                        voting="soft",
                    )
                    model.fit(X_train, y_train)

                    # Evaluate
                    test_accuracy = model.score(X_test, y_test)
                    preds = model.predict_proba(X_test)[:, 1]

                    # Calculate returns-based sharpe
                    pred_returns = preds * test_df["returns"].fillna(0).values
                    sharpe = (
                        np.mean(pred_returns) / (np.std(pred_returns) + 1e-8) * np.sqrt(252)
                        if len(pred_returns) > 1 else 0
                    )

                    # Trade metrics
                    trades = (preds > 0.5).sum()
                    win_rate = (
                        (preds[y_test == 1] > 0.5).sum() / (y_test == 1).sum()
                        if (y_test == 1).sum() > 0 else 0
                    )

                    split_result = {
                        "split": split_idx + 1,
                        "period": f"{train_df.iloc[0]['timestamp'] if 'timestamp' in train_df.columns else 'N/A'} → {test_df.iloc[-1]['timestamp'] if 'timestamp' in test_df.columns else 'N/A'}",
                        "train_bars": len(train_df),
                        "test_bars": len(test_df),
                        "accuracy": round(test_accuracy, 4),
                        "sharpe": round(sharpe, 4),
                        "trades": int(trades),
                        "win_rate": round(win_rate, 4),
                        "max_dd": self._calculate_max_dd(pred_returns),
                    }
                    splits_results.append(split_result)
                    logger.info(f"Split {split_idx + 1}: accuracy={test_accuracy:.3f}, sharpe={sharpe:.3f}")

                except Exception as e:
                    logger.warning(f"Failed split {split_idx}: {e}")
                    continue

            # Aggregate results
            if splits_results:
                accuracies = [s["accuracy"] for s in splits_results]
                sharpes = [s["sharpe"] for s in splits_results]
                trades_total = sum(s["trades"] for s in splits_results)
                win_rates = [s["win_rate"] for s in splits_results]

                aggregate = {
                    "avg_accuracy": round(np.mean(accuracies), 4),
                    "std_accuracy": round(np.std(accuracies), 4),
                    "avg_sharpe": round(np.mean(sharpes), 4),
                    "std_sharpe": round(np.std(sharpes), 4),
                    "total_trades": int(trades_total),
                    "avg_win_rate": round(np.mean(win_rates), 4),
                    "consistency": "CONSISTENT" if np.std(sharpes) < np.mean([abs(s) for s in sharpes]) else "VOLATILE",
                }
            else:
                aggregate = {"error": "No splits completed"}

            return {
                "symbol": self.symbol,
                "total_bars": len(df),
                "n_splits": self.n_splits,
                "splits": splits_results,
                "aggregate": aggregate,
            }

        except Exception as e:
            logger.error(f"Backtest failed for {self.symbol}: {e}")
            return {"symbol": self.symbol, "error": str(e)}

    def _add_indicators_to_df(self, df: pd.DataFrame, pipeline: RealDataPipeline) -> pd.DataFrame:
        """Add technical indicators to dataframe"""
        return pipeline._add_indicators(df, self.symbol)

    @staticmethod
    def _calculate_max_dd(returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        if len(returns) < 2:
            return 0.0
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return round(np.min(drawdown), 4)


class MultiStrategyComparator:
    """Compare multiple symbols/strategies side-by-side"""

    def __init__(self, symbols: list[str], artifact_root: Optional[str] = None):
        self.symbols = symbols
        self.artifact_root = artifact_root
        self.backtests = {
            symbol: WalkForwardBacktester(symbol, artifact_root, n_splits=3)
            for symbol in symbols
        }

    def run_all(self) -> dict:
        """Run backtest for all symbols

        Returns:
          {
            timestamp: str,
            results: {symbol: backtest_result, ...},
            ranking: [(symbol, avg_sharpe, accuracy), ...]
          }
        """
        results = {}
        for symbol, bt in self.backtests.items():
            logger.info(f"Running backtest for {symbol}...")
            results[symbol] = bt.run()

        # Rank by sharpe ratio
        ranking = []
        for symbol, result in results.items():
            if "aggregate" in result and "avg_sharpe" in result["aggregate"]:
                sharpe = result["aggregate"]["avg_sharpe"]
                accuracy = result["aggregate"].get("avg_accuracy", 0)
                ranking.append((symbol, sharpe, accuracy))

        ranking.sort(key=lambda x: x[1], reverse=True)

        return {
            "timestamp": pd.Timestamp.now().isoformat(),
            "symbols_tested": len(self.symbols),
            "results": results,
            "ranking": ranking,
            "best_symbol": ranking[0][0] if ranking else None,
            "best_sharpe": ranking[0][1] if ranking else 0,
        }
