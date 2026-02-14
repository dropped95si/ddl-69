"""End-to-end pipeline: Real data -> ML models -> Live predictions -> Dashboard"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from ddl69.data.loaders import DataLoader, SupabaseCache
from ddl69.indicators.talib_wrapper import TALibWrapper
from ddl69.labeling.lopez_prado import PurgedKFold, frac_diff_ffd, get_sample_weights
from ddl69.simulations.monte_carlo import (
    monte_carlo_returns,
    monte_carlo_portfolio,
    value_at_risk_mc,
    conditional_var_mc,
)
from ddl69.agents.sklearn_ensemble import SklearnEnsemble
from ddl69.nlp.fingpt import FinGPTAnalyzer

logger = logging.getLogger(__name__)


class RealDataPipeline:
    """Production pipeline: Load -> Preprocess -> Features -> Predict -> Cache."""

    def __init__(
        self,
        artifact_root: Optional[str] = None,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
    ):
        self.loader = DataLoader(artifact_root=artifact_root)
        self.cache = SupabaseCache(url=supabase_url, key=supabase_key)

        # Initialize ML tools
        self.talib = TALibWrapper()
        self.ensemble = None

        # Settings
        self.ta_indicators = [
            "SMA_20", "SMA_50", "EMA_20", "RSI_14", "MACD", "MACD_SIGNAL",
            "BB_UPPER", "BB_LOWER", "ATR_14", "ADX_14", "CCI_14", "ROC_20",
            "STOCH_K", "STOCH_D", "OBV",
        ]

    def run(
        self,
        symbols: list[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        train_split: float = 0.7,
    ) -> dict:
        """Run full pipeline for symbols."""
        results = {}

        for symbol in symbols:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {symbol}")
            logger.info(f"{'='*60}")

            try:
                # Step 1: Load data
                raw_df = self.loader.load(symbol, start_date, end_date)
                logger.info(f"Loaded {len(raw_df)} bars")

                # Step 2: Preprocess
                df = self._preprocess(raw_df)
                logger.info(f"After preprocessing: {len(df)} bars, {len(df.columns)} columns")

                # Step 3: Add technical indicators
                df = self._add_indicators(df, symbol)
                logger.info(f"Added {len(self.ta_indicators)} indicators")

                # Step 4: Create labels for training
                df = self._create_labels(df)
                logger.info(f"Created labels, {(df['label'] != 0).sum()} non-neutral samples")

                # Step 5: Split train/test
                split_idx = int(len(df) * train_split)
                train_df = df.iloc[:split_idx].copy()
                test_df = df.iloc[split_idx:].copy()

                logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")

                # Step 6: Train ensemble
                metrics = self._train_models(train_df, test_df, symbol)

                # Step 7: Make predictions on latest data
                latest_predictions = self._predict(df.iloc[-1:].copy(), symbol)

                # Step 8: Risk analysis
                returns = df["returns"].dropna()
                risk_metrics = self._analyze_risk(returns)

                # Step 9: Save to cache
                self._save_results(symbol, df, latest_predictions, metrics)

                results[symbol] = {
                    "status": "success",
                    "bars_processed": len(df),
                    "train_bars": len(train_df),
                    "test_bars": len(test_df),
                    "metrics": metrics,
                    "latest_predictions": latest_predictions,
                    "risk_metrics": risk_metrics,
                    "data": df,
                }

            except Exception as e:
                logger.error(f"Failed to process {symbol}: {e}", exc_info=True)
                results[symbol] = {"status": "error", "error": str(e)}

        return results

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values, outliers, etc."""
        df = df.copy()

        # Forward fill for small gaps
        df[["open", "high", "low", "close", "volume"]] = df[
            ["open", "high", "low", "close", "volume"]
        ].fillna(method="ffill", limit=2)

        # Drop rows with NaN
        df = df.dropna(subset=["close", "volume"])

        # Calculate simple returns
        df["returns"] = df["close"].pct_change()

        # Remove outliers (> 5% move in single bar)
        df = df[df["returns"].abs() < 0.05]

        return df.reset_index(drop=True)

    def _add_indicators(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add technical indicators."""
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        volume = df["volume"].values

        try:
            df["SMA_20"] = self.talib.SMA(close, period=20)
            df["SMA_50"] = self.talib.SMA(close, period=50)
            df["EMA_20"] = self.talib.EMA(close, period=20)
            df["RSI_14"] = self.talib.RSI(close, period=14)

            macd_result = self.talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            if macd_result is not None:
                df["MACD"], df["MACD_SIGNAL"], df["MACD_HIST"] = macd_result

            bb_result = self.talib.BBANDS(close, timeperiod=20)
            if bb_result is not None:
                df["BB_UPPER"], df["BB_MIDDLE"], df["BB_LOWER"] = bb_result

            df["ATR_14"] = self.talib.ATR(high, low, close, period=14)
            df["ADX_14"] = self.talib.ADX(high, low, close, period=14)
            df["CCI_14"] = self.talib.CCI(high, low, close, period=14)
            df["ROC_20"] = self.talib.ROC(close, period=20)

            stoch_result = self.talib.STOCH(high, low, close, fastk_period=14, slowk_period=3)
            if stoch_result is not None:
                df["STOCH_K"], df["STOCH_D"] = stoch_result

            df["OBV"] = self.talib.OBV(close, volume)

        except Exception as e:
            logger.warning(f"Failed to calculate some indicators: {e}")

        # Fill NaN from indicators
        df = df.fillna(method="bfill").fillna(method="ffill").fillna(0)

        return df

    def _create_labels(self, df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
        """Create trading labels: 1 (BUY), 0 (HOLD), -1 (SELL)."""
        df = df.copy()

        # Forward-looking returns
        df["forward_return"] = (df["close"].shift(-horizon) - df["close"]) / df["close"]

        # Triple barrier: profitable trade (1), unprofitable (-1), or hold (0)
        upper_barrier = df["close"] * 1.02  # 2% profit target
        lower_barrier = df["close"] * 0.98  # 2% stop loss

        labels = []
        for i in range(len(df) - horizon):
            high_touch = (df.iloc[i : i + horizon]["high"] >= upper_barrier.iloc[i]).any()
            low_touch = (df.iloc[i : i + horizon]["low"] <= lower_barrier.iloc[i]).any()

            if high_touch and not low_touch:
                labels.append(1)
            elif low_touch and not high_touch:
                labels.append(-1)
            else:
                labels.append(0)

        # Pad labels for final rows
        labels.extend([0] * (len(df) - len(labels)))
        df["label"] = labels

        return df

    def _train_models(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, symbol: str
    ) -> dict:
        """Train ensemble on raw features."""
        # Use only numeric feature columns (not timestamp, price, label)
        feature_cols = [
            c
            for c in train_df.columns
            if c
            not in ["timestamp", "open", "high", "low", "close", "volume", "returns", "label", "forward_return"]
            and train_df[c].dtype in ["float64", "int64"]
        ]

        X_train = train_df[feature_cols].fillna(0)
        y_train = (train_df["label"] > 0).astype(int)  # Binary: UP vs DOWN/HOLD

        X_test = test_df[feature_cols].fillna(0)
        y_test = (test_df["label"] > 0).astype(int)

        # Initialize and train ensemble
        self.ensemble = SklearnEnsemble(
            task="classification",
            models=["rf", "xgb", "lgb"],
            voting="soft",
        )

        self.ensemble.fit(X_train, y_train)

        # Evaluate
        train_score = self.ensemble.score(X_train, y_train)
        test_score = self.ensemble.score(X_test, y_test)
        test_proba = self.ensemble.predict_proba(X_test)[:, 1]

        # Sharpe ratio of predictions vs actuals
        pred_returns = test_proba * test_df["returns"].values
        sharpe = (
            np.mean(pred_returns) / (np.std(pred_returns) + 1e-8) * np.sqrt(252)
            if len(pred_returns) > 1
            else 0
        )

        return {
            "train_accuracy": round(train_score, 4),
            "test_accuracy": round(test_score, 4),
            "test_auc": round(
                self._auc_score(y_test, test_proba), 4
            ),
            "sharpe_ratio": round(sharpe, 4),
            "feature_count": len(feature_cols),
        }

    def _predict(self, latest_row: pd.DataFrame, symbol: str) -> dict:
        """Get latest predictions."""
        if self.ensemble is None:
            return {"error": "Model not trained"}

        feature_cols = [
            c
            for c in latest_row.columns
            if c
            not in ["timestamp", "open", "high", "low", "close", "volume", "returns", "label", "forward_return"]
            and latest_row[c].dtype in ["float64", "int64"]
        ]

        X = latest_row[feature_cols].fillna(0)
        pred_proba = self.ensemble.predict_proba(X)[0]

        confidence = np.max(pred_proba)
        signal = "BUY" if pred_proba[1] > 0.55 else "SELL" if pred_proba[1] < 0.45 else "HOLD"

        return {
            "symbol": symbol,
            "timestamp": latest_row.iloc[0]["timestamp"],
            "signal": signal,
            "buy_probability": round(pred_proba[1], 4),
            "confidence": round(confidence, 4),
            "current_price": float(latest_row.iloc[0]["close"]),
        }

    def _analyze_risk(self, returns: pd.Series) -> dict:
        """Quick Monte Carlo risk metrics."""
        if len(returns) < 30:
            return {"error": "Insufficient data"}

        try:
            # Value at Risk (95%)
            var_95 = value_at_risk_mc(returns, confidence_level=0.95, n_simulations=1000)

            # Conditional VaR (Expected Shortfall)
            cvar_95 = conditional_var_mc(returns, confidence_level=0.95, n_simulations=1000)

            # Sharpe ratio
            sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0

            return {
                "sharpe_ratio": round(sharpe, 4),
                "var_95": round(var_95, 4),
                "cvar_95": round(cvar_95, 4),
                "daily_volatility": round(np.std(returns), 4),
                "max_drawdown": round(float(np.min((np.cumprod(1 + returns) - np.maximum.accumulate(np.cumprod(1 + returns))) / np.maximum.accumulate(np.cumprod(1 + returns)))), 4),
            }
        except Exception as e:
            logger.warning(f"Risk analysis failed: {e}")
            return {"error": str(e)}

    def _save_results(
        self,
        symbol: str,
        df: pd.DataFrame,
        predictions: dict,
        metrics: dict,
    ) -> None:
        """Save to Supabase and local parquet."""
        # Save bars to Supabase
        self.cache.save_bars(symbol, df[["timestamp", "open", "high", "low", "close", "volume"]])

        # Save recent bars to parquet
        self.loader.save_parquet(df.tail(252), symbol, kind="bars_recent")

        logger.info(f"Saved results for {symbol}")

    @staticmethod
    def _auc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Simple AUC calculation."""
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_true, y_pred)
