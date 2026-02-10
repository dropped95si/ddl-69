"""Expert wrapper for ML predictions - integrates with probability ledger"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from ddl69.core.settings import Settings
from ddl69.ledger.supabase_ledger import SupabaseLedger
from ddl69.data.loaders import DataLoader
from ddl69.core.real_pipeline import RealDataPipeline

logger = logging.getLogger(__name__)


class PredictionExpert:
    """Expert instance wrapping ML ensemble predictions"""

    def __init__(
        self,
        symbol: str,
        artifact_root: Optional[str] = None,
        settings: Optional[Settings] = None,
    ):
        self.symbol = symbol
        self.artifact_root = artifact_root
        self.settings = settings or Settings()
        self.ledger = SupabaseLedger(self.settings)

        # Lazy load pipeline
        self._pipeline = None
        self._df = None

    @property
    def pipeline(self) -> RealDataPipeline:
        """Lazy load and train pipeline"""
        if self._pipeline is None:
            try:
                from ddl69.agents.sklearn_ensemble import SklearnEnsemble
                import numpy as np

                loader = DataLoader(artifact_root=self.artifact_root)
                df = loader.load(self.symbol)

                pipeline = RealDataPipeline(artifact_root=self.artifact_root)
                df = pipeline._preprocess(df)
                df = pipeline._add_indicators(df, self.symbol)
                df = pipeline._create_labels(df)

                # Train ensemble
                split_idx = int(len(df) * 0.7)
                train_df = df.iloc[:split_idx].copy()
                test_df = df.iloc[split_idx:].copy()

                feature_cols = [
                    c for c in df.columns
                    if c not in [
                        "timestamp", "open", "high", "low", "close", "volume",
                        "returns", "label", "forward_return"
                    ]
                    and df[c].dtype in ["float64", "int64"]
                ]

                X_train = train_df[feature_cols].fillna(0)
                y_train = (train_df["label"] > 0).astype(int)
                X_test = test_df[feature_cols].fillna(0)
                y_test = (test_df["label"] > 0).astype(int)

                pipeline.ensemble = SklearnEnsemble(
                    task="classification",
                    models=["rf", "xgb", "lgb"],
                    voting="soft",
                )
                pipeline.ensemble.fit(X_train, y_train)

                pipeline._test_accuracy = pipeline.ensemble.score(X_test, y_train)
                pipeline._df = df
                pipeline._feature_cols = feature_cols

                self._pipeline = pipeline
                self._df = df

            except Exception as e:
                logger.error(f"Failed to load pipeline for {self.symbol}: {e}")
                raise

        return self._pipeline

    def predict(self, horizon_days: int = 5) -> dict:
        """Generate prediction as expert forecast for ledger

        Returns:
          {
            probs: {REJECT: 0.2, BREAK_FAIL: 0.1, ACCEPT_CONTINUE: 0.7},
            confidence: 0.7,
            signal: BUY/SELL/HOLD,
            raw_probability: 0.75,
            supporting_indicators: [features],
            uncertainty: {entropy: 0.85}
          }
        """
        try:
            pipeline = self.pipeline
            latest_row = pipeline._df.iloc[-1:].copy()
            pred = pipeline._predict(latest_row, self.symbol)

            if "error" in pred:
                logger.error(f"Prediction error: {pred['error']}")
                # Return neutral forecast on error
                return {
                    "probs": {"REJECT": 0.33, "BREAK_FAIL": 0.33, "ACCEPT_CONTINUE": 0.34},
                    "confidence": 0.1,
                    "signal": "HOLD",
                    "raw_probability": 0.5,
                    "supporting_indicators": [],
                    "uncertainty": {"entropy": 1.5},
                }

            # Map ML signal to probability forecast
            buy_prob = pred["buy_probability"]
            confidence = pred["confidence"]

            # Triple-state forecast for ledger
            if pred["signal"] == "BUY" and buy_prob > 0.6:
                probs = {
                    "REJECT": buy_prob * 0.05,  # Low reject
                    "BREAK_FAIL": buy_prob * 0.15,  # Low stop-loss
                    "ACCEPT_CONTINUE": buy_prob * 0.80,  # High acceptance
                }
            elif pred["signal"] == "SELL" and buy_prob < 0.4:
                probs = {
                    "REJECT": (1 - buy_prob) * 0.80,  # High reject (sell)
                    "BREAK_FAIL": (1 - buy_prob) * 0.15,
                    "ACCEPT_CONTINUE": (1 - buy_prob) * 0.05,  # Low acceptance
                }
            else:  # HOLD
                probs = {
                    "REJECT": 0.25,
                    "BREAK_FAIL": 0.25,
                    "ACCEPT_CONTINUE": 0.50,  # Balanced, slight lean to continue
                }

            # Calculate entropy
            from scipy.stats import entropy as calc_entropy
            ent = calc_entropy(list(probs.values()))

            return {
                "probs": probs,
                "confidence": confidence,
                "signal": pred["signal"],
                "raw_probability": buy_prob,
                "supporting_indicators": [
                    f"Accuracy: {getattr(pipeline, '_test_accuracy', 0):.3f}",
                    f"Sharpe: {getattr(pipeline, '_test_sharpe', 0):.3f}",
                    f"Price: ${pred['current_price']:.2f}",
                ],
                "uncertainty": {"entropy": float(ent)},
            }

        except Exception as e:
            logger.error(f"Error in predict: {e}")
            raise

    def to_ledger(self, event_id: str, run_id: str) -> dict:
        """Convert prediction to ledger-compatible expert forecast

        Args:
          event_id: Event ID from state event
          run_id: Run ID from ledger.create_run()

        Returns:
          Dict ready for ledger.insert_expert_forecast()
        """
        pred = self.predict()

        return {
            "run_id": run_id,
            "event_id": event_id,
            "expert_name": f"ml_ensemble_{self.symbol}",
            "expert_version": "0.8",
            "probs_json": pred["probs"],
            "confidence": pred["confidence"],
            "uncertainty_json": pred["uncertainty"],
            "loss_hint": "logloss",
            "supports_calibration": True,
            "calibration_group": "ml_ensemble",
            "features_uri": None,
            "artifact_uris": [],
            "reasons_json": pred["supporting_indicators"],
            "debug_json": {
                "signal": pred["signal"],
                "raw_probability": pred["raw_probability"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        }

    def upsert_forecast(self, subject_id: str, horizon_days: int = 5) -> str:
        """Generate event + insert expert forecast into ledger

        Args:
          subject_id: Ticker symbol (or subject for prediction)
          horizon_days: Forecast horizon

        Returns:
          event_id created
        """
        try:
            now = datetime.now(timezone.utc)

            # Create run
            run_id = self.ledger.create_run(
                asof_ts=now,
                mode="live",
                config_hash="ml_ensemble",
                code_version="0.8",
            )

            # Create state event
            event_id = f"{subject_id}|ml_signal|{now.date().isoformat()}"
            self.ledger.upsert_event(
                event_id=event_id,
                subject_type="ticker",
                subject_id=subject_id,
                event_type="state_event",
                asof_ts=now,
                horizon_json={
                    "type": "time",
                    "value": horizon_days,
                    "unit": "d",
                },
                event_params_json={"source": "ml_ensemble"},
                context_json={"method": "ensemble", "version": "0.8"},
            )

            # Insert expert forecast
            forecast = self.to_ledger(event_id, run_id)
            self.ledger.insert_expert_forecast(**forecast)

            logger.info(f"Upserted forecast for {subject_id}: {event_id}")
            return event_id

        except Exception as e:
            logger.error(f"Error upserting forecast: {e}")
            raise


class EnsembleExpertPortfolio:
    """Portfolio of experts, one per symbol"""

    def __init__(
        self,
        symbols: list[str],
        artifact_root: Optional[str] = None,
        settings: Optional[Settings] = None,
    ):
        self.symbols = symbols
        self.artifact_root = artifact_root
        self.settings = settings or Settings()
        self.ledger = SupabaseLedger(self.settings)

        # Create expert per symbol
        self.experts = {
            symbol: PredictionExpert(symbol, artifact_root, settings)
            for symbol in symbols
        }

    def predict_all(self) -> dict:
        """Get predictions from all experts

        Returns:
          {symbol: {probs, confidence, signal, ...}, ...}
        """
        results = {}
        for symbol, expert in self.experts.items():
            try:
                results[symbol] = expert.predict()
            except Exception as e:
                logger.warning(f"Failed to predict for {symbol}: {e}")
                results[symbol] = None

        return results

    def upsert_all(self) -> list[str]:
        """Upsert forecasts for all symbols

        Returns:
          List of event_ids created
        """
        event_ids = []
        for symbol, expert in self.experts.items():
            try:
                event_id = expert.upsert_forecast(symbol)
                event_ids.append(event_id)
            except Exception as e:
                logger.warning(f"Failed to upsert forecast for {symbol}: {e}")

        return event_ids

    def consensus(self) -> dict:
        """Aggregate portfolio-level consensus"""
        preds = self.predict_all()

        buy_count = sum(1 for p in preds.values() if p and p["signal"] == "BUY")
        sell_count = sum(1 for p in preds.values() if p and p["signal"] == "SELL")
        hold_count = sum(1 for p in preds.values() if p and p["signal"] == "HOLD")
        total = len([p for p in preds.values() if p])

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_symbols": total,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "hold_count": hold_count,
            "buy_pct": buy_count / total if total > 0 else 0,
            "sell_pct": sell_count / total if total > 0 else 0,
            "predictions": preds,
        }
