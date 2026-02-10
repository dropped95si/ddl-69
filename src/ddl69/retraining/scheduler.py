"""Real-time model retraining with drift detection"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
import pickle

import pandas as pd
import numpy as np

from ddl69.data.loaders import DataLoader
from ddl69.core.real_pipeline import RealDataPipeline
from ddl69.agents.sklearn_ensemble import SklearnEnsemble

logger = logging.getLogger(__name__)


class ModelDriftDetector:
    """Detect model performance degradation"""

    def __init__(self, drift_threshold: float = 0.05, window_size: int = 50):
        """
        Args:
          drift_threshold: accuracy drop % to trigger retrain (0.05 = 5%)
          window_size: samples to evaluate for drift (recent test set)
        """
        self.drift_threshold = drift_threshold
        self.window_size = window_size
        self.baseline_accuracy = None
        self.accuracy_history = []

    def record_accuracy(self, accuracy: float) -> None:
        """Add accuracy sample to history"""
        self.accuracy_history.append({
            "timestamp": datetime.now(timezone.utc),
            "accuracy": accuracy,
        })

    def set_baseline(self, accuracy: float) -> None:
        """Set baseline accuracy from training"""
        self.baseline_accuracy = accuracy
        logger.info(f"Baseline accuracy: {accuracy:.3f}")

    def detect_drift(self) -> dict:
        """Check if model has drifted

        Returns:
          {
            has_drifted: bool,
            current_accuracy: float,
            baseline_accuracy: float,
            degradation: float,  # % drop
            recent_trend: str,  # UP, DOWN, STABLE
          }
        """
        if not self.accuracy_history or self.baseline_accuracy is None:
            return {
                "has_drifted": False,
                "reason": "Not enough data",
                "current_accuracy": None,
                "baseline_accuracy": self.baseline_accuracy,
                "degradation": 0,
            }

        # Recent accuracy (last window_size samples)
        recent = self.accuracy_history[-self.window_size:]
        current_accuracy = np.mean([s["accuracy"] for s in recent])

        # Degradation
        degradation = (self.baseline_accuracy - current_accuracy) / self.baseline_accuracy

        # Trend
        if len(recent) > 5:
            trend_old = np.mean([s["accuracy"] for s in recent[:-5]])
            trend_new = np.mean([s["accuracy"] for s in recent[-5:]])
            if trend_new > trend_old:
                trend = "UP"
            elif trend_new < trend_old:
                trend = "DOWN"
            else:
                trend = "STABLE"
        else:
            trend = "INSUFFICIENT_DATA"

        has_drifted = degradation > self.drift_threshold

        return {
            "has_drifted": has_drifted,
            "current_accuracy": round(current_accuracy, 4),
            "baseline_accuracy": round(self.baseline_accuracy, 4),
            "degradation": round(degradation, 4),
            "degradation_pct": f"{degradation*100:.1f}%",
            "recent_trend": trend,
            "why_retrain": "Accuracy dropped >5%" if has_drifted else None,
        }


class ModelVersionManager:
    """Manage model versions with history"""

    def __init__(self, symbol: str, artifact_root: Optional[str] = None):
        self.symbol = symbol
        self.artifact_root = Path(artifact_root or ".artifacts")
        self.model_dir = self.artifact_root / "models" / symbol
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_path = self.model_dir / "metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> dict:
        """Load existing metadata or create new"""
        if self.metadata_path.exists():
            with open(self.metadata_path) as f:
                return json.load(f)
        return {
            "symbol": self.symbol,
            "versions": [],
            "current_version": None,
        }

    def save_model(
        self,
        model: SklearnEnsemble,
        accuracy: float,
        sharpe: float,
    ) -> str:
        """Save model and metadata

        Returns:
          version_id (e.g., "v1_20260210_143000")
        """
        version_id = f"v{len(self.metadata['versions'])+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Save model
        model_path = self.model_dir / f"{version_id}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Update metadata
        version_info = {
            "version_id": version_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "accuracy": accuracy,
            "sharpe": sharpe,
            "path": str(model_path),
        }
        self.metadata["versions"].append(version_info)
        self.metadata["current_version"] = version_id

        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

        logger.info(f"Saved model {version_id}: accuracy={accuracy:.3f}, sharpe={sharpe:.3f}")
        return version_id

    def load_model(self, version_id: Optional[str] = None) -> Optional[SklearnEnsemble]:
        """Load model by version (current if not specified)"""
        if version_id is None:
            version_id = self.metadata.get("current_version")

        if not version_id:
            return None

        # Find version
        version_info = next(
            (v for v in self.metadata["versions"] if v["version_id"] == version_id),
            None,
        )

        if not version_info:
            logger.warning(f"Version {version_id} not found")
            return None

        try:
            with open(version_info["path"], "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load {version_id}: {e}")
            return None

    def get_history(self, limit: int = 10) -> list[dict]:
        """Get recent model versions"""
        return self.metadata["versions"][-limit:]


class AutoRetrainingScheduler:
    """Schedule and execute model retraining with drift detection"""

    def __init__(self, symbol: str, artifact_root: Optional[str] = None):
        self.symbol = symbol
        self.artifact_root = artifact_root
        self.loader = DataLoader(artifact_root)
        self.drift_detector = ModelDriftDetector()
        self.version_manager = ModelVersionManager(symbol, artifact_root)
        self.last_retrain = None

    def should_retrain(self, min_hours_since_last: int = 24) -> tuple[bool, str]:
        """Decide if retraining is needed

        Returns:
          (should_retrain, reason)
        """
        # Check drift
        drift_status = self.drift_detector.detect_drift()
        if drift_status["has_drifted"]:
            return True, f"Drift detected: {drift_status['degradation_pct']} degradation"

        # Check time since last retrain
        if self.last_retrain:
            hours_since = (datetime.now(timezone.utc) - self.last_retrain).total_seconds() / 3600
            if hours_since >= min_hours_since_last:
                return True, f"Scheduled retrain ({hours_since:.0f} hours since last)"

        return False, "No retrain needed"

    def retrain(self) -> dict:
        """Execute retraining with latest data

        Returns:
          {
            success: bool,
            version_id: str,
            old_accuracy: float,
            new_accuracy: float,
            improvement: float,
            drift_status: {...}
          }
        """
        try:
            logger.info(f"Starting retrain for {self.symbol}...")

            # Load fresh data
            df = self.loader.load(self.symbol)
            pipeline = RealDataPipeline(artifact_root=self.artifact_root)
            df = pipeline._preprocess(df)
            df = pipeline._add_indicators(df, self.symbol)
            df = pipeline._create_labels(df)

            # Train/test split
            split_idx = int(len(df) * 0.7)
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]

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

            # Train new model
            new_model = SklearnEnsemble(
                task="classification",
                models=["rf", "xgb", "lgb"],
                voting="soft",
            )
            new_model.fit(X_train, y_train)

            new_accuracy = new_model.score(X_test, y_test)

            # Calculate sharpe
            preds = new_model.predict_proba(X_test)[:, 1]
            pred_returns = preds * test_df["returns"].fillna(0).values
            new_sharpe = (
                np.mean(pred_returns) / (np.std(pred_returns) + 1e-8) * np.sqrt(252)
                if len(pred_returns) > 1 else 0
            )

            # Get old model accuracy
            old_model = self.version_manager.load_model()
            old_accuracy = 0.0
            if old_model:
                old_accuracy = old_model.score(X_test, y_test)

            improvement = new_accuracy - old_accuracy

            # Save new model
            version_id = self.version_manager.save_model(new_model, new_accuracy, new_sharpe)

            # Update drift detector
            self.drift_detector.set_baseline(new_accuracy)
            self.last_retrain = datetime.now(timezone.utc)

            result = {
                "success": True,
                "symbol": self.symbol,
                "version_id": version_id,
                "old_accuracy": round(old_accuracy, 4),
                "new_accuracy": round(new_accuracy, 4),
                "improvement": round(improvement, 4),
                "improvement_pct": f"{improvement*100:+.1f}%",
                "new_sharpe": round(new_sharpe, 4),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            logger.info(f"Retrain complete: {result}")
            return result

        except Exception as e:
            logger.error(f"Retrain failed for {self.symbol}: {e}")
            return {
                "success": False,
                "symbol": self.symbol,
                "error": str(e),
            }

    def get_status(self) -> dict:
        """Get current model status"""
        drift = self.drift_detector.detect_drift()
        history = self.version_manager.get_history(limit=5)

        return {
            "symbol": self.symbol,
            "current_version": self.version_manager.metadata.get("current_version"),
            "drift_status": drift,
            "last_retrain": self.last_retrain.isoformat() if self.last_retrain else None,
            "recent_versions": history,
        }
