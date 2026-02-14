"""
Scikit-learn ensemble models for prediction.

Supports: RandomForest, XGBoost, LightGBM, CatBoost, ExtraTrees, GradientBoosting
"""
from __future__ import annotations

from typing import Optional, Any, Literal
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

# Try XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Install via: pip install xgboost")

# Try LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM not available. Install via: pip install lightgbm")

# Try CatBoost
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    warnings.warn("CatBoost not available. Install via: pip install catboost")


class SklearnEnsemble:
    """
    Unified ensemble of scikit-learn and gradient boosting models.

    Supports both classification and regression tasks.
    """

    def __init__(
        self,
        task: Literal["classification", "regression"] = "classification",
        models: Optional[list[str]] = None,
        voting: Literal["hard", "soft", "weighted"] = "soft",
        weights: Optional[dict[str, float]] = None,
    ):
        self.task = task
        self.voting = voting
        self.models_config = models or ["rf", "xgb", "lgb"]
        self.weights = weights
        self.models = {}
        self.ensemble = None
        self.feature_names = None

    def _create_model(self, model_name: str, **kwargs: Any) -> Any:
        """Create a single model."""
        if self.task == "classification":
            if model_name == "rf":
                return RandomForestClassifier(
                    n_estimators=kwargs.get("n_estimators", 100),
                    max_depth=kwargs.get("max_depth", 10),
                    min_samples_split=kwargs.get("min_samples_split", 5),
                    random_state=42,
                    n_jobs=-1,
                )
            elif model_name == "et":
                return ExtraTreesClassifier(
                    n_estimators=kwargs.get("n_estimators", 100),
                    max_depth=kwargs.get("max_depth", 10),
                    min_samples_split=kwargs.get("min_samples_split", 5),
                    random_state=42,
                    n_jobs=-1,
                )
            elif model_name == "gb":
                return GradientBoostingClassifier(
                    n_estimators=kwargs.get("n_estimators", 100),
                    learning_rate=kwargs.get("learning_rate", 0.1),
                    max_depth=kwargs.get("max_depth", 5),
                    random_state=42,
                )
            elif model_name == "xgb":
                if not XGBOOST_AVAILABLE:
                    raise RuntimeError("XGBoost not installed")
                return xgb.XGBClassifier(
                    n_estimators=kwargs.get("n_estimators", 100),
                    learning_rate=kwargs.get("learning_rate", 0.1),
                    max_depth=kwargs.get("max_depth", 6),
                    random_state=42,
                    n_jobs=-1,
                    use_label_encoder=False,
                    eval_metric="logloss",
                )
            elif model_name == "lgb":
                if not LIGHTGBM_AVAILABLE:
                    raise RuntimeError("LightGBM not installed")
                return lgb.LGBMClassifier(
                    n_estimators=kwargs.get("n_estimators", 100),
                    learning_rate=kwargs.get("learning_rate", 0.1),
                    max_depth=kwargs.get("max_depth", -1),
                    num_leaves=kwargs.get("num_leaves", 31),
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                )
            elif model_name == "cb":
                if not CATBOOST_AVAILABLE:
                    raise RuntimeError("CatBoost not installed")
                return cb.CatBoostClassifier(
                    iterations=kwargs.get("n_estimators", 100),
                    learning_rate=kwargs.get("learning_rate", 0.1),
                    depth=kwargs.get("max_depth", 6),
                    random_state=42,
                    verbose=False,
                )
            else:
                raise ValueError(f"Unknown model: {model_name}")

        else:  # regression
            if model_name == "rf":
                return RandomForestRegressor(
                    n_estimators=kwargs.get("n_estimators", 100),
                    max_depth=kwargs.get("max_depth", 10),
                    min_samples_split=kwargs.get("min_samples_split", 5),
                    random_state=42,
                    n_jobs=-1,
                )
            elif model_name == "et":
                return ExtraTreesRegressor(
                    n_estimators=kwargs.get("n_estimators", 100),
                    max_depth=kwargs.get("max_depth", 10),
                    min_samples_split=kwargs.get("min_samples_split", 5),
                    random_state=42,
                    n_jobs=-1,
                )
            elif model_name == "gb":
                return GradientBoostingRegressor(
                    n_estimators=kwargs.get("n_estimators", 100),
                    learning_rate=kwargs.get("learning_rate", 0.1),
                    max_depth=kwargs.get("max_depth", 5),
                    random_state=42,
                )
            elif model_name == "xgb":
                if not XGBOOST_AVAILABLE:
                    raise RuntimeError("XGBoost not installed")
                return xgb.XGBRegressor(
                    n_estimators=kwargs.get("n_estimators", 100),
                    learning_rate=kwargs.get("learning_rate", 0.1),
                    max_depth=kwargs.get("max_depth", 6),
                    random_state=42,
                    n_jobs=-1,
                )
            elif model_name == "lgb":
                if not LIGHTGBM_AVAILABLE:
                    raise RuntimeError("LightGBM not installed")
                return lgb.LGBMRegressor(
                    n_estimators=kwargs.get("n_estimators", 100),
                    learning_rate=kwargs.get("learning_rate", 0.1),
                    max_depth=kwargs.get("max_depth", -1),
                    num_leaves=kwargs.get("num_leaves", 31),
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                )
            elif model_name == "cb":
                if not CATBOOST_AVAILABLE:
                    raise RuntimeError("CatBoost not installed")
                return cb.CatBoostRegressor(
                    iterations=kwargs.get("n_estimators", 100),
                    learning_rate=kwargs.get("learning_rate", 0.1),
                    depth=kwargs.get("max_depth", 6),
                    random_state=42,
                    verbose=False,
                )
            else:
                raise ValueError(f"Unknown model: {model_name}")

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> None:
        """
        Train all models in the ensemble.

        Args:
            X: Feature matrix
            y: Target vector
            **kwargs: Hyperparameters for models
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Create and train individual models
        for model_name in self.models_config:
            print(f"Training {model_name.upper()}...")
            model = self._create_model(model_name, **kwargs)
            model.fit(X, y)
            self.models[model_name] = model

        # Create ensemble
        estimators = [(name, model) for name, model in self.models.items()]

        if self.weights:
            weights_list = [self.weights.get(name, 1.0) for name in self.models.keys()]
        else:
            weights_list = None

        if self.task == "classification":
            self.ensemble = VotingClassifier(
                estimators=estimators,
                voting=self.voting if self.voting != "weighted" else "soft",
                weights=weights_list,
                n_jobs=-1,
            )
        else:
            self.ensemble = VotingRegressor(
                estimators=estimators,
                weights=weights_list,
                n_jobs=-1,
            )

        self.ensemble.fit(X, y)
        print("Ensemble training complete")

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict using ensemble."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.ensemble.predict(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict probabilities (classification only)."""
        if self.task != "classification":
            raise RuntimeError("predict_proba only available for classification")
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.ensemble.predict_proba(X)

    def evaluate(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> dict[str, float]:
        """Evaluate ensemble performance."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        y_pred = self.predict(X)

        if self.task == "classification":
            metrics = {
                "accuracy": accuracy_score(y, y_pred),
                "precision": precision_score(y, y_pred, average="weighted", zero_division=0),
                "recall": recall_score(y, y_pred, average="weighted", zero_division=0),
                "f1": f1_score(y, y_pred, average="weighted", zero_division=0),
            }
            try:
                y_proba = self.predict_proba(X)
                if y_proba.shape[1] == 2:
                    metrics["roc_auc"] = roc_auc_score(y, y_proba[:, 1])
                else:
                    metrics["roc_auc"] = roc_auc_score(y, y_proba, multi_class="ovr", average="weighted")
            except Exception:
                metrics["roc_auc"] = None
        else:
            mse = mean_squared_error(y, y_pred)
            metrics = {
                "mse": mse,
                "rmse": np.sqrt(mse),
                "mae": mean_absolute_error(y, y_pred),
                "r2": r2_score(y, y_pred),
            }

        return metrics

    def cross_validate(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        cv: int = 5,
        time_series: bool = True,
    ) -> dict[str, Any]:
        """
        Cross-validate the ensemble.

        Args:
            X: Feature matrix
            y: Target vector
            cv: Number of folds
            time_series: Use TimeSeriesSplit if True

        Returns:
            dict with mean scores and individual model scores
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        if time_series:
            cv_splitter = TimeSeriesSplit(n_splits=cv)
        else:
            cv_splitter = cv

        # Score ensemble
        scoring = "accuracy" if self.task == "classification" else "neg_mean_squared_error"
        ensemble_scores = cross_val_score(self.ensemble, X, y, cv=cv_splitter, scoring=scoring, n_jobs=-1)

        # Score individual models
        individual_scores = {}
        for name, model in self.models.items():
            scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=scoring, n_jobs=-1)
            individual_scores[name] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "scores": scores.tolist(),
            }

        return {
            "ensemble_mean": float(np.mean(ensemble_scores)),
            "ensemble_std": float(np.std(ensemble_scores)),
            "ensemble_scores": ensemble_scores.tolist(),
            "individual_models": individual_scores,
        }

    def feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from tree-based models.

        Returns:
            DataFrame with feature importances
        """
        if self.feature_names is None:
            raise RuntimeError("Feature names not available. Fit with DataFrame.")

        importances = {}
        for name, model in self.models.items():
            if hasattr(model, "feature_importances_"):
                importances[name] = model.feature_importances_
            else:
                print(f"{name} does not have feature_importances_")

        if not importances:
            raise RuntimeError("No models with feature_importances_ found")

        # Average across models
        avg_importance = np.mean(list(importances.values()), axis=0)

        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": avg_importance,
        })
        df = df.sort_values("importance", ascending=False).head(top_n)

        # Add individual model importances
        for name, imp in importances.items():
            df[f"importance_{name}"] = df["feature"].map(dict(zip(self.feature_names, imp)))

        return df.reset_index(drop=True)


def train_sklearn_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task: str = "classification",
    models: Optional[list[str]] = None,
    **kwargs: Any,
) -> tuple[SklearnEnsemble, dict[str, Any]]:
    """
    Convenience function to train and evaluate sklearn ensemble.

    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        task: "classification" or "regression"
        models: List of model names (rf, xgb, lgb, cb, et, gb)
        **kwargs: Hyperparameters

    Returns:
        (trained_ensemble, test_metrics)
    """
    ensemble = SklearnEnsemble(task=task, models=models)
    ensemble.fit(X_train, y_train, **kwargs)
    metrics = ensemble.evaluate(X_test, y_test)
    return ensemble, metrics
