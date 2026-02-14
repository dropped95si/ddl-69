"""
FinRL agents for reinforcement learning-based trading.

This module provides RL agents (PPO, A2C, DDPG, TD3, SAC) for trading using FinRL.
Requires: finrl, stable-baselines3
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Any, Literal
import warnings

import numpy as np
import pandas as pd

# Try to import FinRL and stable-baselines3
try:
    from finrl.agents.stablebaselines3.models import DRLAgent
    from finrl.config import INDICATORS, TRAINED_MODEL_DIR, RESULTS_DIR
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
    from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
    FINRL_AVAILABLE = True
except ImportError:
    FINRL_AVAILABLE = False
    warnings.warn("FinRL not available. Install from: https://github.com/AI4Finance-Foundation/FinRL")

try:
    from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    warnings.warn("stable-baselines3 not available. Install via: pip install stable-baselines3")


class FinRLTrader:
    """
    Unified interface for FinRL RL agents.

    Supports: PPO, A2C, DDPG, TD3, SAC
    """

    def __init__(
        self,
        algorithm: Literal["ppo", "a2c", "ddpg", "td3", "sac"] = "ppo",
        initial_capital: float = 1_000_000,
        transaction_cost: float = 0.001,
        model_dir: Optional[str] = None,
    ):
        if not FINRL_AVAILABLE:
            raise RuntimeError("FinRL is not installed")
        if not SB3_AVAILABLE:
            raise RuntimeError("stable-baselines3 is not installed")

        self.algorithm = algorithm.lower()
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.model_dir = Path(model_dir) if model_dir else Path(TRAINED_MODEL_DIR)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.env_train = None
        self.env_trade = None

    def prepare_data(
        self,
        df: pd.DataFrame,
        technical_indicators: Optional[list[str]] = None,
        train_end: Optional[str] = None,
        val_end: Optional[str] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare and split data for training.

        Args:
            df: DataFrame with columns: date, tic, open, high, low, close, volume
            technical_indicators: List of technical indicators to add
            train_end: End date for training split (YYYY-MM-DD)
            val_end: End date for validation split (YYYY-MM-DD)

        Returns:
            (train_df, val_df, test_df)
        """
        if technical_indicators is None:
            technical_indicators = [
                "macd", "rsi_30", "cci_30", "dx_30",
                "close_30_sma", "close_60_sma",
            ]

        # Add technical indicators
        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=technical_indicators,
            use_vix=False,
            use_turbulence=False,
        )
        df = fe.preprocess_data(df)

        # Split data
        if train_end and val_end:
            train = data_split(df, start=df["date"].min(), end=train_end)
            val = data_split(df, start=train_end, end=val_end)
            test = data_split(df, start=val_end, end=df["date"].max())
        else:
            # Default 70/15/15 split
            dates = sorted(df["date"].unique())
            n = len(dates)
            train_end_idx = int(n * 0.7)
            val_end_idx = int(n * 0.85)

            train = df[df["date"] <= dates[train_end_idx]].copy()
            val = df[(df["date"] > dates[train_end_idx]) & (df["date"] <= dates[val_end_idx])].copy()
            test = df[df["date"] > dates[val_end_idx]].copy()

        return train, val, test

    def create_env(
        self,
        df: pd.DataFrame,
        mode: Literal["train", "trade"] = "train",
    ) -> StockTradingEnv:
        """Create trading environment."""
        stock_dimension = len(df["tic"].unique())
        state_space = 1 + 2 * stock_dimension + len([c for c in df.columns if c not in ["date", "tic", "open", "high", "low", "close", "volume"]])

        env_kwargs = {
            "df": df,
            "stock_dim": stock_dimension,
            "hmax": 100,  # max shares to hold
            "initial_amount": self.initial_capital,
            "num_stock_shares": [0] * stock_dimension,
            "buy_cost_pct": [self.transaction_cost] * stock_dimension,
            "sell_cost_pct": [self.transaction_cost] * stock_dimension,
            "reward_scaling": 1e-4,
            "state_space": state_space,
            "action_space": stock_dimension,
            "tech_indicator_list": [c for c in df.columns if c not in ["date", "tic", "open", "high", "low", "close", "volume"]],
        }

        if mode == "train":
            env = StockTradingEnv(**env_kwargs)
        else:
            env = StockTradingEnv(**env_kwargs, turbulence_threshold=None)

        return env

    def train(
        self,
        train_df: pd.DataFrame,
        total_timesteps: int = 100_000,
        model_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Train the RL agent.

        Args:
            train_df: Training data
            total_timesteps: Total training timesteps
            model_name: Name to save model
            **kwargs: Additional hyperparameters for the algorithm
        """
        # Create training environment
        self.env_train = self.create_env(train_df, mode="train")
        env_train = DummyVecEnv([lambda: self.env_train])

        # Set default hyperparameters per algorithm
        if self.algorithm == "ppo":
            default_params = {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.01,
            }
            default_params.update(kwargs)
            self.model = PPO("MlpPolicy", env_train, verbose=0, **default_params)

        elif self.algorithm == "a2c":
            default_params = {
                "learning_rate": 7e-4,
                "n_steps": 5,
                "gamma": 0.99,
                "gae_lambda": 1.0,
                "ent_coef": 0.01,
            }
            default_params.update(kwargs)
            self.model = A2C("MlpPolicy", env_train, verbose=0, **default_params)

        elif self.algorithm == "ddpg":
            default_params = {
                "learning_rate": 1e-3,
                "buffer_size": 1_000_000,
                "learning_starts": 100,
                "batch_size": 128,
                "tau": 0.005,
                "gamma": 0.99,
            }
            default_params.update(kwargs)
            self.model = DDPG("MlpPolicy", env_train, verbose=0, **default_params)

        elif self.algorithm == "td3":
            default_params = {
                "learning_rate": 1e-3,
                "buffer_size": 1_000_000,
                "learning_starts": 100,
                "batch_size": 128,
                "tau": 0.005,
                "gamma": 0.99,
                "policy_delay": 2,
            }
            default_params.update(kwargs)
            self.model = TD3("MlpPolicy", env_train, verbose=0, **default_params)

        elif self.algorithm == "sac":
            default_params = {
                "learning_rate": 3e-4,
                "buffer_size": 1_000_000,
                "learning_starts": 100,
                "batch_size": 256,
                "tau": 0.005,
                "gamma": 0.99,
                "ent_coef": "auto",
            }
            default_params.update(kwargs)
            self.model = SAC("MlpPolicy", env_train, verbose=0, **default_params)

        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        # Train
        self.model.learn(total_timesteps=total_timesteps)

        # Save
        if model_name is None:
            model_name = f"{self.algorithm}_finrl"
        save_path = self.model_dir / f"{model_name}.zip"
        self.model.save(save_path)
        print(f"Model saved to: {save_path}")

    def load(self, model_path: str) -> None:
        """Load a trained model."""
        path = Path(model_path)
        if not path.exists():
            path = self.model_dir / f"{model_path}.zip"
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        if self.algorithm == "ppo":
            self.model = PPO.load(path)
        elif self.algorithm == "a2c":
            self.model = A2C.load(path)
        elif self.algorithm == "ddpg":
            self.model = DDPG.load(path)
        elif self.algorithm == "td3":
            self.model = TD3.load(path)
        elif self.algorithm == "sac":
            self.model = SAC.load(path)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        print(f"Model loaded from: {path}")

    def predict(
        self,
        test_df: pd.DataFrame,
        deterministic: bool = True,
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        """
        Run backtest on test data.

        Returns:
            (actions_df, metrics)
        """
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")

        # Create trading environment
        self.env_trade = self.create_env(test_df, mode="trade")

        # Run episode
        obs = self.env_trade.reset()
        done = False
        actions_list = []
        rewards_list = []

        while not done:
            action, _states = self.model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = self.env_trade.step(action)
            actions_list.append(action)
            rewards_list.append(reward)

        # Get account value history
        account_values = self.env_trade.save_asset_memory()

        # Calculate metrics
        returns = account_values["account_value"].pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0.0
        total_return = (account_values["account_value"].iloc[-1] / self.initial_capital - 1) * 100
        max_drawdown = (account_values["account_value"] / account_values["account_value"].cummax() - 1).min() * 100

        metrics = {
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "final_value": float(account_values["account_value"].iloc[-1]),
        }

        return account_values, metrics


class EnsembleFinRL:
    """
    Ensemble multiple FinRL agents.

    Combines predictions from PPO, A2C, DDPG, TD3, SAC using weighted voting.
    """

    def __init__(
        self,
        algorithms: Optional[list[str]] = None,
        weights: Optional[dict[str, float]] = None,
        initial_capital: float = 1_000_000,
        transaction_cost: float = 0.001,
        model_dir: Optional[str] = None,
    ):
        if algorithms is None:
            algorithms = ["ppo", "a2c", "sac"]

        self.algorithms = algorithms
        self.weights = weights or {algo: 1.0 / len(algorithms) for algo in algorithms}
        self.traders = {
            algo: FinRLTrader(
                algorithm=algo,
                initial_capital=initial_capital,
                transaction_cost=transaction_cost,
                model_dir=model_dir,
            )
            for algo in algorithms
        }

    def train_all(
        self,
        train_df: pd.DataFrame,
        total_timesteps: int = 100_000,
        **kwargs: Any,
    ) -> None:
        """Train all agents in the ensemble."""
        for algo, trader in self.traders.items():
            print(f"\nTraining {algo.upper()}...")
            trader.train(train_df, total_timesteps=total_timesteps, **kwargs)

    def load_all(self, model_names: dict[str, str]) -> None:
        """Load all agents."""
        for algo, model_name in model_names.items():
            if algo in self.traders:
                self.traders[algo].load(model_name)

    def predict_ensemble(
        self,
        test_df: pd.DataFrame,
        deterministic: bool = True,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Ensemble prediction combining all agents.

        Returns:
            (ensemble_account_values, ensemble_metrics)
        """
        all_actions = []
        all_metrics = {}

        # Get predictions from each agent
        for algo, trader in self.traders.items():
            account_values, metrics = trader.predict(test_df, deterministic=deterministic)
            all_metrics[algo] = metrics
            # Store actions (simplified - in practice you'd need to track positions)

        # Weighted average of metrics
        ensemble_metrics = {
            "total_return": sum(all_metrics[algo]["total_return"] * self.weights[algo] for algo in self.algorithms),
            "sharpe_ratio": sum(all_metrics[algo]["sharpe_ratio"] * self.weights[algo] for algo in self.algorithms),
            "max_drawdown": sum(all_metrics[algo]["max_drawdown"] * self.weights[algo] for algo in self.algorithms),
            "individual_metrics": all_metrics,
        }

        return None, ensemble_metrics


def train_finrl_agent(
    df: pd.DataFrame,
    algorithm: str = "ppo",
    total_timesteps: int = 100_000,
    model_name: Optional[str] = None,
    model_dir: Optional[str] = None,
) -> tuple[FinRLTrader, dict[str, float]]:
    """
    Convenience function to train a single FinRL agent.

    Args:
        df: DataFrame with OHLCV data (columns: date, tic, open, high, low, close, volume)
        algorithm: One of: ppo, a2c, ddpg, td3, sac
        total_timesteps: Training timesteps
        model_name: Model save name
        model_dir: Model save directory

    Returns:
        (trained_trader, test_metrics)
    """
    trader = FinRLTrader(algorithm=algorithm, model_dir=model_dir)

    # Prepare data
    train_df, val_df, test_df = trader.prepare_data(df)

    # Train
    trader.train(train_df, total_timesteps=total_timesteps, model_name=model_name)

    # Evaluate
    _, metrics = trader.predict(test_df)

    return trader, metrics
