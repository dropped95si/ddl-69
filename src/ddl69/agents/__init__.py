"""
RL and ML agents for trading: FinRL RL agents and sklearn ensembles.
"""

from __future__ import annotations

__all__ = [
    "FinRLTrader",
    "EnsembleFinRL",
    "train_finrl_agent",
    "SklearnEnsemble",
    "train_sklearn_ensemble",
]

try:
    from .finrl_agents import FinRLTrader, EnsembleFinRL, train_finrl_agent
except ImportError:
    FinRLTrader = None
    EnsembleFinRL = None
    train_finrl_agent = None

try:
    from .sklearn_ensemble import SklearnEnsemble, train_sklearn_ensemble
except ImportError:
    SklearnEnsemble = None
    train_sklearn_ensemble = None
