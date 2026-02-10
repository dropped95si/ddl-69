"""
Monte Carlo simulations for strategy testing and risk analysis
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Callable


def monte_carlo_returns(
    returns: pd.Series,
    n_simulations: int = 1000,
    n_periods: int = 252,
    method: str = "bootstrap"
) -> np.ndarray:
    """
    Monte Carlo simulation of returns
    Args:
        returns: historical returns
        n_simulations: number of simulation paths
        n_periods: periods to simulate
        method: "bootstrap" or "parametric"
    Returns:
        array of shape (n_simulations, n_periods)
    """
    if method == "bootstrap":
        # Bootstrap resampling
        sims = np.zeros((n_simulations, n_periods))
        for i in range(n_simulations):
            sims[i] = np.random.choice(returns.dropna(), size=n_periods, replace=True)
        return sims

    elif method == "parametric":
        # Assume normal distribution
        mu = returns.mean()
        sigma = returns.std()
        sims = np.random.normal(mu, sigma, size=(n_simulations, n_periods))
        return sims

    else:
        raise ValueError(f"Unknown method: {method}")


def monte_carlo_portfolio(
    returns: pd.Series,
    initial_capital: float = 10000.0,
    n_simulations: int = 1000,
    n_periods: int = 252,
    method: str = "bootstrap"
) -> pd.DataFrame:
    """
    Monte Carlo simulation of portfolio value paths
    Returns:
        DataFrame with columns: [sim_0, sim_1, ..., sim_n]
    """
    ret_sims = monte_carlo_returns(returns, n_simulations, n_periods, method)

    # Convert returns to cumulative portfolio values
    portfolios = np.zeros_like(ret_sims)
    portfolios[:, 0] = initial_capital
    for t in range(1, n_periods):
        portfolios[:, t] = portfolios[:, t-1] * (1 + ret_sims[:, t])

    df = pd.DataFrame(portfolios.T, columns=[f"sim_{i}" for i in range(n_simulations)])
    return df


def sharpe_ratio_distribution(
    returns: pd.Series,
    n_simulations: int = 1000,
    n_periods: int = 252,
    risk_free_rate: float = 0.02
) -> np.ndarray:
    """
    Monte Carlo distribution of Sharpe ratios
    """
    ret_sims = monte_carlo_returns(returns, n_simulations, n_periods, "bootstrap")

    sharpes = np.zeros(n_simulations)
    for i in range(n_simulations):
        r = ret_sims[i]
        excess = r - (risk_free_rate / 252)
        sharpes[i] = np.sqrt(252) * excess.mean() / (excess.std() + 1e-9)

    return sharpes


def value_at_risk_mc(
    returns: pd.Series,
    confidence_level: float = 0.95,
    n_simulations: int = 10000,
    n_periods: int = 1
) -> float:
    """
    Value at Risk via Monte Carlo
    Returns:
        VaR at given confidence level (negative = loss)
    """
    ret_sims = monte_carlo_returns(returns, n_simulations, n_periods, "bootstrap")
    terminal_returns = ret_sims.sum(axis=1)
    var = np.percentile(terminal_returns, (1 - confidence_level) * 100)
    return var


def conditional_var_mc(
    returns: pd.Series,
    confidence_level: float = 0.95,
    n_simulations: int = 10000,
    n_periods: int = 1
) -> float:
    """
    Conditional Value at Risk (Expected Shortfall) via Monte Carlo
    """
    ret_sims = monte_carlo_returns(returns, n_simulations, n_periods, "bootstrap")
    terminal_returns = ret_sims.sum(axis=1)
    var = value_at_risk_mc(returns, confidence_level, n_simulations, n_periods)
    cvar = terminal_returns[terminal_returns <= var].mean()
    return cvar


def drawdown_distribution(
    returns: pd.Series,
    n_simulations: int = 1000,
    n_periods: int = 252
) -> dict:
    """
    Monte Carlo distribution of max drawdowns
    Returns:
        dict with percentiles of max drawdown
    """
    portfolios = monte_carlo_portfolio(returns, 10000, n_simulations, n_periods)

    max_dds = []
    for col in portfolios.columns:
        cummax = portfolios[col].cummax()
        dd = (portfolios[col] - cummax) / cummax
        max_dds.append(dd.min())

    max_dds = np.array(max_dds)
    return {
        "mean": max_dds.mean(),
        "median": np.median(max_dds),
        "5th_percentile": np.percentile(max_dds, 5),
        "95th_percentile": np.percentile(max_dds, 95),
        "worst": max_dds.min(),
    }


def permutation_test(
    returns_a: pd.Series,
    returns_b: pd.Series,
    n_permutations: int = 10000,
    statistic: Callable = None
) -> dict:
    """
    Permutation test to compare two strategies
    Args:
        returns_a, returns_b: strategy returns
        n_permutations: number of permutations
        statistic: function to compute (default: mean difference)
    Returns:
        dict with p_value and observed statistic
    """
    if statistic is None:
        statistic = lambda a, b: a.mean() - b.mean()

    observed = statistic(returns_a, returns_b)

    combined = pd.concat([returns_a, returns_b])
    n_a = len(returns_a)

    perm_stats = []
    for _ in range(n_permutations):
        shuffled = combined.sample(frac=1.0)
        perm_a = shuffled.iloc[:n_a]
        perm_b = shuffled.iloc[n_a:]
        perm_stats.append(statistic(perm_a, perm_b))

    perm_stats = np.array(perm_stats)
    p_value = (np.abs(perm_stats) >= np.abs(observed)).mean()

    return {
        "observed": observed,
        "p_value": p_value,
        "significant": p_value < 0.05
    }
