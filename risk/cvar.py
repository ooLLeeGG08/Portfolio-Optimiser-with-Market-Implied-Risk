"""Conditional Value at Risk (Expected Shortfall). See METHODOLOGY §5.

CVaR_a = E[L | L >= VaR_a]: the average loss on the days that breach the VaR threshold.
Same loss convention as risk.var (positive = loss). By construction CVaR_a >= VaR_a.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import integrate, stats

from risk.var import _check_confidence, cornish_fisher_var, historical_var


def historical_cvar(returns: pd.Series | np.ndarray, confidence: float) -> float:
    """Mean of the empirical losses at or beyond the historical VaR."""
    var = historical_var(returns, confidence)
    losses = -np.asarray(returns, dtype=float)
    return float(losses[losses >= var].mean())


def parametric_cvar(mu: float, sigma: float, confidence: float) -> float:
    """Normal Expected Shortfall: -mu + sigma * phi(z_a) / (1 - a)."""
    _check_confidence(confidence)
    z = stats.norm.ppf(confidence)
    return float(-mu + sigma * stats.norm.pdf(z) / (1 - confidence))


def cornish_fisher_cvar(mu: float, sigma: float, skew: float, excess_kurtosis: float,
                        confidence: float) -> float:
    """ES as the average of the Cornish-Fisher VaR over all tail levels:
    CVaR_a = 1/(1-a) * integral_a^1 VaR_u du (numerical quadrature)."""
    _check_confidence(confidence)
    val, _ = integrate.quad(
        lambda u: cornish_fisher_var(mu, sigma, skew, excess_kurtosis, u),
        confidence, 1 - 1e-12, limit=200)
    return float(val / (1 - confidence))


def monte_carlo_cvar(simulated_returns: np.ndarray, confidence: float) -> float:
    _check_confidence(confidence)
    losses = -simulated_returns
    var = np.quantile(losses, confidence)
    return float(losses[losses >= var].mean())
