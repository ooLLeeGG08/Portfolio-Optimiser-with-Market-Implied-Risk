"""Value at Risk. See METHODOLOGY §5.

Loss convention: L = -R. VaR_a is the a-quantile of the LOSS distribution, reported as a
positive number when it is a loss. A 1-day 95% VaR of 0.021 means: on 5% of days the loss
is expected to exceed 2.1% of portfolio value. VaR is a threshold, not a worst case.

``confidence`` is the VaR confidence level a (e.g. 0.95, 0.99).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _check_confidence(confidence: float) -> None:
    if not 0.5 < confidence < 1:
        raise ValueError("Confidence level must be between 50% and 100% (e.g. 0.95).")


# --- historical -----------------------------------------------------------------------------

def historical_var(returns: pd.Series | np.ndarray, confidence: float) -> float:
    """Empirical a-quantile of losses (linear interpolation between order statistics)."""
    _check_confidence(confidence)
    losses = -np.asarray(returns, dtype=float)
    if losses.size < 20:
        raise ValueError("Historical VaR needs at least 20 observations.")
    return float(np.quantile(losses, confidence))


# --- parametric ---------------------------------------------------------------------------

def parametric_var(mu: float, sigma: float, confidence: float) -> float:
    """Normal VaR: -mu + sigma * z_a, z_a = Phi^-1(a)."""
    _check_confidence(confidence)
    return float(-mu + sigma * stats.norm.ppf(confidence))


def cornish_fisher_z(p: float, skew: float, excess_kurtosis: float) -> float:
    """Cornish-Fisher adjusted standard-normal quantile at probability p:
    z + (z^2 - 1) S/6 + (z^3 - 3z) K/24 - (2z^3 - 5z) S^2/36."""
    z = stats.norm.ppf(p)
    s, k = skew, excess_kurtosis
    return float(z + (z ** 2 - 1) * s / 6 + (z ** 3 - 3 * z) * k / 24
                 - (2 * z ** 3 - 5 * z) * s ** 2 / 36)


def cornish_fisher_is_monotone(skew: float, excess_kurtosis: float) -> bool:
    """The expansion is only a valid quantile function where it is increasing. For large
    skew/kurtosis it is not, and the CF VaR must not be trusted."""
    z = np.linspace(-4, 4, 801)
    s, k = skew, excess_kurtosis
    dz = 1 + 2 * z * s / 6 + (3 * z ** 2 - 3) * k / 24 - (6 * z ** 2 - 5) * s ** 2 / 36
    return bool(np.all(dz > 0))


def cornish_fisher_var(mu: float, sigma: float, skew: float, excess_kurtosis: float,
                       confidence: float) -> float:
    """VaR from the skew/kurtosis-adjusted lower-tail quantile of returns."""
    _check_confidence(confidence)
    return float(-(mu + sigma * cornish_fisher_z(1 - confidence, skew, excess_kurtosis)))


# --- Monte Carlo ----------------------------------------------------------------------------

def student_t_dof_from_kurtosis(excess_kurtosis: float, lo: float = 4.1, hi: float = 50.0) -> float | None:
    """Method of moments: a Student-t with nu > 4 has excess kurtosis 6 / (nu - 4).
    Returns None (use normal) if the sample shows no excess kurtosis."""
    if excess_kurtosis <= 6 / (hi - 4):
        return None
    return float(np.clip(4 + 6 / excess_kurtosis, lo, hi))


def simulate_portfolio_returns(weights: np.ndarray, mean: np.ndarray, cov: np.ndarray,
                               n_sims: int = 100_000, dof: float | None = None,
                               seed: int = 0) -> np.ndarray:
    """Simulate one-period asset returns from a multivariate normal (dof=None) or a
    multivariate Student-t scaled so its covariance equals ``cov``, and aggregate them
    linearly into portfolio simple returns (w . R)."""
    rng = np.random.default_rng(seed)
    n = len(weights)
    chol = np.linalg.cholesky(cov + 1e-14 * np.eye(n))
    z = rng.standard_normal((n_sims, n)) @ chol.T
    if dof is not None:
        if dof <= 2:
            raise ValueError("Student-t degrees of freedom must exceed 2 for a finite variance.")
        mix = np.sqrt(rng.chisquare(dof, size=(n_sims, 1)) / dof)
        z = z / mix * np.sqrt((dof - 2) / dof)  # rescale: Var(t_nu) = nu / (nu - 2)
    return (mean + z) @ weights


def monte_carlo_var(simulated_returns: np.ndarray, confidence: float) -> float:
    _check_confidence(confidence)
    return float(np.quantile(-simulated_returns, confidence))
