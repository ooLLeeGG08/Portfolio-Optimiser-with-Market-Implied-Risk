"""Covariance estimation. See METHODOLOGY §4 and §6.

All estimators take ALIGNED daily simple returns and return ANNUALISED covariance
(x252, core.conventions). Callers are responsible for passing only data available at the
decision date (see portfolio.optimization.estimate_inputs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from core import conventions as cv

CovMethod = Literal["sample", "ledoit_wolf"]


@dataclass(frozen=True)
class CovarianceEstimate:
    matrix: pd.DataFrame  # annualised
    method: CovMethod
    n_observations: int
    shrinkage: float | None = None  # Ledoit-Wolf intensity in [0, 1]


def _check_returns(returns: pd.DataFrame, min_obs: int = 2) -> None:
    if returns.isna().any().any():
        raise ValueError("Returns contain missing values; align prices first (no filling).")
    if len(returns) < max(min_obs, returns.shape[1] + 1):
        raise ValueError(f"Need more observations ({len(returns)}) than assets "
                         f"({returns.shape[1]}) to estimate a covariance matrix.")


def sample_covariance(returns: pd.DataFrame) -> CovarianceEstimate:
    _check_returns(returns)
    cov = cv.annualize_cov(returns.cov(ddof=1))
    return CovarianceEstimate(cov, "sample", len(returns))


def ledoit_wolf_covariance(returns: pd.DataFrame) -> CovarianceEstimate:
    """Ledoit-Wolf (2004) shrinkage towards a scaled identity. Reduces estimation noise
    that otherwise drives mean-variance optimisers to extreme weights."""
    _check_returns(returns)
    lw = LedoitWolf(assume_centered=False).fit(returns.to_numpy())
    # sklearn uses the 1/T (MLE) normalisation; rescale to the unbiased 1/(T-1) convention
    # so a zero-shrinkage estimate coincides with sample_covariance.
    t = len(returns)
    daily = lw.covariance_ * t / (t - 1)
    cov = pd.DataFrame(cv.annualize_cov(daily), index=returns.columns, columns=returns.columns)
    return CovarianceEstimate(cov, "ledoit_wolf", t, float(lw.shrinkage_))


def estimate_covariance(returns: pd.DataFrame, method: CovMethod = "ledoit_wolf") -> CovarianceEstimate:
    if method == "sample":
        return sample_covariance(returns)
    if method == "ledoit_wolf":
        return ledoit_wolf_covariance(returns)
    raise ValueError(f"Unknown covariance method: {method}")


def cov_to_corr(cov: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Split a covariance matrix into volatilities and a correlation matrix."""
    vol = np.sqrt(np.diag(cov))
    if np.any(vol <= 0):
        raise ValueError("Covariance matrix has a non-positive variance.")
    corr = cov / np.outer(vol, vol)
    return pd.Series(vol, index=cov.index), pd.DataFrame(corr, index=cov.index, columns=cov.columns)


def is_positive_semidefinite(m: pd.DataFrame | np.ndarray, tol: float = 1e-10) -> bool:
    a = np.asarray(m, dtype=float)
    return bool(np.allclose(a, a.T, atol=1e-12) and np.linalg.eigvalsh(a).min() >= -tol)
