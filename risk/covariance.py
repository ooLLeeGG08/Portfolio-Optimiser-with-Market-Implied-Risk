"""Covariance estimation. See METHODOLOGY §4 and §6.

All estimators take ALIGNED daily simple returns and return ANNUALISED covariance
(x252, core.conventions). Callers are responsible for passing only data available at the
decision date (see portfolio.optimization.estimate_inputs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from core import conventions as cv
from core.types import RiskBasis

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


# =====================================================================================
# Forward-looking covariance: Sigma = D . R . D  (METHODOLOGY §6)
# =====================================================================================

# Precedence when combining bases: the "least observed" component determines the label.
_BASIS_PRECEDENCE = [RiskBasis.HYPOTHETICAL, RiskBasis.IMPLIED_PROXY,
                     RiskBasis.MARKET_IMPLIED, RiskBasis.FORECAST, RiskBasis.OBSERVED]


def combined_basis(bases) -> RiskBasis:
    bases = set(bases)
    for b in _BASIS_PRECEDENCE:
        if b in bases:
            return b
    raise ValueError("No basis supplied.")


@dataclass(frozen=True)
class VolatilityVector:
    """Per-asset annualised volatilities that all refer to the SAME horizon."""

    values: pd.Series  # annualised decimal vols
    horizon_days: int  # the forecast / implied horizon these vols describe
    basis: dict[str, RiskBasis]  # per-asset basis (e.g. some implied, some proxy)
    description: str

    def __post_init__(self):
        if self.values.isna().any():
            raise ValueError(f"Missing volatility for: {', '.join(self.values[self.values.isna()].index)}.")
        if (self.values <= 0).any():
            raise ValueError("Volatilities must be positive.")
        if set(self.basis) != set(self.values.index):
            raise ValueError("A basis must be given for every asset.")


@dataclass(frozen=True)
class ForwardCovariance:
    matrix: pd.DataFrame  # annualised
    vols: VolatilityVector
    correlation: pd.DataFrame
    correlation_source: str
    basis: RiskBasis
    assumptions: tuple[str, ...] = field(default_factory=tuple)


FORWARD_COV_ASSUMPTIONS = (
    "Correlations are estimated from historical returns; historical correlations may not "
    "represent future market conditions and tend to rise in market stress.",
    "Volatility and correlation are treated as separable: changing a volatility does not "
    "change the correlation matrix.",
    "All volatilities refer to the same horizon and annualisation convention.",
    "The result inherits the basis of its least-observed input (e.g. any proxy volatility "
    "makes the whole covariance an implied-vol proxy estimate).",
)


def build_forward_covariance(vols: VolatilityVector, correlation: pd.DataFrame,
                             correlation_source: str = "historical") -> ForwardCovariance:
    """Sigma = D R D with D = diag(vols). R must be a valid correlation matrix; if it is not
    positive semi-definite it is projected to the nearest one (with a recorded note)."""
    tickers = list(vols.values.index)
    missing = set(tickers) - set(correlation.index)
    if missing:
        raise ValueError(f"No correlation estimates for: {', '.join(sorted(missing))}.")
    R = correlation.loc[tickers, tickers]
    notes = list(FORWARD_COV_ASSUMPTIONS)
    if not is_valid_correlation(R):
        R = nearest_correlation(R)
        notes.append("The correlation matrix was not positive semi-definite and was replaced "
                     "by the nearest valid correlation matrix (Higham 2002).")
    d = vols.values.to_numpy()
    cov = pd.DataFrame(np.outer(d, d) * R.to_numpy(), index=tickers, columns=tickers)
    basis = combined_basis(vols.basis.values())
    if basis is RiskBasis.MARKET_IMPLIED:
        notes.append("Market-implied (risk-neutral) volatilities are combined with historical "
                     "correlations; the result is not a pure market-implied covariance.")
    return ForwardCovariance(cov, vols, R, correlation_source, basis, tuple(notes))


def check_same_horizon(*vectors: VolatilityVector) -> None:
    horizons = {v.horizon_days for v in vectors}
    if len(horizons) > 1:
        raise ValueError(f"Volatility estimates refer to different horizons {sorted(horizons)} "
                         f"trading days; align them before comparing or combining.")


# --- correlation utilities ----------------------------------------------------------------

def is_valid_correlation(R: pd.DataFrame | np.ndarray, tol: float = 1e-10) -> bool:
    a = np.asarray(R, dtype=float)
    return (np.allclose(np.diag(a), 1.0, atol=1e-10) and np.all(np.abs(a) <= 1 + 1e-12)
            and is_positive_semidefinite(a, tol))


def stress_correlation(R: pd.DataFrame, lam: float) -> pd.DataFrame:
    """R_lambda = (1 - lambda) R + lambda 11'. A convex combination of two valid correlation
    matrices, so it is a valid correlation matrix for every lambda in [0, 1]."""
    if not 0.0 <= lam <= 1.0:
        raise ValueError("Correlation stress lambda must be between 0 and 1.")
    ones = np.ones_like(R.to_numpy())
    return pd.DataFrame((1 - lam) * R.to_numpy() + lam * ones, index=R.index, columns=R.columns)


def nearest_correlation(R: pd.DataFrame | np.ndarray, max_iter: int = 500,
                        tol: float = 1e-10) -> pd.DataFrame:
    """Higham (2002) alternating projections with Dykstra's correction: nearest (Frobenius)
    symmetric PSD matrix with unit diagonal."""
    A = np.asarray(R, dtype=float)
    A = (A + A.T) / 2
    Y, dS = A.copy(), np.zeros_like(A)
    for _ in range(max_iter):
        Rk = Y - dS
        vals, vecs = np.linalg.eigh(Rk)
        X = vecs @ np.diag(np.maximum(vals, 0)) @ vecs.T
        dS = X - Rk
        Y_new = X.copy()
        np.fill_diagonal(Y_new, 1.0)
        if np.linalg.norm(Y_new - Y, "fro") < tol:
            Y = Y_new
            break
        Y = Y_new
    # Final clean-up: tiny negative eigenvalues from round-off.
    vals, vecs = np.linalg.eigh((Y + Y.T) / 2)
    Y = vecs @ np.diag(np.maximum(vals, 1e-12)) @ vecs.T
    d = np.sqrt(np.diag(Y))
    Y = Y / np.outer(d, d)
    idx = R.index if isinstance(R, pd.DataFrame) else None
    return pd.DataFrame(Y, index=idx, columns=idx)


def correlation_stress_profile(weights: pd.Series, vols: pd.Series, R: pd.DataFrame,
                               lambdas=np.linspace(0, 1, 11)) -> pd.DataFrame:
    """Portfolio volatility as correlations converge towards 1 (HYPOTHETICAL)."""
    w = weights.reindex(vols.index).fillna(0.0).to_numpy()
    d = vols.to_numpy()
    rows = []
    for lam in lambdas:
        Rl = stress_correlation(R.loc[vols.index, vols.index], float(lam)).to_numpy()
        S = np.outer(d, d) * Rl
        rows.append({"lambda": float(lam), "portfolio_volatility": float(np.sqrt(w @ S @ w)),
                     "average_correlation": float(Rl[np.triu_indices(len(d), 1)].mean())
                     if len(d) > 1 else 1.0})
    return pd.DataFrame(rows)
