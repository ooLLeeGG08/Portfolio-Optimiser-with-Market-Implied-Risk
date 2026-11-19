"""Euler risk decomposition of portfolio volatility. See METHODOLOGY §5.

sigma_p = sqrt(w' S w)
MCR_i   = d sigma_p / d w_i = (S w)_i / sigma_p      (marginal contribution)
CCR_i   = w_i * MCR_i                               (component contribution)
sum_i CCR_i = sigma_p exactly (sigma_p is homogeneous of degree 1 in w)
%CR_i   = CCR_i / sigma_p                           (sums to 100%)

Works with any covariance (historical, forecast D.R.D, stressed); the result inherits the
covariance's basis. Under a zero-mean normal model, VaR_a = z_a sigma_p, so each asset's
VaR contribution is %CR_i x VaR.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.types import RiskBasis


@dataclass(frozen=True)
class RiskContribution:
    table: pd.DataFrame  # weight, volatility, MCR, CCR, pct_contribution
    portfolio_volatility: float
    basis: RiskBasis


def euler_contributions(weights: pd.Series, annual_cov: pd.DataFrame,
                        basis: RiskBasis = RiskBasis.OBSERVED) -> RiskContribution:
    w = weights.reindex(annual_cov.index).fillna(0.0)
    S = annual_cov.to_numpy()
    wv = w.to_numpy()
    sigma = float(np.sqrt(wv @ S @ wv))
    if sigma <= 0:
        raise ValueError("Portfolio volatility is zero; risk contributions are undefined.")
    mcr = S @ wv / sigma
    ccr = wv * mcr
    table = pd.DataFrame({
        "weight": wv,
        "volatility": np.sqrt(np.diag(S)),
        "marginal_contribution": mcr,
        "component_contribution": ccr,
        "pct_contribution": ccr / sigma,
    }, index=annual_cov.index)
    return RiskContribution(table, sigma, basis)
