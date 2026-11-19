"""Assemble VaR / CVaR tables across methods and confidence levels. See METHODOLOGY §5.

All estimates are computed at a 1-day horizon. Multi-day figures are the 1-day figures
scaled by sqrt(h) — an i.i.d. approximation that is labelled on every multi-day number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

import config
from core import conventions as cv
from core.types import RiskBasis
from risk import cvar as C
from risk import var as V

CONFIDENCES = (0.95, 0.99)


@dataclass(frozen=True)
class TailRiskEstimate:
    method: str
    confidence: float
    horizon_days: int
    var: float  # fraction of portfolio value, positive = loss
    cvar: float
    portfolio_value: float
    basis: RiskBasis
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def var_amount(self) -> float:
        return self.var * self.portfolio_value

    @property
    def cvar_amount(self) -> float:
        return self.cvar * self.portfolio_value


def _scale(x: float, h: int) -> float:
    return float(cv.scale_to_horizon(x, h))


def _horizon_note(h: int) -> tuple[str, ...]:
    return () if h == 1 else (f"{h}-day figure = 1-day figure x sqrt({h}) (i.i.d. approximation).",)


def historical_tail_risk(portfolio_returns: pd.Series, asset_returns: pd.DataFrame,
                         weights: pd.Series, portfolio_value: float,
                         confidences=CONFIDENCES, horizon_days: int = 1,
                         n_sims: int = 100_000, seed: int = config.RANDOM_SEED,
                         mc_distribution: str = "student_t") -> list[TailRiskEstimate]:
    """OBSERVED-basis VaR/CVaR from the historical return sample.

    * Historical: empirical distribution of the portfolio's daily returns.
    * Parametric normal and Cornish-Fisher: sample mean, std, skew, excess kurtosis.
    * Monte Carlo: multivariate Student-t (dof from the portfolio's excess kurtosis) or
      normal, with the sample mean vector and covariance of daily asset returns.
    """
    r = portfolio_returns.dropna()
    mu, sigma = float(r.mean()), float(r.std(ddof=1))
    skew, kurt = float(stats.skew(r, bias=False)), float(stats.kurtosis(r, bias=False))
    cf_ok = V.cornish_fisher_is_monotone(skew, kurt)

    w = weights.reindex(asset_returns.columns).fillna(0.0).to_numpy()
    dof = V.student_t_dof_from_kurtosis(kurt) if mc_distribution == "student_t" else None
    sims = V.simulate_portfolio_returns(w, asset_returns.mean().to_numpy(),
                                        asset_returns.cov().to_numpy(), n_sims, dof, seed)
    mc_label = (f"Monte Carlo (Student-t, nu={dof:.1f})" if dof else "Monte Carlo (normal)")
    sample_note = f"Sample: {r.index[0].date()} to {r.index[-1].date()} ({len(r)} days)."

    out: list[TailRiskEstimate] = []
    h, hn = horizon_days, _horizon_note(horizon_days)
    for a in confidences:
        tail_n = int(round(len(r) * (1 - a)))
        out.append(TailRiskEstimate("Historical", a, h,
                                    _scale(V.historical_var(r, a), h), _scale(C.historical_cvar(r, a), h),
                                    portfolio_value, RiskBasis.OBSERVED,
                                    hn + (sample_note, f"Tail rests on ~{tail_n} observations.")))
        out.append(TailRiskEstimate("Parametric (normal)", a, h,
                                    _scale(V.parametric_var(mu, sigma, a), h),
                                    _scale(C.parametric_cvar(mu, sigma, a), h),
                                    portfolio_value, RiskBasis.OBSERVED,
                                    hn + ("Assumes normally distributed returns (thin tails).",)))
        cf_notes = (f"Skew {skew:.2f}, excess kurtosis {kurt:.2f}.",)
        if cf_ok:
            cf_var = _scale(V.cornish_fisher_var(mu, sigma, skew, kurt, a), h)
            cf_cvar = _scale(C.cornish_fisher_cvar(mu, sigma, skew, kurt, a), h)
        else:
            # The expansion is not a valid quantile function for these moments (typically
            # excess kurtosis above ~8): its output would be misleading, so it is withheld.
            cf_var = cf_cvar = float("nan")
            cf_notes += ("Not reported: the Cornish-Fisher expansion is not a valid (monotone) "
                         "quantile function for this skew/kurtosis, so its VaR would be "
                         "misleading. Use the historical or Monte Carlo estimates.",)
        out.append(TailRiskEstimate("Cornish-Fisher", a, h, cf_var, cf_cvar,
                                    portfolio_value, RiskBasis.OBSERVED, hn + cf_notes))
        out.append(TailRiskEstimate(mc_label, a, h,
                                    _scale(V.monte_carlo_var(sims, a), h),
                                    _scale(C.monte_carlo_cvar(sims, a), h),
                                    portfolio_value, RiskBasis.OBSERVED,
                                    hn + (f"{n_sims:,} simulations, seed {seed}.",)))
    return out


def covariance_tail_risk(weights: pd.Series, annual_cov: pd.DataFrame, portfolio_value: float,
                         basis: RiskBasis, confidences=CONFIDENCES, horizon_days: int = 1,
                         dof: float | None = None, n_sims: int = 100_000,
                         seed: int = config.RANDOM_SEED) -> list[TailRiskEstimate]:
    """VaR/CVaR implied by a (typically forward-looking) covariance matrix, zero mean.
    Basis is inherited from how the covariance was built (forecast, implied, proxy,
    hypothetical). No historical method is possible here."""
    w = weights.reindex(annual_cov.index).fillna(0.0).to_numpy()
    daily_cov = annual_cov.to_numpy() / cv.TRADING_DAYS
    sigma = float(np.sqrt(w @ daily_cov @ w))
    sims = V.simulate_portfolio_returns(w, np.zeros(len(w)), daily_cov, n_sims, dof, seed)
    h, hn = horizon_days, _horizon_note(horizon_days)
    note = ("Zero expected return assumed.",)
    mc_label = f"Monte Carlo (Student-t, nu={dof:.1f})" if dof else "Monte Carlo (normal)"
    out = []
    for a in confidences:
        out.append(TailRiskEstimate("Parametric (normal)", a, h,
                                    _scale(V.parametric_var(0.0, sigma, a), h),
                                    _scale(C.parametric_cvar(0.0, sigma, a), h),
                                    portfolio_value, basis, hn + note))
        out.append(TailRiskEstimate(mc_label, a, h,
                                    _scale(V.monte_carlo_var(sims, a), h),
                                    _scale(C.monte_carlo_cvar(sims, a), h),
                                    portfolio_value, basis, hn + note))
    return out


def to_frame(estimates: list[TailRiskEstimate]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Method": e.method, "Confidence": e.confidence, "Horizon (days)": e.horizon_days,
        "VaR": e.var, "CVaR": e.cvar, "VaR amount": e.var_amount, "CVaR amount": e.cvar_amount,
        "Basis": e.basis.label, "Notes": " ".join(e.notes),
    } for e in estimates])
