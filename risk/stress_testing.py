"""Hypothetical stress scenarios. See METHODOLOGY §10.

Every result here has basis HYPOTHETICAL. Scenarios are "what if" calculations under
stated assumptions — not predictions, and never labelled as such.

Two families:
* Price-shock scenarios (market decline, rate shock, single-asset, custom): instantaneous
  simple-return shocks per asset; portfolio return = sum_i w_i shock_i (linear in weights,
  exact for an instantaneous shock). Shocks are floored at -100%.
* Risk-regime scenarios (volatility increase, correlation convergence): the covariance is
  stressed and risk (volatility, parametric VaR) is compared with the baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from core import conventions as cv
from core.types import RiskBasis
from risk.covariance import cov_to_corr, stress_correlation
from risk.risk_contribution import euler_contributions

HYPOTHETICAL_LABEL = "Hypothetical scenario — not a prediction"


@dataclass(frozen=True)
class RiskSnapshot:
    """Zero-mean normal risk implied by a covariance matrix (annualised volatility and
    1-day parametric VaR at 95% / 99%, as fractions of portfolio value)."""

    volatility: float
    var_95_1d: float
    var_99_1d: float


def risk_snapshot(weights: pd.Series, annual_cov: pd.DataFrame) -> RiskSnapshot:
    w = weights.reindex(annual_cov.index).fillna(0.0).to_numpy()
    vol = float(np.sqrt(w @ annual_cov.to_numpy() @ w))
    d = float(cv.deannualize_vol(vol))
    return RiskSnapshot(vol, float(d * stats.norm.ppf(0.95)), float(d * stats.norm.ppf(0.99)))


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    description: str
    asset_shocks: pd.Series | None  # simple-return shock per asset (price scenarios)
    asset_pnl: pd.Series | None  # currency P&L per asset (negative = loss)
    portfolio_return: float | None
    portfolio_value_before: float
    baseline_risk: RiskSnapshot | None = None
    stressed_risk: RiskSnapshot | None = None
    stressed_contributions: pd.Series | None = None  # % risk contribution under stress
    assumptions: tuple[str, ...] = field(default_factory=tuple)
    basis: RiskBasis = RiskBasis.HYPOTHETICAL
    label: str = HYPOTHETICAL_LABEL

    @property
    def portfolio_loss(self) -> float | None:
        """Positive = loss, as a fraction of value."""
        return None if self.portfolio_return is None else -self.portfolio_return

    @property
    def portfolio_value_after(self) -> float | None:
        if self.portfolio_return is None:
            return None
        return self.portfolio_value_before * (1 + self.portfolio_return)


def _price_scenario(name: str, description: str, weights: pd.Series, shocks: pd.Series,
                    value: float, assumptions: tuple[str, ...]) -> ScenarioResult:
    shocks = shocks.reindex(weights.index).fillna(0.0).clip(lower=-1.0)
    pnl = weights * shocks * value
    return ScenarioResult(name, description, shocks, pnl, float((weights * shocks).sum()),
                          value, assumptions=assumptions)


def market_decline(weights: pd.Series, betas: pd.Series, market_move: float,
                   value: float, benchmark: str = "benchmark") -> ScenarioResult:
    """Each asset moves by beta_i x market_move (e.g. market_move = -0.20)."""
    shocks = betas.reindex(weights.index) * market_move
    if shocks.isna().any():
        raise ValueError(f"Missing beta for: {', '.join(shocks[shocks.isna()].index)}.")
    return _price_scenario(
        f"Market {market_move:+.0%}", f"Instantaneous {market_move:+.0%} move in {benchmark}, "
        f"transmitted to each asset through its historical beta.", weights, shocks, value,
        ("Asset moves = historical beta x market move; idiosyncratic moves ignored.",
         "Betas are estimated in normal markets and can change sharply in a crisis "
         "(e.g. bonds and equities can fall together, as in 2022).",
         "Positive-beta assets fall with the market; negative-beta assets rise."))


def rate_shock(weights: pd.Series, rate_sensitivities: pd.Series, shift_bp: float,
               value: float) -> ScenarioResult:
    """Asset move = sensitivity_i x yield change (sensitivity per 1 percentage point)."""
    shocks = rate_sensitivities.reindex(weights.index) * (shift_bp / 100.0)
    if shocks.isna().any():
        raise ValueError(f"Missing rate sensitivity for: {', '.join(shocks[shocks.isna()].index)}.")
    return _price_scenario(
        f"Rates {shift_bp:+.0f} bp", f"Parallel {shift_bp:+.0f} bp move in the 10-year Treasury "
        f"yield, transmitted through each asset's estimated rate sensitivity.", weights, shocks,
        value, ("Sensitivities are OLS slopes of daily returns on daily 10-year-yield changes; "
                "for equities they are typically small and statistically unstable.",
                "Linear (duration-only) response; convexity ignored."))


def asset_shock(weights: pd.Series, shocks: dict[str, float], value: float) -> ScenarioResult:
    unknown = set(shocks) - set(weights.index)
    if unknown:
        raise ValueError(f"Not in the portfolio: {', '.join(sorted(unknown))}.")
    desc = ", ".join(f"{t} {s:+.0%}" for t, s in shocks.items())
    return _price_scenario(f"Asset shock ({desc})", "Instantaneous shock to selected assets; "
                           "other assets unchanged.", weights, pd.Series(shocks, dtype=float), value,
                           ("Other assets assumed unaffected (no contagion).",))


def custom_shock(weights: pd.Series, shocks: dict[str, float], value: float,
                 name: str = "Custom scenario") -> ScenarioResult:
    r = asset_shock(weights, shocks, value)
    return ScenarioResult(name, "User-defined per-asset shocks.", r.asset_shocks, r.asset_pnl,
                          r.portfolio_return, value, assumptions=r.assumptions)


def _risk_scenario(name: str, description: str, weights: pd.Series, base_cov: pd.DataFrame,
                   stressed_cov: pd.DataFrame, value: float,
                   assumptions: tuple[str, ...]) -> ScenarioResult:
    contrib = euler_contributions(weights, stressed_cov, RiskBasis.HYPOTHETICAL)
    return ScenarioResult(name, description, None, None, None, value,
                          baseline_risk=risk_snapshot(weights, base_cov),
                          stressed_risk=risk_snapshot(weights, stressed_cov),
                          stressed_contributions=contrib.table["pct_contribution"],
                          assumptions=assumptions)


def volatility_increase(weights: pd.Series, annual_cov: pd.DataFrame, multiplier: float,
                        value: float) -> ScenarioResult:
    """All volatilities x multiplier, correlations unchanged (covariance x multiplier^2)."""
    if multiplier <= 0:
        raise ValueError("Volatility multiplier must be positive.")
    return _risk_scenario(f"Volatility x{multiplier:g}", f"All asset volatilities multiplied by "
                          f"{multiplier:g}; correlations unchanged.", weights, annual_cov,
                          annual_cov * multiplier ** 2, value,
                          ("Correlations held at baseline (in practice they usually rise "
                           "together with volatility — combine with correlation stress).",))


def correlation_convergence(weights: pd.Series, annual_cov: pd.DataFrame, lam: float,
                            value: float, vol_multiplier: float = 1.0) -> ScenarioResult:
    """R -> (1 - lam) R + lam 11', optionally with volatilities scaled too."""
    vols, R = cov_to_corr(annual_cov)
    Rs = stress_correlation(R, lam)
    d = vols.to_numpy() * vol_multiplier
    stressed = pd.DataFrame(np.outer(d, d) * Rs.to_numpy(), index=annual_cov.index,
                            columns=annual_cov.columns)
    name = f"Correlation stress (lambda={lam:.2f})"
    if vol_multiplier != 1.0:
        name += f", volatility x{vol_multiplier:g}"
    return _risk_scenario(name, "Pairwise correlations pushed towards 1, reducing "
                          "diversification as typically observed in market stress.",
                          weights, annual_cov, stressed, value,
                          ("lambda = 0 is the historical correlation matrix; lambda = 1 means "
                           "all assets perfectly correlated (no diversification).",))


def scenario_table(results: list[ScenarioResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "Scenario": r.name,
            "Portfolio return": r.portfolio_return,
            "Value after": r.portfolio_value_after,
            "Baseline vol": r.baseline_risk.volatility if r.baseline_risk else None,
            "Stressed vol": r.stressed_risk.volatility if r.stressed_risk else None,
            "Baseline 1d VaR 99%": r.baseline_risk.var_99_1d if r.baseline_risk else None,
            "Stressed 1d VaR 99%": r.stressed_risk.var_99_1d if r.stressed_risk else None,
            "Basis": r.basis.label,
        })
    return pd.DataFrame(rows)
