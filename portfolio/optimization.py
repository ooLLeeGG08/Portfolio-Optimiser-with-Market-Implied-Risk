"""Portfolio construction strategies. See METHODOLOGY §4.

Inputs
------
* Expected returns: annualised ARITHMETIC mean of daily simple returns (252 x mean). This is
  the quantity that enters mean-variance optimisation and the Sharpe ratio consistently.
  It is the noisiest input; max-Sharpe weights are highly sensitive to it
  (see ``max_sharpe_sensitivity``).
* Covariance: Ledoit-Wolf shrinkage by default, annualised.
* No look-ahead: ``estimate_inputs`` uses only returns dated on or before ``as_of``.

All reported statistics (return, volatility, Sharpe) are computed by ``portfolio_stats`` —
not by the solver library — so every strategy is measured with the same conventions.
These are IN-SAMPLE ESTIMATES, not predictions; out-of-sample results come from the
walk-forward backtest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier

from core import conventions as cv
from risk.covariance import CovarianceEstimate, CovMethod, estimate_covariance

SOLVER = "CLARABEL"
WEIGHT_TOL = 1e-6


class OptimizationError(ValueError):
    """The requested optimisation is infeasible or failed; message is user-readable."""


# --- inputs ------------------------------------------------------------------------------

@dataclass(frozen=True)
class OptimizationInputs:
    mu: pd.Series  # annualised arithmetic expected returns
    cov: CovarianceEstimate  # annualised
    risk_free_rate: float  # annualised
    start: pd.Timestamp
    end: pd.Timestamp  # last date used (the decision date)

    @property
    def tickers(self) -> list[str]:
        return list(self.mu.index)

    @property
    def n_obs(self) -> int:
        return self.cov.n_observations

    @property
    def years(self) -> float:
        return self.n_obs / cv.TRADING_DAYS

    @property
    def mu_standard_error(self) -> pd.Series:
        """Standard error of each annualised mean: sigma_annual / sqrt(years)."""
        vol = np.sqrt(np.diag(self.cov.matrix))
        return pd.Series(vol / np.sqrt(self.years), index=self.mu.index)


def estimate_inputs(returns: pd.DataFrame, risk_free_rate: float,
                    as_of: pd.Timestamp | None = None, lookback_days: int | None = None,
                    cov_method: CovMethod = "ledoit_wolf") -> OptimizationInputs:
    """Estimate mu and covariance from daily simple returns dated <= as_of (no look-ahead),
    optionally limited to the most recent ``lookback_days`` observations."""
    r = returns if as_of is None else returns.loc[returns.index <= pd.Timestamp(as_of)]
    if lookback_days is not None:
        r = r.iloc[-lookback_days:]
    if len(r) < 2:
        raise OptimizationError("Not enough return observations before the decision date.")
    if r.isna().any().any():
        raise OptimizationError("Returns contain missing values; align prices first.")
    return OptimizationInputs(
        mu=cv.annualize_mean(r.mean()),
        cov=estimate_covariance(r, cov_method),
        risk_free_rate=risk_free_rate,
        start=r.index[0], end=r.index[-1],
    )


# --- constraints -------------------------------------------------------------------------

@dataclass(frozen=True)
class Constraints:
    """Long-only, fully invested; per-asset lower/upper bounds (scalars or per-ticker)."""

    min_weight: float | dict[str, float] = 0.0
    max_weight: float | dict[str, float] = 1.0

    def bounds(self, tickers: list[str]) -> list[tuple[float, float]]:
        def get(v, t, default):
            return float(v.get(t, default)) if isinstance(v, dict) else float(v)
        return [(get(self.min_weight, t, 0.0), get(self.max_weight, t, 1.0)) for t in tickers]

    def validate(self, tickers: list[str]) -> list[tuple[float, float]]:
        b = self.bounds(tickers)
        for t, (lo, hi) in zip(tickers, b):
            if lo < 0:
                raise OptimizationError(f"{t}: minimum weight cannot be negative (long-only).")
            if hi > 1:
                raise OptimizationError(f"{t}: maximum weight cannot exceed 100%.")
            if lo > hi:
                raise OptimizationError(f"{t}: minimum weight {lo:.1%} exceeds maximum {hi:.1%}.")
        lo_sum, hi_sum = sum(x for x, _ in b), sum(y for _, y in b)
        if lo_sum > 1 + WEIGHT_TOL:
            raise OptimizationError(f"Minimum weights sum to {lo_sum:.1%}; they must sum to at "
                                    f"most 100% for a fully invested portfolio.")
        if hi_sum < 1 - WEIGHT_TOL:
            raise OptimizationError(f"Maximum weights sum to {hi_sum:.1%}; with {len(tickers)} "
                                    f"assets they must allow at least 100% in total.")
        return b


# --- results -------------------------------------------------------------------------------

@dataclass(frozen=True)
class PortfolioStats:
    expected_return: float  # annualised arithmetic
    volatility: float  # annualised
    sharpe: float


def portfolio_stats(weights: pd.Series, inputs: OptimizationInputs) -> PortfolioStats:
    w = weights.reindex(inputs.tickers).fillna(0.0).to_numpy()  # absent asset = zero weight
    mu = inputs.mu.to_numpy()
    cov = inputs.cov.matrix.to_numpy()
    ret = float(w @ mu)
    vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
    sharpe = (ret - inputs.risk_free_rate) / vol if vol > 0 else float("nan")
    return PortfolioStats(ret, vol, sharpe)


class Strategy(str, Enum):
    EQUAL_WEIGHT = "Equal weight"
    MIN_VARIANCE = "Minimum variance"
    MAX_SHARPE = "Maximum Sharpe"
    CUSTOM = "Custom (constrained)"
    USER = "User-defined"


@dataclass(frozen=True)
class OptimizedPortfolio:
    strategy: Strategy
    weights: pd.Series
    stats: PortfolioStats
    notes: tuple[str, ...] = field(default_factory=tuple)


def _clean(raw: dict | pd.Series, tickers: list[str],
           bounds: list[tuple[float, float]]) -> pd.Series:
    """Remove solver noise: clip to bounds, zero out dust, renormalise, re-verify."""
    w = pd.Series(raw, dtype=float).reindex(tickers).fillna(0.0)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    v = np.clip(w.to_numpy(), lo, hi)
    v[(np.abs(v) < 1e-8) & (lo == 0)] = 0.0
    v = v / v.sum()
    if np.any(v < lo - WEIGHT_TOL) or np.any(v > hi + WEIGHT_TOL):
        raise OptimizationError("Solver returned weights that violate the constraints.")
    return pd.Series(v, index=tickers)


def _frontier(inputs: OptimizationInputs, bounds) -> EfficientFrontier:
    return EfficientFrontier(inputs.mu, inputs.cov.matrix, weight_bounds=bounds, solver=SOLVER)


def equal_weight(inputs: OptimizationInputs) -> OptimizedPortfolio:
    n = len(inputs.tickers)
    w = pd.Series(1.0 / n, index=inputs.tickers)
    return OptimizedPortfolio(Strategy.EQUAL_WEIGHT, w, portfolio_stats(w, inputs),
                              ("Baseline: 1/N allocation, ignores the constraints.",))


def min_variance(inputs: OptimizationInputs, constraints: Constraints = Constraints(),
                 strategy: Strategy = Strategy.MIN_VARIANCE) -> OptimizedPortfolio:
    bounds = constraints.validate(inputs.tickers)
    try:
        raw = _frontier(inputs, bounds).min_volatility()
    except Exception as exc:
        raise OptimizationError(f"Minimum-variance optimisation failed: {exc}") from exc
    w = _clean(raw, inputs.tickers, bounds)
    return OptimizedPortfolio(strategy, w, portfolio_stats(w, inputs))


def max_sharpe(inputs: OptimizationInputs, constraints: Constraints = Constraints(),
               strategy: Strategy = Strategy.MAX_SHARPE) -> OptimizedPortfolio:
    bounds = constraints.validate(inputs.tickers)
    if inputs.mu.max() <= inputs.risk_free_rate:
        raise OptimizationError(
            f"No asset has an estimated expected return above the risk-free rate "
            f"({inputs.risk_free_rate:.2%}), so the maximum-Sharpe portfolio is undefined. "
            f"Use minimum variance instead, or a longer estimation window.")
    try:
        raw = _frontier(inputs, bounds).max_sharpe(risk_free_rate=inputs.risk_free_rate)
    except Exception as exc:
        raise OptimizationError(f"Maximum-Sharpe optimisation failed: {exc}") from exc
    w = _clean(raw, inputs.tickers, bounds)
    return OptimizedPortfolio(strategy, w, portfolio_stats(w, inputs), (
        "Highly sensitive to expected-return estimates, which have large standard errors; "
        "see the sensitivity analysis.",))


def target_return(inputs: OptimizationInputs, target: float,
                  constraints: Constraints = Constraints(),
                  strategy: Strategy = Strategy.CUSTOM) -> OptimizedPortfolio:
    """Minimum-variance portfolio achieving at least ``target`` expected return."""
    bounds = constraints.validate(inputs.tickers)
    hi = max_feasible_return(inputs, bounds)
    if target > hi + 1e-9:
        raise OptimizationError(f"Target return {target:.2%} is not attainable under the "
                                f"constraints (maximum {hi:.2%}).")
    try:
        raw = _frontier(inputs, bounds).efficient_return(target)
    except Exception as exc:
        raise OptimizationError(f"Target-return optimisation failed: {exc}") from exc
    w = _clean(raw, inputs.tickers, bounds)
    return OptimizedPortfolio(strategy, w, portfolio_stats(w, inputs))


def target_volatility(inputs: OptimizationInputs, target: float,
                      constraints: Constraints = Constraints(),
                      strategy: Strategy = Strategy.CUSTOM) -> OptimizedPortfolio:
    """Maximum-return portfolio with volatility at most ``target``."""
    bounds = constraints.validate(inputs.tickers)
    floor = min_variance(inputs, constraints).stats.volatility
    if target < floor - 1e-9:
        raise OptimizationError(f"Target volatility {target:.2%} is below the minimum attainable "
                                f"{floor:.2%} under the constraints.")
    try:
        raw = _frontier(inputs, bounds).efficient_risk(target)
    except Exception as exc:
        raise OptimizationError(f"Target-volatility optimisation failed: {exc}") from exc
    w = _clean(raw, inputs.tickers, bounds)
    return OptimizedPortfolio(strategy, w, portfolio_stats(w, inputs))


def custom(inputs: OptimizationInputs, constraints: Constraints, objective: str = "max_sharpe",
           target: float | None = None) -> OptimizedPortfolio:
    """User-constrained optimisation. objective in
    {'min_variance', 'max_sharpe', 'target_return', 'target_volatility'}."""
    if objective == "min_variance":
        return min_variance(inputs, constraints, Strategy.CUSTOM)
    if objective == "max_sharpe":
        return max_sharpe(inputs, constraints, Strategy.CUSTOM)
    if objective in ("target_return", "target_volatility"):
        if target is None:
            raise OptimizationError(f"Objective '{objective}' needs a target value.")
        fn = target_return if objective == "target_return" else target_volatility
        return fn(inputs, target, constraints, Strategy.CUSTOM)
    raise OptimizationError(f"Unknown objective: {objective}")


def user_defined(inputs: OptimizationInputs, weights: pd.Series) -> OptimizedPortfolio:
    missing = set(weights.index) - set(inputs.tickers)
    if missing:
        raise OptimizationError(f"No estimates for: {', '.join(sorted(missing))}.")
    w = weights.reindex(inputs.tickers).fillna(0.0)
    if abs(w.sum() - 1) > WEIGHT_TOL or (w < -WEIGHT_TOL).any():
        raise OptimizationError("User weights must be non-negative and sum to 100%.")
    return OptimizedPortfolio(Strategy.USER, w, portfolio_stats(w, inputs))


def max_feasible_return(inputs: OptimizationInputs, bounds: list[tuple[float, float]]) -> float:
    """Maximum expected return of a fully invested portfolio within bounds. The LP
    max mu.w s.t. sum w = 1, lo <= w <= hi is solved exactly by greedy filling:
    start from the lower bounds and allocate the remaining budget to the highest-mu
    assets up to their upper bounds."""
    mu = inputs.mu.to_numpy()
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    w = lo.copy()
    budget = 1.0 - w.sum()
    for i in np.argsort(-mu):
        add = min(hi[i] - w[i], budget)
        w[i] += add
        budget -= add
        if budget <= 0:
            break
    return float(w @ mu)


def compare_strategies(inputs: OptimizationInputs, constraints: Constraints = Constraints(),
                       user_weights: pd.Series | None = None,
                       custom_objective: str | None = None,
                       custom_target: float | None = None) -> tuple[list[OptimizedPortfolio], list[str]]:
    """Run every strategy; failures (e.g. max-Sharpe undefined) are reported, not hidden."""
    results: list[OptimizedPortfolio] = [equal_weight(inputs)]
    errors: list[str] = []
    runs = [("Minimum variance", lambda: min_variance(inputs)),
            ("Maximum Sharpe", lambda: max_sharpe(inputs))]
    if custom_objective:
        runs.append(("Custom", lambda: custom(inputs, constraints, custom_objective, custom_target)))
    if user_weights is not None:
        runs.append(("User-defined", lambda: user_defined(inputs, user_weights)))
    for name, fn in runs:
        try:
            results.append(fn())
        except OptimizationError as exc:
            errors.append(f"{name}: {exc}")
    return results, errors


def comparison_table(results: list[OptimizedPortfolio]) -> pd.DataFrame:
    rows = {r.strategy.value: {"Expected return (ann.)": r.stats.expected_return,
                               "Volatility (ann.)": r.stats.volatility,
                               "Sharpe ratio": r.stats.sharpe,
                               **{f"w[{t}]": w for t, w in r.weights.items()}}
            for r in results}
    return pd.DataFrame(rows).T


# --- sensitivity -------------------------------------------------------------------------

@dataclass(frozen=True)
class SensitivityResult:
    draws: pd.DataFrame  # one row of max-Sharpe weights per resampled mu
    summary: pd.DataFrame  # point estimate, mean, 5th / 95th percentile per asset
    n_requested: int
    n_failed: int  # draws where no asset beat the risk-free rate


def max_sharpe_sensitivity(inputs: OptimizationInputs, constraints: Constraints = Constraints(),
                           n_draws: int = 200, seed: int = 0) -> SensitivityResult:
    """How much do max-Sharpe weights move if expected returns are perturbed by their own
    estimation error? mu* ~ N(mu, Sigma / years) — the sampling distribution of the
    annualised mean — and the portfolio is re-optimised for each draw."""
    bounds = constraints.validate(inputs.tickers)
    point = max_sharpe(inputs, constraints).weights
    rng = np.random.default_rng(seed)
    sampling_cov = inputs.cov.matrix.to_numpy() / inputs.years
    mus = rng.multivariate_normal(inputs.mu.to_numpy(), sampling_cov, size=n_draws)
    rows, failed = [], 0
    for m in mus:
        mu_s = pd.Series(m, index=inputs.tickers)
        if mu_s.max() <= inputs.risk_free_rate:
            failed += 1
            continue
        try:
            raw = EfficientFrontier(mu_s, inputs.cov.matrix, weight_bounds=bounds,
                                    solver=SOLVER).max_sharpe(risk_free_rate=inputs.risk_free_rate)
            rows.append(_clean(raw, inputs.tickers, bounds))
        except Exception:
            failed += 1
    draws = pd.DataFrame(rows)
    summary = pd.DataFrame({
        "point_estimate": point,
        "mean": draws.mean() if len(draws) else np.nan,
        "p05": draws.quantile(0.05) if len(draws) else np.nan,
        "p95": draws.quantile(0.95) if len(draws) else np.nan,
    })
    return SensitivityResult(draws, summary, n_draws, failed)
