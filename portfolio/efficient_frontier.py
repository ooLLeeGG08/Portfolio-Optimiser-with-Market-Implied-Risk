"""Efficient frontier under long-only, fully-invested, per-asset-bound constraints.

The frontier is traced by minimising variance for a grid of target returns from the
minimum-variance portfolio's return up to the maximum feasible return. Only the efficient
(upper) branch is produced. Grid points where the solver fails are dropped and counted —
never interpolated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio.optimization import (Constraints, OptimizationError, OptimizationInputs,
                                    max_feasible_return, min_variance, portfolio_stats,
                                    target_return)


@dataclass(frozen=True)
class Frontier:
    points: pd.DataFrame  # columns: expected_return, volatility, sharpe
    weights: pd.DataFrame  # one row per point
    n_failed: int


def efficient_frontier(inputs: OptimizationInputs, constraints: Constraints = Constraints(),
                       n_points: int = 40) -> Frontier:
    if n_points < 2:
        raise ValueError("n_points must be at least 2.")
    bounds = constraints.validate(inputs.tickers)
    mv = min_variance(inputs, constraints)
    r_lo = mv.stats.expected_return
    r_hi = max_feasible_return(inputs, bounds)

    rows, weights, failed = [], [], 0
    if r_hi - r_lo < 1e-10:  # degenerate: a single feasible return level
        targets = []
        rows.append(mv.stats)
        weights.append(mv.weights)
    else:
        # Tiny inset at the top end avoids numerically infeasible boundary solves.
        targets = np.linspace(r_lo, r_hi - 1e-7 * (r_hi - r_lo), n_points)
        rows.append(mv.stats)
        weights.append(mv.weights)
    for tgt in targets[1:]:
        try:
            p = target_return(inputs, float(tgt), constraints)
        except OptimizationError:
            failed += 1
            continue
        rows.append(p.stats)
        weights.append(p.weights)

    pts = pd.DataFrame([{"expected_return": s.expected_return, "volatility": s.volatility,
                         "sharpe": s.sharpe} for s in rows])
    return Frontier(pts, pd.DataFrame(weights).reset_index(drop=True), failed)


def random_feasible_portfolios(inputs: OptimizationInputs, n: int = 2000,
                               seed: int = 0) -> pd.DataFrame:
    """Random long-only fully-invested portfolios (Dirichlet), for context in charts
    and as a test oracle (no random portfolio may lie above the frontier)."""
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.ones(len(inputs.tickers)), size=n)
    stats = [portfolio_stats(pd.Series(x, index=inputs.tickers), inputs) for x in w]
    return pd.DataFrame([{"expected_return": s.expected_return, "volatility": s.volatility,
                          "sharpe": s.sharpe} for s in stats])
