import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linprog
from sklearn.covariance import LedoitWolf

from core import conventions as cv
from portfolio.efficient_frontier import efficient_frontier, random_feasible_portfolios
from portfolio.optimization import (Constraints, OptimizationError, OptimizationInputs,
                                    compare_strategies, custom, equal_weight, estimate_inputs,
                                    max_feasible_return, max_sharpe, max_sharpe_sensitivity,
                                    min_variance, portfolio_stats, target_return,
                                    target_volatility, user_defined)
from risk.covariance import (CovarianceEstimate, cov_to_corr, is_positive_semidefinite,
                             ledoit_wolf_covariance, sample_covariance)
from tests.conftest import make_gbm_prices

TICKERS = ["A", "B", "C", "D"]


def make_inputs(mu, vols, corr=0.2, rf=0.02, n_obs=1260) -> OptimizationInputs:
    vols = np.asarray(vols, dtype=float)
    R = np.full((len(vols), len(vols)), corr)
    np.fill_diagonal(R, 1.0)
    cov = pd.DataFrame(np.outer(vols, vols) * R, index=TICKERS[:len(vols)],
                       columns=TICKERS[:len(vols)])
    return OptimizationInputs(
        mu=pd.Series(mu, index=cov.index, dtype=float),
        cov=CovarianceEstimate(cov, "sample", n_obs),
        risk_free_rate=rf, start=pd.Timestamp("2020-01-01"), end=pd.Timestamp("2024-12-31"))


@pytest.fixture
def inputs():
    return make_inputs([0.06, 0.08, 0.11, 0.14], [0.10, 0.15, 0.20, 0.28])


def assert_valid(w, bounds=None):
    assert w.sum() == pytest.approx(1.0, abs=1e-9)
    assert (w >= -1e-9).all()
    if bounds:
        for x, (lo, hi) in zip(w, bounds):
            assert lo - 1e-6 <= x <= hi + 1e-6


# --- covariance --------------------------------------------------------------------------

def test_sample_covariance_annualised():
    r = cv.simple_returns(make_gbm_prices(n=300))
    est = sample_covariance(r)
    np.testing.assert_allclose(est.matrix, r.cov() * 252)


def test_ledoit_wolf_is_convex_combination_of_sample_and_scaled_identity():
    r = cv.simple_returns(make_gbm_prices(("A", "B", "C", "D", "E"), n=120, seed=3))
    est = ledoit_wolf_covariance(r)
    lw = LedoitWolf().fit(r.to_numpy())
    S = np.cov(r.to_numpy(), rowvar=False, ddof=0)
    target = np.trace(S) / S.shape[0] * np.eye(S.shape[0])
    expected_daily = ((1 - lw.shrinkage_) * S + lw.shrinkage_ * target) * len(r) / (len(r) - 1)
    np.testing.assert_allclose(est.matrix, expected_daily * 252, rtol=1e-10)
    assert 0 <= est.shrinkage <= 1
    assert is_positive_semidefinite(est.matrix)


def test_cov_to_corr_roundtrip(inputs):
    vol, corr = cov_to_corr(inputs.cov.matrix)
    np.testing.assert_allclose(vol, [0.10, 0.15, 0.20, 0.28])
    np.testing.assert_allclose(np.diag(corr), 1.0)
    np.testing.assert_allclose(np.outer(vol, vol) * corr, inputs.cov.matrix)


def test_covariance_rejects_too_few_observations():
    r = cv.simple_returns(make_gbm_prices(n=4))
    with pytest.raises(ValueError, match="more observations"):
        sample_covariance(r)


# --- no look-ahead -----------------------------------------------------------------------

def test_estimate_inputs_ignores_data_after_as_of():
    r = cv.simple_returns(make_gbm_prices(n=500))
    as_of = r.index[299]
    a = estimate_inputs(r, 0.02, as_of=as_of)
    tampered = r.copy()
    tampered.iloc[300:] *= 50  # wildly different future
    b = estimate_inputs(tampered, 0.02, as_of=as_of)
    pd.testing.assert_series_equal(a.mu, b.mu)
    pd.testing.assert_frame_equal(a.cov.matrix, b.cov.matrix)
    assert a.end == as_of and a.n_obs == 300


def test_estimate_inputs_lookback_and_arithmetic_mean():
    r = cv.simple_returns(make_gbm_prices(n=500))
    est = estimate_inputs(r, 0.0, lookback_days=252)
    assert est.n_obs == 252 and est.start == r.index[-252]
    np.testing.assert_allclose(est.mu, r.iloc[-252:].mean() * 252)


# --- analytic solutions -------------------------------------------------------------------

def test_min_variance_matches_closed_form_when_unconstrained_solution_is_long_only(inputs):
    S = inputs.cov.matrix.to_numpy()
    x = np.linalg.solve(S, np.ones(4))
    gmv = x / x.sum()
    assert (gmv > 0).all()  # so the long-only constraint is not binding
    res = min_variance(inputs)
    assert_valid(res.weights)
    np.testing.assert_allclose(res.weights, gmv, atol=1e-5)


def test_max_sharpe_matches_tangency_portfolio(inputs):
    S = inputs.cov.matrix.to_numpy()
    x = np.linalg.solve(S, inputs.mu.to_numpy() - inputs.risk_free_rate)
    tangency = x / x.sum()
    assert (tangency > 0).all()
    res = max_sharpe(inputs)
    assert_valid(res.weights)
    np.testing.assert_allclose(res.weights, tangency, atol=1e-4)


def test_portfolio_stats_hand_calculation(inputs):
    w = pd.Series([0.25, 0.25, 0.25, 0.25], index=TICKERS)
    s = portfolio_stats(w, inputs)
    S = inputs.cov.matrix.to_numpy()
    assert s.expected_return == pytest.approx(0.0975)
    assert s.volatility == pytest.approx(np.sqrt(w @ S @ w))
    assert s.sharpe == pytest.approx((0.0975 - 0.02) / s.volatility)


# --- optimality against random feasible portfolios ---------------------------------------

def test_no_random_portfolio_beats_optimisers(inputs):
    rnd = random_feasible_portfolios(inputs, n=3000, seed=1)
    mv = min_variance(inputs).stats
    ms = max_sharpe(inputs).stats
    assert (rnd["volatility"] >= mv.volatility - 1e-8).all()
    assert (rnd["sharpe"] <= ms.sharpe + 1e-8).all()


def test_frontier_dominates_random_portfolios(inputs):
    f = efficient_frontier(inputs, n_points=30).points
    rnd = random_feasible_portfolios(inputs, n=3000, seed=2)
    in_range = rnd[(rnd.expected_return >= f.expected_return.min()) &
                   (rnd.expected_return <= f.expected_return.max())]
    # Chords of the convex frontier lie ABOVE it, so interpolation is only a screen: take the
    # random portfolios closest to (or below) the interpolated frontier and solve exactly.
    gap = in_range.volatility - np.interp(in_range.expected_return, f.expected_return, f.volatility)
    for idx in gap.nsmallest(25).index:
        exact = target_return(inputs, float(in_range.loc[idx, "expected_return"])).stats.volatility
        assert in_range.loc[idx, "volatility"] >= exact - 1e-7


# --- constraints ---------------------------------------------------------------------------

def test_custom_bounds_respected(inputs):
    c = Constraints(min_weight=0.05, max_weight=0.40)
    for res in (min_variance(inputs, c), max_sharpe(inputs, c)):
        assert_valid(res.weights, c.bounds(TICKERS))
    c2 = Constraints(min_weight={"A": 0.3}, max_weight={"D": 0.1})
    res = custom(inputs, c2, "max_sharpe")
    assert res.weights["A"] >= 0.3 - 1e-6 and res.weights["D"] <= 0.1 + 1e-6


@pytest.mark.parametrize("c, msg", [
    (Constraints(min_weight=0.3), "Minimum weights sum"),
    (Constraints(max_weight=0.2), "Maximum weights sum"),
    (Constraints(min_weight={"A": 0.5}, max_weight={"A": 0.4}), "exceeds maximum"),
    (Constraints(min_weight=-0.1), "negative"),
])
def test_infeasible_constraints_raise_clear_errors(inputs, c, msg):
    with pytest.raises(OptimizationError, match=msg):
        min_variance(inputs, c)


def test_max_sharpe_undefined_when_no_asset_beats_risk_free():
    inp = make_inputs([0.01, 0.015, 0.02, 0.01], [0.1, 0.1, 0.1, 0.1], rf=0.05)
    with pytest.raises(OptimizationError, match="risk-free"):
        max_sharpe(inp)
    results, errors = compare_strategies(inp)
    assert [r.strategy.value for r in results] == ["Equal weight", "Minimum variance"]
    assert len(errors) == 1 and "Maximum Sharpe" in errors[0]


def test_target_return_and_volatility(inputs):
    p = target_return(inputs, 0.10)
    assert p.stats.expected_return == pytest.approx(0.10, abs=1e-6)
    with pytest.raises(OptimizationError, match="not attainable"):
        target_return(inputs, 0.20)
    q = target_volatility(inputs, 0.15)
    assert q.stats.volatility <= 0.15 + 1e-6
    with pytest.raises(OptimizationError, match="below the minimum"):
        target_volatility(inputs, 0.01)


def test_max_feasible_return_matches_linear_program(inputs):
    bounds = [(0.05, 0.5), (0.0, 0.3), (0.1, 0.25), (0.0, 0.2)]
    res = linprog(-inputs.mu.to_numpy(), A_eq=np.ones((1, 4)), b_eq=[1.0], bounds=bounds)
    assert max_feasible_return(inputs, bounds) == pytest.approx(-res.fun)


# --- efficient frontier ------------------------------------------------------------------

def test_frontier_shape(inputs):
    f = efficient_frontier(inputs, n_points=25)
    pts = f.points
    assert f.n_failed == 0 and len(pts) == 25
    assert np.all(np.diff(pts.expected_return) > 0)
    assert np.all(np.diff(pts.volatility) >= -1e-9)  # efficient branch: more return, more risk
    mv = min_variance(inputs).stats
    assert pts.volatility.iloc[0] == pytest.approx(mv.volatility, abs=1e-8)
    assert max_sharpe(inputs).stats.sharpe >= pts.sharpe.max() - 1e-6
    np.testing.assert_allclose(f.weights.sum(axis=1), 1.0)


def test_frontier_respects_bounds(inputs):
    c = Constraints(max_weight=0.35)
    f = efficient_frontier(inputs, c, n_points=15)
    assert (f.weights <= 0.35 + 1e-6).all().all()
    assert f.points.expected_return.max() == pytest.approx(
        max_feasible_return(inputs, c.bounds(TICKERS)), abs=1e-5)


# --- misc ------------------------------------------------------------------------------------

def test_equal_weight_and_user_defined(inputs):
    ew = equal_weight(inputs)
    np.testing.assert_allclose(ew.weights, 0.25)
    u = user_defined(inputs, pd.Series({"A": 0.7, "D": 0.3}))
    assert u.weights["B"] == 0.0
    assert u.stats.expected_return == pytest.approx(0.7 * 0.06 + 0.3 * 0.14)
    with pytest.raises(OptimizationError):
        user_defined(inputs, pd.Series({"A": 0.7}))


def test_sensitivity_is_reproducible_and_valid(inputs):
    a = max_sharpe_sensitivity(inputs, n_draws=40, seed=7)
    b = max_sharpe_sensitivity(inputs, n_draws=40, seed=7)
    pd.testing.assert_frame_equal(a.draws, b.draws)
    assert len(a.draws) + a.n_failed == 40
    np.testing.assert_allclose(a.draws.sum(axis=1), 1.0)
    assert (a.summary["p05"] <= a.summary["p95"] + 1e-12).all()
    # Standard error of annualised mean: sigma / sqrt(years), years = 1260 / 252 = 5.
    np.testing.assert_allclose(inputs.mu_standard_error, np.array([0.10, 0.15, 0.20, 0.28]) / np.sqrt(5))
