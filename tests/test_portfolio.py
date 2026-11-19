import numpy as np
import pandas as pd
import pytest

from core import conventions as cv
from core.types import RebalanceFrequency as RF
from portfolio.portfolio import (PortfolioSpec, performance_summary, rebalance_dates,
                                 simulate_portfolio)
from tests.conftest import make_gbm_prices

W = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})


# --- specification ----------------------------------------------------------------

def test_spec_from_amounts():
    spec = PortfolioSpec.from_amounts({"A": 6000, "B": 3000, "C": 1000, "D": 0})
    assert spec.initial_value == 10_000
    assert spec.tickers == ["A", "B", "C"]  # zero amount dropped
    np.testing.assert_allclose(spec.weights, [0.6, 0.3, 0.1])
    assert spec.weights.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("kwargs, msg", [
    (dict(weights={"A": 0.6, "B": 0.6}), "sum to 100%"),
    (dict(weights={"A": 1.0}, horizon_days=0), "horizon"),
    (dict(weights={"A": 1.0}, risk_free_rate=4.0), "Risk-free"),
])
def test_spec_rejects_invalid_inputs(kwargs, msg):
    w = kwargs.pop("weights")
    with pytest.raises(ValueError, match=msg):
        PortfolioSpec.from_weights(w, **kwargs)


def test_without_renormalises_and_reports():
    spec = PortfolioSpec.from_weights({"A": 0.5, "B": 0.3, "C": 0.2})
    new, msg = spec.without(["B"])
    np.testing.assert_allclose(new.weights, [0.5 / 0.7, 0.2 / 0.7])
    assert "B" in msg and "30.0%" in msg
    same, none = spec.without(["Z"])
    assert same is spec and none is None


# --- rebalancing dates ---------------------------------------------------------------

def test_monthly_rebalance_dates_are_first_trading_day_of_month():
    idx = pd.bdate_range("2024-01-01", "2024-04-30")
    d = rebalance_dates(idx, RF.MONTHLY)
    assert list(d) == list(pd.to_datetime(["2024-02-01", "2024-03-01", "2024-04-01"]))
    assert len(rebalance_dates(idx, RF.QUARTERLY)) == 1
    assert len(rebalance_dates(idx, RF.NONE)) == 0


# --- value path --------------------------------------------------------------------

def test_buy_and_hold_matches_hand_calculation(gbm_prices):
    path = simulate_portfolio(gbm_prices, W, RF.NONE, initial_value=1000.0)
    expected = 1000.0 * (gbm_prices / gbm_prices.iloc[0] * W).sum(axis=1)
    np.testing.assert_allclose(path.values, expected)
    assert path.turnover.empty


def test_portfolio_return_equals_weighted_asset_returns(gbm_prices):
    """R_p,t = sum_i w_i,t-1 R_i,t with previous end-of-day (drifted or rebalanced) weights."""
    path = simulate_portfolio(gbm_prices, W, RF.MONTHLY)
    asset_r = cv.simple_returns(gbm_prices)[W.index]
    implied = (path.weights.shift(1).iloc[1:] * asset_r).sum(axis=1)
    np.testing.assert_allclose(path.returns, implied, atol=1e-12)


def test_weights_reset_to_target_on_rebalance_dates(gbm_prices):
    path = simulate_portfolio(gbm_prices, W, RF.MONTHLY)
    assert len(path.rebalance_dates) > 0
    for d in path.rebalance_dates:
        np.testing.assert_allclose(path.weights.loc[d], W, atol=1e-12)
    np.testing.assert_allclose(path.weights.sum(axis=1), 1.0)
    # Weights drift away from target between rebalances.
    assert not np.allclose(path.weights.iloc[-1], W)


def test_turnover_hand_calculation():
    idx = pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-01"])
    prices = pd.DataFrame({"A": [100.0, 200.0, 200.0], "B": [100.0, 100.0, 100.0]}, index=idx)
    path = simulate_portfolio(prices, pd.Series({"A": 0.5, "B": 0.5}), RF.MONTHLY, 100.0)
    # Before the Feb-1 rebalance A doubled: drifted weights (2/3, 1/3) -> target (1/2, 1/2).
    assert path.turnover.loc["2024-02-01"] == pytest.approx(1 / 6)
    assert path.values.iloc[-1] == pytest.approx(150.0)  # rebalancing itself costs nothing here


def test_constant_prices_give_constant_value():
    idx = pd.bdate_range("2024-01-01", periods=80)
    prices = pd.DataFrame({"A": 50.0, "B": 20.0}, index=idx)
    path = simulate_portfolio(prices, pd.Series({"A": 0.4, "B": 0.6}), RF.MONTHLY, 500.0)
    np.testing.assert_allclose(path.values, 500.0)
    np.testing.assert_allclose(path.turnover, 0.0, atol=1e-15)


def test_simulation_rejects_missing_prices(gbm_prices):
    p = gbm_prices.copy()
    p.iloc[5, 0] = np.nan
    with pytest.raises(ValueError, match="missing"):
        simulate_portfolio(p, W)
    with pytest.raises(ValueError, match="No prices"):
        simulate_portfolio(gbm_prices, pd.Series({"A": 0.5, "Z": 0.5}))


# --- performance -------------------------------------------------------------------

def test_performance_summary_known_path():
    idx = pd.bdate_range("2024-01-01", periods=253)
    values = pd.Series(100.0 * 2 ** (np.arange(253) / 252), index=idx)  # doubles in 252 steps
    s = performance_summary(values)
    assert s.total_return == pytest.approx(1.0)
    assert s.cagr == pytest.approx(1.0)
    assert s.annualized_volatility == pytest.approx(0.0, abs=1e-12)
    assert s.annualized_mean_return == pytest.approx(252 * (2 ** (1 / 252) - 1))
