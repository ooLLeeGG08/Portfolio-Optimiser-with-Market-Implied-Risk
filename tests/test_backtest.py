import numpy as np
import pandas as pd
import pytest

from backtest.walk_forward import (backtest_metrics, equal_weight_rule, fixed_weights_rule,
                                   max_sharpe_rule, min_variance_rule, walk_forward_backtest)
from core.types import RebalanceFrequency as RF
from portfolio.portfolio import simulate_portfolio
from tests.conftest import make_gbm_prices


@pytest.fixture
def prices():
    return make_gbm_prices(("A", "B", "C"), n=700, seed=4)


def test_zero_cost_equal_weight_matches_portfolio_simulation(prices):
    bt = walk_forward_backtest(prices, equal_weight_rule, "EW", 0.0, lookback=252, cost_bps=0)
    start = bt.values.index[0]
    sim = simulate_portfolio(prices.loc[start:], pd.Series(1 / 3, index=prices.columns),
                             RF.MONTHLY, initial_value=1.0)
    # Same first-of-month rebalancing, same drift -> identical value paths.
    np.testing.assert_allclose(bt.values.to_numpy(), sim.values.to_numpy(), rtol=1e-12)


def test_costs_equal_rate_times_traded_notional(prices):
    free = walk_forward_backtest(prices, equal_weight_rule, "EW", 0.0, cost_bps=0)
    paid = walk_forward_backtest(prices, equal_weight_rule, "EW", 0.0, cost_bps=25)
    np.testing.assert_allclose(paid.costs, 0.0025 * 2 * free.turnover)
    assert paid.costs.iloc[0] == 0.0  # initial allocation not charged
    assert paid.values.iloc[-1] < free.values.iloc[-1]
    expected = free.values.iloc[-1] * np.prod(1 - paid.costs.to_numpy())
    assert paid.values.iloc[-1] == pytest.approx(expected, rel=1e-12)


def test_weights_do_not_use_future_data(prices):
    a = walk_forward_backtest(prices, min_variance_rule(), "MV", 0.0)
    cut = prices.index[500]
    tampered = prices.copy()
    tampered.loc[tampered.index > cut] *= np.linspace(1, 3, (tampered.index > cut).sum())[:, None]
    b = walk_forward_backtest(tampered, min_variance_rule(), "MV", 0.0)
    pd.testing.assert_frame_equal(a.target_weights.loc[:cut], b.target_weights.loc[:cut])
    pd.testing.assert_series_equal(a.values.loc[:cut], b.values.loc[:cut])


def test_first_allocation_uses_full_lookback(prices):
    bt = walk_forward_backtest(prices, equal_weight_rule, "EW", 0.0, lookback=252)
    rets_before = prices.index.get_loc(bt.values.index[0])  # returns available = position
    assert rets_before >= 252
    np.testing.assert_allclose(bt.target_weights.sum(axis=1), 1.0)


def test_max_sharpe_falls_back_and_reports():
    p = make_gbm_prices(("A", "B"), n=600, seed=1, mu=-0.002)  # every asset loses money
    bt = walk_forward_backtest(p, max_sharpe_rule(), "MS", 0.05)
    assert bt.fallbacks > 0 and "fell back" in bt.notes[0]


def test_metrics_and_windows(prices):
    bt = walk_forward_backtest(prices, fixed_weights_rule(pd.Series({"A": 1.0})), "A", 0.0, cost_bps=0)
    m = backtest_metrics(bt, 0.0)
    assert m["Annual turnover (one-way)"] == pytest.approx(0.0)
    a = prices["A"].loc[bt.values.index[0]:]
    assert m["Max drawdown"] == pytest.approx((a / a.cummax() - 1).min())
    late = backtest_metrics(bt, 0.0, start=prices.index[600])
    assert late["Start"] == prices.index[600].date()
