import numpy as np
import pandas as pd
import pytest

from core.types import RiskBasis
from risk.covariance import cov_to_corr, stress_correlation
from risk.stress_testing import (asset_shock, correlation_convergence, custom_shock,
                                 market_decline, rate_shock, risk_snapshot, scenario_table,
                                 volatility_increase)

W = pd.Series({"EQ": 0.5, "HIBETA": 0.3, "BOND": 0.2})
VALUE = 1_000_000.0
COV = pd.DataFrame([[0.04, 0.03, -0.004], [0.03, 0.09, -0.006], [-0.004, -0.006, 0.01]],
                   index=W.index, columns=W.index)


def test_market_decline_through_betas():
    betas = pd.Series({"EQ": 1.0, "HIBETA": 2.0, "BOND": -0.5})
    r = market_decline(W, betas, -0.10, VALUE, "SPY")
    np.testing.assert_allclose(r.asset_shocks, [-0.10, -0.20, 0.05])
    assert r.portfolio_return == pytest.approx(0.5 * -0.1 + 0.3 * -0.2 + 0.2 * 0.05)
    assert r.portfolio_loss == pytest.approx(0.10)
    assert r.portfolio_value_after == pytest.approx(900_000)
    assert r.asset_pnl.sum() == pytest.approx(-100_000)
    assert r.basis is RiskBasis.HYPOTHETICAL and "not a prediction" in r.label


def test_shocks_floored_at_total_loss():
    r = market_decline(W, pd.Series({"EQ": 1.0, "HIBETA": 6.0, "BOND": 0.0}), -0.20, VALUE)
    assert r.asset_shocks["HIBETA"] == -1.0  # cannot lose more than 100% long-only


def test_rate_shock_uses_sensitivity_per_percentage_point():
    sens = pd.Series({"EQ": -0.01, "HIBETA": -0.02, "BOND": -0.17})
    r = rate_shock(W, sens, +100, VALUE)
    np.testing.assert_allclose(r.asset_shocks, [-0.01, -0.02, -0.17])
    r50 = rate_shock(W, sens, -50, VALUE)
    assert r50.asset_shocks["BOND"] == pytest.approx(0.085)


def test_asset_and_custom_shock():
    r = asset_shock(W, {"EQ": -0.3}, VALUE)
    assert r.portfolio_return == pytest.approx(-0.15)
    assert r.asset_shocks["BOND"] == 0.0
    with pytest.raises(ValueError, match="Not in the portfolio"):
        asset_shock(W, {"ZZZ": -0.3}, VALUE)
    c = custom_shock(W, {"EQ": -0.1, "BOND": 0.02}, VALUE, "My scenario")
    assert c.name == "My scenario" and c.portfolio_return == pytest.approx(-0.046)


def test_volatility_increase_scales_risk_linearly():
    r = volatility_increase(W, COV, 1.5, VALUE)
    assert r.stressed_risk.volatility == pytest.approx(1.5 * r.baseline_risk.volatility)
    assert r.stressed_risk.var_99_1d == pytest.approx(1.5 * r.baseline_risk.var_99_1d)
    assert r.stressed_contributions.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(r.stressed_contributions,
                               volatility_increase(W, COV, 1.0, VALUE).stressed_contributions)


def test_correlation_convergence_matches_stressed_matrix():
    lam = 0.6
    r = correlation_convergence(W, COV, lam, VALUE)
    vols, R = cov_to_corr(COV)
    Rs = stress_correlation(R, lam).to_numpy()
    S = np.outer(vols, vols) * Rs
    assert r.stressed_risk.volatility == pytest.approx(np.sqrt(W @ S @ W))
    assert r.stressed_risk.volatility > r.baseline_risk.volatility  # diversification lost
    full = correlation_convergence(W, COV, 1.0, VALUE)
    assert full.stressed_risk.volatility == pytest.approx((W * vols).sum())
    both = correlation_convergence(W, COV, lam, VALUE, vol_multiplier=2.0)
    assert both.stressed_risk.volatility == pytest.approx(2 * r.stressed_risk.volatility)


def test_risk_snapshot_and_table():
    snap = risk_snapshot(W, COV)
    assert snap.var_99_1d > snap.var_95_1d > 0
    t = scenario_table([market_decline(W, pd.Series({"EQ": 1, "HIBETA": 1, "BOND": 0}), -0.2, VALUE),
                        volatility_increase(W, COV, 2.0, VALUE)])
    assert len(t) == 2 and t["Basis"].str.contains("Hypothetical").all()
