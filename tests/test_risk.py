import numpy as np
import pandas as pd
import pytest
from scipy import stats

from core import conventions as cv
from core.types import RiskBasis
from risk import cvar as C
from risk import var as V
from risk.covariance import (VolatilityVector, build_forward_covariance, check_same_horizon,
                             combined_basis, correlation_stress_profile, cov_to_corr,
                             is_valid_correlation, nearest_correlation, stress_correlation)
from risk.risk_contribution import euler_contributions
from risk.risk_metrics import (beta, downside_deviation, drawdown_series, factor_sensitivities,
                               max_drawdown, risk_summary, sharpe_ratio, sortino_ratio)
from risk.tail_risk import covariance_tail_risk, historical_tail_risk, to_frame
from tests.conftest import make_gbm_prices

IDX = pd.bdate_range("2020-01-01", periods=6)


def cov_from(vols, corr):
    vols = np.asarray(vols, dtype=float)
    R = np.full((len(vols), len(vols)), corr)
    np.fill_diagonal(R, 1.0)
    names = list("ABCDEFG"[:len(vols)])
    return pd.DataFrame(np.outer(vols, vols) * R, index=names, columns=names)


# --- standard metrics -----------------------------------------------------------------------

def test_max_drawdown_hand_calculation():
    v = pd.Series([100, 120, 90, 130, 65, 140], index=IDX, dtype=float)
    md = max_drawdown(v)
    assert md.depth == pytest.approx(65 / 130 - 1)
    assert md.peak_date == IDX[3] and md.trough_date == IDX[4] and md.recovery_date == IDX[5]
    assert md.peak_to_trough_days == 1 and md.trough_to_recovery_days == 1
    assert (drawdown_series(v) <= 0).all()


def test_max_drawdown_unrecovered():
    v = pd.Series([100, 110, 80, 90], index=IDX[:4], dtype=float)
    md = max_drawdown(v)
    assert md.recovery_date is None and md.trough_to_recovery_days is None


def test_sharpe_and_sortino_hand_calculation():
    r = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01, 0.02])
    rf = 0.0252  # daily 0.0001
    mu_a, vol_a = 252 * r.mean(), np.sqrt(252) * r.std(ddof=1)
    assert sharpe_ratio(r, rf) == pytest.approx((mu_a - rf) / vol_a)
    shortfall = np.minimum(r - 0.0001, 0)
    dd = np.sqrt(np.mean(shortfall ** 2)) * np.sqrt(252)
    assert downside_deviation(r, rf) == pytest.approx(dd)
    assert sortino_ratio(r, rf) == pytest.approx((mu_a - rf) / dd)


def test_beta_recovers_known_exposure():
    rng = np.random.default_rng(0)
    b = pd.Series(rng.normal(0, 0.01, 500), index=pd.bdate_range("2020-01-01", periods=500))
    assert beta(2.0 * b, b) == pytest.approx(2.0)
    assert beta(b, b) == pytest.approx(1.0)
    noisy = -0.5 * b + pd.Series(rng.normal(0, 0.001, 500), index=b.index)
    assert beta(noisy, b) == pytest.approx(-0.5, abs=0.02)


def test_factor_sensitivity_recovers_duration_like_slope():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2020-01-01", periods=1000)
    dy = pd.Series(rng.normal(0, 0.06, 1000), index=idx)  # daily yield change, pct points
    assets = pd.DataFrame({"BOND": -0.17 * dy + rng.normal(0, 0.002, 1000),
                           "STOCK": rng.normal(0, 0.01, 1000)}, index=idx)
    s = factor_sensitivities(assets, dy)
    assert s.coefficient["BOND"] == pytest.approx(-0.17, abs=0.01)
    assert abs(s.t_stat["BOND"]) > 20 and abs(s.t_stat["STOCK"]) < 4
    assert s.r_squared["BOND"] > 0.9


def test_risk_summary_basis_is_observed():
    p = make_gbm_prices(("P", "B"), n=300)
    r = cv.simple_returns(p)
    s = risk_summary(r["P"], p["P"], 0.02, r["B"])
    assert s.basis is RiskBasis.OBSERVED
    assert -1 <= s.correlation_with_benchmark <= 1


# --- VaR / CVaR --------------------------------------------------------------------------------

def test_historical_var_cvar_on_known_sample():
    r = -np.arange(1, 101) / 1000.0  # losses 0.001 .. 0.100
    assert V.historical_var(r, 0.95) == pytest.approx(np.quantile(np.arange(1, 101) / 1000, 0.95))
    tail = np.arange(1, 101)[np.arange(1, 101) / 1000 >= V.historical_var(r, 0.95)] / 1000
    assert C.historical_cvar(r, 0.95) == pytest.approx(tail.mean())


def test_parametric_closed_form():
    mu, s = 0.0005, 0.012
    assert V.parametric_var(mu, s, 0.99) == pytest.approx(-mu + s * 2.3263478740, rel=1e-8)
    assert C.parametric_cvar(mu, s, 0.975) == pytest.approx(-mu + s * stats.norm.pdf(1.95996) / 0.025, rel=1e-4)


def test_cornish_fisher_reduces_to_normal_without_skew_or_kurtosis():
    for a in (0.95, 0.99):
        assert V.cornish_fisher_var(0.0, 0.01, 0, 0, a) == pytest.approx(V.parametric_var(0, 0.01, a))
        assert C.cornish_fisher_cvar(0.0, 0.01, 0, 0, a) == pytest.approx(C.parametric_cvar(0, 0.01, a), rel=1e-6)
    # Fat tails raise the 99% VaR; negative skew raises it further.
    assert V.cornish_fisher_var(0, 0.01, 0, 3, 0.99) > V.parametric_var(0, 0.01, 0.99)
    assert V.cornish_fisher_var(0, 0.01, -0.5, 3, 0.99) > V.cornish_fisher_var(0, 0.01, 0, 3, 0.99)
    assert V.cornish_fisher_is_monotone(-0.3, 2.0)
    assert not V.cornish_fisher_is_monotone(-2.5, 1.0)


def test_large_normal_sample_methods_agree():
    rng = np.random.default_rng(3)
    r = rng.normal(0.0003, 0.01, 400_000)
    for a in (0.95, 0.99):
        p = V.parametric_var(0.0003, 0.01, a)
        assert V.historical_var(r, a) == pytest.approx(p, rel=0.01)
        assert C.historical_cvar(r, a) == pytest.approx(C.parametric_cvar(0.0003, 0.01, a), rel=0.01)


def test_monte_carlo_normal_matches_parametric_and_t_matches_covariance():
    cov = cov_from([0.012, 0.018], 0.3).to_numpy()
    w = np.array([0.6, 0.4])
    sig = np.sqrt(w @ cov @ w)
    sims = V.simulate_portfolio_returns(w, np.zeros(2), cov, 400_000, None, seed=1)
    assert V.monte_carlo_var(sims, 0.99) == pytest.approx(V.parametric_var(0, sig, 0.99), rel=0.01)
    t_sims = V.simulate_portfolio_returns(w, np.zeros(2), cov, 400_000, dof=6, seed=1)
    assert t_sims.std() == pytest.approx(sig, rel=0.01)  # covariance matched despite fat tails
    assert V.monte_carlo_var(t_sims, 0.99) > V.monte_carlo_var(sims, 0.99)  # fatter 99% tail
    again = V.simulate_portfolio_returns(w, np.zeros(2), cov, 1000, dof=6, seed=1)
    np.testing.assert_array_equal(again, V.simulate_portfolio_returns(w, np.zeros(2), cov, 1000, 6, 1))


def test_dof_from_kurtosis():
    assert V.student_t_dof_from_kurtosis(1.0) == pytest.approx(10.0)
    assert V.student_t_dof_from_kurtosis(0.0) is None
    assert V.student_t_dof_from_kurtosis(1000.0) == pytest.approx(4.1)


def test_tail_risk_table_consistency():
    p = make_gbm_prices(("A", "B", "C"), n=1500, seed=5)
    r = cv.simple_returns(p)
    w = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    port = (r * w).sum(axis=1)
    est = historical_tail_risk(port, r, w, 1_000_000, n_sims=50_000)
    df = to_frame(est)
    assert len(df) == 8
    assert (df["CVaR"] >= df["VaR"] - 1e-12).all()  # CVaR >= VaR for every method
    for m in df["Method"].unique():
        sub = df[df["Method"] == m].set_index("Confidence")
        assert sub.loc[0.99, "VaR"] > sub.loc[0.95, "VaR"]
    assert (df["VaR"] > 0).all()  # these are losses
    np.testing.assert_allclose(df["VaR amount"], df["VaR"] * 1_000_000)
    assert all(e.basis is RiskBasis.OBSERVED for e in est)
    h10 = to_frame(historical_tail_risk(port, r, w, 1.0, horizon_days=10, n_sims=50_000))
    np.testing.assert_allclose(h10["VaR"], df["VaR"] * np.sqrt(10))
    assert h10["Notes"].str.contains("sqrt\\(10\\)").all()


def test_cornish_fisher_withheld_when_expansion_invalid():
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2015-01-01", periods=3000)
    r = pd.DataFrame({"A": rng.standard_t(3, 3000) * 0.01}, index=idx)  # very fat tails
    est = historical_tail_risk(r["A"], r, pd.Series({"A": 1.0}), 1.0, n_sims=20_000)
    cf = [e for e in est if e.method == "Cornish-Fisher"]
    assert all(np.isnan(e.var) and np.isnan(e.cvar) for e in cf)
    assert all(any("Not reported" in n for n in e.notes) for e in cf)
    others = [e for e in est if e.method != "Cornish-Fisher"]
    assert all(np.isfinite(e.var) for e in others)


def test_covariance_tail_risk_inherits_basis():
    cov = cov_from([0.2, 0.3], 0.4)
    est = covariance_tail_risk(pd.Series({"A": 0.5, "B": 0.5}), cov, 100.0, RiskBasis.FORECAST,
                               n_sims=20_000)
    assert {e.basis for e in est} == {RiskBasis.FORECAST}
    sig_d = np.sqrt(np.array([.5, .5]) @ cov.to_numpy() @ np.array([.5, .5]) / 252)
    pn = [e for e in est if e.method == "Parametric (normal)" and e.confidence == 0.99][0]
    assert pn.var == pytest.approx(sig_d * stats.norm.ppf(0.99))


# --- risk contributions ----------------------------------------------------------------------

def test_euler_contributions_reconcile_to_total_volatility():
    cov = cov_from([0.15, 0.25, 0.35, 0.10], 0.3)
    w = pd.Series([0.4, 0.3, 0.2, 0.1], index=cov.index)
    rc = euler_contributions(w, cov)
    t = rc.table
    assert t["component_contribution"].sum() == pytest.approx(rc.portfolio_volatility, rel=1e-12)
    assert t["pct_contribution"].sum() == pytest.approx(1.0)
    assert rc.portfolio_volatility == pytest.approx(np.sqrt(w @ cov @ w))
    # Marginal contribution equals the numerical derivative d sigma / d w_i.
    eps = 1e-7
    for i, name in enumerate(cov.index):
        bumped = w.copy()
        bumped[name] += eps
        num = (np.sqrt(bumped @ cov @ bumped) - rc.portfolio_volatility) / eps
        assert t.loc[name, "marginal_contribution"] == pytest.approx(num, rel=1e-5)


def test_symmetric_portfolio_has_equal_contributions():
    cov = cov_from([0.2, 0.2, 0.2], 0.0)
    rc = euler_contributions(pd.Series(1 / 3, index=cov.index), cov)
    np.testing.assert_allclose(rc.table["pct_contribution"], 1 / 3)
    single = euler_contributions(pd.Series({"A": 1.0}), cov)
    assert single.table.loc["A", "pct_contribution"] == pytest.approx(1.0)


# --- forward covariance D.R.D ---------------------------------------------------------------

def test_drd_reconstructs_historical_covariance():
    cov = cov_from([0.15, 0.25, 0.35], 0.4)
    vols, R = cov_to_corr(cov)
    vv = VolatilityVector(vols, 21, {t: RiskBasis.OBSERVED for t in vols.index}, "hist")
    fc = build_forward_covariance(vv, R)
    np.testing.assert_allclose(fc.matrix, cov)
    assert fc.basis is RiskBasis.OBSERVED
    assert any("historical correlations may not represent future" in a for a in fc.assumptions)


def test_drd_basis_labelling():
    _, R = cov_to_corr(cov_from([0.2, 0.2], 0.5))
    vals = pd.Series({"A": 0.2, "B": 0.3})
    mixed = VolatilityVector(vals, 21, {"A": RiskBasis.MARKET_IMPLIED, "B": RiskBasis.IMPLIED_PROXY}, "iv")
    assert build_forward_covariance(mixed, R).basis is RiskBasis.IMPLIED_PROXY
    implied = VolatilityVector(vals, 21, {"A": RiskBasis.MARKET_IMPLIED, "B": RiskBasis.MARKET_IMPLIED}, "iv")
    fc = build_forward_covariance(implied, R)
    assert fc.basis is RiskBasis.MARKET_IMPLIED
    assert any("not a pure market-implied" in a for a in fc.assumptions)
    assert combined_basis([RiskBasis.FORECAST, RiskBasis.OBSERVED]) is RiskBasis.FORECAST


def test_horizon_mismatch_rejected():
    s = pd.Series({"A": 0.2})
    a = VolatilityVector(s, 21, {"A": RiskBasis.FORECAST}, "garch")
    b = VolatilityVector(s, 5, {"A": RiskBasis.FORECAST}, "lstm")
    with pytest.raises(ValueError, match="different horizons"):
        check_same_horizon(a, b)
    with pytest.raises(ValueError, match="basis must be given"):
        VolatilityVector(s, 21, {}, "x")


def test_invalid_correlation_is_repaired():
    bad = pd.DataFrame([[1, 0.9, -0.9], [0.9, 1, 0.9], [-0.9, 0.9, 1]], index=list("ABC"), columns=list("ABC"))
    assert not is_valid_correlation(bad)
    fixed = nearest_correlation(bad)
    assert is_valid_correlation(fixed)
    np.testing.assert_allclose(np.diag(fixed), 1.0)
    vv = VolatilityVector(pd.Series({"A": .2, "B": .2, "C": .2}), 21,
                          {t: RiskBasis.FORECAST for t in "ABC"}, "x")
    fc = build_forward_covariance(vv, bad)
    assert any("nearest valid correlation" in a for a in fc.assumptions)


def test_correlation_stress_properties():
    cov = cov_from([0.15, 0.25, 0.10], 0.2)
    vols, R = cov_to_corr(cov)
    w = pd.Series([0.5, 0.3, 0.2], index=cov.index)
    for lam in np.linspace(0, 1, 11):
        assert is_valid_correlation(stress_correlation(R, lam))
    prof = correlation_stress_profile(w, vols, R)
    assert prof.portfolio_volatility.iloc[0] == pytest.approx(np.sqrt(w @ cov @ w))
    assert prof.portfolio_volatility.iloc[-1] == pytest.approx((w * vols).sum())  # no diversification
    assert np.all(np.diff(prof.portfolio_volatility) > 0)
    with pytest.raises(ValueError):
        stress_correlation(R, 1.5)
