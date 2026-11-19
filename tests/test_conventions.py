import numpy as np
import pandas as pd
import pytest

from core import conventions as cv


def test_simple_returns_hand_calculation_and_first_row_dropped():
    p = pd.Series([100.0, 110.0, 99.0], index=pd.bdate_range("2024-01-01", periods=3))
    r = cv.simple_returns(p)
    assert len(r) == 2
    np.testing.assert_allclose(r.to_numpy(), [0.10, -0.10])


def test_log_and_simple_returns_are_consistent():
    p = pd.Series([100.0, 105.0, 95.0, 120.0])
    np.testing.assert_allclose(cv.log_returns(p), cv.simple_to_log(cv.simple_returns(p)))
    # Log returns sum over time to the total log growth.
    assert cv.log_returns(p).sum() == pytest.approx(np.log(120 / 100))


def test_non_positive_prices_rejected():
    with pytest.raises(ValueError):
        cv.simple_returns(pd.Series([100.0, 0.0, 101.0]))


def test_annualisation_conventions():
    assert cv.TRADING_DAYS == 252
    assert cv.annualize_vol(0.01) == pytest.approx(0.01 * np.sqrt(252))
    assert cv.deannualize_vol(cv.annualize_vol(0.013)) == pytest.approx(0.013)
    assert cv.annualize_mean(0.0004) == pytest.approx(0.1008)
    assert cv.daily_rate_from_annual(0.0504) == pytest.approx(0.0002)


def test_cagr():
    assert cv.cagr(2.0, 252) == pytest.approx(1.0)
    assert cv.cagr(2.0, 504) == pytest.approx(np.sqrt(2) - 1)
    assert cv.cagr(0.0, 252) == -1.0


def test_horizon_scaling_and_loss_sign():
    assert cv.scale_to_horizon(0.02, 4) == pytest.approx(0.04)
    with pytest.raises(ValueError):
        cv.scale_to_horizon(0.02, 0)
    assert cv.return_to_loss(-0.03) == pytest.approx(0.03)
