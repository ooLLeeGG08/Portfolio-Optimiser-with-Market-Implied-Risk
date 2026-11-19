"""Single source of truth for financial conventions.

* Trading days per year: 252.
* Volatility annualisation: sigma_annual = sigma_daily * sqrt(252) (assumes i.i.d. returns).
* Mean annualisation: arithmetic, mu_annual = 252 * mu_daily.
* Simple returns aggregate across assets; log returns aggregate across time.
* Losses (VaR, CVaR, stress losses) are reported as POSITIVE numbers.

See docs/METHODOLOGY.md §1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS: int = 252
MONTHS_PER_YEAR: int = 12

ArrayLike = float | np.ndarray | pd.Series | pd.DataFrame


# --- Returns -----------------------------------------------------------------------

def simple_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """R_t = P_t / P_{t-1} - 1. The first row (undefined) is dropped, never filled."""
    _check_positive_prices(prices)
    return prices.pct_change(fill_method=None).iloc[1:]


def log_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """r_t = ln(P_t / P_{t-1}). The first row (undefined) is dropped, never filled."""
    _check_positive_prices(prices)
    return np.log(prices / prices.shift(1)).iloc[1:]


def simple_to_log(r: ArrayLike) -> ArrayLike:
    return np.log1p(r)


def log_to_simple(r: ArrayLike) -> ArrayLike:
    return np.expm1(r)


def _check_positive_prices(prices: pd.DataFrame | pd.Series) -> None:
    values = np.asarray(prices, dtype=float)
    if np.any(values[~np.isnan(values)] <= 0):
        raise ValueError("Prices must be strictly positive to compute returns.")


# --- Annualisation -------------------------------------------------------------------

def annualize_vol(daily_vol: ArrayLike, periods_per_year: int = TRADING_DAYS) -> ArrayLike:
    """Square-root-of-time scaling. Valid under i.i.d. returns only."""
    return daily_vol * np.sqrt(periods_per_year)


def deannualize_vol(annual_vol: ArrayLike, periods_per_year: int = TRADING_DAYS) -> ArrayLike:
    return annual_vol / np.sqrt(periods_per_year)


def annualize_mean(daily_mean: ArrayLike, periods_per_year: int = TRADING_DAYS) -> ArrayLike:
    """Arithmetic annualisation of a mean periodic return."""
    return daily_mean * periods_per_year


def annualize_cov(daily_cov: ArrayLike, periods_per_year: int = TRADING_DAYS) -> ArrayLike:
    return daily_cov * periods_per_year


def daily_rate_from_annual(annual_rate: float, periods_per_year: int = TRADING_DAYS) -> float:
    """Simple de-annualisation used for excess returns: r_daily = r_annual / 252."""
    return annual_rate / periods_per_year


def cagr(total_growth: float, n_periods: int, periods_per_year: int = TRADING_DAYS) -> float:
    """Compound annual growth rate from a growth factor V_T / V_0 over n_periods."""
    if n_periods <= 0:
        raise ValueError("n_periods must be positive.")
    if total_growth <= 0:
        return -1.0
    return float(total_growth ** (periods_per_year / n_periods) - 1.0)


def scale_to_horizon(one_period_value: ArrayLike, horizon: int) -> ArrayLike:
    """sqrt(h) scaling of a 1-period volatility or VaR. Assumes i.i.d. returns —
    callers must label results that use it."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    return one_period_value * np.sqrt(horizon)


# --- Loss convention -------------------------------------------------------------------

def return_to_loss(r: ArrayLike) -> ArrayLike:
    """Loss L = -R; positive values are losses."""
    return -r
