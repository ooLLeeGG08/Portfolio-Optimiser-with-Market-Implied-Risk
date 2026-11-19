"""GARCH(1,1) volatility model (FORECAST basis). See METHODOLOGY §7.

    r_t = mu + e_t,  e_t = sigma_t z_t,  sigma_t^2 = omega + alpha e_{t-1}^2 + beta sigma_{t-1}^2

Fitted with the ``arch`` library to log returns in PERCENT (numerical conditioning).

Timing (verified against a manual recursion): ``fit(last_obs=k)`` uses observations
0..k-1; ``forecast(start=t)`` row t holds sigma^2_{t+1..t+h | t}, i.e. it uses data up to and
including t. Walk-forward refits therefore pass ``last_obs = position(t) + 1``.

h-day forecast of RV_{t,h}:  sqrt( (252/h) * sum_{k=1..h} sigma^2_{t+k|t} ) / 100.
Intervals: simulated return paths from the fitted model -> distribution of realised vol
over the horizon -> 10/50/90 % quantiles. Captures innovation (shock) uncertainty but NOT
parameter-estimation uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox

import config
from backtest.splits import OOSSplit, assert_development_only
from core import conventions as cv

Dist = Literal["normal", "t"]
SCALE = 100.0  # returns are modelled in percent
QUANTILES = (0.10, 0.50, 0.90)


@dataclass(frozen=True)
class GarchFit:
    result: object  # arch ARCHModelResult
    dist: Dist
    params: pd.Series
    std_err: pd.Series
    n_obs: int
    start: pd.Timestamp
    end: pd.Timestamp  # last observation used in estimation
    log_likelihood: float
    aic: float
    bic: float
    ljung_box: pd.DataFrame  # rows: standardised residuals, squared standardised residuals

    @property
    def persistence(self) -> float:
        return float(self.params["alpha[1]"] + self.params["beta[1]"])

    @property
    def half_life_days(self) -> float:
        """Days for a variance shock to decay by half: ln(0.5) / ln(alpha + beta)."""
        p = self.persistence
        return float(np.log(0.5) / np.log(p)) if 0 < p < 1 else float("inf")

    @property
    def unconditional_vol(self) -> float:
        """Long-run annualised volatility sqrt(omega / (1 - alpha - beta)); inf if p >= 1."""
        p = self.persistence
        if p >= 1:
            return float("inf")
        daily_var = self.params["omega"] / (1 - p) / SCALE ** 2
        return float(cv.annualize_vol(np.sqrt(daily_var)))

    @property
    def conditional_volatility(self) -> pd.Series:
        """In-sample annualised conditional volatility sigma_t."""
        cv_pct = self.result.conditional_volatility.dropna()
        return cv.annualize_vol(cv_pct / SCALE).rename("garch_conditional_vol")


def _model(log_returns: pd.Series, dist: Dist):
    return arch_model(log_returns * SCALE, mean="Constant", vol="GARCH", p=1, q=1,
                      dist="t" if dist == "t" else "normal", rescale=False)


def fit_garch(log_returns: pd.Series, dist: Dist = "normal",
              last_date: pd.Timestamp | None = None) -> GarchFit:
    """Fit on observations up to and including ``last_date`` (default: all)."""
    if log_returns.isna().any():
        raise ValueError("Log returns contain missing values.")
    k = len(log_returns) if last_date is None else int(log_returns.index.get_loc(last_date)) + 1
    if k < 250:
        raise ValueError(f"GARCH needs at least 250 observations ({k} available).")
    res = _model(log_returns, dist).fit(last_obs=k, disp="off", show_warning=False)
    std_resid = (res.resid / res.conditional_volatility).dropna()
    lb = pd.concat([acorr_ljungbox(std_resid, lags=[10]), acorr_ljungbox(std_resid ** 2, lags=[10])])
    lb.index = ["standardised residuals (lag 10)", "squared standardised residuals (lag 10)"]
    return GarchFit(res, dist, res.params, res.std_err, int(res.nobs),
                    log_returns.index[0], log_returns.index[k - 1],
                    float(res.loglikelihood), float(res.aic), float(res.bic), lb)


def select_distribution(log_returns: pd.Series, split: OOSSplit) -> tuple[Dist, pd.DataFrame]:
    """Choose normal vs Student-t innovations by BIC on DEVELOPMENT data only."""
    dev = split.development(log_returns)
    assert_development_only(dev, split, "GARCH distribution selection")
    rows = {d: fit_garch(dev, d) for d in ("normal", "t")}
    table = pd.DataFrame({d: {"BIC": f.bic, "AIC": f.aic, "log-likelihood": f.log_likelihood}
                          for d, f in rows.items()}).T
    return ("t" if rows["t"].bic < rows["normal"].bic else "normal"), table


def _horizon_vol(variance_pct2: np.ndarray, h: int) -> np.ndarray:
    """Annualised vol over the horizon from per-step variances (last axis = steps)."""
    return np.sqrt(cv.TRADING_DAYS / h * variance_pct2.sum(axis=-1)) / SCALE


def _innovation_rng(fit: GarchFit, seed: int):
    """Seeded unit-variance innovations matching the fitted distribution. (arch ignores
    ``random_state`` in simulation forecasts, so a custom ``rng`` is required for
    reproducible intervals.)"""
    gen = np.random.default_rng(seed)
    if fit.dist == "t":
        nu = float(fit.params["nu"])
        scale = np.sqrt((nu - 2) / nu)
        return lambda size: gen.standard_t(nu, size=size) * scale
    return lambda size: gen.standard_normal(size)


def _forecast_block(fit: GarchFit, start: pd.Timestamp, h: int, n_sims: int,
                    seed: int) -> pd.DataFrame:
    fc = fit.result.forecast(horizon=h, start=start, reindex=False, method="simulation",
                             simulations=n_sims, rng=_innovation_rng(fit, seed))
    analytic = fit.result.forecast(horizon=h, start=start, reindex=False).variance
    out = pd.DataFrame(index=analytic.index)
    out["point"] = _horizon_vol(analytic.to_numpy(), h)
    sims = fc.simulations.values  # (n_dates, n_sims, h) simulated returns in percent
    rv_paths = np.sqrt(cv.TRADING_DAYS / h * (sims ** 2).sum(axis=2)) / SCALE
    for q in QUANTILES:
        out[f"q{int(q * 100):02d}"] = np.quantile(rv_paths, q, axis=1)
    return out


def walk_forward_garch(log_returns: pd.Series, horizon: int, start: pd.Timestamp,
                       end: pd.Timestamp | None = None, dist: Dist = "normal",
                       refit_every: int = 21, n_sims: int = 500,
                       seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Out-of-sample forecasts of RV_{t,h} for each t in [start, end].

    Parameters are re-estimated every ``refit_every`` days on an expanding window ending at
    the refit date; between refits the variance filter uses the latest returns with fixed
    parameters. Every forecast at t uses data up to and including t only."""
    idx = log_returns.index
    end = end if end is not None else idx[-1]
    dates = idx[(idx >= start) & (idx <= end)]
    if len(dates) == 0:
        raise ValueError("No forecast dates in the requested range.")
    blocks = []
    for b, i in enumerate(range(0, len(dates), refit_every)):
        block = dates[i:i + refit_every]
        fit = fit_garch(log_returns, dist, last_date=block[0])
        fc = _forecast_block(fit, block[0], horizon, n_sims, seed + b)
        blocks.append(fc.loc[block])
    return pd.concat(blocks)


def current_forecast(fit: GarchFit, horizon: int, n_sims: int = 2000,
                     seed: int = config.RANDOM_SEED) -> pd.Series:
    """Forecast of RV over the next h days as of the fit's last observation."""
    return _forecast_block(fit, fit.end, horizon, n_sims, seed).iloc[-1]
