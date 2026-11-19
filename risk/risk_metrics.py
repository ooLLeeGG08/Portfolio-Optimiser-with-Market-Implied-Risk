"""Standard (OBSERVED) portfolio risk measures from historical returns. See METHODOLOGY §5.

Conventions (core.conventions): daily simple returns; annualised mean = 252 x mean;
annualised vol = sqrt(252) x std (ddof=1); risk-free rate is an annual decimal, converted to
a daily rate as r_f / 252 when forming excess returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from core import conventions as cv
from core.types import RiskBasis


def annualized_volatility(returns: pd.Series) -> float:
    return float(cv.annualize_vol(returns.std(ddof=1)))


def annualized_return(returns: pd.Series) -> float:
    """Arithmetic annualised mean of daily simple returns."""
    return float(cv.annualize_mean(returns.mean()))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float) -> float:
    """(252 * mean(R) - r_f) / (sqrt(252) * std(R)). With a constant r_f this equals the
    annualised mean of daily excess returns over their annualised std."""
    vol = annualized_volatility(returns)
    return (annualized_return(returns) - risk_free_rate) / vol if vol > 0 else float("nan")


def downside_deviation(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualised root-mean-square of shortfalls below the daily risk-free rate, averaged
    over ALL observations (the standard Sortino definition)."""
    target = cv.daily_rate_from_annual(risk_free_rate)
    shortfall = np.minimum(returns.to_numpy() - target, 0.0)
    return float(cv.annualize_vol(np.sqrt(np.mean(shortfall ** 2))))


def sortino_ratio(returns: pd.Series, risk_free_rate: float) -> float:
    dd = downside_deviation(returns, risk_free_rate)
    return (annualized_return(returns) - risk_free_rate) / dd if dd > 0 else float("nan")


def drawdown_series(values: pd.Series) -> pd.Series:
    """V_t / max_{s<=t} V_s - 1 (zero at new highs, negative otherwise)."""
    return values / values.cummax() - 1.0


@dataclass(frozen=True)
class MaxDrawdown:
    depth: float  # negative, e.g. -0.34
    peak_date: pd.Timestamp
    trough_date: pd.Timestamp
    recovery_date: pd.Timestamp | None  # first date back at the previous peak; None if not yet
    peak_to_trough_days: int  # trading days
    trough_to_recovery_days: int | None  # trading days; None if not yet recovered


def max_drawdown(values: pd.Series) -> MaxDrawdown:
    if len(values) < 2:
        raise ValueError("Need at least two values.")
    dd = drawdown_series(values)
    trough = dd.idxmin()
    peak = values.loc[:trough].idxmax()
    after = values.loc[trough:]
    recovered = after[after >= values.loc[peak]]
    recovery = recovered.index[0] if len(recovered) else None
    pos = values.index.get_loc
    return MaxDrawdown(
        depth=float(dd.min()), peak_date=peak, trough_date=trough, recovery_date=recovery,
        peak_to_trough_days=pos(trough) - pos(peak),
        trough_to_recovery_days=None if recovery is None else pos(recovery) - pos(trough),
    )


def _aligned(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    df = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(df) < 3:
        raise ValueError("Not enough overlapping observations with the benchmark.")
    return df.iloc[:, 0], df.iloc[:, 1]


def beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Cov(R_p, R_b) / Var(R_b) on dates both series share."""
    r, b = _aligned(returns, benchmark_returns)
    return float(np.cov(r, b, ddof=1)[0, 1] / np.var(b, ddof=1))


def asset_betas(asset_returns: pd.DataFrame, benchmark_returns: pd.Series) -> pd.Series:
    return pd.Series({c: beta(asset_returns[c], benchmark_returns) for c in asset_returns})


@dataclass(frozen=True)
class Sensitivity:
    """OLS slope of asset returns on a factor, with HAC (Newey-West) t-statistics."""

    coefficient: pd.Series
    t_stat: pd.Series
    r_squared: pd.Series
    n_obs: int


def factor_sensitivities(asset_returns: pd.DataFrame, factor: pd.Series,
                         hac_lags: int = 5) -> Sensitivity:
    """Regress each asset's daily return on a factor (e.g. daily change in the 10-year
    yield, in percentage points). Reports t-stats so unstable sensitivities are visible."""
    df = pd.concat([asset_returns, factor.rename("_factor")], axis=1, join="inner").dropna()
    if len(df) < 30:
        raise ValueError("Not enough overlapping observations to estimate sensitivities.")
    X = sm.add_constant(df["_factor"])
    coef, t, r2 = {}, {}, {}
    for c in asset_returns.columns:
        fit = sm.OLS(df[c], X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
        coef[c], t[c], r2[c] = fit.params["_factor"], fit.tvalues["_factor"], fit.rsquared
    return Sensitivity(pd.Series(coef), pd.Series(t), pd.Series(r2), len(df))


@dataclass(frozen=True)
class RiskSummary:
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: MaxDrawdown
    beta: float | None
    correlation_with_benchmark: float | None
    start: pd.Timestamp
    end: pd.Timestamp
    basis: RiskBasis = RiskBasis.OBSERVED


def risk_summary(returns: pd.Series, values: pd.Series, risk_free_rate: float,
                 benchmark_returns: pd.Series | None = None) -> RiskSummary:
    b = corr = None
    if benchmark_returns is not None:
        b = beta(returns, benchmark_returns)
        r, bm = _aligned(returns, benchmark_returns)
        corr = float(np.corrcoef(r, bm)[0, 1])
    return RiskSummary(
        annualized_return=annualized_return(returns),
        annualized_volatility=annualized_volatility(returns),
        sharpe=sharpe_ratio(returns, risk_free_rate),
        sortino=sortino_ratio(returns, risk_free_rate),
        max_drawdown=max_drawdown(values),
        beta=b, correlation_with_benchmark=corr,
        start=returns.index[0], end=returns.index[-1],
    )
