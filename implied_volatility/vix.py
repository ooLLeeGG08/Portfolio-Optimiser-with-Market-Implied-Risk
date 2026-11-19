"""VIX analytics (MARKET_IMPLIED, IVSource.VIX_INDEX). See METHODOLOGY §9.

The CBOE VIX is the risk-neutral expected volatility of the S&P 500 over the next ~30
calendar days, derived from SPX option prices. It is NOT a volatility forecast for
individual stocks, and it is risk-neutral: it typically exceeds subsequently realised
volatility (the variance risk premium), so it is not an unbiased forecast even for the index.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.types import IVSource, RiskBasis
from forecasting.targets import realized_vol_forward

VIX_HORIZON_CALENDAR_DAYS = 30
VIX_DESCRIPTION = ("VIX: risk-neutral expected volatility of the S&P 500 over the next ~30 "
                   "calendar days, derived from S&P 500 index option prices. It is not a "
                   "forecast of any individual stock's volatility.")


def to_decimal(vix_points: pd.Series) -> pd.Series:
    """VIX is quoted in volatility points (e.g. 18.5 = 18.5 % annualised)."""
    return (vix_points / 100.0).rename("vix")


def percentile_rank(history: pd.Series, value: float) -> float:
    """Share of historical observations at or below ``value``."""
    h = history.dropna()
    return float((h <= value).mean()) if len(h) else float("nan")


@dataclass(frozen=True)
class VixSummary:
    latest: float  # decimal
    latest_date: pd.Timestamp
    change_1d: float | None  # volatility points (decimal), e.g. +0.012 = +1.2 vol points
    change_5d: float | None
    change_21d: float | None
    mean_20d: float
    mean_60d: float
    mean_252d: float
    percentile_full: float
    percentile_5y: float
    history_start: pd.Timestamp
    n_obs: int
    basis: RiskBasis = RiskBasis.MARKET_IMPLIED
    source: IVSource = IVSource.VIX_INDEX
    description: str = VIX_DESCRIPTION


def vix_frame(vix: pd.Series) -> pd.DataFrame:
    """VIX (decimal) with 20/60/252-day rolling means."""
    out = pd.DataFrame({"vix": vix})
    for w in (20, 60, 252):
        out[f"mean_{w}d"] = vix.rolling(w, min_periods=w).mean()
    return out


def vix_summary(vix: pd.Series) -> VixSummary:
    v = vix.dropna()
    if len(v) < 253:
        raise ValueError("Need at least one year of VIX history.")
    last, date = float(v.iloc[-1]), v.index[-1]

    def chg(n):
        return float(v.iloc[-1] - v.iloc[-1 - n]) if len(v) > n else None

    five_y = v[v.index >= date - pd.DateOffset(years=5)]
    return VixSummary(
        latest=last, latest_date=date, change_1d=chg(1), change_5d=chg(5), change_21d=chg(21),
        mean_20d=float(v.iloc[-20:].mean()), mean_60d=float(v.iloc[-60:].mean()),
        mean_252d=float(v.iloc[-252:].mean()),
        percentile_full=percentile_rank(v, last), percentile_5y=percentile_rank(five_y, last),
        history_start=v.index[0], n_obs=len(v))


@dataclass(frozen=True)
class VarianceRiskPremium:
    frame: pd.DataFrame  # vix, realized_fwd, spread (vix - subsequently realised S&P vol)
    mean_spread: float
    median_spread: float
    share_vix_above: float
    horizon_days: int


def variance_risk_premium(vix: pd.Series, index_log_returns: pd.Series,
                          horizon: int = 21) -> VarianceRiskPremium:
    """Compare VIX at t with the S&P 500 volatility realised over the next ``horizon``
    trading days (~30 calendar days). Ex-post comparison: realised values are only known
    after the fact."""
    rv = realized_vol_forward(index_log_returns, horizon)
    df = pd.concat({"vix": vix, "realized_fwd": rv}, axis=1, join="inner").dropna()
    if len(df) < 30:
        raise ValueError("Not enough overlapping VIX and index observations.")
    df["spread"] = df.vix - df.realized_fwd
    return VarianceRiskPremium(df, float(df.spread.mean()), float(df.spread.median()),
                               float((df.spread > 0).mean()), horizon)
