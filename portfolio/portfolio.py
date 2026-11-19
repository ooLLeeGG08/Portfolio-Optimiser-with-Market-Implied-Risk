"""Portfolio specification and historical value path. See METHODOLOGY §3.

The value path holds units of each asset. Units are reset to target weights at the close
of each rebalancing date; between rebalances weights drift with prices. Transaction costs
are not charged here (the walk-forward backtest charges them explicitly).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

import config
from core import conventions as cv
from core.types import RebalanceFrequency, RiskBasis
from data.data_validation import validate_amounts, validate_weights

MAX_HORIZON_DAYS = 5 * cv.TRADING_DAYS


@dataclass(frozen=True)
class PortfolioSpec:
    weights: pd.Series  # target weights, index = tickers, sum to 1
    benchmark: str = config.DEFAULT_BENCHMARK
    horizon_days: int = 21  # investment / risk horizon in trading days
    risk_free_rate: float = 0.0  # annualised decimal
    rebalance: RebalanceFrequency = RebalanceFrequency.MONTHLY
    initial_value: float = config.DEFAULT_INITIAL_VALUE
    amounts: pd.Series | None = field(default=None, compare=False)

    def __post_init__(self):
        validate_weights(self.weights)
        if not 1 <= int(self.horizon_days) <= MAX_HORIZON_DAYS:
            raise ValueError(f"Investment horizon must be between 1 and {MAX_HORIZON_DAYS} "
                             f"trading days.")
        if not -0.05 <= self.risk_free_rate <= 0.25:
            raise ValueError("Risk-free rate must be an annual decimal between -5% and 25% "
                             "(e.g. 0.04 for 4%).")
        if self.initial_value <= 0:
            raise ValueError("Portfolio value must be positive.")

    @property
    def tickers(self) -> list[str]:
        return list(self.weights.index)

    @classmethod
    def from_amounts(cls, amounts: Mapping[str, float], **kwargs) -> "PortfolioSpec":
        a = pd.Series(amounts, dtype=float)
        validate_amounts(a)
        nonzero = a[a > 0]
        return cls(weights=nonzero / nonzero.sum(), initial_value=float(nonzero.sum()),
                   amounts=nonzero, **kwargs)

    @classmethod
    def from_weights(cls, weights: Mapping[str, float], **kwargs) -> "PortfolioSpec":
        w = pd.Series(weights, dtype=float)
        return cls(weights=w[w != 0], **kwargs)

    def without(self, excluded: list[str]) -> tuple["PortfolioSpec", str | None]:
        """Remove assets (e.g. failed data validation) and renormalise the remaining
        weights. Returns the new spec and a message the caller MUST show the user."""
        drop = [t for t in excluded if t in self.weights.index]
        if not drop:
            return self, None
        remaining = self.weights.drop(drop)
        if remaining.empty or remaining.sum() <= 0:
            raise ValueError("No assets with usable data remain in the portfolio.")
        removed_weight = float(self.weights[drop].sum())
        new = replace(self, weights=remaining / remaining.sum(),
                      amounts=None if self.amounts is None else self.amounts.drop(drop, errors="ignore"))
        msg = (f"Excluded {', '.join(drop)} ({removed_weight:.1%} of the portfolio) because of "
               f"data problems; the remaining weights were rescaled proportionally to sum to 100%.")
        return new, msg


@dataclass
class PortfolioPath:
    values: pd.Series  # portfolio value at each close
    returns: pd.Series  # daily simple returns (first date dropped)
    weights: pd.DataFrame  # end-of-day weights (after any rebalance on that day)
    rebalance_dates: pd.DatetimeIndex
    turnover: pd.Series  # one-way turnover at each rebalance date
    basis: RiskBasis = RiskBasis.OBSERVED


def rebalance_dates(index: pd.DatetimeIndex, freq: RebalanceFrequency) -> pd.DatetimeIndex:
    """First trading day of each new period (excluding the initial allocation date)."""
    if freq is RebalanceFrequency.NONE or len(index) < 2:
        return pd.DatetimeIndex([])
    code = {RebalanceFrequency.MONTHLY: "M", RebalanceFrequency.QUARTERLY: "Q",
            RebalanceFrequency.ANNUAL: "Y"}[freq]
    periods = index.to_period(code)
    is_new = np.r_[False, periods[1:] != periods[:-1]]
    return index[is_new]


def simulate_portfolio(prices: pd.DataFrame, weights: pd.Series,
                       rebalance: RebalanceFrequency = RebalanceFrequency.MONTHLY,
                       initial_value: float = config.DEFAULT_INITIAL_VALUE) -> PortfolioPath:
    if prices.empty:
        raise ValueError("No prices supplied.")
    if prices.isna().any().any():
        raise ValueError("Prices contain missing values; align them first (no filling).")
    missing = set(weights.index) - set(prices.columns)
    if missing:
        raise ValueError(f"No prices for: {', '.join(sorted(missing))}.")
    validate_weights(weights)

    cols = list(weights.index)
    P = prices[cols].to_numpy(dtype=float)
    w = weights.to_numpy(dtype=float)
    n = len(P)

    reb = rebalance_dates(prices.index, rebalance)
    reb_pos = set(prices.index.get_indexer(reb))

    units = initial_value * w / P[0]
    values = np.empty(n)
    eod_w = np.empty_like(P)
    turnover = []
    for t in range(n):
        v = float(units @ P[t])
        if t in reb_pos:
            drifted = units * P[t] / v
            turnover.append(0.5 * np.abs(w - drifted).sum())
            units = v * w / P[t]
        values[t] = v
        eod_w[t] = units * P[t] / v

    vals = pd.Series(values, index=prices.index, name="portfolio_value")
    return PortfolioPath(
        values=vals,
        returns=cv.simple_returns(vals).rename("portfolio_return"),
        weights=pd.DataFrame(eod_w, index=prices.index, columns=cols),
        rebalance_dates=reb,
        turnover=pd.Series(turnover, index=reb, name="turnover", dtype=float),
    )


@dataclass(frozen=True)
class PerformanceSummary:
    start: pd.Timestamp
    end: pd.Timestamp
    n_returns: int
    start_value: float
    end_value: float
    total_return: float
    cagr: float
    annualized_mean_return: float  # arithmetic, 252 * mean daily simple return
    annualized_volatility: float  # sqrt(252) * std of daily simple returns (ddof=1)
    basis: RiskBasis = RiskBasis.OBSERVED


def performance_summary(values: pd.Series) -> PerformanceSummary:
    if len(values) < 2:
        raise ValueError("Need at least two observations.")
    rets = cv.simple_returns(values)
    growth = float(values.iloc[-1] / values.iloc[0])
    return PerformanceSummary(
        start=values.index[0], end=values.index[-1], n_returns=len(rets),
        start_value=float(values.iloc[0]), end_value=float(values.iloc[-1]),
        total_return=growth - 1.0,
        cagr=cv.cagr(growth, len(rets)),
        annualized_mean_return=float(cv.annualize_mean(rets.mean())),
        annualized_volatility=float(cv.annualize_vol(rets.std(ddof=1))),
    )


def current_allocation(path: PortfolioPath) -> pd.DataFrame:
    """Latest market value and drifted weight per asset."""
    w = path.weights.iloc[-1]
    v = float(path.values.iloc[-1])
    return pd.DataFrame({"weight": w, "market_value": w * v}).sort_values("weight", ascending=False)
