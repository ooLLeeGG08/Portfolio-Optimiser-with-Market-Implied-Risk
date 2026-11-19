"""Walk-forward strategy backtest with turnover and transaction costs. See METHODOLOGY §8.

At the close of each rebalancing date t, target weights are estimated from the ``lookback``
daily returns ending at t (no data after t), applied from t+1, and left to drift until the
next rebalance. Trading cost = cost_rate x sum_i |w_target,i - w_drifted,i| (total traded
notional as a fraction of portfolio value), deducted from portfolio value at t. Reported
turnover is one-way: 1/2 sum |dw|.

The lookback, rebalancing frequency and cost are fixed a priori; metrics are reported
separately for the development window and the locked test window (backtest.splits).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core import conventions as cv
from core.types import RebalanceFrequency
from portfolio.optimization import (Constraints, OptimizationError, OptimizationInputs,
                                    estimate_inputs, max_sharpe, min_variance)
from portfolio.portfolio import rebalance_dates
from risk.risk_metrics import max_drawdown

WeightRule = Callable[[OptimizationInputs], pd.Series]


def equal_weight_rule(inp: OptimizationInputs) -> pd.Series:
    return pd.Series(1.0 / len(inp.tickers), index=inp.tickers)


def min_variance_rule(constraints: Constraints = Constraints()) -> WeightRule:
    return lambda inp: min_variance(inp, constraints).weights


def max_sharpe_rule(constraints: Constraints = Constraints()) -> WeightRule:
    """Falls back to minimum variance when no asset beats the risk-free rate (recorded)."""
    def rule(inp):
        try:
            return max_sharpe(inp, constraints).weights
        except OptimizationError:
            w = min_variance(inp, constraints).weights
            w.attrs["fallback"] = True
            return w
    return rule


def fixed_weights_rule(weights: pd.Series) -> WeightRule:
    return lambda inp: weights.reindex(inp.tickers).fillna(0.0)


@dataclass
class BacktestResult:
    name: str
    values: pd.Series  # net of costs
    returns: pd.Series  # daily net simple returns
    target_weights: pd.DataFrame  # one row per rebalance date
    turnover: pd.Series  # one-way, per rebalance date
    costs: pd.Series  # cost as a fraction of value, per rebalance date
    fallbacks: int = 0
    notes: list[str] = field(default_factory=list)


def walk_forward_backtest(prices: pd.DataFrame, rule: WeightRule, name: str,
                          risk_free_rate: float, lookback: int = 252,
                          rebalance: RebalanceFrequency = RebalanceFrequency.MONTHLY,
                          cost_bps: float = 10.0, initial_value: float = 1.0) -> BacktestResult:
    if prices.isna().any().any():
        raise ValueError("Prices contain missing values; align them first.")
    rets = cv.simple_returns(prices)
    reb_all = rebalance_dates(rets.index, rebalance)
    eligible = [d for d in reb_all if rets.index.get_loc(d) + 1 >= lookback]
    if rebalance is RebalanceFrequency.NONE or not eligible:
        first = rets.index[lookback - 1] if len(rets) >= lookback else None
        if first is None:
            raise ValueError(f"Need at least {lookback} returns before the first allocation.")
        eligible = [first] + ([] if rebalance is RebalanceFrequency.NONE else eligible)
    start = eligible[0]
    reb_set = set(eligible)
    cost_rate = cost_bps / 1e4

    R = rets.to_numpy()
    cols = list(prices.columns)
    pos0 = rets.index.get_loc(start)
    value, w = initial_value, np.zeros(len(cols))
    values, out_rets, tw_rows, to_rows, cost_rows, fallbacks = [], [], {}, {}, {}, 0
    for p in range(pos0, len(rets)):
        d = rets.index[p]
        if p > pos0:
            gross = float(w @ R[p])
            value *= 1 + gross
            w = w * (1 + R[p]) / (1 + gross)  # drift
        if d in reb_set:
            inp = estimate_inputs(rets.iloc[:p + 1], risk_free_rate, lookback_days=lookback)
            target = rule(inp).reindex(cols).fillna(0.0)
            fallbacks += int(bool(target.attrs.get("fallback")))
            t = target.to_numpy()
            traded = 0.0 if p == pos0 else float(np.abs(t - w).sum())  # initial buy not charged
            cost = cost_rate * traded
            value *= 1 - cost
            w = t
            tw_rows[d], to_rows[d], cost_rows[d] = target, traded / 2, cost
        values.append(value)
    vals = pd.Series(values, index=rets.index[pos0:], name=name)
    res = BacktestResult(name, vals, cv.simple_returns(vals).rename(name),
                         pd.DataFrame(tw_rows).T, pd.Series(to_rows, name="turnover"),
                         pd.Series(cost_rows, name="cost"), fallbacks)
    if fallbacks:
        res.notes.append(f"{fallbacks} rebalance(s) fell back to minimum variance because no "
                         f"asset's estimated return exceeded the risk-free rate.")
    return res


def backtest_metrics(res: BacktestResult, risk_free_rate: float,
                     start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> dict:
    v = res.values
    if start is not None:
        v = v[v.index >= start]
    if end is not None:
        v = v[v.index <= end]
    r = cv.simple_returns(v)
    if len(r) < 20:
        raise ValueError("Too few observations in the evaluation window.")
    years = len(r) / cv.TRADING_DAYS
    in_win = lambda s: s[(s.index >= v.index[0]) & (s.index <= v.index[-1])]
    vol = float(cv.annualize_vol(r.std(ddof=1)))
    ann = float(cv.annualize_mean(r.mean()))
    return {
        "Start": v.index[0].date(), "End": v.index[-1].date(),
        "CAGR": cv.cagr(float(v.iloc[-1] / v.iloc[0]), len(r)),
        "Annualised return": ann, "Annualised volatility": vol,
        "Sharpe ratio": (ann - risk_free_rate) / vol if vol > 0 else float("nan"),
        "Max drawdown": max_drawdown(v).depth,
        "Annual turnover (one-way)": float(in_win(res.turnover).sum() / years),
        "Annual cost drag": float(in_win(res.costs).sum() / years),
    }
