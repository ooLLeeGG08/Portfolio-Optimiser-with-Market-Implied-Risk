"""Plain-English "Key takeaways" for the Dashboard, written from the app's own numbers.

Each function turns one result into a sentence; nothing here is hard-coded. Pure functions, so
they are unit-tested without Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.types import RiskBasis
from implied_volatility.vix import VixSummary
from risk.risk_contribution import RiskContribution
from risk.risk_metrics import MaxDrawdown


@dataclass(frozen=True)
class Takeaway:
    title: str
    text: str
    basis: RiskBasis


def _p(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}%}".replace("-", "−")


def _d(x) -> str:
    return f"{pd.Timestamp(x):%d %b %Y}"


def risk_concentration(rc: RiskContribution) -> Takeaway:
    """The asset whose share of portfolio volatility differs most from its share of the money."""
    t = rc.table
    gap = t["pct_contribution"] - t["weight"]
    a = gap.abs().idxmax()
    w, s = float(t.loc[a, "weight"]), float(t.loc[a, "pct_contribution"])
    if s >= w:
        text = (f"{a} is {_p(w)} of the portfolio but {_p(s)} of its volatility, the largest gap between "
                f"weight and share of risk.")
    elif s < 0:
        text = (f"{a} is {_p(w)} of the portfolio but {_p(s)} of its volatility: it moves against the other "
                f"assets and offsets part of their risk.")
    else:
        text = (f"{a} is {_p(w)} of the portfolio but only {_p(s)} of its volatility, the largest gap between "
                f"weight and share of risk.")
    top = gap.idxmax()
    if top != a and gap[top] > 0:
        text += (f" The biggest risk share relative to its weight is {top}: {_p(t.loc[top, 'pct_contribution'])} "
                 f"of volatility from {_p(t.loc[top, 'weight'])} of the money.")
    return Takeaway("Where the risk sits", text, rc.basis)


def vix_level(v: VixSummary | None) -> Takeaway:
    if v is None:
        return Takeaway("Market fear gauge", "VIX data is unavailable, so no market-implied comparison is shown.",
                        RiskBasis.MARKET_IMPLIED)
    pc = v.percentile_full
    word = "low" if pc < 0.25 else ("high" if pc > 0.75 else "middle-of-the-range")
    text = (f"The VIX is {_p(v.latest)} ({_d(v.latest_date)}), higher than {pc:.0%} of trading days since "
            f"{pd.Timestamp(v.history_start):%Y}: a {word} level of expected S&P 500 volatility over the next "
            f"30 days. It describes the US market, not this portfolio.")
    return Takeaway("Market fear gauge", text, RiskBasis.MARKET_IMPLIED)


def worst_drawdown(md: MaxDrawdown) -> Takeaway:
    text = (f"The worst fall was {_p(md.depth)}, from {_d(md.peak_date)} to {_d(md.trough_date)} "
            f"({md.peak_to_trough_days} trading days). ")
    if md.recovery_date is not None:
        text += f"It took until {_d(md.recovery_date)} ({md.trough_to_recovery_days} more trading days) to recover."
    else:
        text += "It has not yet recovered to its previous peak."
    return Takeaway("Worst decline", text, RiskBasis.OBSERVED)


def key_takeaways(rc: RiskContribution, vix: VixSummary | None, md: MaxDrawdown) -> list[Takeaway]:
    return [risk_concentration(rc), vix_level(vix), worst_drawdown(md)]


def _names(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def weight_uncertainty(summary: pd.DataFrame) -> str:
    """One plain-English reading of the max-Sharpe sensitivity results (columns point_estimate,
    mean, p05, p95 per asset), written from the numbers."""
    width = summary["p95"] - summary["p05"]
    a = width.idxmax()
    text = (f"The widest range is {a}: anywhere from {_p(summary.loc[a, 'p05'], 0)} to "
            f"{_p(summary.loc[a, 'p95'], 0)} of the portfolio, depending on small changes in past returns.")
    wide = int((width >= 0.20).sum())
    if wide >= 2:
        share = f"All {wide}" if wide == len(summary) else f"{wide} of the {len(summary)}"
        text += f" {share} assets have a range of 20 percentage points or more."
    dropped = [t for t in summary.index if summary.loc[t, "point_estimate"] < 0.005 and summary.loc[t, "mean"] >= 0.01]
    if dropped:
        avg = (_p(summary.loc[dropped[0], "mean"], 0) if len(dropped) == 1
               else _names([f"{_p(summary.loc[t, 'mean'], 0)} for {t}" for t in dropped]))
        text += (f" {_names(dropped)} {'gets' if len(dropped) == 1 else 'get'} 0% in the “best” mix but "
                 f"about {avg} on average, so leaving {'it' if len(dropped) == 1 else 'them'} out is not a firm result.")
    if width.max() >= 0.20:
        text += " Treat the exact weights as rough, not as a recommendation."
    return text
