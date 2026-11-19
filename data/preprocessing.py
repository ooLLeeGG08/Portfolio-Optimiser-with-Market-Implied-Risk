"""Alignment and return computation. Nothing is forward-filled or interpolated.

See METHODOLOGY §2.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import pandas as pd

from core import conventions as cv
from core.types import DataQualityReport, Severity


def align_prices(
    series: Mapping[str, pd.Series],
    report: DataQualityReport | None = None,
) -> pd.DataFrame:
    """Align on the INTERSECTION of trading dates. Rows with any missing price are
    dropped, never filled (forward-filling fabricates zero returns and biases volatility
    and correlation downward)."""
    report = report if report is not None else DataQualityReport()
    if not series:
        raise ValueError("No price series to align.")
    frame = pd.concat({t: s for t, s in series.items()}, axis=1).sort_index()
    union_rows = len(frame.dropna(how="all"))
    aligned = frame.dropna(how="any")
    report.rows_dropped_in_alignment = union_rows - len(aligned)
    report.n_aligned_observations = len(aligned)
    if aligned.empty:
        report.add(None, Severity.ERROR, "NO_COMMON_DATES",
                   "The selected assets have no trading dates in common.")
        return aligned
    report.aligned_start = aligned.index.min()
    report.aligned_end = aligned.index.max()
    if report.rows_dropped_in_alignment > 0:
        report.add(None, Severity.INFO, "ALIGNMENT_ROWS_DROPPED",
                   f"{report.rows_dropped_in_alignment} date(s) where at least one asset had "
                   f"no price were dropped so all assets share the same dates "
                   f"({report.aligned_start.date()} to {report.aligned_end.date()}).")
    return aligned


def compute_returns(prices: pd.DataFrame | pd.Series,
                    kind: Literal["simple", "log"] = "simple") -> pd.DataFrame | pd.Series:
    if kind == "simple":
        return cv.simple_returns(prices)
    if kind == "log":
        return cv.log_returns(prices)
    raise ValueError(f"Unknown return kind: {kind}")


def month_end_prices(prices: pd.DataFrame | pd.Series,
                     as_of: pd.Timestamp | None = None) -> pd.DataFrame | pd.Series:
    """Last available price in each calendar month, indexed by the actual trading date.

    If ``as_of`` falls inside the final month and before that month's last calendar day,
    the final month is incomplete and is dropped (a partial month is not a monthly return).
    """
    grouped = prices.groupby(prices.index.to_period("M"))
    last_dates = grouped.apply(lambda g: g.index.max())
    monthly = prices.loc[pd.DatetimeIndex(last_dates.to_numpy())]
    if as_of is not None and len(monthly):
        as_of = pd.Timestamp(as_of)
        last_period = monthly.index[-1].to_period("M")
        if as_of.to_period("M") == last_period and as_of < last_period.end_time.normalize():
            monthly = monthly.iloc[:-1]
    return monthly


def monthly_returns(prices: pd.DataFrame | pd.Series,
                    as_of: pd.Timestamp | None = None) -> pd.DataFrame | pd.Series:
    """Simple month-over-month returns from month-end prices (complete months only)."""
    return cv.simple_returns(month_end_prices(prices, as_of))


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation of aligned returns (no pairwise deletion: inputs must be aligned)."""
    if returns.isna().any().any():
        raise ValueError("Returns contain missing values; align prices before computing correlations.")
    return returns.corr()
