"""Input and data-quality validation.

Nothing here modifies or fills data. Functions return a DataQualityReport of errors
(asset excluded / input rejected) and warnings (kept, but reported). See METHODOLOGY §2.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

import config
from core.types import DataQualityReport, Severity

_TICKER_RE = re.compile(r"^\^?[A-Z0-9][A-Z0-9.\-=]{0,14}$")


def normalize_tickers(raw: Iterable[str]) -> tuple[list[str], DataQualityReport]:
    """Upper-case, strip, de-duplicate (order preserved) and reject malformed symbols."""
    report = DataQualityReport()
    clean: list[str] = []
    for item in raw:
        if item is None:
            continue
        t = str(item).strip().upper()
        if not t:
            continue
        if not _TICKER_RE.match(t):
            report.add(t, Severity.ERROR, "MALFORMED_TICKER",
                       f"'{t}' is not a valid ticker symbol format.")
            continue
        if t in clean:
            report.add(t, Severity.WARNING, "DUPLICATE_TICKER",
                       f"'{t}' was entered more than once; using it once.")
            continue
        clean.append(t)
    return clean, report


def validate_price_series(
    series: Mapping[str, pd.Series],
    min_observations: int = config.MIN_OBSERVATIONS,
    stale_days: int = config.STALE_DAYS_THRESHOLD,
    extreme_return: float = config.EXTREME_RETURN_THRESHOLD,
) -> DataQualityReport:
    """Per-asset checks on raw (unaligned) price series within the requested window."""
    report = DataQualityReport()

    usable = {t: s.dropna() for t, s in series.items()}
    for t, s in usable.items():
        if s.empty:
            report.add(t, Severity.ERROR, "NO_DATA",
                       f"{t}: no price data returned — the symbol may be invalid or delisted.")

    non_empty = {t: s for t, s in usable.items() if not s.empty}
    if not non_empty:
        return report

    # Panel calendar = union of all dates observed for any asset in the request.
    calendar = pd.DatetimeIndex(sorted(set().union(*(s.index for s in non_empty.values()))))
    panel_last = calendar.max()

    for t, s in non_empty.items():
        if (s <= 0).any():
            report.add(t, Severity.ERROR, "NON_POSITIVE_PRICE",
                       f"{t}: contains zero or negative prices; excluded.")
            continue

        if len(s) < min_observations:
            report.add(t, Severity.ERROR, "INSUFFICIENT_HISTORY",
                       f"{t}: only {len(s)} daily observations in the selected window "
                       f"(minimum {min_observations}). It may have listed recently.")
            continue

        days_behind = int((calendar > s.index.max()).sum())
        if days_behind > stale_days:
            report.add(t, Severity.WARNING, "STALE_DATA",
                       f"{t}: last price on {s.index.max().date()}, {days_behind} trading days "
                       f"before the latest panel date {panel_last.date()} — possibly delisted "
                       f"or suspended. Aligned analysis will end at its last date.")

        own_span = calendar[(calendar >= s.index.min()) & (calendar <= s.index.max())]
        missing = own_span.difference(s.index)
        if len(missing) > 0:
            report.add(t, Severity.WARNING, "MISSING_OBSERVATIONS",
                       f"{t}: {len(missing)} missing daily observation(s) inside its own history "
                       f"relative to the other assets' calendar. These dates are dropped at "
                       f"alignment, not filled.")

        rets = s.pct_change(fill_method=None).dropna()
        extreme = rets[np.abs(rets) > extreme_return]
        if len(extreme) > 0:
            dates = ", ".join(str(d.date()) for d in extreme.index[:5])
            report.add(t, Severity.WARNING, "EXTREME_RETURN",
                       f"{t}: {len(extreme)} daily move(s) larger than {extreme_return:.0%} "
                       f"({dates}{'…' if len(extreme) > 5 else ''}). Kept — verify they are genuine.")
    return report


def validate_weights(weights: pd.Series, long_only: bool = True, tol: float = 1e-6) -> None:
    """Raise ValueError with a user-readable message if weights are invalid."""
    if weights.empty:
        raise ValueError("Portfolio has no assets.")
    if weights.isna().any():
        raise ValueError(f"Missing weight for: {', '.join(weights[weights.isna()].index)}.")
    if not np.isfinite(weights.to_numpy(dtype=float)).all():
        raise ValueError("Weights must be finite numbers.")
    if long_only and (weights < -tol).any():
        neg = ", ".join(weights[weights < -tol].index)
        raise ValueError(f"Negative weights are not allowed in a long-only portfolio: {neg}.")
    total = float(weights.sum())
    if abs(total - 1.0) > tol:
        raise ValueError(f"Weights must sum to 100% for a fully invested portfolio "
                         f"(they sum to {total:.4%}).")


def validate_amounts(amounts: pd.Series) -> None:
    if amounts.empty:
        raise ValueError("Portfolio has no assets.")
    if amounts.isna().any() or not np.isfinite(amounts.to_numpy(dtype=float)).all():
        raise ValueError("Investment amounts must be finite numbers.")
    if (amounts < 0).any():
        raise ValueError("Investment amounts cannot be negative (short positions are not supported).")
    if float(amounts.sum()) <= 0:
        raise ValueError("Total investment amount must be positive.")
