"""Keeps the Portfolio Builder's weight and amount columns in sync.

The table has a Ticker, a Weight (%) and an Amount ($) column. After every edit:

* an amount changed  -> every weight is recalculated as amount / total amount;
* one weight changed -> the other weights are rescaled in proportion so they still sum to 100%,
                        and every amount becomes weight x the portfolio's total value;
* several weights changed at once (e.g. a paste) -> taken as typed (a warning shows if they no
                        longer sum to 100%); amounts follow;
* a row removed or added -> weights are recalculated from the amounts (a new row starts at 0).

Pure functions on DataFrames, so they are unit-tested without Streamlit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

W, A, T = "Weight (%)", "Amount ($)", "Ticker"
DEFAULT_TOTAL = 100_000.0
_TOL_W, _TOL_A = 1e-6, 0.005


def _clean(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.reset_index(drop=True).copy()
    out[W] = pd.to_numeric(out[W], errors="coerce").fillna(0.0).clip(lower=0.0)
    out[A] = pd.to_numeric(out[A], errors="coerce").fillna(0.0).clip(lower=0.0)
    return out


def weights_from_amounts(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    total = float(out[A].sum())
    out[W] = out[A] / total * 100 if total > 0 else 0.0
    return out


def amounts_from_weights(rows: pd.DataFrame, total: float) -> pd.DataFrame:
    out = rows.copy()
    out[A] = out[W] / 100 * total
    return out


def sync_positions(prev: pd.DataFrame, cur: pd.DataFrame) -> tuple[pd.DataFrame, str | None, bool]:
    """Return (synced rows, message for the user or None, whether anything changed)."""
    p, c = _clean(prev), _clean(cur)
    total_prev = float(p[A].sum()) or DEFAULT_TOTAL

    if len(c) != len(p):  # rows added or removed: amounts are the reference
        out = weights_from_amounts(c)
        msg = "Weights recalculated from the remaining amounts." if len(c) < len(p) else None
        return out, msg, True

    a_changed = ~np.isclose(c[A].to_numpy(), p[A].to_numpy(), atol=_TOL_A, rtol=0)
    w_changed = ~np.isclose(c[W].to_numpy(), p[W].to_numpy(), atol=_TOL_W, rtol=0)

    if a_changed.any():
        out = weights_from_amounts(c)
        return out, None, True

    if w_changed.sum() == 1:
        i = int(np.flatnonzero(w_changed)[0])
        new = float(min(c.at[i, W], 100.0))
        others = [j for j in range(len(c)) if j != i]
        out = c.copy()
        out.at[i, W] = new
        rest = float(p.loc[others, W].sum())
        if others:
            if rest > 0:
                out.loc[others, W] = p.loc[others, W] * (100.0 - new) / rest
            else:
                out.loc[others, W] = (100.0 - new) / len(others)
        out = amounts_from_weights(out, total_prev)
        return out, None, True

    if w_changed.sum() > 1:
        return amounts_from_weights(c, total_prev), None, True

    return c, None, False


def add_position(rows: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, str]:
    """Append a ticker. An equally weighted portfolio stays equally weighted (same total value);
    otherwise the new position starts at 0 for the user to size."""
    r = _clean(rows)
    total = float(r[A].sum()) or DEFAULT_TOTAL
    equal = len(r) > 0 and float(r[W].max() - r[W].min()) < 0.011
    out = pd.concat([r, pd.DataFrame({T: [symbol], W: [0.0], A: [0.0]})], ignore_index=True)
    if equal:
        n = len(out)
        out[W], out[A] = 100.0 / n, total / n
        return out, (f"Added {symbol}. The portfolio was equally weighted, so all {n} assets now get "
                     f"{100 / n:.2f}% (${total / n:,.2f}).")
    return out, f"Added {symbol}. Set its weight or amount below."
