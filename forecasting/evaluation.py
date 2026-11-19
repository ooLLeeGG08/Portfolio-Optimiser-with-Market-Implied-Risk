"""Out-of-sample forecast evaluation. See METHODOLOGY §7.

Metrics (forecast f, realised RV y, both annualised vols):
* MAE = mean |f - y|;  RMSE = sqrt(mean (f - y)^2);  bias = mean (f - y) (positive = over-forecast)
* QLIKE = mean( y^2/f^2 - ln(y^2/f^2) - 1 ): robust to noise in the realised-vol proxy
  (Patton 2011); 0 is perfect, penalises under-forecasting more than over-forecasting.
* Directional accuracy: share of dates where sign(f - c) == sign(y - c), c = trailing 21-day
  realised vol at t (known at t) — "did the model call whether vol would rise or fall?"
* Interval coverage and mean width for 10–90 % intervals (nominal 80 %).

Overlap: daily forecasts of h-day-ahead RV share h-1 days of target, so errors are strongly
autocorrelated. Metrics are reported on all dates and on a non-overlapping subsample (every
h-th date), and the Diebold–Mariano test uses a HAC (Bartlett) variance with h-1 lags plus
the Harvey–Leybourne–Newbold small-sample correction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


def qlike(forecast: np.ndarray, realized: np.ndarray) -> np.ndarray:
    ratio = (np.asarray(realized) / np.asarray(forecast)) ** 2
    return ratio - np.log(ratio) - 1.0


def forecast_metrics(forecast: pd.Series, realized: pd.Series, current: pd.Series | None = None,
                     lower: pd.Series | None = None, upper: pd.Series | None = None) -> dict:
    df = pd.concat({"f": forecast, "y": realized}, axis=1).dropna()
    f, y = df["f"].to_numpy(), df["y"].to_numpy()
    err = f - y
    out = {"n": len(df), "MAE": float(np.mean(np.abs(err))),
           "RMSE": float(np.sqrt(np.mean(err ** 2))), "Bias": float(np.mean(err)),
           "QLIKE": float(np.mean(qlike(f, y)))}
    if current is not None:
        c = current.reindex(df.index).to_numpy()
        ok = ~np.isnan(c) & (np.sign(y - c) != 0)
        out["Directional accuracy"] = float(np.mean(np.sign(f[ok] - c[ok]) == np.sign(y[ok] - c[ok])))
    if lower is not None and upper is not None:
        lo, hi = lower.reindex(df.index).to_numpy(), upper.reindex(df.index).to_numpy()
        out["Interval coverage"] = float(np.mean((y >= lo) & (y <= hi)))
        out["Mean interval width"] = float(np.mean(hi - lo))
    return out


def non_overlapping(index: pd.DatetimeIndex, horizon: int) -> pd.DatetimeIndex:
    """Every h-th date, so consecutive target windows do not overlap."""
    return index[::horizon]


@dataclass(frozen=True)
class DMResult:
    statistic: float
    p_value: float
    mean_loss_difference: float  # mean(loss_a - loss_b); negative => model a is better
    n: int


def diebold_mariano(loss_a: pd.Series, loss_b: pd.Series, horizon: int) -> DMResult:
    """H0: equal expected loss. HAC variance with h-1 Bartlett lags; HLN-corrected statistic
    compared with Student-t(n-1)."""
    d = (loss_a - loss_b).dropna().to_numpy()
    n = len(d)
    if n < 10:
        raise ValueError("Too few observations for a Diebold-Mariano test.")
    dbar = d.mean()
    dc = d - dbar
    lags = max(horizon - 1, 0)
    gamma0 = np.dot(dc, dc) / n
    lrv = gamma0
    for k in range(1, lags + 1):
        lrv += 2 * (1 - k / (lags + 1)) * np.dot(dc[k:], dc[:-k]) / n
    if lrv <= 0:
        return DMResult(float("nan"), float("nan"), float(dbar), n)
    dm = dbar / np.sqrt(lrv / n)
    hln = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    stat = float(dm * hln)
    return DMResult(stat, float(2 * stats.t.sf(abs(stat), df=n - 1)), float(dbar), n)


def compare_models(df: pd.DataFrame, realized_col: str, models: dict[str, str],
                   intervals: dict[str, tuple[str, str]], current_col: str | None,
                   horizon: int, subset: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Metrics table, one row per model. ``models`` maps display name -> forecast column."""
    d = df if subset is None else df.loc[subset]
    rows = {}
    for name, col in models.items():
        lo_hi = intervals.get(name)
        rows[name] = forecast_metrics(
            d[col], d[realized_col], d[current_col] if current_col else None,
            d[lo_hi[0]] if lo_hi else None, d[lo_hi[1]] if lo_hi else None)
    return pd.DataFrame(rows).T


def dm_table(df: pd.DataFrame, realized_col: str, models: dict[str, str], benchmark: str,
             horizon: int) -> pd.DataFrame:
    """DM tests of each model against ``benchmark`` under QLIKE and squared-error loss."""
    y = df[realized_col]
    rows = {}
    for name, col in models.items():
        if name == benchmark:
            continue
        r = {}
        for loss_name, fn in (("QLIKE", lambda f: pd.Series(qlike(f, y), index=df.index)),
                              ("Squared error", lambda f: (f - y) ** 2)):
            res = diebold_mariano(fn(df[col]), fn(df[models[benchmark]]), horizon)
            r[f"{loss_name}: mean loss diff"] = res.mean_loss_difference
            r[f"{loss_name}: DM stat"] = res.statistic
            r[f"{loss_name}: p-value"] = res.p_value
        rows[f"{name} vs {benchmark}"] = r
    return pd.DataFrame(rows).T
