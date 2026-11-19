"""Rolling historical volatility (OBSERVED) and its use as a naive forecast (FORECAST).

Uses the same zero-mean realised-vol definition as the forecasting target
(forecasting.targets) so the naive forecast and the target are directly comparable.
As a forecast of RV_{t,h}, the trailing w-day value at t is used unchanged
("volatility tomorrow equals volatility recently").
"""

from __future__ import annotations

import pandas as pd

from forecasting.targets import realized_vol_backward

DEFAULT_WINDOWS = (20, 60, 252)


def rolling_volatility(log_returns: pd.Series, windows=DEFAULT_WINDOWS) -> pd.DataFrame:
    return pd.concat({f"hist_{w}d": realized_vol_backward(log_returns, w) for w in windows}, axis=1)


def historical_forecast(log_returns: pd.Series, window: int) -> pd.Series:
    """Naive forecast of RV_{t,h} made at t: trailing w-day realised vol at t."""
    return realized_vol_backward(log_returns, window).rename(f"hist_{window}d")
