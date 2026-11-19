"""The single definition of realised volatility used everywhere. See METHODOLOGY §7.

Forward target (what is forecast at the close of day t):
    RV_{t,h} = sqrt( (252 / h) * sum_{k=1..h} r_{t+k}^2 )
Backward realised vol (known at the close of day t, used as features / naive forecast):
    RV^back_{t,w} = sqrt( (252 / w) * sum_{k=0..w-1} r_{t-k}^2 )

r are daily LOG returns; zero-mean (the daily mean is negligible relative to volatility).
Both are annualised decimals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core import conventions as cv


def _check(log_returns: pd.Series, n: int) -> None:
    if n < 1:
        raise ValueError("Window / horizon must be at least 1 day.")
    if log_returns.isna().any():
        raise ValueError("Log returns contain missing values.")


def realized_vol_backward(log_returns: pd.Series, window: int) -> pd.Series:
    """Trailing realised volatility using returns up to and including t."""
    _check(log_returns, window)
    ss = (log_returns ** 2).rolling(window, min_periods=window).sum()
    return np.sqrt(cv.TRADING_DAYS / window * ss).rename(f"rv_back_{window}")


def realized_vol_forward(log_returns: pd.Series, horizon: int) -> pd.Series:
    """Target at t: realised volatility over t+1..t+h. NaN for the last h dates (unknown)."""
    _check(log_returns, horizon)
    ss = (log_returns ** 2).rolling(horizon, min_periods=horizon).sum().shift(-horizon)
    return np.sqrt(cv.TRADING_DAYS / horizon * ss).rename(f"rv_fwd_{horizon}")
