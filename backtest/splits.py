"""The locked out-of-sample test window. See METHODOLOGY §8.

The most recent ``test_days`` dates form the test window. Everything at least ``purge_days``
before it is the development window. Model selection, hyper-parameter choice and early
stopping may only see development data; ``assert_development_only`` enforces this.

The purge gap matters for forecasting: a forecast made at t targets returns t+1..t+h, so
development forecast dates must end h days before the test window starts, or the last
development targets would overlap test-period returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import config


class TestDataLeakageError(RuntimeError):
    """Raised when a model-selection step is handed data from the locked test window."""

    __test__ = False  # not a pytest test class


@dataclass(frozen=True)
class OOSSplit:
    dev_start: pd.Timestamp
    dev_end: pd.Timestamp  # last date usable for development
    test_start: pd.Timestamp  # first date of the locked test window
    test_end: pd.Timestamp
    purge_days: int

    def development(self, obj: pd.Series | pd.DataFrame):
        return obj.loc[obj.index <= self.dev_end]

    def test(self, obj: pd.Series | pd.DataFrame):
        return obj.loc[(obj.index >= self.test_start) & (obj.index <= self.test_end)]


def make_oos_split(index: pd.DatetimeIndex, test_days: int = config.TEST_WINDOW_DAYS,
                   purge_days: int = 0, min_dev_days: int = 252) -> OOSSplit:
    n = len(index)
    if test_days < 1 or purge_days < 0:
        raise ValueError("test_days must be >= 1 and purge_days >= 0.")
    n_dev = n - test_days - purge_days
    if n_dev < min_dev_days:
        raise ValueError(f"Only {max(n_dev, 0)} development observations remain before the "
                         f"{test_days}-day test window (minimum {min_dev_days}). Use a longer "
                         f"history or a shorter test window.")
    return OOSSplit(dev_start=index[0], dev_end=index[n_dev - 1], test_start=index[n - test_days],
                    test_end=index[-1], purge_days=purge_days)


def assert_development_only(index: pd.DatetimeIndex | pd.Series | pd.DataFrame,
                            split: OOSSplit, what: str = "Model selection") -> None:
    idx = index if isinstance(index, pd.DatetimeIndex) else index.index
    if len(idx) and idx.max() > split.dev_end:
        raise TestDataLeakageError(
            f"{what} received data up to {idx.max().date()}, but only development data "
            f"(up to {split.dev_end.date()}) may be used. The test window "
            f"{split.test_start.date()}–{split.test_end.date()} is locked.")
