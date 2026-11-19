import numpy as np
import pandas as pd
import pytest

from core.types import DataQualityReport
from data.preprocessing import (align_prices, compute_returns, correlation_matrix,
                                month_end_prices, monthly_returns)
from tests.conftest import make_gbm_prices


def test_alignment_drops_unshared_dates_and_never_fills():
    p = make_gbm_prices(("A", "B"), n=50)
    a = p["A"]
    b = p["B"].drop(p.index[[10, 20]])  # B missing two dates
    rep = DataQualityReport()
    aligned = align_prices({"A": a, "B": b}, rep)

    assert len(aligned) == 48
    assert rep.rows_dropped_in_alignment == 2
    assert not aligned.isna().any().any()
    # Every aligned value is an original observation (nothing fabricated).
    assert aligned["A"].equals(a.loc[aligned.index])
    assert aligned["B"].equals(b.loc[aligned.index])
    assert p.index[10] not in aligned.index


def test_alignment_with_no_common_dates_reports_error():
    a = pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-01-01", "2024-01-02"]))
    b = pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-02-01", "2024-02-02"]))
    rep = DataQualityReport()
    out = align_prices({"A": a, "B": b}, rep)
    assert out.empty
    assert any(i.code == "NO_COMMON_DATES" for i in rep.errors)


def test_month_end_prices_use_last_trading_day():
    idx = pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-28", "2024-02-29", "2024-03-01"])
    p = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
    me = month_end_prices(p)
    assert list(me.index) == list(pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-01"]))


def test_incomplete_final_month_is_dropped():
    idx = pd.bdate_range("2024-01-01", "2024-03-15")
    p = pd.Series(np.linspace(100, 110, len(idx)), index=idx)
    assert month_end_prices(p).index[-1].month == 3
    assert month_end_prices(p, as_of=pd.Timestamp("2024-03-15")).index[-1].month == 2
    # A completed month is kept even when as_of is its last calendar day.
    p2 = p[p.index <= "2024-02-29"]
    assert month_end_prices(p2, as_of=pd.Timestamp("2024-02-29")).index[-1].month == 2


def test_monthly_returns_compound_daily_returns():
    p = make_gbm_prices(("A",), n=300)["A"]
    daily = compute_returns(p, "simple")
    monthly = monthly_returns(p)
    me = month_end_prices(p)
    for prev, cur in zip(me.index[:-1], me.index[1:]):
        window = daily[(daily.index > prev) & (daily.index <= cur)]
        assert monthly.loc[cur] == pytest.approx(np.prod(1 + window) - 1)


def test_correlation_requires_aligned_data():
    r = compute_returns(make_gbm_prices(), "log")
    c = correlation_matrix(r)
    np.testing.assert_allclose(np.diag(c), 1.0)
    np.testing.assert_allclose(c, c.T)
    r.iloc[3, 0] = np.nan
    with pytest.raises(ValueError):
        correlation_matrix(r)
