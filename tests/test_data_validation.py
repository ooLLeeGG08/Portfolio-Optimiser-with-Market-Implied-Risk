import numpy as np
import pandas as pd
import pytest

from core.types import Severity
from data.data_validation import (normalize_tickers, validate_amounts,
                                  validate_price_series, validate_weights)
from tests.conftest import make_gbm_prices


def codes(report, ticker=None):
    return {i.code for i in report.issues if ticker is None or i.ticker == ticker}


# --- ticker symbols -------------------------------------------------------------------

def test_normalize_tickers_cleans_dedupes_and_rejects_malformed():
    clean, rep = normalize_tickers([" aapl", "MSFT", "msft", "^vix", "BRK-B", "AAPL!!", "", None,
                                    "WAYTOOLONGTICKER123"])
    assert clean == ["AAPL", "MSFT", "^VIX", "BRK-B"]
    assert "DUPLICATE_TICKER" in codes(rep, "MSFT")
    assert "MALFORMED_TICKER" in codes(rep, "AAPL!!")
    assert "MALFORMED_TICKER" in codes(rep, "WAYTOOLONGTICKER123")


# --- price series -------------------------------------------------------------------

@pytest.fixture
def panel():
    return make_gbm_prices(("A", "B"), n=400)


def test_clean_series_produce_no_issues(panel):
    rep = validate_price_series({t: panel[t] for t in panel}, min_observations=252)
    assert rep.issues == []


def test_empty_series_is_error(panel):
    rep = validate_price_series({"A": panel["A"], "BAD": pd.Series(dtype=float)})
    assert "NO_DATA" in codes(rep, "BAD")
    assert rep.excluded_tickers() == ["BAD"]


def test_insufficient_history_is_error(panel):
    rep = validate_price_series({"A": panel["A"], "NEW": panel["B"].iloc[-100:]},
                                min_observations=252)
    assert "INSUFFICIENT_HISTORY" in codes(rep, "NEW")
    assert rep.excluded_tickers() == ["NEW"]


def test_non_positive_price_is_error(panel):
    b = panel["B"].copy()
    b.iloc[50] = 0.0
    rep = validate_price_series({"A": panel["A"], "B": b})
    assert "NON_POSITIVE_PRICE" in codes(rep, "B")


def test_stale_series_warned_not_excluded(panel):
    rep = validate_price_series({"A": panel["A"], "B": panel["B"].iloc[:-30]},
                                min_observations=252, stale_days=10)
    assert "STALE_DATA" in codes(rep, "B")
    assert rep.excluded_tickers() == []


def test_internal_gaps_counted_exactly(panel):
    b = panel["B"].drop(panel.index[[100, 101, 250]])
    rep = validate_price_series({"A": panel["A"], "B": b}, min_observations=252)
    issue = next(i for i in rep.issues if i.code == "MISSING_OBSERVATIONS")
    assert issue.ticker == "B" and issue.severity is Severity.WARNING
    assert issue.message.startswith("B: 3 missing")


def test_extreme_return_flagged_but_kept(panel):
    b = panel["B"].copy()
    b.iloc[200:] *= 0.5  # a -50% day
    rep = validate_price_series({"A": panel["A"], "B": b}, min_observations=252)
    assert "EXTREME_RETURN" in codes(rep, "B")
    assert rep.excluded_tickers() == []


# --- weights and amounts --------------------------------------------------------------

def test_valid_weights_pass():
    validate_weights(pd.Series({"A": 0.5, "B": 0.3, "C": 0.2}))


@pytest.mark.parametrize("w, msg", [
    ({"A": 0.5, "B": 0.3}, "sum to 100%"),
    ({"A": 1.2, "B": -0.2}, "Negative weights"),
    ({"A": 0.5, "B": np.nan}, "Missing weight"),
    ({}, "no assets"),
])
def test_invalid_weights_raise(w, msg):
    with pytest.raises(ValueError, match=msg):
        validate_weights(pd.Series(w, dtype=float))


def test_invalid_amounts_raise():
    with pytest.raises(ValueError, match="negative"):
        validate_amounts(pd.Series({"A": 100.0, "B": -5.0}))
    with pytest.raises(ValueError, match="positive"):
        validate_amounts(pd.Series({"A": 0.0}))
