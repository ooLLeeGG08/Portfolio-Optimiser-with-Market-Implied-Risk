import numpy as np
import pandas as pd
import pytest

from core import conventions as cv
from data.cache import PriceCache
from data.currency import convert_series, fx_candidates, normalize_currency
from data.market_data import MarketDataProvider, MarketDataService

IDX = pd.bdate_range("2020-01-01", periods=400)


# --- pure conversion ------------------------------------------------------------------------

def test_minor_units_and_iso_codes():
    assert normalize_currency("GBp") == ("GBP", 100.0)
    assert normalize_currency("ZAc") == ("ZAR", 100.0)
    assert normalize_currency("jpy") == ("JPY", 1.0)
    assert fx_candidates("JPY", "USD") == [("USDJPY=X", True), ("JPYUSD=X", False)]


def test_base_currency_unchanged_and_pence_divided():
    p = pd.Series(np.linspace(100, 120, 5), index=IDX[:5], name="X")
    same = convert_series(p, "USD", "USD")
    pd.testing.assert_series_equal(same.series, p)
    assert same.fx_ticker is None and same.dates_dropped == 0
    gbp = convert_series(pd.Series([1000.0], index=IDX[:1], name="V"), "GBp", "USD",
                         pd.Series([1.25], index=IDX[:1]), "GBPUSD=X")
    assert gbp.series.iloc[0] == pytest.approx(12.5)  # 1000 pence = £10 = $12.50


def test_unhedged_returns_combine_asset_and_fx_moves():
    rng = np.random.default_rng(0)
    local = pd.Series(1000 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))), index=IDX, name="A")
    fx = pd.Series(0.007 * np.exp(np.cumsum(rng.normal(0, 0.005, 400))), index=IDX)
    conv = convert_series(local, "JPY", "USD", fx, "USDJPY=X")
    r_usd = cv.simple_returns(conv.series)
    expected = (1 + cv.simple_returns(local)) * (1 + cv.simple_returns(fx)) - 1
    np.testing.assert_allclose(r_usd, expected, rtol=1e-12, atol=1e-14)
    # Constant local price: USD returns are exactly the currency's returns.
    flat = convert_series(pd.Series(500.0, index=IDX, name="F"), "JPY", "USD", fx, "x")
    np.testing.assert_allclose(cv.simple_returns(flat.series), cv.simple_returns(fx))


def test_missing_fx_dates_dropped_not_filled():
    p = pd.Series(100.0, index=IDX[:10], name="A")
    fx = pd.Series(0.5, index=IDX[:10]).drop(IDX[[3, 7]])
    conv = convert_series(p, "EUR", "USD", fx, "USDEUR=X")
    assert conv.dates_dropped == 2
    assert len(conv.series) == 8 and IDX[3] not in conv.series.index
    with pytest.raises(ValueError, match="No EUR->USD"):
        convert_series(p, "EUR", "USD", None)


# --- service integration -------------------------------------------------------------------

class FXProvider(MarketDataProvider):
    name = "fake"
    CURRENCY = {"AAA": "USD", "TOYO.T": "JPY", "VOD.L": "GBp", "SAP.DE": "EUR"}

    def __init__(self):
        self.data = {
            "AAA": pd.Series(np.linspace(100, 110, 400), index=IDX),
            "TOYO.T": pd.Series(3000.0, index=IDX),  # flat in yen
            "VOD.L": pd.Series(100.0, index=IDX),  # 100 pence
            "SAP.DE": pd.Series(180.0, index=IDX),
            "NOCUR.T": pd.Series(50.0, index=IDX),
            "NOCUR": pd.Series(50.0, index=IDX),
            "USDJPY=X": pd.Series(np.linspace(150, 160, 400), index=IDX),  # JPY per USD
            "GBPUSD=X": pd.Series(1.25, index=IDX),  # only the direct pair exists for GBP
        }

    def fetch_history(self, tickers):
        return {t: self.data.get(t, pd.Series(dtype=float, name=t)) for t in tickers}

    def fetch_currency(self, ticker):
        return self.CURRENCY.get(ticker)


@pytest.fixture
def svc(tmp_path):
    return MarketDataService(FXProvider(), PriceCache(tmp_path))


def test_service_converts_to_usd_with_inverted_and_direct_pairs(svc):
    pdata = svc.get_price_data(["AAA", "TOYO.T", "VOD.L"], min_observations=100)
    p = pdata.prices
    assert p["AAA"].iloc[-1] == pytest.approx(110.0)  # USD untouched
    assert p["TOYO.T"].iloc[0] == pytest.approx(3000 / 150)  # yen / (JPY per USD)
    assert p["TOYO.T"].iloc[-1] == pytest.approx(3000 / 160)  # yen weakened -> USD value fell
    assert p["VOD.L"].iloc[0] == pytest.approx(100 / 100 * 1.25)
    prov = pdata.provenance
    assert prov["TOYO.T"].currency == "JPY" and prov["TOYO.T"].fx_ticker == "USDJPY=X"
    assert prov["VOD.L"].currency == "GBp" and prov["VOD.L"].fx_ticker == "GBPUSD=X"
    assert prov["AAA"].currency == "USD" and prov["AAA"].fx_ticker is None
    msgs = {i.ticker: i.message for i in pdata.quality.issues if i.code == "CURRENCY_CONVERTED"}
    assert "unhedged" in msgs["TOYO.T"] and "(inverted)" in msgs["TOYO.T"]
    assert "GBp" in msgs["VOD.L"]


def test_unknown_or_unconvertible_currency(svc):
    pdata = svc.get_price_data(["AAA", "NOCUR.T", "NOCUR", "SAP.DE"], min_observations=100)
    codes = {(i.ticker, i.code) for i in pdata.quality.issues}
    assert ("NOCUR.T", "CURRENCY_UNKNOWN") in codes  # foreign suffix: never guess
    assert ("NOCUR", "CURRENCY_ASSUMED") in codes  # US-style symbol: assumed USD, with a warning
    assert ("SAP.DE", "FX_UNAVAILABLE") in codes  # no EUR rate in this fake source
    assert set(pdata.prices.columns) == {"AAA", "NOCUR"}


def test_fx_gaps_counted_only_inside_the_analysis_window(tmp_path):
    prov = FXProvider()
    prov.data["USDJPY=X"] = prov.data["USDJPY=X"].iloc[100:].drop(IDX[[300, 301]])  # FX starts late, 2 gaps
    svc = MarketDataService(prov, PriceCache(tmp_path))
    pdata = svc.get_price_data(["TOYO.T"], start=IDX[200], min_observations=100)
    msg = next(i.message for i in pdata.quality.issues if i.code == "CURRENCY_CONVERTED")
    assert "2 date(s) without an exchange rate" in msg
    assert len(pdata.prices) == 198


def test_currency_lookup_is_cached(svc):
    svc.get_currency("TOYO.T")
    svc.provider.CURRENCY = {}
    assert svc.get_currency("TOYO.T") == "JPY"
