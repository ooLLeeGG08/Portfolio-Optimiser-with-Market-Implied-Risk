from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from core.types import Severity
from data.cache import PriceCache
from data.market_data import (DataSourceError, MarketDataProvider, MarketDataService,
                              drop_incomplete_session, tbill_discount_to_bond_equivalent)
from tests.conftest import make_gbm_prices


class FakeProvider(MarketDataProvider):
    name = "fake"

    def __init__(self, data: dict[str, pd.Series], fail: bool = False):
        self.data = data
        self.fail = fail
        self.calls: list[list[str]] = []

    def fetch_history(self, tickers):
        self.calls.append(list(tickers))
        if self.fail:
            raise DataSourceError("simulated outage")
        return {t: self.data.get(t, pd.Series(dtype=float, name=t)) for t in tickers}  # RangeIndex on purpose


@pytest.fixture
def panel():
    return make_gbm_prices(("AAA", "BBB", "SPY"), n=400)


@pytest.fixture
def service(panel, tmp_path):
    provider = FakeProvider({t: panel[t] for t in panel})
    return MarketDataService(provider, PriceCache(tmp_path), cache_ttl_hours=12)


def test_get_price_data_happy_path_and_provenance(service, panel):
    pdata = service.get_price_data(["aaa", "BBB"])
    assert pdata.tickers == ["AAA", "BBB"]
    pd.testing.assert_frame_equal(pdata.prices, panel[["AAA", "BBB"]], check_freq=False,
                                  check_names=False)
    assert pdata.quality.errors == []
    assert pdata.provenance["AAA"].source == "fake"
    assert pdata.provenance["AAA"].last_date == panel.index[-1]


def test_cache_hit_avoids_second_request_and_ttl_expiry_refetches(service):
    service.get_price_data(["AAA"])
    pdata = service.get_price_data(["AAA"])
    assert service.provider.calls == [["AAA"]]
    assert pdata.provenance["AAA"].from_cache
    service.cache_ttl_hours = 0.0
    service.get_price_data(["AAA"])
    assert len(service.provider.calls) == 2


def test_invalid_ticker_excluded_others_kept(service):
    pdata = service.get_price_data(["AAA", "NOPE"])
    assert pdata.tickers == ["AAA"]
    assert pdata.quality.excluded_tickers() == ["NOPE"]
    assert any(i.code == "NO_DATA" and i.ticker == "NOPE" for i in pdata.quality.issues)


def test_invalid_ticker_with_date_window(service, panel):
    """Regression: an empty series for an invalid ticker must not break date slicing."""
    pdata = service.get_price_data(["AAA", "NOPE"], start=panel.index[10], min_observations=100)
    assert pdata.tickers == ["AAA"]
    assert pdata.quality.excluded_tickers() == ["NOPE"]


def test_outage_uses_stale_cache_with_warning(panel, tmp_path):
    cache = PriceCache(tmp_path)
    old = datetime.now(timezone.utc) - timedelta(days=3)
    cache.put("AAA", panel["AAA"], "fake", fetched_at=old)
    svc = MarketDataService(FakeProvider({}, fail=True), cache, cache_ttl_hours=12)
    pdata = svc.get_price_data(["AAA"])
    assert pdata.tickers == ["AAA"]
    assert any(i.code == "STALE_CACHE_USED" for i in pdata.quality.warnings)


def test_outage_without_cache_reports_unavailable_not_invalid(tmp_path):
    svc = MarketDataService(FakeProvider({}, fail=True), PriceCache(tmp_path))
    pdata = svc.get_price_data(["AAA"])
    codes = {i.code for i in pdata.quality.errors}
    assert "DATA_SOURCE_UNAVAILABLE" in codes
    assert "NO_DATA" not in codes
    assert pdata.prices.empty


def test_all_empty_responses_hint_at_rate_limit(tmp_path):
    svc = MarketDataService(FakeProvider({}), PriceCache(tmp_path))
    pdata = svc.get_price_data(["AAA", "BBB"])
    assert any(i.code == "ALL_REQUESTS_EMPTY" for i in pdata.quality.warnings)


def test_insufficient_common_history(panel, tmp_path):
    # Each asset alone passes (300 and 260 obs) but they share only 160 dates.
    data = {"AAA": panel["AAA"].iloc[:300], "BBB": panel["BBB"].iloc[140:]}
    svc = MarketDataService(FakeProvider(data), PriceCache(tmp_path))
    pdata = svc.get_price_data(["AAA", "BBB"], min_observations=252)
    assert any(i.code == "INSUFFICIENT_COMMON_HISTORY" for i in pdata.quality.errors)


def test_date_window_is_respected(service, panel):
    start, end = panel.index[50], panel.index[349]
    pdata = service.get_price_data(["AAA"], start=start, end=end, min_observations=100)
    assert pdata.prices.index[0] == start and pdata.prices.index[-1] == end


def test_drop_incomplete_session():
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-11-18", "2026-11-19"]))
    during = datetime(2026, 11, 19, 15, 0, tzinfo=timezone.utc)  # 10:00 New York
    after = datetime(2026, 11, 19, 22, 0, tzinfo=timezone.utc)  # 17:00 New York
    next_day = datetime(2026, 11, 20, 14, 0, tzinfo=timezone.utc)
    assert len(drop_incomplete_session(s, during)) == 1
    assert len(drop_incomplete_session(s, after)) == 2
    assert len(drop_incomplete_session(s, next_day)) == 2


def test_tbill_bond_equivalent_conversion():
    assert tbill_discount_to_bond_equivalent(0.04) == pytest.approx(365 * 0.04 / (360 - 0.04 * 91))
    assert tbill_discount_to_bond_equivalent(0.04) > 0.04


def test_latest_risk_free_rate(tmp_path):
    idx = pd.bdate_range("2026-11-02", periods=5)
    irx = pd.Series([4.10, 4.12, 4.11, 4.15, 4.147], index=idx)
    svc = MarketDataService(FakeProvider({"^IRX": irx}), PriceCache(tmp_path))
    rf = svc.latest_risk_free_rate()
    assert rf.observation_date == idx[-1]
    assert rf.value == pytest.approx(tbill_discount_to_bond_equivalent(0.04147))


@pytest.mark.network
def test_live_yfinance_fetch(tmp_path):
    from data.market_data import YFinanceProvider
    svc = MarketDataService(YFinanceProvider(), PriceCache(tmp_path))
    pdata = svc.get_price_data(["SPY", "AAPL", "NOTAREALTICKERZZ"], start="2020-01-01")
    assert set(pdata.tickers) == {"SPY", "AAPL"}
    assert pdata.quality.excluded_tickers() == ["NOTAREALTICKERZZ"]
    assert (pdata.prices > 0).all().all()
    assert len(pdata.prices) > 1000
    assert svc.latest_risk_free_rate() is not None


# --- security search --------------------------------------------------------------------------

class _FakeSearch:
    def __init__(self, query, **kwargs):
        self.quotes = [
            {"symbol": "TM", "longname": "Toyota Motor Corporation", "exchDisp": "NYSE", "quoteType": "EQUITY"},
            {"symbol": "TOY=F", "shortname": "Toyota futures", "exchDisp": "CME", "quoteType": "FUTURE"},
            {"symbol": "7203.T", "shortname": "TOYOTA MOTOR CORP ", "exchDisp": "Tokyo", "quoteType": "EQUITY",
             "sectorDisp": "Consumer Cyclical"},
            {"symbol": "TOYETF", "longname": "Toy ETF", "exchange": "PCX", "quoteType": "ETF"},
        ]


def test_yfinance_search_filters_futures_and_maps_fields(tmp_path):
    from data.market_data import YFinanceProvider
    prov = YFinanceProvider(tz_cache_dir=tmp_path)

    class FakeYF:
        Search = _FakeSearch
    prov._yf = FakeYF
    res = prov.search("toyota")
    assert [m.symbol for m in res] == ["TM", "7203.T", "TOYETF"]  # futures removed
    assert res[1].name == "TOYOTA MOTOR CORP" and res[1].exchange == "Tokyo"
    assert res[2].exchange == "PCX"
    assert res[0].label == "TM — Toyota Motor Corporation · NYSE · Stock"
    assert res[2].label.endswith("· ETF")


def test_service_search_is_cached_and_lookup_is_exact(tmp_path):
    from data.market_data import SecurityMatch

    class SearchProvider(FakeProvider):
        def __init__(self):
            super().__init__({})
            self.queries = []

        def search(self, query, max_results=10):
            self.queries.append(query)
            return [SecurityMatch("TM", "Toyota", "NYSE", "EQUITY"),
                    SecurityMatch("7203.T", "Toyota", "Tokyo", "EQUITY")]

    svc = MarketDataService(SearchProvider(), PriceCache(tmp_path))
    assert len(svc.search("Toyota")) == 2
    svc.search("toyota ")
    assert svc.provider.queries == ["Toyota"]  # case/whitespace-insensitive cache
    assert svc.lookup("7203.t").exchange == "Tokyo"
    assert svc.lookup("72") is None  # partial symbols are not accepted as matches
    assert svc.search("   ") == []
