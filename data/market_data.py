"""Market-data access.

``MarketDataProvider`` is the vendor interface; ``YFinanceProvider`` implements it.
``MarketDataService`` adds caching, validation and alignment and returns ``PriceData``
with provenance (source, fetch time, first/last observation) for every series.

No missing values are ever fabricated: failures surface in the DataQualityReport.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

import pandas as pd

import config
from core.types import DataQualityReport, PriceData, SeriesProvenance, Severity
from data.cache import CacheEntry, PriceCache
from data.data_validation import normalize_tickers, validate_price_series
from data.preprocessing import align_prices

log = logging.getLogger(__name__)

NEW_YORK = ZoneInfo("America/New_York")
# Daily bars dated today are provisional until the close has settled.
SESSION_SETTLED_ET = time(16, 30)


class DataSourceError(RuntimeError):
    """The data source could not be reached or refused the request."""


@dataclass(frozen=True)
class OptionChainRaw:
    """Raw listed-option quotes for one underlying, as delivered by the provider.

    ``quotes`` columns: expiry (Timestamp), type ('call'/'put'), strike, bid, ask, last,
    volume, open_interest, last_trade (UTC Timestamp), provider_iv.
    """

    ticker: str
    spot: float
    quote_time: datetime  # UTC time of the underlying quote
    quotes: pd.DataFrame
    source: str


class MarketDataProvider(ABC):
    name: str = "abstract"
    supports_options: bool = False

    @abstractmethod
    def fetch_history(self, tickers: list[str]) -> dict[str, pd.Series]:
        """Full available history of daily adjusted close prices per ticker.

        Must return an entry for every requested ticker; an empty Series means the
        provider returned no data for it. Must not fill or interpolate missing values.
        Raise DataSourceError if the source itself is unavailable.
        """

    def fetch_option_chain(self, ticker: str, max_days: int = 400) -> OptionChainRaw | None:
        """Listed option quotes for expiries within ``max_days`` calendar days, or None if
        the provider has no option data for the ticker. Providers without option data
        keep this default."""
        return None


def drop_incomplete_session(series: pd.Series, now: datetime | None = None) -> pd.Series:
    """Drop a bar dated today (New York) if the session has not settled yet — during
    market hours the 'close' is a live intraday price, not a closing price."""
    if series.empty:
        return series
    now_et = (now or datetime.now(timezone.utc)).astimezone(NEW_YORK)
    last = series.index.max()
    if last.date() == now_et.date() and now_et.time() < SESSION_SETTLED_ET:
        return series.iloc[:-1]
    return series


class YFinanceProvider(MarketDataProvider):
    name = "Yahoo Finance (yfinance)"
    supports_options = True

    def __init__(self, tz_cache_dir=None):
        import yfinance as yf

        self._yf = yf
        # yfinance keeps an sqlite timezone cache; a private location avoids lock contention.
        tz_dir = tz_cache_dir or (config.CACHE_DIR.parent / "yf_tz")
        tz_dir.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(tz_dir))

    def fetch_history(self, tickers: list[str]) -> dict[str, pd.Series]:
        if not tickers:
            return {}
        try:
            frame = self._yf.download(
                tickers, period="max", interval="1d", auto_adjust=True,
                progress=False, threads=False, multi_level_index=True,
            )
        except Exception as exc:  # network errors, rate limits, parsing failures
            raise DataSourceError(f"Yahoo Finance request failed: {exc}") from exc

        out: dict[str, pd.Series] = {}
        if frame is None or frame.empty or "Close" not in frame.columns.get_level_values(0):
            return {t: empty_series(t) for t in tickers}
        close = frame["Close"]
        for t in tickers:
            if t in close.columns:
                s = close[t].dropna().astype(float)
                s.index = pd.DatetimeIndex(s.index).tz_localize(None).astype("datetime64[ns]")
                s.index.name = "Date"
                out[t] = drop_incomplete_session(s.rename(t))
            else:
                out[t] = empty_series(t)
        return out


    def fetch_option_chain(self, ticker: str, max_days: int = 400) -> OptionChainRaw | None:
        try:
            tk = self._yf.Ticker(ticker)
            expiries = list(tk.options)
        except Exception as exc:
            raise DataSourceError(f"Option expiries for {ticker} could not be retrieved: {exc}") from exc
        if not expiries:
            return None
        now = datetime.now(timezone.utc)
        horizon = pd.Timestamp(now.date()) + pd.Timedelta(days=max_days)
        frames, spot, qtime = [], None, None
        for e in expiries:
            exp = pd.Timestamp(e)
            if exp > horizon:
                break
            try:
                ch = tk.option_chain(e)
            except Exception as exc:
                raise DataSourceError(f"Option chain {ticker} {e} could not be retrieved: {exc}") from exc
            und = ch.underlying if isinstance(ch.underlying, dict) else {}
            if spot is None and und.get("regularMarketPrice"):
                spot = float(und["regularMarketPrice"])
                if und.get("regularMarketTime"):
                    qtime = datetime.fromtimestamp(int(und["regularMarketTime"]), tz=timezone.utc)
            for kind, df in (("call", ch.calls), ("put", ch.puts)):
                if df is None or df.empty:
                    continue
                frames.append(pd.DataFrame({
                    "expiry": exp, "type": kind, "strike": df["strike"].astype(float),
                    "bid": df["bid"].astype(float), "ask": df["ask"].astype(float),
                    "last": df["lastPrice"].astype(float),
                    "volume": pd.to_numeric(df["volume"], errors="coerce"),
                    "open_interest": pd.to_numeric(df["openInterest"], errors="coerce"),
                    "last_trade": pd.to_datetime(df["lastTradeDate"], utc=True),
                    "provider_iv": df["impliedVolatility"].astype(float),
                }))
        if not frames or spot is None:
            return None
        return OptionChainRaw(ticker, spot, qtime or now, pd.concat(frames, ignore_index=True), self.name)


@dataclass(frozen=True)
class RiskFreeRate:
    value: float  # annualised, decimal, bond-equivalent basis
    observation_date: pd.Timestamp
    source: str
    note: str


def tbill_discount_to_bond_equivalent(discount_yield: float, days: int = 91) -> float:
    """Convert a T-bill discount yield (decimal) to a bond-equivalent yield.
    BEY = 365 d / (360 - d * days)."""
    return 365.0 * discount_yield / (360.0 - discount_yield * days)


class MarketDataService:
    def __init__(self, provider: MarketDataProvider | None = None,
                 cache: PriceCache | None = None,
                 cache_ttl_hours: float = config.CACHE_TTL_HOURS):
        self.provider = provider or YFinanceProvider()
        self.cache = cache or PriceCache(config.CACHE_DIR)
        self.cache_ttl_hours = cache_ttl_hours

    # -- raw retrieval with cache --------------------------------------------------
    def _load(self, tickers: list[str], refresh: bool,
              report: DataQualityReport) -> dict[str, tuple[pd.Series, SeriesProvenance]]:
        result: dict[str, tuple[pd.Series, SeriesProvenance]] = {}
        to_fetch: list[str] = []
        for t in tickers:
            entry = None if refresh else self.cache.get_fresh(t, self.cache_ttl_hours)
            if entry is not None:
                result[t] = (entry.series, self._prov(t, entry, from_cache=True))
            else:
                to_fetch.append(t)

        if to_fetch:
            try:
                fetched = self.provider.fetch_history(to_fetch)
                source_error = None
            except DataSourceError as exc:
                fetched, source_error = {}, str(exc)

            for t in to_fetch:
                s = fetched.get(t, empty_series(t))
                if not s.empty:
                    entry = self.cache.put(t, s, self.provider.name)
                    result[t] = (entry.series, self._prov(t, entry, from_cache=False))
                    continue
                stale = self.cache.get(t)
                if stale is not None and not stale.series.empty:
                    report.add(t, Severity.WARNING, "STALE_CACHE_USED",
                               f"{t}: refresh failed; using cached data fetched "
                               f"{stale.fetched_at:%Y-%m-%d %H:%M} UTC.")
                    result[t] = (stale.series, self._prov(t, stale, from_cache=True))
                else:
                    result[t] = (empty_series(t),
                                 SeriesProvenance(t, self.provider.name,
                                                  datetime.now(timezone.utc), None, None, False))
                    if source_error:
                        report.add(t, Severity.ERROR, "DATA_SOURCE_UNAVAILABLE",
                                   f"{t}: data source unavailable ({source_error}).")

            n_failed = sum(1 for t in to_fetch if result[t][0].empty)
            if source_error is None and n_failed == len(to_fetch) and len(to_fetch) > 1:
                report.add(None, Severity.WARNING, "ALL_REQUESTS_EMPTY",
                           "No data was returned for any requested symbol. If the symbols are "
                           "valid, the data source may be rate-limiting requests — retry later.")
        return result

    @staticmethod
    def _prov(t: str, entry: CacheEntry, from_cache: bool) -> SeriesProvenance:
        s = entry.series
        return SeriesProvenance(
            ticker=t, source=entry.source, fetched_at=entry.fetched_at,
            first_date=s.index.min() if len(s) else None,
            last_date=s.index.max() if len(s) else None,
            from_cache=from_cache,
        )

    # -- public API ----------------------------------------------------------------
    def get_price_data(self, tickers: Iterable[str], start: str | pd.Timestamp | None = None,
                       end: str | pd.Timestamp | None = None, refresh: bool = False,
                       min_observations: int = config.MIN_OBSERVATIONS) -> PriceData:
        """Validated, aligned adjusted close prices for ``tickers`` within [start, end].

        Assets failing validation are excluded and listed in ``quality.errors``.
        """
        clean, report = normalize_tickers(tickers)
        if not clean:
            report.add(None, Severity.ERROR, "NO_VALID_TICKERS", "No valid ticker symbols entered.")
            return PriceData(pd.DataFrame(), {}, report)

        loaded = self._load(clean, refresh, report)
        windowed = {t: _slice(s, start, end) for t, (s, _) in loaded.items()}
        provenance = {t: p for t, (_, p) in loaded.items()}

        # A ticker already reported as unavailable must not also be reported as NO_DATA.
        unavailable = {i.ticker for i in report.errors}
        to_validate = {t: s for t, s in windowed.items() if t not in unavailable}
        report.extend(validate_price_series(to_validate, min_observations=min_observations))

        excluded = set(report.excluded_tickers())
        keep = {t: windowed[t] for t in clean if t not in excluded}
        if not keep:
            report.add(None, Severity.ERROR, "NO_USABLE_ASSETS",
                       "None of the selected assets has usable price data.")
            return PriceData(pd.DataFrame(), provenance, report)

        prices = align_prices(keep, report)
        if 0 < len(prices) < min_observations:
            report.add(None, Severity.ERROR, "INSUFFICIENT_COMMON_HISTORY",
                       f"The assets share only {len(prices)} common trading days "
                       f"(minimum {min_observations}). Remove the asset with the shortest "
                       f"history or shorten the analysis window.")
        return PriceData(prices, provenance, report)

    def get_series(self, ticker: str, start=None, end=None,
                   refresh: bool = False) -> tuple[pd.Series, SeriesProvenance, DataQualityReport]:
        """A single reference series (e.g. ^VIX, ^IRX) with no minimum-history rule."""
        report = DataQualityReport()
        loaded = self._load([ticker], refresh, report)
        s, prov = loaded[ticker]
        s = _slice(s, start, end)
        if s.empty and not report.errors:
            report.add(ticker, Severity.ERROR, "NO_DATA", f"{ticker}: no data returned.")
        return s, prov, report

    def get_option_chain(self, ticker: str, max_days: int = 400) -> OptionChainRaw | None:
        """Current option-chain snapshot (not cached: quotes are only meaningful when fresh)."""
        if not self.provider.supports_options:
            return None
        return self.provider.fetch_option_chain(ticker, max_days)

    def latest_risk_free_rate(self, refresh: bool = False) -> RiskFreeRate | None:
        """Latest 13-week T-bill yield (^IRX, quoted as a percent discount yield),
        converted to a bond-equivalent annual rate. None if unavailable."""
        s, prov, _ = self.get_series(config.RISK_FREE_TICKER, refresh=refresh)
        if s.empty:
            return None
        discount = float(s.iloc[-1]) / 100.0
        return RiskFreeRate(
            value=tbill_discount_to_bond_equivalent(discount),
            observation_date=s.index[-1],
            source=f"{config.RISK_FREE_TICKER} via {prov.source}",
            note=f"13-week T-bill discount yield {discount:.3%} converted to bond-equivalent "
                 f"yield; treated as constant over the analysis window.",
        )


def empty_series(name: str) -> pd.Series:
    """An empty price series with a DatetimeIndex (so date filtering still works)."""
    return pd.Series(dtype=float, name=name, index=pd.DatetimeIndex([], name="Date"))


def _slice(s: pd.Series, start, end) -> pd.Series:
    if s.empty:
        return s
    if start is not None:
        s = s[s.index >= pd.Timestamp(start)]
    if end is not None:
        s = s[s.index <= pd.Timestamp(end)]
    return s
