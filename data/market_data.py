"""Market-data access.

``MarketDataProvider`` is the vendor interface; ``YFinanceProvider`` implements it.
``MarketDataService`` adds caching, validation and alignment and returns ``PriceData``
with provenance (source, fetch time, first/last observation) for every series.

No missing values are ever fabricated: failures surface in the DataQualityReport.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

import pandas as pd

import config
from core.types import DataQualityReport, PriceData, SeriesProvenance, Severity
from data.cache import CacheEntry, PriceCache
from data.currency import convert_series, fx_candidates, normalize_currency
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


@dataclass(frozen=True)
class SecurityMatch:
    """One result of a security search."""

    symbol: str
    name: str
    exchange: str
    quote_type: str  # EQUITY, ETF, MUTUALFUND, CRYPTOCURRENCY, INDEX
    sector: str | None = None

    @property
    def label(self) -> str:
        kind = {"EQUITY": "Stock", "ETF": "ETF", "MUTUALFUND": "Mutual fund",
                "CRYPTOCURRENCY": "Crypto", "INDEX": "Index"}.get(self.quote_type, self.quote_type.title())
        return f"{self.symbol} — {self.name} · {self.exchange} · {kind}"


# Price series that behave like investable assets. Futures (contract rolls), options and FX
# pairs are excluded from search results.
SEARCHABLE_TYPES = {"EQUITY", "ETF", "MUTUALFUND", "CRYPTOCURRENCY", "INDEX"}


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

    def fetch_currency(self, ticker: str) -> str | None:
        """Currency code the provider quotes the ticker's prices in (e.g. 'USD', 'JPY',
        'GBp'), or None if unknown."""
        return None

    def search(self, query: str, max_results: int = 10) -> list[SecurityMatch]:
        """Securities matching a name or symbol. Providers without search return []."""
        return []


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


    def search(self, query: str, max_results: int = 10) -> list[SecurityMatch]:
        try:
            res = self._yf.Search(query, max_results=max_results, news_count=0,
                                  enable_fuzzy_query=False).quotes
        except Exception as exc:
            raise DataSourceError(f"Search failed: {exc}") from exc
        out = []
        for r in res:
            sym, qt = r.get("symbol"), (r.get("quoteType") or "").upper()
            if not sym or qt not in SEARCHABLE_TYPES:
                continue
            out.append(SecurityMatch(sym, (r.get("longname") or r.get("shortname") or sym).strip(),
                                     r.get("exchDisp") or r.get("exchange") or "", qt, r.get("sectorDisp")))
        return out

    def fetch_currency(self, ticker: str) -> str | None:
        try:
            cur = self._yf.Ticker(ticker).fast_info["currency"]
        except Exception:
            return None
        return str(cur) if cur else None

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
        self._currency_file = self.cache.directory / "currencies.json"
        self._search_cache: dict[tuple[str, int], list[SecurityMatch]] = {}

    # -- search ---------------------------------------------------------------------------
    def search(self, query: str, max_results: int = 10) -> list[SecurityMatch]:
        q = query.strip()
        if len(q) < 1:
            return []
        key = (q.lower(), max_results)
        if key not in self._search_cache:
            self._search_cache[key] = self.provider.search(q, max_results)
        return self._search_cache[key]

    def lookup(self, ticker: str) -> SecurityMatch | None:
        """Exact-symbol match for a ticker, or None if the search does not know it."""
        t = ticker.strip().upper()
        for m in self.search(t, 10):
            if m.symbol.upper() == t:
                return m
        return None

    # -- currencies ----------------------------------------------------------------------
    def get_currency(self, ticker: str) -> str | None:
        """Quote currency for a ticker, cached on disk (it practically never changes)."""
        try:
            known = json.loads(self._currency_file.read_text()) if self._currency_file.exists() else {}
        except (OSError, ValueError):
            known = {}
        if ticker in known:
            return known[ticker]
        cur = self.provider.fetch_currency(ticker)
        if cur:
            known[ticker] = cur
            self._currency_file.write_text(json.dumps(known, indent=2, sort_keys=True))
        return cur

    def _fx_series(self, iso: str, base: str, refresh: bool,
                   report: DataQualityReport) -> tuple[pd.Series | None, str | None]:
        """Base-currency units per unit of ``iso``, from the first available FX ticker."""
        for fx_ticker, invert in fx_candidates(iso, base):
            loaded = self._load([fx_ticker], refresh, DataQualityReport())
            s = loaded[fx_ticker][0]
            if not s.empty and (s > 0).all():
                return (1.0 / s if invert else s).rename(f"{iso}->{base}"), fx_ticker
        return None, None

    def _to_base_currency(self, loaded: dict, base: str, refresh: bool,
                          report: DataQualityReport) -> dict:
        """Convert every non-empty series in ``loaded`` into the base currency (unhedged)."""
        out = {}
        for t, (s, prov) in loaded.items():
            if s.empty:
                out[t] = (s, prov)
                continue
            cur = self.get_currency(t)
            if cur is None:
                if re.search(r"\.[A-Z]{1,4}$", t):
                    report.add(t, Severity.ERROR, "CURRENCY_UNKNOWN",
                               f"{t}: the quote currency could not be determined, so its prices cannot be "
                               f"converted to {base}; excluded.")
                    out[t] = (empty_series(t), prov)
                    continue
                report.add(t, Severity.WARNING, "CURRENCY_ASSUMED",
                           f"{t}: quote currency unknown; prices are assumed to be in {base}.")
                out[t] = (s, replace(prov, currency=f"{base} (assumed)"))
                continue
            iso, _ = normalize_currency(cur)
            fx, fx_ticker = (None, None) if iso == base else self._fx_series(iso, base, refresh, report)
            try:
                conv = convert_series(s, cur, base, fx, fx_ticker)
            except ValueError as exc:
                report.add(t, Severity.ERROR, "FX_UNAVAILABLE", f"{t}: priced in {iso}; {exc} Excluded.")
                out[t] = (empty_series(t), replace(prov, currency=cur))
                continue
            if conv.fx_ticker or conv.divisor != 1.0:
                how = []
                if conv.divisor != 1.0:
                    how.append(f"quoted in {cur} (1/{conv.divisor:g} {iso})")
                if conv.fx_ticker:
                    how.append(f"converted {iso}→{base} with daily rates from {conv.fx_ticker}"
                               + (" (inverted)" if conv.fx_ticker.startswith(base) else ""))
                msg = f"{t}: " + "; ".join(how) + " — unhedged, so returns include exchange-rate moves."
                if conv.dates_dropped:
                    msg += f" {conv.dates_dropped} date(s) without an exchange rate were dropped (not filled)."
                report.add(t, Severity.INFO, "CURRENCY_CONVERTED", msg)
            out[t] = (conv.series, replace(prov, currency=cur, fx_ticker=conv.fx_ticker))
        return out

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
                       min_observations: int = config.MIN_OBSERVATIONS,
                       base_currency: str = config.BASE_CURRENCY) -> PriceData:
        """Validated, aligned adjusted close prices for ``tickers`` within [start, end],
        converted into ``base_currency`` (unhedged).

        Assets failing validation are excluded and listed in ``quality.errors``.
        """
        clean, report = normalize_tickers(tickers)
        if not clean:
            report.add(None, Severity.ERROR, "NO_VALID_TICKERS", "No valid ticker symbols entered.")
            return PriceData(pd.DataFrame(), {}, report)

        raw = self._load(clean, refresh, report)
        # Slice to the analysis window before converting, so any dates dropped for lack of an
        # exchange rate are counted within the window the user asked for.
        sliced = {t: (_slice(s, start, end), prov) for t, (s, prov) in raw.items()}
        loaded = self._to_base_currency(sliced, base_currency, refresh, report)
        windowed = {t: s for t, (s, _) in loaded.items()}
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
