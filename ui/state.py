"""Streamlit state and caching around app_core.pipeline.

Results are cached per (settings, calendar day): changing the portfolio or a new trading day
recomputes; navigating between pages does not. The two slow stages — volatility forecasts
(LSTM training, ~15 s per series) and option chains (~3 s per asset) — are stored in
server-side stores so they run once and are then shared by every page.
"""

from __future__ import annotations

from datetime import date

import streamlit as st

from app_core import pipeline as pl
from core.types import RebalanceFrequency
from data.market_data import MarketDataService
from portfolio.optimization import Constraints


@st.cache_resource(show_spinner=False)
def service() -> MarketDataService:
    return MarketDataService()


def day() -> str:
    return date.today().isoformat()


# --- portfolio settings, mirrored in the URL ----------------------------------------------------
# The URL carries the portfolio so a reconnect to a fresh server instance (common on serverless
# hosting) restores it, and so a portfolio can be shared as a link.

def _to_query(s: pl.AnalysisSettings) -> dict[str, str]:
    return {"t": ",".join(s.tickers), "w": ",".join(f"{w:.6f}" for w in s.weights), "b": s.benchmark,
            "h": str(s.horizon_days), "rb": s.rebalance.value, "y": str(s.history_years),
            "rf": "auto" if s.risk_free_rate is None else f"{s.risk_free_rate:.6f}", "v": f"{s.initial_value:.2f}"}


def _from_query(q) -> pl.AnalysisSettings | None:
    try:
        tickers = tuple(x for x in q["t"].split(",") if x)
        weights = tuple(float(x) for x in q["w"].split(","))
        if not tickers or len(tickers) != len(weights) or len(tickers) > 100:
            return None
        total = sum(weights)
        if total <= 0 or any(w < 0 for w in weights):
            return None
        return pl.AnalysisSettings(
            tickers=tickers, weights=tuple(w / total for w in weights), benchmark=q.get("b", "SPY"),
            horizon_days=int(q.get("h", 21)), rebalance=RebalanceFrequency(q.get("rb", "monthly")),
            history_years=min(max(int(q.get("y", 10)), 3), 20),
            risk_free_rate=None if q.get("rf", "auto") == "auto" else float(q["rf"]),
            initial_value=float(q.get("v", 100_000)))
    except (KeyError, ValueError):
        return None


def settings() -> pl.AnalysisSettings:
    if "settings" not in st.session_state:
        st.session_state["settings"] = _from_query(st.query_params) or pl.AnalysisSettings()
    s = st.session_state["settings"]
    if s != pl.AnalysisSettings() or "t" in st.query_params:
        st.query_params.update(_to_query(s))
    return s


# --- long jobs that survive a dropped connection --------------------------------------------------
# On serverless hosting the browser's connection can be cut during a multi-minute run; the page then
# reconnects as a new session and the run's result is lost. A job marker in the URL (tied to the
# portfolio) lets the reconnected page continue the job automatically instead of waiting for a
# second click. Finished assets are cached, so the continuation resumes where the run stopped.

def _portfolio_key() -> str:
    import hashlib
    return hashlib.sha1(repr(settings()).encode()).hexdigest()[:10]


def mark_pending(job: str, extra: str = "") -> None:
    st.query_params[job] = _portfolio_key() + extra


def pending(job: str) -> str | None:
    """The job's extra data if it was started for the current portfolio and not finished, else None."""
    v = st.query_params.get(job)
    key = _portfolio_key()
    return v[len(key):] if v and v.startswith(key) else None


def clear_pending(job: str) -> None:
    if job in st.query_params:
        del st.query_params[job]


def set_settings(s: pl.AnalysisSettings) -> None:
    st.session_state["settings"] = s
    st.query_params.update(_to_query(s))


# --- security search -------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=24 * 3600, max_entries=500)
def search(query: str):
    try:
        return service().search(query, 10)
    except Exception:
        return []


@st.cache_data(show_spinner=False, ttl=24 * 3600, max_entries=2000)
def lookup(ticker: str):
    try:
        return service().lookup(ticker)
    except Exception:
        return None


# --- cached stages -------------------------------------------------------------------------

@st.cache_resource(show_spinner=False, max_entries=6)
def _market(s: pl.AnalysisSettings, d: str) -> pl.MarketBundle:
    return pl.load_market_data(s, service())


def market() -> pl.MarketBundle:
    try:
        with st.spinner("Loading market data…"):
            return _market(settings(), day())
    except pl.AnalysisError as exc:
        st.error(f"**Analysis cannot run:** {exc}")
        st.page_link("views/builder.py", label="Edit the portfolio", icon=":material/edit:")
        st.stop()


@st.cache_resource(show_spinner=False, max_entries=6)
def _portfolio(s: pl.AnalysisSettings, d: str) -> pl.PortfolioAnalysis:
    return pl.analyze_portfolio(_market(s, d))


def portfolio() -> pl.PortfolioAnalysis:
    market()
    with st.spinner("Computing portfolio analytics…"):
        return _portfolio(settings(), day())


@st.cache_resource(show_spinner=False, max_entries=12)
def _optimization(s, d, min_w, max_w, objective, target) -> pl.OptimizationAnalysis:
    return pl.run_optimization(_market(s, d), Constraints(min_w, max_w), objective, target)


def optimization(min_w: float, max_w: float, objective: str | None, target: float | None):
    market()
    with st.spinner("Optimising…"):
        return _optimization(settings(), day(), min_w, max_w, objective, target)


@st.cache_resource(show_spinner=False, max_entries=6)
def _backtests(s, d) -> pl.BacktestBundle:
    return pl.run_backtests(_market(s, d))


def backtests() -> pl.BacktestBundle:
    market()
    with st.spinner("Running walk-forward backtests…"):
        return _backtests(settings(), day())


# --- slow stages (explicit stores) ---------------------------------------------------------------

MAX_STORED = 6  # portfolios whose slow results are kept in server memory


def _remember(store: dict, key, value) -> None:
    """Insert into a bounded store, evicting the oldest entries."""
    store.pop(key, None)
    store[key] = value
    while len(store) > MAX_STORED:
        store.pop(next(iter(store)))


@st.cache_resource(show_spinner=False)
def _forecast_store() -> dict:
    return {}


@st.cache_resource(show_spinner=False, max_entries=60)
def _asset_study(ticker: str, horizon: int, history_years: int, d: str):
    return pl.asset_volatility_study(service(), ticker, horizon, history_years)


@st.cache_resource(show_spinner=False, max_entries=MAX_STORED)
def _portfolio_study(s: pl.AnalysisSettings, d: str):
    from core import conventions as cv
    from forecasting.study import VolStudyConfig, run_volatility_study
    return run_volatility_study(cv.log_returns(_portfolio(s, d).path.values), "Portfolio",
                                VolStudyConfig(horizon=s.horizon_days))


def forecasts() -> pl.ForecastBundle | None:
    return _forecast_store().get((settings(), day()))


def run_forecasts() -> pl.ForecastBundle:
    """Each finished series is cached on its own, so an interrupted run (e.g. a dropped
    connection on serverless hosting) resumes from where it stopped."""
    mb, pa = market(), portfolio()
    s, d = settings(), day()
    bar = st.progress(0.0, text="Starting volatility forecasts…")
    fb = pl.run_forecasts(mb, pa, service(), lambda p, m: bar.progress(min(p, 1.0), text=m),
                          asset_study=lambda t: _asset_study(t, s.horizon_days, s.forecast_history_years, d),
                          portfolio_study=lambda: _portfolio_study(s, d))
    bar.empty()
    _remember(_forecast_store(), (s, d), fb)
    return fb


@st.cache_resource(show_spinner=False)
def _implied_store() -> dict:
    return {}


def implied(fetch: bool = True) -> pl.ImpliedBundle | None:
    """Option chains are fetched once per portfolio and day. ``fetch=False`` only returns an
    existing result (used by the dashboard so it never blocks on option downloads)."""
    key = (settings(), day())
    store = _implied_store()
    if key not in store:
        if not fetch:
            return None
        mb = market()
        bar = st.progress(0.0, text="Downloading option chains…")
        _remember(store, key, pl.run_implied(mb, service(), None, lambda p, m: bar.progress(min(p, 1.0), text=m)))
        bar.empty()
    ib = store[key]
    fb = forecasts()
    if fb is not None and not any(d.estimates.get("GARCH") for d in ib.disagreement):
        ib.disagreement = pl.compute_disagreement(market(), ib.estimates, fb)
    return ib


def forward_risk(source: str, lam: float) -> pl.ForwardRisk:
    return pl.forward_risk(market(), portfolio(), source, forecasts(),
                           implied(fetch=False) if source == "Implied" else None, lam)
