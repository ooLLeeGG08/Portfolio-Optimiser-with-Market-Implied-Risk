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
from data.market_data import MarketDataService
from portfolio.optimization import Constraints


@st.cache_resource(show_spinner=False)
def service() -> MarketDataService:
    return MarketDataService()


def day() -> str:
    return date.today().isoformat()


def settings() -> pl.AnalysisSettings:
    if "settings" not in st.session_state:
        st.session_state["settings"] = pl.AnalysisSettings()
    return st.session_state["settings"]


def set_settings(s: pl.AnalysisSettings) -> None:
    st.session_state["settings"] = s


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

@st.cache_resource(show_spinner=False)
def _forecast_store() -> dict:
    return {}


def forecasts() -> pl.ForecastBundle | None:
    return _forecast_store().get((settings(), day()))


def run_forecasts() -> pl.ForecastBundle:
    mb, pa = market(), portfolio()
    bar = st.progress(0.0, text="Starting volatility forecasts…")
    fb = pl.run_forecasts(mb, pa, service(), lambda p, m: bar.progress(min(p, 1.0), text=m))
    bar.empty()
    _forecast_store()[(settings(), day())] = fb
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
        store[key] = pl.run_implied(mb, service(), None, lambda p, m: bar.progress(min(p, 1.0), text=m))
        bar.empty()
    ib = store[key]
    fb = forecasts()
    if fb is not None and not any(d.estimates.get("GARCH") for d in ib.disagreement):
        ib.disagreement = pl.compute_disagreement(market(), ib.estimates, fb)
    return ib


def forward_risk(source: str, lam: float) -> pl.ForwardRisk:
    return pl.forward_risk(market(), portfolio(), source, forecasts(),
                           implied(fetch=False) if source == "Implied" else None, lam)
