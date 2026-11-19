import pandas as pd
import streamlit as st

from core.types import IVSource, RiskBasis
from ui import components as ui
from ui import state
from visualization import charts

mb = state.market()
pa = state.portfolio()
s = mb.settings

ui.hero(
    "risk & volatility", "Portfolio",
    f"A {len(mb.spec.tickers)}-asset portfolio measured against {s.benchmark}. Prices to {mb.as_of:%d %b %Y}; "
    f"risk-free rate {ui.pct(mb.risk_free_rate, 2)} ({mb.risk_free_source}).",
    [(ui.money(pa.current_value), "portfolio value", True),
     (ui.pct(pa.risk.annualized_volatility), "volatility, observed", False),
     (ui.num(pa.risk.sharpe), "Sharpe ratio, observed", False),
     (f"{s.horizon_days}d", "risk horizon", False)])
ui.basis_legend()
if mb.exclusion_message:
    st.warning(mb.exclusion_message, icon=":material/warning:")
for e in mb.errors:
    st.warning(e, icon=":material/warning:")

# --- portfolio overview -----------------------------------------------------------------------
ui.section("overview", "Portfolio")
r, perf = pa.risk, pa.performance
c = st.columns(6)
with c[0]:
    ui.kpi("Portfolio value", ui.money(pa.current_value), RiskBasis.OBSERVED,
           help=f"Value of the simulated portfolio today, from {ui.money(mb.spec.initial_value)} invested on {perf.start:%d %b %Y}.")
with c[1]:
    ui.kpi("Assets", str(len(mb.spec.tickers)))
with c[2]:
    ui.kpi("Expected return (ann.)", ui.pct(r.annualized_return), RiskBasis.OBSERVED,
           help="Historical arithmetic mean daily return × 252. An estimate from past data, not a prediction.")
with c[3]:
    ui.kpi("Volatility (ann.)", ui.pct(r.annualized_volatility), RiskBasis.OBSERVED)
with c[4]:
    ui.kpi("Sharpe ratio", ui.num(r.sharpe), RiskBasis.OBSERVED,
           help="(Annualised return − risk-free rate) / annualised volatility.")
with c[5]:
    ui.kpi("Max drawdown", ui.pct(r.max_drawdown.depth), RiskBasis.OBSERVED,
           help=f"Peak {r.max_drawdown.peak_date:%d %b %Y} → trough {r.max_drawdown.trough_date:%d %b %Y}.")

left, right = st.columns([2, 1])
with left:
    ui.chart(charts.performance_chart(pa.path.values, pa.benchmark_values, s.benchmark))
with right:
    ui.chart(charts.allocation_donut(pa.current_weights, "Current allocation (after drift)"))

# --- risk overview ----------------------------------------------------------------------------
ui.section("overview", "Risk")
tr = {(e.method, e.confidence): e for e in pa.tail_risk}
h95, h99 = tr[("Historical", 0.95)], tr[("Historical", 0.99)]
c = st.columns(4)
with c[0]:
    ui.kpi("1-day VaR 95%", ui.pct(h95.var, 2), RiskBasis.OBSERVED,
           help=f"On 5% of days the loss is expected to exceed {ui.pct(h95.var, 2)} ({ui.money(h95.var_amount)}). Historical method.")
with c[1]:
    ui.kpi("1-day CVaR 95%", ui.pct(h95.cvar, 2), RiskBasis.OBSERVED,
           help="Average loss on the days beyond the 95% VaR.")
with c[2]:
    ui.kpi("1-day VaR 99%", ui.pct(h99.var, 2), RiskBasis.OBSERVED)
with c[3]:
    ui.kpi("1-day CVaR 99%", ui.pct(h99.cvar, 2), RiskBasis.OBSERVED)
left, right = st.columns(2)
with left:
    ui.chart(charts.risk_contribution_chart(pa.contributions.table, RiskBasis.OBSERVED))
with right:
    ui.chart(charts.drawdown_chart(pa.path.values, pa.benchmark_values, s.benchmark))

# --- volatility overview -----------------------------------------------------------------------
ui.section("overview", "Volatility")
st.caption(f"Annualised volatility of the portfolio over the next {s.horizon_days} trading days, by source.")
fb = state.forecasts()
ib = state.implied(fetch=False)
cards = []
hist60 = float(pa.rolling_vol["Portfolio 60d"].iloc[-1])
cards.append(lambda: ui.kpi("Historical (60-day)", ui.pct(hist60), RiskBasis.OBSERVED,
                            help="Trailing 60-day realised volatility."))
if fb is not None and "Portfolio" in fb.studies:
    L = fb.studies["Portfolio"].latest
    cards.append(lambda: ui.kpi("GARCH(1,1) forecast", ui.pct(L.loc["GARCH(1,1)", "point"]), RiskBasis.FORECAST,
                                help=f"10–90% interval {ui.pct(L.loc['GARCH(1,1)', 'lower'])} – {ui.pct(L.loc['GARCH(1,1)', 'upper'])}."))
    cards.append(lambda: ui.kpi("LSTM forecast", ui.pct(L.loc["LSTM", "point"]), RiskBasis.FORECAST,
                                help=f"Calibrated 10–90% interval {ui.pct(L.loc['LSTM', 'lower'])} – {ui.pct(L.loc['LSTM', 'upper'])}."))
else:
    def _todo():
        with st.container(border=True):
            st.markdown("**GARCH and LSTM forecasts**")
            st.caption("Not yet computed (about 2 minutes).")
            st.page_link("views/forecasting.py", label="Run forecasts", icon=":material/play_arrow:")
    cards.append(_todo)
if ib is not None:
    def _implied():
        try:
            fr = state.forward_risk("Implied", 0.0)
        except ValueError as exc:
            st.caption(f"Implied volatility unavailable: {exc}")
            return
        n_proxy = sum(e.source is IVSource.PROXY for e in ib.estimates.values())
        ui.kpi("Implied (portfolio)", ui.pct(fr.contributions.portfolio_volatility), fr.covariance.basis,
               help="Asset implied vols combined with historical correlations (Σ = D·R·D)"
                    + (f"; {n_proxy} asset(s) use a proxy estimate." if n_proxy else "."))
    cards.append(_implied)
else:
    def _load():
        with st.container(border=True):
            st.markdown("**Market-implied volatility**")
            if mb.vix is not None:
                st.caption(f"VIX {ui.pct(mb.vix.iloc[-1])} on {mb.vix.index[-1]:%d %b %Y} (S&P 500, not this portfolio).")
            st.page_link("views/implied.py", label="Load option data", icon=":material/download:")
    cards.append(_load)
for col, card in zip(st.columns(4), cards):
    with col:
        card()

ui.quality_panel(mb.quality)
