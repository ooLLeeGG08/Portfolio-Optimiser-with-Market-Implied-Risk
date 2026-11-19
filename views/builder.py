import pandas as pd
import streamlit as st

from app_core.pipeline import AnalysisSettings
from core.types import RebalanceFrequency, RiskBasis
from data.data_validation import normalize_tickers
from portfolio.portfolio import PortfolioSpec, current_allocation
from ui import components as ui
from ui import state
from visualization import charts

ui.header("builder", "Define the portfolio. Every other section analyses this portfolio.", dim="Portfolio")
s = state.settings()

with st.form("portfolio"):
    mode = st.radio("Specify positions by", ["Weights (%)", "Investment amounts ($)"], horizontal=True)
    default = pd.DataFrame({"Ticker": list(s.tickers),
                            "Weight (%)": [round(w * 100, 2) for w in s.weights],
                            "Amount ($)": [round(w * s.initial_value, 2) for w in s.weights]})
    table = st.data_editor(default, num_rows="dynamic", width="stretch", hide_index=True,
                           column_config={"Ticker": st.column_config.TextColumn(required=True),
                                          "Weight (%)": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, format="%.2f"),
                                          "Amount ($)": st.column_config.NumberColumn(min_value=0.0, format="%.2f")})
    c = st.columns(4)
    benchmark = c[0].text_input("Benchmark", s.benchmark)
    horizon = c[1].selectbox("Investment / risk horizon (trading days)", [5, 10, 21, 63],
                             index=[5, 10, 21, 63].index(s.horizon_days) if s.horizon_days in (5, 10, 21, 63) else 2)
    rebalance = c[2].selectbox("Rebalancing", list(RebalanceFrequency),
                               index=list(RebalanceFrequency).index(s.rebalance), format_func=lambda r: r.value.title())
    years = c[3].slider("History (years)", 3, 20, s.history_years)
    c = st.columns(3)
    auto_rf = c[0].toggle("Use latest 13-week T-bill yield as risk-free rate", s.risk_free_rate is None)
    rf_pct = c[1].number_input("Manual risk-free rate (% p.a.)", -5.0, 25.0,
                               value=float((s.risk_free_rate or 0.04) * 100), step=0.25)
    initial = c[2].number_input("Portfolio value for weights mode ($)", 1000.0, 1e10, float(s.initial_value), step=1000.0)
    submitted = st.form_submit_button("Apply portfolio", type="primary")

if submitted:
    try:
        tickers, rep = normalize_tickers(table["Ticker"].fillna(""))
        for i in rep.issues:
            (st.error if i.severity.value == "error" else st.warning)(i.message)
        if rep.errors:
            st.stop()
        t = table.assign(Ticker=table["Ticker"].fillna("").str.strip().str.upper())
        t = t[t.Ticker.isin(tickers)].drop_duplicates("Ticker")
        if mode.startswith("Weights"):
            spec = PortfolioSpec.from_weights(dict(zip(t.Ticker, t["Weight (%)"].fillna(0) / 100)),
                                              initial_value=initial)
        else:
            spec = PortfolioSpec.from_amounts(dict(zip(t.Ticker, t["Amount ($)"].fillna(0))))
        bench, brep = normalize_tickers([benchmark])
        if not bench:
            st.error("Enter a valid benchmark ticker.")
            st.stop()
        state.set_settings(AnalysisSettings(
            tickers=tuple(spec.tickers), weights=tuple(float(w) for w in spec.weights),
            benchmark=bench[0], horizon_days=int(horizon),
            risk_free_rate=None if auto_rf else rf_pct / 100, rebalance=rebalance,
            initial_value=float(spec.initial_value), history_years=int(years)))
        st.success("Portfolio updated.")
        st.rerun()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

mb = state.market()
pa = state.portfolio()
if mb.exclusion_message:
    st.warning(mb.exclusion_message)
ui.quality_panel(mb.quality, expanded=bool(mb.quality.warnings))

ui.section("composition", "Portfolio")
alloc = current_allocation(pa.path)
alloc.insert(0, "target_weight", mb.spec.weights.reindex(alloc.index))
left, right = st.columns([1, 1])
with left:
    st.dataframe(alloc.rename(columns={"target_weight": "Target weight", "weight": "Current weight (drifted)",
                                       "market_value": "Market value ($)"}),
                 width="stretch",
                 column_config={"Target weight": st.column_config.NumberColumn(format="percent"),
                                "Current weight (drifted)": st.column_config.NumberColumn(format="percent"),
                                "Market value ($)": st.column_config.NumberColumn(format="dollar")})
    ui.basis_line(RiskBasis.OBSERVED, note=f"Simulated from {ui.money(mb.spec.initial_value)} on "
                  f"{pa.performance.start:%d %b %Y}, {mb.spec.rebalance.value} rebalancing, no costs.")
with right:
    ui.chart(charts.allocation_donut(pa.current_weights))

ui.section("performance", "Historical")
perf = pd.DataFrame([
    {"Series": name, "Start": x.start.date(), "End": x.end.date(), "Total return": x.total_return,
     "CAGR": x.cagr, "Arithmetic annual return": x.annualized_mean_return,
     "Annualised volatility": x.annualized_volatility}
    for name, x in (("Portfolio", pa.performance), (mb.settings.benchmark, pa.benchmark_performance))
]).set_index("Series")
st.dataframe(perf, width="stretch", column_config=ui.pct_columns(
    perf, ["Total return", "CAGR", "Arithmetic annual return", "Annualised volatility"]))
ui.chart(charts.performance_chart(pa.path.values, pa.benchmark_values, mb.settings.benchmark))

ui.section("correlations", "Returns and")
rets = mb.asset_returns[mb.spec.tickers]
stats = pd.DataFrame({"Mean daily return": rets.mean(), "Daily volatility": rets.std(),
                      "Mean monthly return": pa.monthly_returns.mean().reindex(rets.columns),
                      "Observations": rets.count()})
monthly_assets = rets.add(1).groupby(rets.index.to_period("M")).prod().sub(1)
stats["Mean monthly return"] = monthly_assets.iloc[:-1].mean()
stats["Monthly volatility"] = monthly_assets.iloc[:-1].std()
st.dataframe(stats, width="stretch", column_config=ui.pct_columns(stats, ["Mean daily return", "Daily volatility",
                                                                          "Mean monthly return", "Monthly volatility"]))
st.caption("Simple returns. Monthly figures exclude the current, incomplete month.")
ui.chart(charts.correlation_heatmap(pa.correlation))

with st.expander("Data sources and observation dates"):
    prov = pd.DataFrame([{"Ticker": t, "Source": p.source, "Fetched (UTC)": f"{p.fetched_at:%Y-%m-%d %H:%M}",
                          "First observation": p.first_date.date() if p.first_date is not None else None,
                          "Last observation": p.last_date.date() if p.last_date is not None else None,
                          "From cache": p.from_cache} for t, p in mb.price_data.provenance.items()])
    st.dataframe(prov, hide_index=True, width="stretch")
