import pandas as pd
import streamlit as st

import config

from app_core.pipeline import AnalysisSettings
from core.types import RebalanceFrequency, RiskBasis
from data.data_validation import normalize_tickers
from portfolio.portfolio import PortfolioSpec, current_allocation
from ui import components as ui
from ui import state
from visualization import charts

ui.header("builder", "Define the portfolio. Every other section analyses this portfolio.", dim="Portfolio")
s = state.settings()

# --- draft positions (kept across reruns so searches and edits do not overwrite each other) ----------
if "draft_rows" not in st.session_state or st.session_state.get("draft_for") != s:
    st.session_state["draft_rows"] = pd.DataFrame({
        "Ticker": list(s.tickers),
        "Weight (%)": [round(w * 100, 2) for w in s.weights],
        "Amount ($)": [round(w * s.initial_value, 2) for w in s.weights]})
    st.session_state["draft_for"] = s
    st.session_state["draft_version"] = st.session_state.get("draft_version", 0) + 1


def add_ticker(symbol: str) -> None:
    rows = st.session_state["draft_current"]
    if symbol.upper() in set(rows["Ticker"].fillna("").str.upper()):
        st.session_state["search_msg"] = f"{symbol} is already in the portfolio."
        return
    st.session_state["draft_rows"] = pd.concat(
        [rows, pd.DataFrame({"Ticker": [symbol], "Weight (%)": [0.0], "Amount ($)": [0.0]})], ignore_index=True)
    st.session_state["draft_version"] += 1
    st.session_state["search_msg"] = f"Added {symbol}. Set its weight or amount below."


ui.section("assets", "Find")
c1, c2 = st.columns([2, 3])
query = c1.text_input("Search by company, fund or ticker", placeholder="e.g. Toyota, Vanguard 500, bitcoin, VOD.L",
                      key="search_query")
matches = state.search(query) if query.strip() else []
if query.strip() and not matches:
    c2.caption("No matching stocks, ETFs, funds, indices or crypto found.")
if matches:
    choice = c2.selectbox(f"{len(matches)} match(es)", matches, format_func=lambda m: m.label, key="search_choice")
    if c2.button("Add to portfolio", icon=":material/add:", key="search_add"):
        add_ticker(choice.symbol)
        st.rerun()
if st.session_state.get("search_msg"):
    st.caption(st.session_state.pop("search_msg"))

ui.section("positions", "Portfolio")
mode = st.radio("Specify positions by", ["Weights (%)", "Investment amounts ($)"], horizontal=True, key="pos_mode")
table = st.data_editor(st.session_state["draft_rows"], num_rows="dynamic", width="stretch", hide_index=True,
                       key=f"draft_editor_{st.session_state['draft_version']}",
                       column_config={"Ticker": st.column_config.TextColumn(required=True),
                                      "Weight (%)": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, format="%.2f"),
                                      "Amount ($)": st.column_config.NumberColumn(min_value=0.0, format="dollar")})
st.session_state["draft_current"] = table

ROUNDING_TOLERANCE_PCT = 0.1  # totals this close to 100% are treated as rounding and rescaled exactly
col_total = "Weight (%)" if mode.startswith("Weights") else "Amount ($)"
total = float(table[col_total].fillna(0).sum())
if mode.startswith("Weights"):
    gap = abs(total - 100)
    if gap < 1e-9:
        st.caption("Weights sum to 100%.")
    elif gap <= ROUNDING_TOLERANCE_PCT:
        st.caption(f"Weights sum to {total:.2f}% (rounding); they will be rescaled to exactly 100% when applied.")
    else:
        st.warning(f"Weights sum to {total:.2f}% — they must sum to 100% (±{ROUNDING_TOLERANCE_PCT}) before applying.")
else:
    st.caption(f"Total invested: ${total:,.2f}. Weights are the amounts divided by the total.")

entered = [t for t in table["Ticker"].fillna("").str.strip().str.upper() if t]
if entered:
    checks = []
    for t in dict.fromkeys(entered):
        m = state.lookup(t)
        checks.append(f"✓ **{t}** — {m.name} · {m.exchange}" if m else
                      f"⚠ **{t}** — not found by search; it will still be checked against price data when applied")
    with st.expander(f"Ticker check ({sum(c.startswith('✓') for c in checks)} of {len(checks)} recognised)",
                     expanded=any(c.startswith("⚠") for c in checks)):
        for c in checks:
            st.markdown(c)

ui.section("settings", "Analysis")
with st.form("portfolio"):
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
            w = t["Weight (%)"].fillna(0).astype(float)
            if (w < 0).any():
                raise ValueError("Weights cannot be negative (short positions are not supported).")
            if abs(w.sum() - 100) > ROUNDING_TOLERANCE_PCT:
                raise ValueError(f"Weights sum to {w.sum():.2f}%; they must sum to 100% "
                                 f"(within ±{ROUNDING_TOLERANCE_PCT} percentage points of rounding).")
            spec = PortfolioSpec.from_weights(dict(zip(t.Ticker, w / w.sum())), initial_value=initial)
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

with st.expander("Data sources, currencies and observation dates"):
    st.caption(f"All prices are in {config.BASE_CURRENCY}. Assets quoted in other currencies are converted with daily "
               f"exchange rates, unhedged, so their returns include currency moves.")
    prov = pd.DataFrame([{"Ticker": t, "Source": p.source, "Fetched (UTC)": f"{p.fetched_at:%Y-%m-%d %H:%M}",
                          "First observation": p.first_date.date() if p.first_date is not None else None,
                          "Last observation": p.last_date.date() if p.last_date is not None else None,
                          "Quote currency": p.currency, "Converted with": p.fx_ticker or "—",
                          "From cache": p.from_cache} for t, p in mb.price_data.provenance.items()])
    st.dataframe(prov, hide_index=True, width="stretch")
