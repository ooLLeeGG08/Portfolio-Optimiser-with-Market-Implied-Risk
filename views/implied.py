import pandas as pd
import streamlit as st

from core.types import IVSource, RiskBasis
from implied_volatility.comparison import VDI_DISCLAIMER, disagreement_table, index_implied_vs_forecast
from implied_volatility.options_data import trading_to_calendar_days
from implied_volatility.vix import VIX_DESCRIPTION
from ui import components as ui
from ui import state
from visualization import charts

mb = state.market()
h = mb.settings.horizon_days
ui.header("volatility", dim="Market-implied", subtitle=
          "What option prices imply about future volatility — kept separate from historical and forecast estimates.")
st.info("Implied volatility is **risk-neutral**: it includes a premium investors pay for protection against "
        "volatility, so it usually exceeds the volatility that is subsequently realised. A gap between implied and "
        "forecast volatility is not, by itself, evidence that options are mispriced.", icon=":material/info:")

ib = state.implied()

# --- VIX ---------------------------------------------------------------------------------------------
ui.section("VIX", "S&P 500 implied volatility ·")
st.caption(VIX_DESCRIPTION)
if ib.vix_summary is None:
    st.warning("VIX data unavailable.")
else:
    v = ib.vix_summary
    c = st.columns(5)
    with c[0]:
        ui.kpi("VIX (latest)", ui.pct(v.latest), RiskBasis.MARKET_IMPLIED, help=f"Observation date {v.latest_date:%d %b %Y}.")
    with c[1]:
        ui.kpi("Change, 1 day", f"{v.change_1d * 100:+.1f} pts", delta=f"21 days: {v.change_21d * 100:+.1f} pts",
               help="Change in VIX in volatility points.")
    with c[2]:
        ui.kpi(f"Percentile since {v.history_start:%Y}", f"{v.percentile_full:.0%}",
               help=f"Share of days since {v.history_start:%d %b %Y} with VIX at or below today's level.")
    with c[3]:
        ui.kpi("Percentile, last 5 years", f"{v.percentile_5y:.0%}")
    with c[4]:
        ui.kpi("252-day average", ui.pct(v.mean_252d), RiskBasis.MARKET_IMPLIED)
    yrs = st.segmented_control("Period", [1, 5, 10, 30], default=5, format_func=lambda y: f"{y}Y", key="vix_years") or 5
    ui.chart(charts.vix_chart(ib.vix_frame, yrs))
    if ib.vrp is not None:
        vr = ib.vrp
        ui.chart(charts.vrp_chart(vr.frame, yrs))
        st.caption(f"Since {vr.frame.index[0]:%Y}: VIX exceeded the S&P 500 volatility realised over the next "
                   f"{vr.horizon_days} trading days on {vr.share_vix_above:.0%} of days, by {vr.mean_spread * 100:.1f} "
                   f"volatility points on average (median {vr.median_spread * 100:.1f}). This ex-post comparison "
                   f"illustrates the variance risk premium.")

# --- option chains -----------------------------------------------------------------------------------
ui.section("by asset", "Option-implied volatility")
ui.basis_line(RiskBasis.MARKET_IMPLIED, RiskBasis.IMPLIED_PROXY,
              note=f" Annualised at-the-money implied vol interpolated to {trading_to_calendar_days(h):.0f} calendar "
                   f"days (= {h} trading days). Proxies are estimates, not market data.")
rows = []
for t, e in ib.estimates.items():
    rows.append({"Ticker": t, "Implied volatility": e.value,
                 "Source": {IVSource.OPTIONS_CHAIN: "Listed options (market data)",
                            IVSource.PROXY: "PROXY ESTIMATE — not market data",
                            IVSource.UNAVAILABLE: "Unavailable"}[e.source],
                 "Reliable": "yes" if e.reliable else "limited",
                 "Valid quotes": float(ib.chains[t].quotes.valid.sum()) if t in ib.chains else None})
iv_tab = pd.DataFrame(rows).set_index("Ticker")
iv_tab[["Implied volatility", "Valid quotes"]] = iv_tab[["Implied volatility", "Valid quotes"]].astype(float)
ui.gap_table(iv_tab, {"Implied volatility": "{:.2%}", "Valid quotes": "{:,.0f}"}, width="stretch")
# The explanations are full sentences; a grid cell would cut them off, so they are listed in full here.
with st.expander("How each estimate was made"):
    for t, e in ib.estimates.items():
        st.markdown(f"- **{t}**: {e.detail}")
for n in ib.notes:
    st.caption(n)

if ib.chains:
    t = st.selectbox("Inspect option chain", list(ib.chains))
    ch = ib.chains[t]
    left, right = st.columns(2)
    with left:
        ui.chart(charts.term_structure_chart(ch.term_structure, t, trading_to_calendar_days(h)))
    with right:
        valid_exp = [e for e in ch.expiries.index if ch.smile(e).shape[0] >= 5]
        if valid_exp:
            default = min(valid_exp, key=lambda e: abs(ch.expiries.loc[e, "T_days"] - 30))
            exp = st.selectbox("Expiry", valid_exp, index=valid_exp.index(default),
                               format_func=lambda e: f"{pd.Timestamp(e):%d %b %Y} ({ch.expiries.loc[e, 'T_days']:.0f} days)")
            ui.chart(charts.smile_chart(ch.smile(exp), t, exp, ch.spot))
        else:
            st.caption("Too few valid quotes in any expiry to show a smile.")
    excl = ch.exclusions.rename("Quotes excluded").to_frame()
    excl.index.name = "Reason"
    c1, c2 = st.columns([1, 1])
    c1.dataframe(excl, width="stretch")
    c2.caption(f"{int(ch.quotes.valid.sum()):,} of {len(ch.quotes):,} quotes passed the filters. Excluded quotes "
               f"are not used and nothing is filled in.")
    for n in ch.notes:
        c2.caption(n)
    fwd = ch.expiries[["T_days", "forward", "forward_method", "atm_iv", "n_valid"]].rename(
        columns={"T_days": "Days", "forward": "Forward", "forward_method": "Forward from", "atm_iv": "ATM IV",
                 "n_valid": "Valid OTM quotes"})
    with st.expander("Per-expiry detail"):
        st.dataframe(fwd, width="stretch", column_config={"ATM IV": st.column_config.NumberColumn(format="percent"),
                                                          "Days": st.column_config.NumberColumn(format="%.0f"),
                                                          "Forward": st.column_config.NumberColumn(format="%.2f")})

# --- implied vs forecast -------------------------------------------------------------------------------
ui.section("vs historical vs forecast", "Implied")
fb = state.forecasts()
if fb is None:
    st.caption("GARCH and LSTM forecasts not computed yet — comparing historical and implied only.")
    st.page_link("views/forecasting.py", label="Run volatility forecasts", icon=":material/play_arrow:")
else:
    ui.precomputed_note(fb)
cmp = pd.DataFrame(index=list(ib.estimates))
for d in ib.disagreement:
    for lab, val in d.estimates.items():
        lab2 = "Historical 60d" if lab == "Historical" else ("GARCH(1,1)" if lab == "GARCH" else lab)
        cmp.loc[d.ticker, lab2] = val
        cmp.loc[d.ticker, f"{lab2} basis"] = d.bases[lab]
if fb is not None:
    for t in cmp.index:
        if t in fb.studies:
            L = fb.studies[t].latest
            cmp.loc[t, "GARCH(1,1) lo"], cmp.loc[t, "GARCH(1,1) hi"] = L.loc["GARCH(1,1)", "lower"], L.loc["GARCH(1,1)", "upper"]
            cmp.loc[t, "LSTM lo"], cmp.loc[t, "LSTM hi"] = L.loc["LSTM", "lower"], L.loc["LSTM", "upper"]
ui.chart(charts.volatility_comparison_chart(cmp, h))
st.caption("Marker colour shows the basis of each estimate: observed, forecast, market-implied or proxy (dashed in tables).")

if fb is not None and "SPY" in fb.studies and ib.vix_frame is not None:
    st.subheader("S&P 500: VIX vs model forecasts in the out-of-sample window")
    df = index_implied_vs_forecast(ib.vix_frame["vix"], fb.studies["SPY"].forecasts)
    ui.chart(charts.rolling_vol_chart(df, "VIX, SPY forecasts and subsequently realised volatility"))

# --- disagreement ---------------------------------------------------------------------------------
ui.section("disagreement indicator", "Volatility")
st.warning(VDI_DISCLAIMER, icon=":material/info:")
if fb is not None and fb.precomputed:
    st.caption(f"GARCH and LSTM columns: {ui.precomputed_text(fb)} Option-implied and historical columns are live.")
tab = disagreement_table(ib.disagreement)
for i, c in enumerate(("GARCH", "LSTM"), start=1):  # always shown, so a missing model is visible
    if c not in tab:
        tab.insert(i, c, float("nan"))
ratio_cols = ("Dispersion (all)", "Dispersion (models)", "IV / mean forecast")
num_cols = [c for c in tab.columns if c != "Disagreement"]
tab[num_cols] = tab[num_cols].apply(pd.to_numeric, errors="coerce")
formats = {c: "{:.3f}" if c in ratio_cols else ("{:.0%}" if c == "Model dispersion percentile" else "{:.2%}")
           for c in num_cols}
# Why a cell is empty: forecasts not run for this portfolio, or an implied proxy (no ratio by design).
reasons = pd.DataFrame(index=tab.index, columns=tab.columns, dtype=object)
if fb is None:
    reasons[["GARCH", "LSTM", "Dispersion (models)", "IV / mean forecast", "Model dispersion percentile"]] = ui.NOT_RUN
else:
    for tk, e in ib.estimates.items():
        if tk in reasons.index and e.source is IVSource.PROXY:
            reasons.loc[tk, "IV / mean forecast"] = "proxy: not shown"
ui.gap_table(tab, formats, reasons, width="stretch")
if fb is None:
    st.caption(f"“{ui.NOT_RUN}”: these columns need the GARCH and LSTM forecasts, which have not been run for this "
               f"portfolio. “n/a”: no value applies.")
ui.chart(charts.disagreement_chart(ib.disagreement))
ui.notes([
    "Dispersion = standard deviation of the logs of the available estimates; 0.10 ≈ estimates spread ±10% around "
    "their geometric mean. Bands (fixed in advance): Low < 0.10 ≤ Moderate < 0.25 ≤ High.",
    "IV / mean forecast is shown only when actual option-implied volatility is available, never for proxies.",
    "Model dispersion percentile places today's historical/GARCH/LSTM disagreement within its own history over the "
    "out-of-sample test window. Implied volatility has no single-stock history in this data source.",
    *[n for d in ib.disagreement for n in d.notes][:2],
], expanded=False)
