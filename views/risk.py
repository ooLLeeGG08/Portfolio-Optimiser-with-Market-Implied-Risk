import pandas as pd
import streamlit as st

from app_core.pipeline import VOL_SOURCES, tail_risk_at_horizon
from core.types import RiskBasis
from risk.tail_risk import to_frame
from ui import components as ui
from ui import state
from visualization import charts

mb = state.market()
pa = state.portfolio()
r = pa.risk
ui.header("analytics", dim="Risk", subtitle= "Observed risk from history, and forward-looking risk under alternative volatility assumptions.")

# --- standard measures ----------------------------------------------------------------------------
ui.section("risk measures", "Observed")
ui.basis_line(RiskBasis.OBSERVED, note=f" Daily portfolio returns {r.start:%d %b %Y} – {r.end:%d %b %Y}.")
c = st.columns(6)
items = [("Volatility (ann.)", ui.pct(r.annualized_volatility)), ("Sharpe ratio", ui.num(r.sharpe)),
         ("Sortino ratio", ui.num(r.sortino)), (f"Beta vs {mb.settings.benchmark}", ui.num(r.beta)),
         ("Correlation with benchmark", ui.num(r.correlation_with_benchmark)),
         ("Max drawdown", ui.pct(r.max_drawdown.depth))]
for col, (lab, val) in zip(c, items):
    with col:
        ui.kpi(lab, val)
md = r.max_drawdown
st.caption(f"Maximum drawdown: peak {md.peak_date:%d %b %Y} → trough {md.trough_date:%d %b %Y} "
           f"({md.peak_to_trough_days} trading days); "
           + (f"recovered {md.recovery_date:%d %b %Y} ({md.trough_to_recovery_days} trading days later)."
              if md.recovery_date is not None else "not yet recovered."))

# --- VaR / CVaR -----------------------------------------------------------------------------------
ui.section("and expected shortfall", "Value at risk")
st.markdown("**VaR** is a loss threshold: at 95%, the loss is expected to exceed it on 5% of days. It is not a worst "
            "case. **CVaR** (Expected Shortfall) is the average loss on the days beyond that threshold. Both are "
            "shown as positive numbers representing losses.")
hz = st.segmented_control("Horizon (trading days)", [1, 5, 10, 21], default=1, key="var_h") or 1
est = pa.tail_risk if hz == 1 else tail_risk_at_horizon(mb, pa, hz)
df = to_frame(est).drop(columns=["Horizon (days)", "Basis"])
notes_by_method = df[["Method", "Confidence", "Notes"]]
df.loc[df["VaR"].isna(), "Method"] = df.loc[df["VaR"].isna(), "Method"] + " — not reported (see notes below)"
show = df.drop(columns="Notes").rename(columns={"VaR": "VaR (% of value)", "CVaR": "CVaR (% of value)"})
ui.gap_table(show, {"Confidence": "{:.0%}", "VaR (% of value)": "{:.2%}", "CVaR (% of value)": "{:.2%}",
                    "VaR amount": "${:,.0f}", "CVaR amount": "${:,.0f}"}, hide_index=True, width="stretch")
# Notes are full sentences, listed below the table so they are never cut off. Sentences shared by both
# confidence levels are shown once; the rest are prefixed with their confidence level.
lines = []
for method, g in notes_by_method.groupby("Method", sort=False):
    per_row = [[s.strip().rstrip(".") + "." for s in str(n).split(". ") if s.strip()] for n in g["Notes"].fillna("")]
    common = [s for s in per_row[0] if all(s in r for r in per_row)]
    extra = [f"{c:.0%}: " + " ".join(s for s in r if s not in common) for c, r in zip(g["Confidence"], per_row)
             if any(s not in common for s in r)]
    lines.append(f"**{method}**: " + " ".join(common + extra))
st.markdown("\n".join(f"- {x}" for x in lines))
ui.basis_line(RiskBasis.OBSERVED, note=f" Hypothetical portfolio value {ui.money(pa.current_value)} ({ui.money(mb.spec.initial_value)} invested on {pa.performance.start:%d %b %Y}). "
              + ("" if hz == 1 else f"{hz}-day figures scale the 1-day estimates by √{hz} (assumes independent returns)."))
h = {(e.method, e.confidence): e for e in pa.tail_risk}
ui.chart(charts.return_distribution_chart(pa.path.returns, h[("Historical", 0.95)].var, h[("Historical", 0.99)].var))

# --- risk contribution ------------------------------------------------------------------------------
ui.section("contribution", "Risk")
t = pa.contributions.table
left, right = st.columns([3, 2])
with left:
    ui.chart(charts.risk_contribution_chart(t, RiskBasis.OBSERVED))
with right:
    show = t.drop(columns="volatility").rename(columns={
        "weight": "Weight", "marginal_contribution": "Marginal (∂σ/∂w)",
        "component_contribution": "Component", "pct_contribution": "% of risk"})
    st.dataframe(show, width="stretch", column_config=ui.pct_columns(show, show.columns))
    st.caption(f"Components sum to the portfolio volatility {ui.pct(pa.contributions.portfolio_volatility, 2)} "
               f"exactly (Euler decomposition). A negative contribution means the asset hedges the rest of the portfolio.")

# --- forward-looking -----------------------------------------------------------------------------
ui.section("forward-looking risk", "Volatility-adjusted")
st.markdown("Each asset's volatility is replaced by an alternative estimate and combined with historical "
            "correlations: **Σ = D·R·D**. Correlations are not re-estimated, so the stress slider below shows what "
            "happens if they rise, as they typically do in market stress.")
fb, ib = state.forecasts(), state.implied(fetch=False)
available = ["Historical 60d"] + (["GARCH(1,1)", "LSTM"] if fb is not None else []) + (["Implied"] if ib is not None else [])
if len(available) < len(VOL_SOURCES):
    st.caption("More sources become available after running the volatility forecasts and loading option data.")
c1, c2 = st.columns([2, 3])
src = c1.radio("Volatility source", available, horizontal=True)
lam = c2.slider("Correlation stress λ (0 = historical correlations, 1 = perfect correlation)", 0.0, 1.0, 0.0, 0.05)
try:
    fr = state.forward_risk(src, lam)
except ValueError as exc:
    st.error(str(exc))
    st.stop()
ui.basis_line(fr.tail_risk[0].basis, note=f" {fr.covariance.vols.description}; correlations: {fr.covariance.correlation_source}.")
if src in ("GARCH(1,1)", "LSTM") and fb.precomputed:
    st.caption(ui.precomputed_text(fb))

compare = []
for s in available:
    try:
        x = state.forward_risk(s, lam)
        e99 = next(e for e in x.tail_risk if e.method == "Parametric (normal)" and e.confidence == 0.99)
        compare.append({"Volatility source": s, "Portfolio volatility (ann.)": x.contributions.portfolio_volatility,
                        "1-day VaR 99% (normal)": e99.var, "1-day CVaR 99% (normal)": e99.cvar,
                        "Basis": e99.basis.label})
    except ValueError as exc:
        compare.append({"Volatility source": s, "Basis": f"unavailable: {exc}"})
cmp = pd.DataFrame(compare).set_index("Volatility source")
st.dataframe(cmp, width="stretch", column_config=ui.pct_columns(cmp, [c for c in cmp.columns if c != "Basis"]))

left, right = st.columns(2)
with left:
    ui.chart(charts.risk_contribution_chart(fr.contributions.table, fr.contributions.basis,
                                            f"Risk contribution — {src}"))
with right:
    ui.chart(charts.correlation_stress_chart(fr.stress_profile, fr.covariance.basis))
vt = pd.DataFrame({"Volatility used": fr.covariance.vols.values,
                   "Basis": [b.label for b in fr.covariance.vols.basis.values()]})
st.dataframe(vt, width="stretch", column_config={"Volatility used": st.column_config.NumberColumn(format="percent")})
ui.notes(list(fr.covariance.assumptions) + ["Forward-looking VaR assumes zero expected return; no historical method "
                                             "exists for a forecast covariance."], "Σ = D·R·D assumptions", expanded=False)
