import pandas as pd
import streamlit as st

from app_core.pipeline import default_scenarios
from core.types import RiskBasis
from risk import stress_testing as stt
from ui import components as ui
from ui import state
from visualization import charts

mb = state.market()
pa = state.portfolio()
w, v, cov = pa.current_weights, pa.current_value, pa.covariance
ui.header("analysis", dim="Scenario", subtitle= "Hypothetical stress tests on the current portfolio.")
st.warning("**Hypothetical scenarios — not predictions.** Each result shows what would happen under the stated "
         "assumptions, not what is expected to happen.", icon=":material/bolt:")


def show_tables(results):
    price = [r for r in results if r.portfolio_return is not None]
    regime = [r for r in results if r.stressed_risk is not None]
    if price:
        st.markdown("**Price shocks** — instantaneous moves in asset prices")
        t = pd.DataFrame([{"Scenario": r.name, "Portfolio return": r.portfolio_return,
                           "Loss ($)": -r.asset_pnl.sum(), "Value after": r.portfolio_value_after} for r in price]).set_index("Scenario")
        st.dataframe(t, width="stretch", column_config={"Portfolio return": st.column_config.NumberColumn(format="percent"),
                                                        "Loss ($)": st.column_config.NumberColumn(format="dollar"),
                                                        "Value after": st.column_config.NumberColumn(format="dollar")})
    if regime:
        st.markdown("**Risk regimes** — higher volatility and/or correlations; risk compared with the baseline")
        t = pd.DataFrame([{"Scenario": r.name, "Vol (ann.) base": r.baseline_risk.volatility,
                           "Vol (ann.) stressed": r.stressed_risk.volatility,
                           "VaR 99% 1d base": r.baseline_risk.var_99_1d,
                           "VaR 99% 1d stressed": r.stressed_risk.var_99_1d,
                           "VaR 99% stressed ($)": r.stressed_risk.var_99_1d * r.portfolio_value_before}
                          for r in regime]).set_index("Scenario")
        cfg = ui.pct_columns(t, [c for c in t.columns if "($)" not in c])
        cfg["VaR 99% stressed ($)"] = st.column_config.NumberColumn(format="dollar")
        st.dataframe(t, width="stretch", column_config=cfg)
        st.caption("Baseline uses the historical (Ledoit-Wolf) covariance; VaR is zero-mean normal.")


ui.section("scenarios", "Standard")
ui.basis_line(RiskBasis.HYPOTHETICAL, note=f" Portfolio value {ui.money(v)}.")
std = default_scenarios(mb, pa)
show_tables(std)
ui.chart(charts.scenario_chart(std))
with st.expander("Inputs: betas and rate sensitivities"):
    sens = pd.DataFrame({f"Beta vs {mb.settings.benchmark}": pa.betas})
    if pa.rate_sensitivity is not None:
        rs = pa.rate_sensitivity
        sens["Return per +1pp in 10y yield"] = rs.coefficient
        sens["t-statistic (HAC)"] = rs.t_stat
        sens["R²"] = rs.r_squared
    st.dataframe(sens, width="stretch", column_config={
        "Return per +1pp in 10y yield": st.column_config.NumberColumn(format="percent"),
        **{c: st.column_config.NumberColumn(format="%.2f") for c in sens.columns if c != "Return per +1pp in 10y yield"}})
    st.caption("Estimated from daily returns over the selected history. Rate sensitivities with |t| < 2 are "
               "statistically indistinguishable from zero.")

ui.section("a scenario", "Build")
with st.form("scenario"):
    c = st.columns(4)
    mkt = c[0].slider(f"Market move ({mb.settings.benchmark}, %)", -50, 30, -15) / 100
    rates = c[1].slider("10-year yield change (bp)", -300, 300, 0, 25)
    volm = c[2].slider("Volatility multiplier", 0.5, 4.0, 1.0, 0.1)
    lam = c[3].slider("Correlation stress λ", 0.0, 1.0, 0.0, 0.05)
    shocks = st.data_editor(pd.DataFrame({"Ticker": w.index, "Additional shock (%)": 0.0}), hide_index=True,
                            width="stretch", disabled=["Ticker"],
                            column_config={"Additional shock (%)": st.column_config.NumberColumn(min_value=-100.0, max_value=200.0, format="%.1f")})
    name = st.text_input("Scenario name", "Custom scenario")
    go = st.form_submit_button("Run scenario", type="primary")

if go:
    total = pa.betas.reindex(w.index) * mkt
    if rates and pa.rate_sensitivity is not None:
        total = total + pa.rate_sensitivity.coefficient.reindex(w.index) * rates / 100
    total = total + shocks.set_index("Ticker")["Additional shock (%)"].reindex(w.index).fillna(0) / 100
    price = stt.custom_shock(w, total.to_dict(), v, name)
    risk = stt.correlation_convergence(w, cov, lam, v, vol_multiplier=volm)
    c = st.columns(4)
    with c[0]:
        ui.kpi("Portfolio return", ui.signed_pct(price.portfolio_return, 2), RiskBasis.HYPOTHETICAL)
    with c[1]:
        ui.kpi("Value after", ui.money(price.portfolio_value_after), RiskBasis.HYPOTHETICAL)
    with c[2]:
        ui.kpi("Volatility: baseline → stressed",
               f"{ui.pct(risk.baseline_risk.volatility)} → {ui.pct(risk.stressed_risk.volatility)}", RiskBasis.HYPOTHETICAL)
    with c[3]:
        ui.kpi("1-day VaR 99%: baseline → stressed",
               f"{ui.pct(risk.baseline_risk.var_99_1d, 2)} → {ui.pct(risk.stressed_risk.var_99_1d, 2)}", RiskBasis.HYPOTHETICAL)
    left, right = st.columns(2)
    with left:
        ui.chart(charts.asset_pnl_chart(price))
    with right:
        rc = pd.DataFrame({"Asset shock": price.asset_shocks, "P&L ($)": price.asset_pnl,
                           "% of risk (stressed)": risk.stressed_contributions})
        st.dataframe(rc, width="stretch", column_config={"Asset shock": st.column_config.NumberColumn(format="percent"),
                                                         "P&L ($)": st.column_config.NumberColumn(format="dollar"),
                                                         "% of risk (stressed)": st.column_config.NumberColumn(format="percent")})
    ui.notes(["Asset move = beta × market move + rate sensitivity × yield change + additional shock; floored at −100%.",
              "Stressed risk scales all volatilities by the multiplier and pushes correlations towards 1 by λ; "
              "VaR is zero-mean normal from the stressed covariance.",
              *stt.market_decline(w, pa.betas, mkt, v).assumptions], expanded=True)
