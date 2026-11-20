import pandas as pd
import streamlit as st

import config

from core.types import RiskBasis
from ui import components as ui
from ui import state
from visualization import charts

mb = state.market()
pa = state.portfolio()
h = mb.settings.horizon_days
ui.header("forecasting", dim="Volatility", subtitle=
          f"Historical, GARCH(1,1) and LSTM forecasts of realised volatility over the next {h} trading days.")

st.markdown(
    f"**Target.** Every model forecasts the same quantity: annualised realised volatility over the next {h} "
    f"trading days, √(252/{h} · Σ r²) of daily log returns. Forecasts made at the close of day *t* use data up "
    f"to *t* only.")

ui.section("historical volatility", "Rolling")
ui.basis_line(RiskBasis.OBSERVED, note="Trailing realised volatility, same definition as the forecast target.")
ui.chart(charts.rolling_vol_chart(pa.rolling_vol))

fb = state.forecasts()
if fb is None:
    with st.container(border=True):
        st.markdown("**Model forecasts have not been computed for this portfolio yet.**")
        st.caption(f"Fits GARCH and trains one LSTM per asset plus the portfolio on up to "
                   f"{mb.settings.forecast_history_years} years of data, then evaluates all models on a locked "
                   f"out-of-sample window. Takes about {15 * (len(mb.spec.tickers) + 1) // 60 + 1} minutes; "
                   f"results are kept for the rest of the day.")
        if config.HOSTED:
            st.caption("On the hosted version the connection can reset during a long run. If that happens, click "
                       "the button again: every finished asset is kept, so the run resumes where it stopped.")
        if st.button("Run volatility forecasts", type="primary", icon=":material/play_arrow:"):
            state.run_forecasts()
            st.rerun()
    st.stop()

for name, err in fb.failures.items():
    st.warning(f"{name}: forecasting failed — {err}", icon=":material/warning:")

ui.section("forecasts", "Latest")
ui.basis_line(RiskBasis.OBSERVED, RiskBasis.FORECAST,
              note=f" Historical columns are observed; GARCH and LSTM are forecasts with 10–90% intervals. "
                   f"As of {pd.Timestamp(fb.latest['As of'].iloc[0]):%d %b %Y}.")
lt = fb.latest.drop(columns="As of").astype(float)
st.dataframe(lt, width="stretch", column_config=ui.pct_columns(lt, lt.columns))

rows = lt[["Historical 60d", "GARCH(1,1)", "LSTM"]].copy()
for lab, lo, hi in (("GARCH(1,1)", "GARCH 10%", "GARCH 90%"), ("LSTM", "LSTM 10%", "LSTM 90%")):
    rows[f"{lab} lo"], rows[f"{lab} hi"] = lt[lo], lt[hi]
rows["Historical 60d basis"] = RiskBasis.OBSERVED
rows["GARCH(1,1) basis"] = RiskBasis.FORECAST
rows["LSTM basis"] = RiskBasis.FORECAST
ui.chart(charts.volatility_comparison_chart(rows, h))

ui.section("detail", "Model")
name = st.selectbox("Series", list(fb.studies), index=len(fb.studies) - 1)
study = fb.studies[name]
ui.chart(charts.forecast_test_chart(study.forecasts, name, h))
st.caption("The locked test window was not used to fit, tune or calibrate any model. "
           "Accuracy statistics are on the Model Evaluation page.")

left, right = st.columns(2)
with left:
    st.subheader("GARCH(1,1) diagnostics (development-window fit)")
    g = study.garch_dev_fit
    params = pd.DataFrame({"Estimate": g.params, "Std. error": g.std_err})
    st.dataframe(params, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.4f") for c in params})
    d = pd.DataFrame({"Value": [g.dist, f"{g.persistence:.4f}", f"{g.half_life_days:.0f} trading days",
                                ui.pct(g.unconditional_vol), f"{g.n_obs:,}", f"{g.start:%Y-%m-%d} – {g.end:%Y-%m-%d}",
                                f"{g.bic:,.1f}"]},
                     index=["Innovation distribution", "Persistence α+β", "Half-life of shocks",
                            "Long-run volatility", "Observations", "Estimation sample", "BIC"])
    st.dataframe(d, width="stretch")
    lb = g.ljung_box.rename(columns={"lb_stat": "Ljung-Box statistic", "lb_pvalue": "p-value"})
    st.dataframe(lb, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.3f") for c in lb})
    st.caption("A low p-value for squared standardised residuals would indicate volatility clustering the model "
               "has not captured. Returns are modelled in percent; parameters refer to that scale.")
    if study.garch_dist_table is not None:
        st.caption("Distribution chosen by BIC on development data: "
                   + ", ".join(f"{i} BIC {r.BIC:,.1f}" for i, r in study.garch_dist_table.iterrows()))
with right:
    st.subheader("LSTM training")
    ls = study.lstm
    ui.chart(charts.training_chart(ls.history, ls.best_epoch))
    seg = ls.prep.segments
    cfg = ls.cfg
    st.dataframe(pd.DataFrame({"Value": [
        f"{seg.train[0]:%Y-%m-%d} – {seg.train[-1]:%Y-%m-%d} ({len(seg.train):,})",
        f"{seg.val[0]:%Y-%m-%d} – {seg.val[-1]:%Y-%m-%d} ({len(seg.val):,})",
        f"{seg.cal[0]:%Y-%m-%d} – {seg.cal[-1]:%Y-%m-%d} ({len(seg.cal):,})",
        f"{seg.test[0]:%Y-%m-%d} – {seg.test[-1]:%Y-%m-%d} ({len(seg.test):,})",
        f"{cfg.seq_len} days, {cfg.hidden_size} units × {cfg.num_layers} layer, dropout {cfg.dropout}",
        f"epoch {ls.best_epoch} of {len(ls.history)} (patience {cfg.patience})",
        f"{ls.conformal_adjustment:+.3f} (log-vol units)",
    ]}, index=["Training samples", "Validation (early stopping)", "Calibration (conformal)", "Locked test",
               "Architecture", "Best epoch (checkpoint)", "Conformal interval adjustment"]), width="stretch")
    st.caption(f"Consecutive segments are separated by a {h}-day purge gap; features are scaled with "
               f"statistics from the training period only. A negative conformal adjustment means the raw "
               f"intervals over-covered on the calibration data and were narrowed.")

ui.notes(study.notes + [
    "GARCH intervals come from simulated return paths and omit parameter-estimation uncertainty.",
    "LSTM intervals are split-conformal; their coverage guarantee assumes exchangeable data, which a shift "
    "in the volatility regime violates — test-window coverage is reported on the Model Evaluation page.",
])
