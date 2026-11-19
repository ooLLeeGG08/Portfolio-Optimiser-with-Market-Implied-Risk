import pandas as pd
import streamlit as st

from core.types import RiskBasis
from ui import components as ui
from ui import state
from visualization import charts

mb = state.market()
ui.header("evaluation", dim="Model", subtitle= "Out-of-sample evidence only: chronological splits, a locked test window, no tuning on test data.")

# --- strategy backtest ---------------------------------------------------------------------------
ui.section("walk-forward backtest", "Portfolio strategies:")
bb = state.backtests()
ui.chart(charts.backtest_chart(bb.results, bb.test_start))
cols = ["CAGR", "Annualised return", "Annualised volatility", "Max drawdown", "Annual turnover (one-way)", "Annual cost drag"]


def table(df: pd.DataFrame):
    cfg = ui.pct_columns(df, cols)
    cfg["Sharpe ratio"] = st.column_config.NumberColumn(format="%.2f")
    st.dataframe(df, width="stretch", column_config=cfg)


def window(df):
    return f"{df['Start'].iloc[0]:%d %b %Y} – {df['End'].iloc[0]:%d %b %Y}"


st.subheader(f"Locked test window · {window(bb.metrics_test)}")
table(bb.metrics_test.drop(columns=["Start", "End"]))
st.subheader(f"Development window · {window(bb.metrics_development)}")
table(bb.metrics_development.drop(columns=["Start", "End"]))
ui.notes(bb.notes, "Backtest design and limitations", expanded=True)

# --- volatility forecasts -------------------------------------------------------------------------
ui.section("out-of-sample accuracy", "Volatility forecasts:")
fb = state.forecasts()
if fb is None:
    st.caption("Volatility forecasts have not been computed yet.")
    st.page_link("views/forecasting.py", label="Run volatility forecasts", icon=":material/play_arrow:")
    st.stop()

summary = []
for name, sres in fb.studies.items():
    m = sres.metrics_all
    best_q = m["QLIKE"].idxmin()
    dm = sres.dm_tests.loc["LSTM vs GARCH(1,1)"]
    summary.append({"Series": name, "Lowest QLIKE": best_q, "Lowest RMSE": m["RMSE"].idxmin(),
                    "GARCH coverage (80% nominal)": m.loc["GARCH(1,1)", "Interval coverage"],
                    "LSTM coverage (80% nominal)": m.loc["LSTM", "Interval coverage"],
                    "LSTM vs GARCH: QLIKE p-value": dm["QLIKE: p-value"],
                    "LSTM vs GARCH: squared-error p-value": dm["Squared error: p-value"]})
sm = pd.DataFrame(summary).set_index("Series")
cfg = ui.pct_columns(sm, ["GARCH coverage (80% nominal)", "LSTM coverage (80% nominal)"])
cfg.update({c: st.column_config.NumberColumn(format="%.3f") for c in sm.columns if "p-value" in c})
st.dataframe(sm, width="stretch", column_config=cfg)
diffs = pd.Series({n: r.dm_tests.loc["LSTM vs GARCH(1,1)", "QLIKE: mean loss diff"] for n, r in fb.studies.items()})
sig = sm["LSTM vs GARCH: QLIKE p-value"] < 0.05
lstm_better = [n for n in sm.index if sig[n] and diffs[n] < 0]
garch_better = [n for n in sm.index if sig[n] and diffs[n] > 0]
st.markdown(f"**Summary.** Lowest QLIKE (the loss most robust to noise in realised volatility): "
            f"{', '.join(f'{k} for {v} series' for k, v in sm['Lowest QLIKE'].value_counts().items())}. "
            f"Diebold–Mariano (QLIKE, 5%): LSTM significantly better than GARCH for "
            f"{len(lstm_better)} series{' (' + ', '.join(lstm_better) + ')' if lstm_better else ''}, GARCH significantly "
            f"better for {len(garch_better)}{' (' + ', '.join(garch_better) + ')' if garch_better else ''}, no "
            f"significant difference for {len(sm) - len(lstm_better) - len(garch_better)}. "
            f"Results are reported as obtained; no model is assumed to win.")

name = st.selectbox("Series detail", list(fb.studies), index=len(fb.studies) - 1)
sres = fb.studies[name]
ui.basis_line(RiskBasis.FORECAST, note=f" Test window {sres.split.test_start:%d %b %Y} – {sres.split.test_end:%d %b %Y}.")
metric_cfg = {c: st.column_config.NumberColumn(format="%.4f") for c in ["MAE", "RMSE", "Bias", "QLIKE"]}
metric_cfg.update(ui.pct_columns(sres.metrics_all, ["Directional accuracy", "Interval coverage", "Mean interval width"]))
metric_cfg["n"] = st.column_config.NumberColumn(format="%d")
left, right = st.columns(2)
with left:
    st.markdown(f"**All {int(sres.metrics_all['n'].iloc[0])} forecast dates** (overlapping targets)")
    st.dataframe(sres.metrics_all, width="stretch", column_config=metric_cfg)
with right:
    st.markdown(f"**Non-overlapping subsample** (every {sres.horizon}th date, "
                f"{int(sres.metrics_non_overlapping['n'].iloc[0])} independent targets)")
    st.dataframe(sres.metrics_non_overlapping, width="stretch", column_config=metric_cfg)
st.markdown("**Diebold–Mariano tests against GARCH(1,1)** (negative mean loss difference = the model beat GARCH)")
st.dataframe(sres.dm_tests, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.3f") for c in sres.dm_tests})
ui.chart(charts.forecast_test_chart(sres.forecasts, name, sres.horizon))
ui.notes([
    "MAE, RMSE and bias are in annualised volatility units (0.01 = 1 volatility point); bias > 0 means over-forecasting.",
    "QLIKE = y²/f² − ln(y²/f²) − 1 penalises under-forecasting more than over-forecasting and is robust to noise in "
    "the realised-volatility proxy.",
    f"Overlapping horizons: consecutive {sres.horizon}-day targets share {sres.horizon - 1} days, so errors are strongly "
    "autocorrelated and the effective sample is about one observation per horizon. The Diebold–Mariano test uses a "
    f"HAC variance with {sres.horizon - 1} lags and the Harvey–Leybourne–Newbold correction.",
    "Directional accuracy compares the forecast and the outcome with today's 21-day realised volatility; for the 20-day "
    "historical forecast this is close to a coin toss by construction.",
    "Interval coverage is the share of outcomes inside the 10–90% interval (nominal 80%).",
    *sres.notes,
], "Interpretation and limitations", expanded=True)
