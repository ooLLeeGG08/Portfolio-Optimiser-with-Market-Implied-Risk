import pandas as pd
import streamlit as st

from app_core.pipeline import AnalysisSettings
from core.types import RiskBasis
from portfolio.optimization import Strategy, comparison_table, identical_strategy_notes
from ui import components as ui
from ui import state
from visualization import charts

ui.header("optimisation", dim="Portfolio", subtitle=
          "Equal-weight, minimum-variance, maximum-Sharpe, custom and your own portfolio, estimated "
          "from the selected history.")
mb = state.market()

with st.container(border=True):
    st.markdown("**Custom portfolio constraints** (long-only, fully invested)")
    c = st.columns(4)
    min_w = c[0].number_input("Minimum weight per asset (%)", 0.0, 100.0, 0.0, 1.0) / 100
    max_w = c[1].number_input("Maximum weight per asset (%)", 1.0, 100.0, 40.0, 1.0) / 100
    objective = c[2].selectbox("Objective", ["max_sharpe", "min_variance", "target_return", "target_volatility"],
                               format_func=lambda o: {"max_sharpe": "Maximum Sharpe", "min_variance": "Minimum variance",
                                                      "target_return": "Target return", "target_volatility": "Target volatility"}[o])
    target = None
    if objective in ("target_return", "target_volatility"):
        target = c[3].number_input(f"{'Return' if objective == 'target_return' else 'Volatility'} target (% p.a.)",
                                   0.0, 200.0, 12.0, 0.5) / 100

oa = state.optimization(min_w, max_w, objective, target)
for e in oa.errors:
    st.warning(e, icon=":material/warning:")

st.info("All figures on this page are **in-sample estimates**: the weights were chosen using the same "
        "history that measures them, so they overstate what the portfolio would have achieved. See "
        "Model Evaluation for out-of-sample (walk-forward) results.", icon=":material/info:")
ui.basis_line(RiskBasis.OBSERVED, note=f"Estimation window {oa.inputs.start:%d %b %Y} – {oa.inputs.end:%d %b %Y} "
              f"({oa.inputs.n_obs:,} days); Ledoit-Wolf shrinkage intensity {oa.inputs.cov.shrinkage:.3f}; "
              f"risk-free rate {ui.pct(oa.inputs.risk_free_rate, 2)}.")

ui.section("comparison", "Strategy")
tab = comparison_table(oa.results)
metric_cols = [ui.RETURN_LABEL, "Excess return over risk-free (ann.)", "Volatility (ann.)", "Sharpe ratio"]
cfg = ui.pct_columns(tab, metric_cols[:3])
cfg[ui.RETURN_LABEL] = st.column_config.NumberColumn(format="percent", help=ui.RETURN_HELP)
cfg["Sharpe ratio"] = st.column_config.NumberColumn(format="%.2f")
st.dataframe(tab[metric_cols], width="stretch", column_config=cfg)
for n in identical_strategy_notes(oa.results, oa.constraints):
    st.caption(n)
weights = tab[[c for c in tab.columns if c.startswith("w[")]].rename(columns=lambda c: c[2:-1])
st.markdown("**Weights**")
st.dataframe(weights, width="stretch", column_config=ui.pct_columns(weights, weights.columns))

left, right = st.columns([3, 2])
with left:
    ui.chart(charts.efficient_frontier_chart(oa.frontier.points, oa.random_portfolios, oa.results))
    if oa.frontier.n_failed:
        st.caption(f"{oa.frontier.n_failed} frontier point(s) failed to solve and were omitted (not interpolated).")
with right:
    ui.chart(charts.weights_chart(oa.results))

c1, c2 = st.columns([1, 3])
apply_to = c1.selectbox("Use a strategy's weights as the portfolio",
                        [r.strategy.value for r in oa.results if r.strategy is not Strategy.USER])
if c2.button("Apply to portfolio", icon=":material/check:"):
    w = next(r.weights for r in oa.results if r.strategy.value == apply_to)
    w = w[w > 1e-4]
    s = state.settings()
    state.set_settings(AnalysisSettings(**{**s.__dict__, "tickers": tuple(w.index),
                                           "weights": tuple(float(x) for x in w / w.sum())}))
    st.rerun()

ui.section("are these weights?", "How reliable")
inp = oa.inputs
est = pd.DataFrame({ui.RETURN_LABEL: inp.mu, "Standard error of the average return": inp.mu_standard_error,
                    "Volatility (ann.)": pd.Series(inp.cov.matrix.values.diagonal() ** 0.5, index=inp.tickers)})
est_cfg = ui.pct_columns(est, est.columns)
est_cfg[ui.RETURN_LABEL] = st.column_config.NumberColumn(format="percent", help=ui.RETURN_HELP)
st.dataframe(est, width="stretch", column_config=est_cfg)
st.caption("Average returns are estimated very imprecisely: for a typical stock the uncertainty is several percentage "
           "points a year, which is why the optimal weights below move so much.")
ui.methodology([
    "**Historical average return** = the arithmetic mean of daily returns × 252 (total, not excess).",
    "**Sharpe ratio** = (average return − risk-free rate) / volatility.",
    "**Standard error of an average return** ≈ volatility / √years: over 10 years, a stock with 30% volatility has a "
    "standard error of about 9 percentage points.",
    "**Sensitivity test.** Average returns are redrawn from N(μ̂, Σ/years) and the maximum-Sharpe portfolio is "
    "re-solved for each draw.",
])
if oa.sensitivity is not None:
    ui.chart(charts.sensitivity_chart(oa.sensitivity.summary))
    st.caption(f"Historical average returns were redrawn {oa.sensitivity.n_requested} times from their sampling distribution "
               f"N(μ̂, Σ/years) and the maximum-Sharpe portfolio re-solved each time"
               + (f" ({oa.sensitivity.n_failed} draws had no asset above the risk-free rate and were skipped)." if oa.sensitivity.n_failed else ".")
               + " Wide ranges mean the optimal weights are not statistically well determined.")
elif oa.sensitivity_error:
    st.caption(f"Sensitivity analysis unavailable: {oa.sensitivity_error}")

ui.notes([
    "Covariance: Ledoit-Wolf shrinkage of daily simple returns, annualised ×252.",
    "Return input: historical average (arithmetic mean daily return ×252), used by the optimiser as its estimate of "
    "expected return — the noisiest input of mean-variance optimisation.",
    "Constraints: long-only, fully invested; custom bounds apply only to the custom portfolio. The frontier and "
    "the standard strategies use 0–100% bounds; equal weight ignores constraints by definition.",
    "No look-ahead: estimates use only returns up to the last date shown.",
])
