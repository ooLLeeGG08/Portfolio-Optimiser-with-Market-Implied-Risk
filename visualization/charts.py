"""Interactive Plotly charts. Every chart has a title, axis titles with units and a legend
where more than one series is shown. Percentages are displayed as percentages."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.types import RiskBasis
from visualization import theme as T

PCT = ".1%"


def _fig(title: str, x: str | None = None, y: str | None = None, height: int = 380) -> go.Figure:
    f = go.Figure()
    f.update_layout(title=title, height=height, xaxis_title=x, yaxis_title=y)
    return f


# --- portfolio ---------------------------------------------------------------------------------

def allocation_donut(weights: pd.Series, title: str = "Current allocation") -> go.Figure:
    w = weights[weights > 1e-6].sort_values(ascending=False)
    f = _fig(title, height=340)
    f.add_trace(go.Pie(labels=w.index, values=w.values, hole=0.66, sort=False,
                       marker=dict(colors=T.SERIES[:len(w)], line=dict(color=T.SURFACE, width=3)),
                       textinfo="label+percent", textposition="outside", automargin=True,
                       textfont=dict(size=11, color=T.MUTED),
                       hovertemplate="%{label}: %{percent}<extra></extra>"))
    f.update_layout(showlegend=False, margin=dict(l=40, r=40, t=60, b=30))
    return f


def performance_chart(values: pd.Series, benchmark: pd.Series, bench_name: str,
                      title: str = "Portfolio value vs benchmark") -> go.Figure:
    f = _fig(title, None, "Value ($)")
    f.add_trace(go.Scatter(x=values.index, y=values, name="Portfolio", line=dict(width=2, color=T.ACCENT),
                           fill="tozeroy", fillgradient=dict(type="vertical", colorscale=[
                               [0, "rgba(255,106,26,0)"], [1, "rgba(255,106,26,0.28)"]])))
    f.add_trace(go.Scatter(x=benchmark.index, y=benchmark, name=f"{bench_name} (same start value)",
                           line=dict(width=1.4, color=T.MUTED)))
    f.update_yaxes(tickprefix="$", tickformat=",.0f")
    return f


def drawdown_chart(values: pd.Series, benchmark: pd.Series | None = None, bench_name: str = "",
                   title: str = "Drawdown from previous peak") -> go.Figure:
    f = _fig(title, None, "Drawdown (%)", height=300)
    dd = values / values.cummax() - 1
    f.add_trace(go.Scatter(x=dd.index, y=dd, name="Portfolio", fill="tozeroy",
                           line=dict(width=1.2, color=T.NEGATIVE), fillcolor="rgba(229,86,107,0.18)"))
    if benchmark is not None:
        bd = benchmark / benchmark.cummax() - 1
        f.add_trace(go.Scatter(x=bd.index, y=bd, name=bench_name, line=dict(width=1, color=T.MUTED)))
    f.update_yaxes(tickformat=".0%")
    return f


def correlation_heatmap(corr: pd.DataFrame, title: str = "Correlation of daily returns") -> go.Figure:
    f = _fig(title, height=60 + 42 * len(corr) + 80)
    f.add_trace(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
                           colorscale=[[0, "#7C8CA8"], [0.5, T.SURFACE_2], [1, T.ACCENT]],
                           text=np.round(corr.values, 2), texttemplate="%{text}",
                           colorbar=dict(title="ρ", thickness=10),
                           hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>"))
    f.update_yaxes(autorange="reversed")
    return f


def rolling_vol_chart(df: pd.DataFrame, title: str = "Rolling realised volatility (annualised)") -> go.Figure:
    f = _fig(title, None, "Annualised volatility (%)")
    for i, c in enumerate(df.columns):
        f.add_trace(go.Scatter(x=df.index, y=df[c], name=c, line=dict(width=1.4, color=T.SERIES[i % 10])))
    f.update_yaxes(tickformat=".0%")
    return f


# --- optimisation ------------------------------------------------------------------------------

def efficient_frontier_chart(points: pd.DataFrame, random: pd.DataFrame, strategies) -> go.Figure:
    f = _fig("Efficient frontier (in-sample estimates)", "Annualised volatility (%)",
             "Historical average annual return (%)", height=460)
    f.add_trace(go.Scatter(x=random.volatility, y=random.expected_return, mode="markers",
                           name="Random long-only portfolios",
                           # Each point outlined in the background colour so overlapping portfolios read as
                           # separate dots rather than one orange cloud.
                           marker=dict(size=6, color=random.sharpe, colorscale=[[0, "#2A2A2D"], [1, T.ACCENT_SOFT]], opacity=0.9,
                                       line=dict(color=T.BG, width=0.8),
                                       colorbar=dict(title="Sharpe", thickness=10, x=1.02)),
                           hovertemplate="vol %{x:.1%}, return %{y:.1%}<extra></extra>"))
    f.add_trace(go.Scatter(x=points.volatility, y=points.expected_return, mode="lines",
                           name="Efficient frontier", line=dict(color=T.TEXT, width=2.4)))
    symbols = ["circle", "diamond", "star", "square", "triangle-up"]
    for i, s in enumerate(strategies):
        f.add_trace(go.Scatter(x=[s.stats.volatility], y=[s.stats.expected_return], mode="markers",
                               name=s.strategy.value,
                               marker=dict(size=13, symbol=symbols[i % 5], color=T.SERIES[(i + 1) % 10],
                                           line=dict(color=T.BG, width=1)),
                               hovertemplate=f"{s.strategy.value}<br>vol %{{x:.1%}}<br>return %{{y:.1%}}"
                                             f"<br>Sharpe {s.stats.sharpe:.2f}<extra></extra>"))
    f.update_xaxes(tickformat=".0%")
    f.update_yaxes(tickformat=".0%")
    # Legend below the chart, not on top of the points; the bottom margin leaves room for it to wrap.
    f.update_layout(height=600, margin=dict(b=150),
                    legend=dict(orientation="h", yanchor="top", y=-0.16, x=0, xanchor="left"))
    return f


def weights_chart(strategies, title: str = "Asset allocation by strategy") -> go.Figure:
    f = _fig(title, None, "Weight (%)")
    for i, s in enumerate(strategies):
        f.add_trace(go.Bar(x=s.weights.index, y=s.weights.values, name=s.strategy.value,
                           marker_color=T.SERIES[(i + 1) % 10]))
    f.update_layout(barmode="group", height=440,
                    legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0))
    f.update_yaxes(tickformat=".0%")
    return f


def sensitivity_chart(summary: pd.DataFrame) -> go.Figure:
    """Maximum-Sharpe weights and how much they move when past returns are nudged within their
    uncertainty. Labels are written for readers without a finance background."""
    f = _fig("How much should go into each asset, and how sure is that?", None, "Share of the portfolio (%)",
             height=470)
    rng = [f"{lo:.0%} to {hi:.0%}" for lo, hi in zip(summary["p05"], summary["p95"])]
    f.add_trace(go.Bar(x=summary.index, y=summary["point_estimate"], name="“Best” mix, using the past exactly as it was",
                       marker_color=T.ACCENT, customdata=rng,
                       hovertemplate="%{x}<br>“Best” mix: %{y:.0%}<extra></extra>"))
    f.add_trace(go.Scatter(x=summary.index, y=summary["mean"], mode="markers",
                           name="Typical answer (dot) and likely range (line)",
                           marker=dict(color=T.TEXT, size=9, line=dict(color=T.BG, width=1)), customdata=rng,
                           hovertemplate="%{x}<br>Typical answer: %{y:.0%}<br>Likely range: %{customdata}<extra></extra>",
                           error_y=dict(type="data", symmetric=False, thickness=2, width=7,
                                        array=summary["p95"] - summary["mean"],
                                        arrayminus=summary["mean"] - summary["p05"], color=T.MUTED)))
    f.update_yaxes(tickformat=".0%")
    f.update_layout(margin=dict(b=90), legend=dict(orientation="h", yanchor="top", y=-0.12, x=0, xanchor="left"))
    return f


# --- risk --------------------------------------------------------------------------------------

def risk_contribution_chart(table: pd.DataFrame, basis: RiskBasis,
                            title: str = "Weight vs share of portfolio risk") -> go.Figure:
    t = table.sort_values("pct_contribution", ascending=False)
    f = _fig(f"{title} · {T.BASIS_SHORT[basis]}", None, "Share (%)", height=340)
    f.add_trace(go.Bar(x=t.index, y=t["weight"], name="Portfolio weight", marker_color=T.NEUTRAL_BAR))
    f.add_trace(go.Bar(x=t.index, y=t["pct_contribution"], name="Share of volatility (Euler)",
                       marker_color=T.BASIS_COLORS[basis]))
    f.update_layout(barmode="group")
    f.update_yaxes(tickformat=".0%")
    return f


def return_distribution_chart(returns: pd.Series, var95: float, var99: float) -> go.Figure:
    f = _fig("Distribution of daily portfolio returns with historical VaR", "Daily return (%)",
             "Number of days", height=340)
    f.add_trace(go.Histogram(x=returns, nbinsx=120, name="Daily returns", marker_color=T.ACCENT,
                             opacity=0.75))
    for v, lab, c, pos in ((var95, "95% VaR", T.SERIES[2], "top right"), (var99, "99% VaR", T.NEGATIVE, "top left")):
        f.add_vline(x=-v, line=dict(color=c, dash="dash", width=1.5), annotation_position=pos,
                    annotation=dict(text=f"{lab}: −{v:.2%}", font=dict(color=c, size=11)))
    f.update_xaxes(tickformat=".0%")
    f.update_layout(showlegend=False)
    return f


def correlation_stress_chart(profile: pd.DataFrame, base_basis: RiskBasis) -> go.Figure:
    f = _fig("Portfolio volatility as correlations converge (hypothetical)",
             "Correlation stress λ (0 = historical, 1 = perfect)", "Annualised volatility (%)",
             height=320)
    f.add_trace(go.Scatter(x=profile["lambda"], y=profile["portfolio_volatility"], mode="lines+markers",
                           name="Portfolio volatility", line=dict(color=T.BASIS_COLORS[RiskBasis.HYPOTHETICAL])))
    f.add_trace(go.Scatter(x=[0], y=[profile["portfolio_volatility"].iloc[0]], mode="markers",
                           name=f"Baseline ({T.BASIS_SHORT[base_basis]})",
                           marker=dict(size=11, color=T.BASIS_COLORS[base_basis])))
    f.update_yaxes(tickformat=".0%")
    return f


def scenario_chart(results) -> go.Figure:
    price = [r for r in results if r.portfolio_return is not None]
    f = _fig("Hypothetical scenario impact on portfolio value", None, "Portfolio return (%)", height=320)
    f.add_trace(go.Bar(x=[r.name for r in price], y=[r.portfolio_return for r in price],
                       marker_color=[T.NEGATIVE if r.portfolio_return < 0 else T.POSITIVE for r in price],
                       name="Portfolio return", hovertemplate="%{x}: %{y:.2%}<extra></extra>"))
    f.update_yaxes(tickformat=".1%")
    f.update_layout(showlegend=False)
    return f


def asset_pnl_chart(result) -> go.Figure:
    p = result.asset_pnl.sort_values()
    f = _fig(f"Asset P&L — {result.name} (hypothetical)", None, "P&L ($)", height=320)
    f.add_trace(go.Bar(x=p.index, y=p.values, marker_color=[T.NEGATIVE if v < 0 else T.POSITIVE for v in p],
                       hovertemplate="%{x}: $%{y:,.0f}<extra></extra>", name="P&L"))
    f.update_yaxes(tickprefix="$", tickformat=",.0f")
    f.update_layout(showlegend=False)
    return f


# --- volatility --------------------------------------------------------------------------------

def forecast_test_chart(df: pd.DataFrame, name: str, horizon: int) -> go.Figure:
    f = _fig(f"{name}: {horizon}-day volatility forecasts vs subsequently realised (locked test window)",
             "Forecast date", "Annualised volatility (%)", height=420)
    for lo, hi, col, lab in (("garch_q10", "garch_q90", T.BASIS_COLORS[RiskBasis.FORECAST], "GARCH 10–90%"),
                             ("lstm_q10", "lstm_q90", T.ACCENT, "LSTM 10–90% (conformal)")):
        rgba = _rgba(col, 0.14)
        f.add_trace(go.Scatter(x=df.index, y=df[hi], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        f.add_trace(go.Scatter(x=df.index, y=df[lo], fill="tonexty", fillcolor=rgba, line=dict(width=0),
                               name=lab, hoverinfo="skip"))
    f.add_trace(go.Scatter(x=df.index, y=df["realized"], name="Realised over next h days (ex post)",
                           line=dict(color=T.TEXT, width=2)))
    f.add_trace(go.Scatter(x=df.index, y=df["hist_60d"], name="Historical 60d", line=dict(color=T.MUTED, width=1.2, dash="dot")))
    f.add_trace(go.Scatter(x=df.index, y=df["garch"], name="GARCH(1,1)", line=dict(color=T.BASIS_COLORS[RiskBasis.FORECAST], width=1.6)))
    f.add_trace(go.Scatter(x=df.index, y=df["lstm"], name="LSTM (median)", line=dict(color=T.ACCENT, width=1.6)))
    f.update_yaxes(tickformat=".0%")
    return f


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def as_basis(b) -> RiskBasis | None:
    return RiskBasis.coerce(b)


def volatility_comparison_chart(rows: pd.DataFrame, horizon: int) -> go.Figure:
    """rows: index = ticker; columns = label -> value; plus '<label> basis' columns."""
    f = _fig(f"Volatility estimates by source — {horizon}-day horizon, annualised",
             None, "Annualised volatility (%)", height=400)
    labels = [c for c in rows.columns if not c.endswith(" basis") and not c.endswith(" lo") and not c.endswith(" hi")]
    symbols = {"Historical 60d": "circle", "GARCH(1,1)": "diamond", "LSTM": "square",
               "Implied (options)": "star", "Implied proxy (estimate)": "star-open"}
    for lab in labels:
        basis = rows.get(f"{lab} basis")
        color = [T.BASIS_COLORS.get(as_basis(b), T.MUTED)
                 for b in (basis if basis is not None else [None] * len(rows))]
        err = None
        if f"{lab} lo" in rows and f"{lab} hi" in rows:
            err = dict(type="data", symmetric=False, array=(rows[f"{lab} hi"] - rows[lab]).to_numpy(),
                       arrayminus=(rows[lab] - rows[f"{lab} lo"]).to_numpy(), color=T.MUTED, thickness=1)
        f.add_trace(go.Scatter(x=rows.index, y=rows[lab], mode="markers", name=lab, error_y=err,
                               marker=dict(size=12, symbol=symbols.get(lab, "circle"), color=color,
                                           line=dict(color=T.BG, width=1)),
                               hovertemplate=f"{lab}: %{{y:.1%}}<extra>%{{x}}</extra>"))
    f.update_yaxes(tickformat=".0%")
    f.update_layout(legend=dict(orientation="h", y=1.0))
    return f


def training_chart(history: pd.DataFrame, best_epoch: int) -> go.Figure:
    f = _fig("LSTM training (pinball loss on log volatility)", "Epoch", "Loss", height=300)
    f.add_trace(go.Scatter(x=history.epoch, y=history.train_loss, name="Training"))
    f.add_trace(go.Scatter(x=history.epoch, y=history.val_loss, name="Validation"))
    f.add_vline(x=best_epoch, line=dict(color=T.MUTED, dash="dot"),
                annotation=dict(text="best (checkpoint)", font=dict(color=T.MUTED, size=10)))
    return f


# --- implied volatility -------------------------------------------------------------------------

def vix_chart(frame: pd.DataFrame, years: int = 5) -> go.Figure:
    d = frame[frame.index >= frame.index[-1] - pd.DateOffset(years=years)]
    f = _fig(f"VIX (S&P 500 30-day implied volatility) — last {years} years", None, "Implied volatility (%)")
    f.add_trace(go.Scatter(x=d.index, y=d.vix, name="VIX", line=dict(color=T.BASIS_COLORS[RiskBasis.MARKET_IMPLIED], width=1.4)))
    for c, col in (("mean_60d", T.MUTED), ("mean_252d", T.SERIES[2])):
        f.add_trace(go.Scatter(x=d.index, y=d[c], name=c.replace("mean_", "").replace("d", "-day mean"),
                               line=dict(color=col, width=1.2, dash="dot")))
    f.update_yaxes(tickformat=".0%")
    return f


def vrp_chart(frame: pd.DataFrame, years: int = 5) -> go.Figure:
    d = frame[frame.index >= frame.index[-1] - pd.DateOffset(years=years)]
    f = _fig("VIX vs S&P 500 volatility realised over the following ~30 days (ex post)", "Date of VIX observation",
             "Annualised volatility (%)")
    f.add_trace(go.Scatter(x=d.index, y=d.vix, name="VIX (implied, risk-neutral)",
                           line=dict(color=T.BASIS_COLORS[RiskBasis.MARKET_IMPLIED], width=1.3)))
    f.add_trace(go.Scatter(x=d.index, y=d.realized_fwd, name="Subsequently realised (SPY)",
                           line=dict(color=T.TEXT, width=1.1)))
    f.update_yaxes(tickformat=".0%")
    return f


def term_structure_chart(ts: pd.DataFrame, ticker: str, target_days: float) -> go.Figure:
    f = _fig(f"{ticker}: at-the-money implied volatility term structure", "Days to expiry (calendar)",
             "Implied volatility (%)", height=340)
    f.add_trace(go.Scatter(x=ts.T_days, y=ts.atm_iv, mode="lines+markers", name="ATM implied vol",
                           line=dict(color=T.BASIS_COLORS[RiskBasis.MARKET_IMPLIED])))
    f.add_vline(x=target_days, line=dict(color=T.MUTED, dash="dot"),
                annotation=dict(text=f"forecast horizon ≈ {target_days:.0f} days", font=dict(color=T.MUTED, size=10)))
    f.update_yaxes(tickformat=".0%")
    return f


def smile_chart(smile: pd.DataFrame, ticker: str, expiry, spot: float) -> go.Figure:
    f = _fig(f"{ticker}: implied volatility smile — expiry {pd.Timestamp(expiry).date()}", "Strike",
             "Implied volatility (%)", height=340)
    for kind, col in (("put", T.SERIES[5]), ("call", T.SERIES[0])):
        s = smile[smile.type == kind]
        f.add_trace(go.Scatter(x=s.strike, y=s.iv, mode="markers", name=f"{kind.title()}s (OTM filled, ITM hollow)",
                               marker=dict(color=col, size=7, symbol=["circle" if o else "circle-open" for o in s.otm])))
    f.add_vline(x=spot, line=dict(color=T.MUTED, dash="dot"), annotation=dict(text="spot", font=dict(color=T.MUTED, size=10)))
    f.update_yaxes(tickformat=".0%")
    return f


def disagreement_chart(indicators) -> go.Figure:
    f = _fig("Volatility Disagreement Indicator — dispersion of estimates (analytical, not a signal)",
             None, "Log-dispersion of estimates", height=320)
    t = [d.ticker for d in indicators]
    f.add_trace(go.Bar(x=t, y=[d.dispersion for d in indicators], name="All estimates", marker_color=T.ACCENT))
    f.add_trace(go.Bar(x=t, y=[d.dispersion_models for d in indicators], name="Models only", marker_color=T.MUTED))
    for y, lab in ((0.10, "Low / Moderate"), (0.25, "Moderate / High")):
        f.add_hline(y=y, line=dict(color=T.SERIES[2], dash="dot", width=1),
                    annotation=dict(text=lab, font=dict(color=T.SERIES[2], size=10)))
    f.update_layout(barmode="group")
    return f


# --- backtest ----------------------------------------------------------------------------------

def backtest_chart(results: dict, test_start: pd.Timestamp) -> go.Figure:
    f = _fig("Walk-forward backtest, net of costs (growth of $1, log scale)", None, "Growth of $1", height=420)
    start = max(r.values.index[0] for r in results.values())
    for i, (n, r) in enumerate(results.items()):
        v = r.values.loc[start:]
        f.add_trace(go.Scatter(x=v.index, y=v / v.iloc[0], name=n, line=dict(width=1.6, color=T.SERIES[i % 10])))
    f.add_vrect(x0=test_start, x1=max(r.values.index[-1] for r in results.values()),
                fillcolor=T.ACCENT, opacity=0.07, line_width=0,
                annotation=dict(text="locked test window", font=dict(color=T.MUTED, size=10)))
    f.update_yaxes(type="log", tickprefix="$")
    return f
