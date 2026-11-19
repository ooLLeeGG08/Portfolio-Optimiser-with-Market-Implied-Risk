"""Static (matplotlib) charts for the PDF report, styled for print.

Every chart has a title, axis labels with units and a legend where more than one series is
shown. The basis colours match the app's meaning (darkened for paper); orange is the
portfolio / brand colour only.
"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

from core.types import RiskBasis  # noqa: E402

TEXT = "#16161A"
MUTED = "#6B6B72"
GRID = "#E7E7EA"
ACCENT = "#E85A0C"
SERIES = ["#E85A0C", "#1F1F23", "#F2A06B", "#8C8C93", "#A63A08", "#C4C4CA", "#4A4A50",
          "#D9824A", "#B0B0B6", "#6E2A08"]
BASIS = {
    RiskBasis.OBSERVED: "#55555E",
    RiskBasis.FORECAST: "#4F5FE0",
    RiskBasis.MARKET_IMPLIED: "#0F9580",
    RiskBasis.IMPLIED_PROXY: "#C29B00",
    RiskBasis.HYPOTHETICAL: "#D2334F",
}
WIDTH_IN = 6.7

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.labelsize": 8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 7.5, "text.color": TEXT,
    "axes.titlepad": 8, "figure.dpi": 100,
})

PCT = PercentFormatter(1.0, decimals=0)
PCT1 = PercentFormatter(1.0, decimals=1)
MONEY = FuncFormatter(lambda x, _: f"${x:,.0f}")


def _png(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def _fig(h: float = 2.7, w: float = WIDTH_IN):
    return plt.subplots(figsize=(w, h))


def allocation_bar(weights: pd.Series, title: str) -> io.BytesIO:
    w = weights.sort_values()
    fig, ax = _fig(0.32 * len(w) + 0.8)
    ax.barh(w.index, w.values, color=ACCENT, height=0.6)
    for i, v in enumerate(w.values):
        ax.text(v + 0.003, i, f"{v:.1%}", va="center", fontsize=7.5, color=TEXT)
    ax.set_title(title)
    ax.set_xlabel("Weight (% of portfolio value)")
    ax.xaxis.set_major_formatter(PCT)
    ax.grid(axis="y", visible=False)
    return _png(fig)


def performance(values: pd.Series, bench: pd.Series, bench_name: str) -> io.BytesIO:
    fig, ax = _fig()
    ax.plot(values.index, values, color=ACCENT, lw=1.4, label="Portfolio")
    ax.plot(bench.index, bench, color=MUTED, lw=1.0, label=f"{bench_name} (same start value)")
    ax.fill_between(values.index, values, values.min() * 0.97, color=ACCENT, alpha=0.08)
    ax.set_title("Portfolio value vs benchmark (observed)")
    ax.set_ylabel("Value ($)")
    ax.yaxis.set_major_formatter(MONEY)
    ax.legend(loc="upper left")
    return _png(fig)


def drawdown(values: pd.Series, bench: pd.Series, bench_name: str) -> io.BytesIO:
    fig, ax = _fig(2.2)
    dd = values / values.cummax() - 1
    bd = bench / bench.cummax() - 1
    ax.fill_between(dd.index, dd, 0, color=ACCENT, alpha=0.35, lw=0, label="Portfolio")
    ax.plot(bd.index, bd, color=MUTED, lw=0.8, label=bench_name)
    ax.set_title("Drawdown from previous peak (observed)")
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(PCT)
    ax.legend(loc="lower left")
    return _png(fig)


def correlation(corr: pd.DataFrame) -> io.BytesIO:
    n = len(corr)
    fig, ax = _fig(0.36 * n + 1.1, min(WIDTH_IN, 0.5 * n + 2.2))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("pi", ["#5B6B8C", "#FFFFFF", ACCENT])
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(n), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(n), corr.index)
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color=TEXT)
    ax.set_title("Correlation of daily returns (observed)")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Correlation ρ")
    return _png(fig)


def frontier(points: pd.DataFrame, random: pd.DataFrame, strategies) -> io.BytesIO:
    fig, ax = _fig(3.2)
    sc = ax.scatter(random.volatility, random.expected_return, c=random.sharpe, cmap="Oranges", s=4,
                    alpha=0.5, label="Random long-only portfolios", lw=0)
    ax.plot(points.volatility, points.expected_return, color=TEXT, lw=1.6, label="Efficient frontier")
    markers = ["o", "D", "*", "s", "^"]
    fills = ["#FFFFFF", "#16161A", ACCENT, "#F2A06B", "#8C8C93"]
    for i, s in enumerate(strategies):
        ax.scatter([s.stats.volatility], [s.stats.expected_return], marker=markers[i % 5],
                   s=180 if markers[i % 5] == "*" else 85, color=fills[i % 5], edgecolor=TEXT, lw=1.0, zorder=6,
                   label=s.strategy.value)
    ax.set_title("Efficient frontier (in-sample estimates)")
    ax.set_xlabel("Annualised volatility (%)")
    ax.set_ylabel("Expected annual return (%)")
    ax.xaxis.set_major_formatter(PCT)
    ax.yaxis.set_major_formatter(PCT)
    fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02, label="Sharpe ratio")
    ax.legend(loc="lower right", fontsize=7)
    return _png(fig)


def return_distribution(returns: pd.Series, var95: float, var99: float) -> io.BytesIO:
    fig, ax = _fig(2.4)
    ax.hist(returns, bins=120, color=ACCENT, alpha=0.75)
    for v, lab, c in ((var95, "95% VaR", "#B07A00"), (var99, "99% VaR", BASIS[RiskBasis.HYPOTHETICAL])):
        ax.axvline(-v, color=c, ls="--", lw=1.1, label=f"{lab} = {v:.2%} loss")
    ax.set_title("Distribution of daily portfolio returns with historical VaR (observed)")
    ax.set_xlabel("Daily return (%)")
    ax.set_ylabel("Number of days")
    ax.xaxis.set_major_formatter(PCT)
    ax.legend(loc="upper left")
    return _png(fig)


def risk_contribution(table: pd.DataFrame, basis: RiskBasis, title: str) -> io.BytesIO:
    t = table.sort_values("pct_contribution", ascending=False)
    x = np.arange(len(t))
    fig, ax = _fig(2.5)
    ax.bar(x - 0.2, t["weight"], 0.4, color="#C4C4CA", label="Portfolio weight")
    ax.bar(x + 0.2, t["pct_contribution"], 0.4, color=BASIS[basis], label="Share of volatility (Euler)")
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xticks(x, t.index)
    ax.set_title(title)
    ax.set_ylabel("Share (%)")
    ax.yaxis.set_major_formatter(PCT)
    ax.legend(loc="upper right")
    return _png(fig)


def vol_comparison(rows: pd.DataFrame, horizon: int) -> io.BytesIO:
    """rows: index ticker; columns = estimate labels, plus '<label> basis', '<label> lo/hi'."""
    labels = [c for c in rows.columns if not c.endswith((" basis", " lo", " hi"))]
    fig, ax = _fig(2.9)
    x = np.arange(len(rows))
    offsets = np.linspace(-0.27, 0.27, max(len(labels), 1))
    markers = {"Historical 60d": "o", "GARCH(1,1)": "D", "LSTM": "s", "Implied (options)": "*",
               "Implied proxy (estimate)": "*"}
    for off, lab in zip(offsets, labels):
        vals = rows[lab].astype(float)
        bases = rows.get(f"{lab} basis", pd.Series([None] * len(rows), index=rows.index))
        colors = [BASIS.get(RiskBasis.coerce(b), MUTED) for b in bases]
        if f"{lab} lo" in rows:
            lo, hi = rows[f"{lab} lo"].astype(float), rows[f"{lab} hi"].astype(float)
            ax.vlines(x + off, lo, hi, color="#B0B0B6", lw=0.9)
        face = "none" if lab.startswith("Implied proxy") else colors
        ax.scatter(x + off, vals, marker=markers.get(lab, "o"), s=40 if "Implied" in lab else 26,
                   c=colors if face != "none" else "none", edgecolors=colors, lw=1, zorder=4, label=lab)
    ax.set_xticks(x, rows.index)
    ax.set_title(f"Volatility estimates by source, {horizon}-day horizon (annualised)")
    ax.set_ylabel("Annualised volatility (%)")
    ax.yaxis.set_major_formatter(PCT)
    leg = ax.legend(loc="upper left", ncol=min(len(labels), 5), fontsize=7)
    for h in leg.legend_handles:
        h.set_color(MUTED)
    return _png(fig)


def forecast_test(df: pd.DataFrame, name: str, horizon: int) -> io.BytesIO:
    fig, ax = _fig(2.9)
    ax.fill_between(df.index, df.garch_q10, df.garch_q90, color=BASIS[RiskBasis.FORECAST], alpha=0.12,
                    lw=0, label="GARCH 10–90%")
    ax.fill_between(df.index, df.lstm_q10, df.lstm_q90, color=ACCENT, alpha=0.10, lw=0,
                    label="LSTM 10–90% (conformal)")
    ax.plot(df.index, df.realized, color=TEXT, lw=1.3, label="Realised over next h days (ex post)")
    ax.plot(df.index, df.hist_60d, color=MUTED, lw=0.9, ls=":", label="Historical 60d")
    ax.plot(df.index, df.garch, color=BASIS[RiskBasis.FORECAST], lw=1.1, label="GARCH(1,1)")
    ax.plot(df.index, df.lstm, color=ACCENT, lw=1.1, label="LSTM (median)")
    ax.set_title(f"{name}: {horizon}-day volatility forecasts vs realised, locked test window")
    ax.set_ylabel("Annualised volatility (%)")
    ax.yaxis.set_major_formatter(PCT)
    ax.legend(loc="upper right", ncol=3, fontsize=6.8)
    return _png(fig)


def vix_history(frame: pd.DataFrame, vrp: pd.DataFrame | None, years: int = 5) -> io.BytesIO:
    d = frame[frame.index >= frame.index[-1] - pd.DateOffset(years=years)]
    fig, ax = _fig(2.6)
    ax.plot(d.index, d.vix, color=BASIS[RiskBasis.MARKET_IMPLIED], lw=1.0, label="VIX (implied, risk-neutral)")
    ax.plot(d.index, d.mean_252d, color=MUTED, lw=0.9, ls="--", label="252-day mean")
    if vrp is not None:
        r = vrp[vrp.index >= d.index[0]]
        ax.plot(r.index, r.realized_fwd, color=TEXT, lw=0.8, alpha=0.8,
                label="S&P 500 realised over following ~30 days (ex post)")
    ax.set_title(f"VIX and subsequently realised S&P 500 volatility, last {years} years")
    ax.set_ylabel("Annualised volatility (%)")
    ax.yaxis.set_major_formatter(PCT)
    ax.legend(loc="upper left", fontsize=7)
    return _png(fig)


def scenarios(results) -> io.BytesIO:
    price = [r for r in results if r.portfolio_return is not None]
    fig, ax = _fig(2.3)
    vals = [r.portfolio_return for r in price]
    ax.bar([r.name for r in price], vals,
           color=[BASIS[RiskBasis.HYPOTHETICAL] if v < 0 else "#2E9E6B" for v in vals], width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:+.1%}", ha="center", va="top" if v < 0 else "bottom", fontsize=7.5)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_title("Hypothetical price-shock scenarios: portfolio return (not predictions)")
    ax.set_ylabel("Portfolio return (%)")
    ax.yaxis.set_major_formatter(PCT1)
    return _png(fig)


def correlation_stress(profile: pd.DataFrame) -> io.BytesIO:
    fig, ax = _fig(2.2)
    ax.plot(profile["lambda"], profile["portfolio_volatility"], marker="o", ms=3,
            color=BASIS[RiskBasis.HYPOTHETICAL])
    ax.set_title("Portfolio volatility as correlations converge (hypothetical)")
    ax.set_xlabel("Correlation stress λ (0 = historical, 1 = all correlations equal 1)")
    ax.set_ylabel("Annualised volatility (%)")
    ax.yaxis.set_major_formatter(PCT)
    return _png(fig)


def backtest(results: dict, test_start: pd.Timestamp) -> io.BytesIO:
    fig, ax = _fig(2.8)
    start = max(r.values.index[0] for r in results.values())
    for i, (n, r) in enumerate(results.items()):
        v = r.values.loc[start:]
        ax.plot(v.index, v / v.iloc[0], lw=1.1, color=SERIES[i % 10], label=n)
    ax.axvspan(test_start, max(r.values.index[-1] for r in results.values()), color=ACCENT, alpha=0.07,
               label="Locked test window")
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    ticks = [t for t in (0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 8, 10) if lo <= t <= hi]
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:g}"))
    from matplotlib.ticker import NullFormatter
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_title("Walk-forward backtest, net of costs (growth of $1, log scale)")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left", fontsize=7)
    return _png(fig)
