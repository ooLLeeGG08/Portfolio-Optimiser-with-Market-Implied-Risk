"""PDF risk report (ReportLab). Built entirely from a ``ReportData`` produced by the analysis
pipeline — no static example content. Every figure carries its basis (observed, forecast,
market-implied, proxy estimate, hypothetical).
"""

from __future__ import annotations

import io
import math
import os
from xml.sax.saxutils import escape

import matplotlib
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import (CondPageBreak, Image, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

import config
from app_core.pipeline import ReportData
from core.types import IVSource, RiskBasis
from implied_volatility.comparison import VDI_DISCLAIMER
from implied_volatility.vix import VIX_DESCRIPTION
from portfolio.optimization import comparison_table
from reporting import report_charts as rc

# --- fonts (DejaVu Sans ships with matplotlib and covers Greek and math symbols) ------------------
_FONT_DIR = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
pdfmetrics.registerFont(TTFont("DejaVu", os.path.join(_FONT_DIR, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", os.path.join(_FONT_DIR, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Oblique", os.path.join(_FONT_DIR, "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu-Oblique",
                              boldItalic="DejaVu-Bold")

INK = colors.HexColor(rc.TEXT)
MUTED = colors.HexColor(rc.MUTED)
RULE = colors.HexColor("#DDDDE1")
ZEBRA = colors.HexColor("#F6F6F7")
ACCENT = colors.HexColor(rc.ACCENT)
PAGE_W, PAGE_H = A4
MARGIN = 0.75 * inch
FRAME_W = PAGE_W - 2 * MARGIN

S = {
    "title": ParagraphStyle("title", fontName="DejaVu-Bold", fontSize=24, leading=29, textColor=INK, spaceAfter=6),
    "subtitle": ParagraphStyle("subtitle", fontName="DejaVu", fontSize=11, leading=15, textColor=MUTED, spaceAfter=14),
    "eyebrow": ParagraphStyle("eyebrow", fontName="DejaVu-Bold", fontSize=8, leading=10, textColor=ACCENT, spaceAfter=4),
    "h1": ParagraphStyle("h1", fontName="DejaVu-Bold", fontSize=14, leading=18, textColor=INK, spaceBefore=4, spaceAfter=8,
                         keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="DejaVu-Bold", fontSize=10, leading=13, textColor=INK, spaceBefore=10, spaceAfter=4,
                         keepWithNext=1),
    "body": ParagraphStyle("body", fontName="DejaVu", fontSize=8.8, leading=12.4, textColor=INK, spaceAfter=5, alignment=TA_LEFT),
    "small": ParagraphStyle("small", fontName="DejaVu", fontSize=7.4, leading=9.8, textColor=MUTED, spaceAfter=4),
    "bullet": ParagraphStyle("bullet", fontName="DejaVu", fontSize=8.6, leading=12, textColor=INK, leftIndent=10,
                             bulletIndent=0, spaceAfter=3),
    "cell": ParagraphStyle("cell", fontName="DejaVu", fontSize=7.3, leading=9.2, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="DejaVu-Bold", fontSize=7.3, leading=9.2, textColor=INK),
    "head": ParagraphStyle("head", fontName="DejaVu-Bold", fontSize=7.1, leading=9, textColor=MUTED),
}


# --- formatting --------------------------------------------------------------------------------

def _bad(x) -> bool:
    return x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))


def pct(x, d=1):
    return "—" if _bad(x) else f"{x:.{d}%}"


def spct(x, d=1):
    return "—" if _bad(x) else f"{x:+.{d}%}"


def money(x):
    return "—" if _bad(x) else f"${x:,.0f}"


def num(x, d=2):
    return "—" if _bad(x) else f"{x:.{d}f}"


def date(x):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{pd.Timestamp(x):%d %b %Y}"


def tag(basis: RiskBasis | None) -> str:
    if basis is None:
        return '<font color="#6B6B72">[unavailable]</font>'
    return f'<font color="{rc.BASIS[basis]}"><b>[{basis.label}]</b></font>'


def P(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def E(text) -> str:
    return escape(str(text))


def bullets(items) -> list:
    return [Paragraph(i, S["bullet"], bulletText="•") for i in items if i]


def img(buf: io.BytesIO, width: float = FRAME_W) -> Image:
    from PIL import Image as PILImage
    w, h = PILImage.open(buf).size
    buf.seek(0)
    return Image(buf, width=width, height=width * h / w)


def table(rows: list[list], header: list[str], widths: list[float] | None = None,
          align_right_from: int = 1, bold_first: bool = True) -> Table:
    data = [[Paragraph(E(h), S["head"]) for h in header]]
    for r in rows:
        data.append([c if isinstance(c, Paragraph) else
                     Paragraph(E(c), S["cellb" if (j == 0 and bold_first) else "cell"]) for j, c in enumerate(r)])
    if widths is None:
        widths = [FRAME_W / len(header)] * len(header)
    t = Table(data, colWidths=widths, repeatRows=1)
    style = [
        ("LINEBELOW", (0, 0), (-1, 0), 0.9, ACCENT),
        ("LINEBELOW", (0, -1), (-1, -1), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    t.setStyle(TableStyle(style))
    for row in data[1:]:
        for j, cell in enumerate(row):
            if j >= align_right_from and isinstance(cell, Paragraph):
                cell.style = ParagraphStyle("r", parent=cell.style, alignment=2)
    return t


# --- page decoration --------------------------------------------------------------------------

class NumberedCanvas(rl_canvas.Canvas):
    """Two-pass canvas so every page can show 'Page x of y'."""

    def __init__(self, *args, report_meta: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved = []
        self.meta = report_meta or {}

    def showPage(self):
        self._saved.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        n = len(self._saved)
        for state in self._saved:
            self.__dict__.update(state)
            self._decorate(n)
            super().showPage()
        super().save()

    def _decorate(self, total: int):
        page = self._pageNumber
        self.setStrokeColor(RULE)
        self.setLineWidth(0.5)
        if page > 1:
            self.setFont("DejaVu-Bold", 7)
            self.setFillColor(ACCENT)
            self.drawString(MARGIN, PAGE_H - 0.5 * inch, "PORTFOLIO INTELLIGENCE")
            self.setFont("DejaVu", 7)
            self.setFillColor(MUTED)
            self.drawString(MARGIN + 105, PAGE_H - 0.5 * inch, "Portfolio Risk Report")
            self.drawRightString(PAGE_W - MARGIN, PAGE_H - 0.5 * inch, self.meta.get("generated", ""))
            self.line(MARGIN, PAGE_H - 0.56 * inch, PAGE_W - MARGIN, PAGE_H - 0.56 * inch)
        self.line(MARGIN, 0.6 * inch, PAGE_W - MARGIN, 0.6 * inch)
        self.setFont("DejaVu", 7)
        self.setFillColor(MUTED)
        self.drawString(MARGIN, 0.45 * inch, "Not investment advice. Figures are labelled observed, forecast, "
                                             "market-implied, proxy estimate or hypothetical.")
        self.drawRightString(PAGE_W - MARGIN, 0.45 * inch, f"Page {page} of {total}")


# --- sections ---------------------------------------------------------------------------------

SECTIONS = [
    "Executive summary", "Portfolio composition", "Portfolio performance", "Optimisation results",
    "Efficient frontier", "Historical risk metrics", "VaR and CVaR analysis", "Asset-level risk contributions",
    "Volatility forecast comparison", "Market-implied volatility analysis", "Stress-test results",
    "Model evaluation", "Methodology and assumptions", "Model limitations and data quality",
    "Data limitations and disclaimers",
]


def h1(i: int) -> list:
    return [CondPageBreak(2.2 * inch), Paragraph(f'<font color="{rc.ACCENT}">{i:02d}</font>&nbsp;&nbsp;{SECTIONS[i - 1]}', S["h1"])]


def _tail(rd: ReportData):
    return {(e.method, e.confidence): e for e in rd.pa.tail_risk}


def cover(rd: ReportData) -> list:
    mb, s = rd.mb, rd.mb.settings
    out = [Spacer(1, 0.25 * inch), P("PORTFOLIO INTELLIGENCE", "eyebrow"),
           P("Portfolio Risk Report", "title"),
           P(f"{len(mb.spec.tickers)}-asset portfolio vs {E(s.benchmark)} · "
             + ", ".join(f"{E(t)} {w:.1%}" for t, w in mb.spec.weights.items()), "subtitle")]
    q = mb.quality
    fetched = [p.fetched_at for p in mb.price_data.provenance.values()]
    facts = [
        ["Report generated", f"{rd.generated_at:%d %b %Y %H:%M} UTC"],
        ["Price data", f"{date(q.aligned_start)} – {date(mb.as_of)} ({q.n_aligned_observations:,} trading days); "
                       f"downloaded {min(fetched):%d %b %Y %H:%M}–{max(fetched):%H:%M} UTC" if fetched else "—"],
        ["Data source", E(next(iter(mb.price_data.provenance.values())).source) if mb.price_data.provenance else "—"],
        ["Currency", f"All values in {config.BASE_CURRENCY}"
                     + ("; converted unhedged: " + ", ".join(f"{E(t)} via {E(fx)}" for t, fx in mb.foreign_assets.items())
                        if mb.foreign_assets else "; all assets quoted in " + config.BASE_CURRENCY)],
        ["Risk-free rate", f"{pct(mb.risk_free_rate, 2)} — {E(mb.risk_free_source)}"],
        ["VIX observation", f"{pct(mb.vix.iloc[-1])} on {date(mb.vix.index[-1])}" if mb.vix is not None else "unavailable"],
        ["Option quotes", (", ".join(sorted({f"{pd.Timestamp(c.quote_time):%d %b %Y %H:%M} UTC" for c in rd.ib.chains.values()}))
                           if rd.ib and rd.ib.chains else "not included")],
        ["Volatility forecasts", f"as of {date(rd.fb.latest['As of'].iloc[0])}, horizon {rd.fb.horizon_days} trading days"
                                 if rd.fb else "not included"],
        ["Settings", f"{s.rebalance.value} rebalancing; risk horizon {s.horizon_days} trading days; "
                     f"{s.history_years}-year history; backtest cost {s.cost_bps:g} bps"],
    ]
    out += [table(facts, ["Item", "Value"], [1.6 * inch, FRAME_W - 1.6 * inch], align_right_from=9), Spacer(1, 14)]
    out += [P("How to read this report", "h2"),
            P("Every figure is labelled by basis. The distinction matters: observed figures describe the past, "
              "forecasts are model outputs with uncertainty, market-implied figures come from option prices and are "
              "risk-neutral, proxy estimates are not market data, and hypothetical figures are stress scenarios, "
              "not predictions.")]
    out += bullets([f"{tag(b)} {d}" for b, d in (
        (RiskBasis.OBSERVED, "computed from historical prices"),
        (RiskBasis.FORECAST, "output of a statistical or machine-learning model"),
        (RiskBasis.MARKET_IMPLIED, "derived from traded option prices (risk-neutral)"),
        (RiskBasis.IMPLIED_PROXY, "estimate used where no usable option data exist — not market data"),
        (RiskBasis.HYPOTHETICAL, "result of a stated stress scenario — not a prediction"))])
    out += [Spacer(1, 8), P("Contents", "h2")]
    entries = [P(f'<font color="{rc.ACCENT}">{i:02d}</font>&nbsp;&nbsp;{t}', "body") for i, t in enumerate(SECTIONS, 1)]
    half = (len(entries) + 1) // 2
    cols = [[entries[i], entries[i + half] if i + half < len(entries) else ""] for i in range(half)]
    toc = Table(cols, colWidths=[FRAME_W / 2] * 2)
    toc.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 0),
                             ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    out += [toc, PageBreak()]
    return out


def executive_summary(rd: ReportData) -> list:
    mb, pa, r = rd.mb, rd.pa, rd.pa.risk
    t = _tail(rd)
    h99 = t[("Historical", 0.99)]
    top = pa.contributions.table.sort_values("pct_contribution", ascending=False).iloc[0]
    top_name = pa.contributions.table["pct_contribution"].idxmax()
    items = [
        f"{tag(RiskBasis.OBSERVED)} From {money(mb.spec.initial_value)} on {date(pa.performance.start)} the simulated "
        f"portfolio grew to {money(pa.current_value)} ({spct(pa.performance.total_return)}; CAGR {pct(pa.performance.cagr)}), "
        f"against {spct(pa.benchmark_performance.total_return)} for {E(mb.settings.benchmark)}. Annualised volatility "
        f"{pct(r.annualized_volatility)}, Sharpe {num(r.sharpe)}, maximum drawdown {pct(r.max_drawdown.depth)} "
        f"({date(r.max_drawdown.peak_date)} to {date(r.max_drawdown.trough_date)}).",
        f"{tag(RiskBasis.OBSERVED)} One-day historical VaR at 99% is {pct(h99.var, 2)} ({money(h99.var_amount)}); "
        f"on the days beyond it the average loss (CVaR) is {pct(h99.cvar, 2)} ({money(h99.cvar_amount)}).",
        f"{tag(RiskBasis.OBSERVED)} {E(top_name)} carries {pct(top['pct_contribution'])} of portfolio volatility "
        f"with a {pct(top['weight'])} weight — the largest risk contributor.",
    ]
    if rd.fb is not None and "Portfolio" in rd.fb.studies:
        L = rd.fb.studies["Portfolio"].latest
        items.append(f"{tag(RiskBasis.FORECAST)} Over the next {rd.fb.horizon_days} trading days, GARCH forecasts "
                     f"{pct(L.loc['GARCH(1,1)', 'point'])} annualised volatility (10–90%: {pct(L.loc['GARCH(1,1)', 'lower'])}–"
                     f"{pct(L.loc['GARCH(1,1)', 'upper'])}) and the LSTM {pct(L.loc['LSTM', 'point'])} "
                     f"({pct(L.loc['LSTM', 'lower'])}–{pct(L.loc['LSTM', 'upper'])}), versus "
                     f"{pct(L.loc['Historical 60d', 'point'])} realised over the last 60 days.")
    if "Implied" in rd.forward:
        fr = rd.forward["Implied"]
        items.append(f"{tag(fr.covariance.basis)} Combining each asset's implied volatility with historical "
                     f"correlations gives portfolio volatility of {pct(fr.contributions.portfolio_volatility)}. "
                     f"Implied volatility is risk-neutral and normally exceeds subsequently realised volatility.")
    if rd.ib is not None and rd.ib.vix_summary is not None:
        v = rd.ib.vix_summary
        items.append(f"{tag(RiskBasis.MARKET_IMPLIED)} VIX is {pct(v.latest)} ({date(v.latest_date)}), at the "
                     f"{v.percentile_full:.0%} percentile of its history since {v.history_start:%Y}. VIX describes the "
                     f"S&amp;P 500, not this portfolio.")
    price = [x for x in rd.scenarios if x.portfolio_return is not None]
    worst = min(price, key=lambda x: x.portfolio_return)
    items.append(f"{tag(RiskBasis.HYPOTHETICAL)} The most severe standard scenario, {E(worst.name)}, would reduce the "
                 f"portfolio by {pct(-worst.portfolio_return)} ({money(-worst.asset_pnl.sum())}). This is a stress "
                 f"calculation, not a prediction.")
    if rd.fb is not None:
        best = pd.Series({n: s.metrics_all["QLIKE"].idxmin() for n, s in rd.fb.studies.items()}).value_counts()
        items.append(f"{tag(RiskBasis.FORECAST)} Out of sample, the lowest QLIKE loss came from "
                     + ", ".join(f"{E(k)} ({v} series)" for k, v in best.items())
                     + ". Differences between the leading models are rarely statistically significant; see section 12.")
    m = rd.bb.metrics_test
    items.append(f"In the locked backtest test window ({date(rd.bb.test_start)} onwards), Sharpe ratios were "
                 + ", ".join(f"{E(n)} {num(m.loc[n, 'Sharpe ratio'])}" for n in m.index)
                 + ". Two years is a short window; no strategy is claimed to outperform reliably.")
    return h1(1) + bullets(items)


def composition(rd: ReportData) -> list:
    pa, mb = rd.pa, rd.mb
    rows = []
    for t in pa.current_weights.sort_values(ascending=False).index:
        rows.append([t, pct(mb.spec.weights.get(t), 2), pct(pa.current_weights[t], 2),
                     money(pa.current_weights[t] * pa.current_value),
                     num(pa.betas.get(t)), pct(pa.contributions.table.loc[t, "volatility"])])
    out = h1(2) + [P(f"Target weights and current (drifted) weights after {mb.spec.rebalance.value} rebalancing. "
                     f"{tag(RiskBasis.OBSERVED)}"),
                   table(rows, ["Asset", "Target weight", "Current weight", "Market value", f"Beta vs {mb.settings.benchmark}",
                                "Volatility (ann.)"])]
    if mb.exclusion_message:
        out.append(P(f"<b>Note:</b> {E(mb.exclusion_message)}", "small"))
    out.append(img(rc.allocation_bar(pa.current_weights, "Current allocation (after drift)"), FRAME_W * 0.8))
    return out


def performance(rd: ReportData) -> list:
    pa, mb = rd.pa, rd.mb
    rows = [[n, date(x.start), date(x.end), spct(x.total_return), pct(x.cagr), pct(x.annualized_mean_return),
             pct(x.annualized_volatility)]
            for n, x in (("Portfolio", pa.performance), (mb.settings.benchmark, pa.benchmark_performance))]
    return h1(3) + [P(f"Simulated value path without transaction costs. {tag(RiskBasis.OBSERVED)}"),
                    table(rows, ["Series", "Start", "End", "Total return", "CAGR", "Arithmetic return (ann.)",
                                 "Volatility (ann.)"]),
                    Spacer(1, 6), img(rc.performance(pa.path.values, pa.benchmark_values, mb.settings.benchmark)),
                    img(rc.drawdown(pa.path.values, pa.benchmark_values, mb.settings.benchmark))]


def optimisation(rd: ReportData) -> list:
    oa = rd.oa
    tab = comparison_table(oa.results)
    rows = [[n, pct(r["Expected return (ann.)"]), pct(r["Excess return over risk-free (ann.)"]),
             pct(r["Volatility (ann.)"]), num(r["Sharpe ratio"])] for n, r in tab.iterrows()]
    wcols = [c for c in tab.columns if c.startswith("w[")]
    wrows = [[n] + [pct(r[c]) for c in wcols] for n, r in tab.iterrows()]
    out = h1(4) + [
        P(f"<b>In-sample estimates.</b> Weights were chosen using the same history that measures them, so these figures "
          f"overstate achievable performance. Estimation window {date(oa.inputs.start)} – {date(oa.inputs.end)}; "
          f"Ledoit–Wolf shrinkage {oa.inputs.cov.shrinkage:.3f}; risk-free rate {pct(oa.inputs.risk_free_rate, 2)}. "
          f"{tag(RiskBasis.OBSERVED)}"),
        table(rows, ["Strategy", "Expected return", "Excess over risk-free", "Volatility", "Sharpe"]),
        P("Expected return is the total annualised arithmetic mean; Sharpe = excess return / volatility.", "small"),
        P("Weights", "h2"),
        table(wrows, ["Strategy"] + [c[2:-1] for c in wcols],
              [1.5 * inch] + [(FRAME_W - 1.5 * inch) / len(wcols)] * len(wcols)),
    ]
    for e in oa.errors:
        out.append(P(f"<b>Not computed:</b> {E(e)}", "small"))
    if oa.sensitivity is not None:
        sm = oa.sensitivity.summary
        se = oa.inputs.mu_standard_error
        out += [P("Sensitivity of maximum-Sharpe weights to estimation error", "h2"),
                P(f"Expected returns were redrawn {oa.sensitivity.n_requested} times from N(μ̂, Σ/years) and the "
                  f"maximum-Sharpe portfolio re-solved each time. Wide ranges mean the optimal weights are not "
                  f"statistically well determined."),
                table([[t, pct(oa.inputs.mu[t]), pct(se[t]), pct(r.point_estimate), pct(r["mean"]), pct(r.p05), pct(r.p95)]
                       for t, r in sm.iterrows()],
                      ["Asset", "Expected return", "Std. error", "Point weight", "Mean weight", "5th pct", "95th pct"])]
    return out


def frontier(rd: ReportData) -> list:
    oa = rd.oa
    return h1(5) + [P(f"Minimum-variance portfolios for target returns between the minimum-variance return and the "
                      f"maximum feasible return (long-only, fully invested). {tag(RiskBasis.OBSERVED)}"),
                    img(rc.frontier(oa.frontier.points, oa.random_portfolios, oa.results)),
                    P(f"{len(oa.frontier.points)} frontier points; {oa.frontier.n_failed} failed solves omitted "
                      f"(not interpolated). Random portfolios are shown for context.", "small")]


def historical_risk(rd: ReportData) -> list:
    r = rd.pa.risk
    md = r.max_drawdown
    rows = [["Annualised return (arithmetic)", pct(r.annualized_return)],
            ["Annualised volatility", pct(r.annualized_volatility)],
            ["Sharpe ratio", num(r.sharpe)], ["Sortino ratio", num(r.sortino)],
            [f"Beta vs {rd.mb.settings.benchmark}", num(r.beta)],
            ["Correlation with benchmark", num(r.correlation_with_benchmark)],
            ["Maximum drawdown", f"{pct(md.depth)} (peak {date(md.peak_date)}, trough {date(md.trough_date)}, "
                                 + (f"recovered {date(md.recovery_date)})" if md.recovery_date is not None else "not recovered)")]]
    return h1(6) + [P(f"Daily portfolio returns {date(r.start)} – {date(r.end)}. {tag(RiskBasis.OBSERVED)}"),
                    table(rows, ["Measure", "Value"], [2.6 * inch, FRAME_W - 2.6 * inch]),
                    img(rc.correlation(rd.pa.correlation), FRAME_W * 0.62)]


def var_cvar(rd: ReportData) -> list:
    from app_core.pipeline import tail_risk_at_horizon
    h = rd.mb.settings.horizon_days
    rows = []
    for e in rd.pa.tail_risk:
        rows.append([e.method, f"{e.confidence:.0%}", pct(e.var, 2) if not _bad(e.var) else "not reported",
                     pct(e.cvar, 2) if not _bad(e.cvar) else "not reported", money(e.var_amount), money(e.cvar_amount)])
    hz = tail_risk_at_horizon(rd.mb, rd.pa, h)
    hrows = [[e.method, f"{e.confidence:.0%}", pct(e.var, 2), pct(e.cvar, 2), money(e.var_amount)]
             for e in hz if e.method == "Historical" or e.method.startswith("Monte")]
    cf_notes = [n for e in rd.pa.tail_risk if e.method == "Cornish-Fisher" for n in e.notes if "Not reported" in n][:1]
    t = _tail(rd)
    out = h1(7) + [
        P("<b>Interpretation.</b> VaR is a loss threshold: at 99%, the one-day loss is expected to exceed it on about one "
          "day in a hundred. It is not a worst case. CVaR (Expected Shortfall) is the average loss on the days beyond "
          f"that threshold. Losses are shown as positive numbers. {tag(RiskBasis.OBSERVED)}"),
        table(rows, ["Method", "Confidence", "1-day VaR", "1-day CVaR", "VaR ($)", "CVaR ($)"],
              [2.1 * inch, 0.8 * inch] + [(FRAME_W - 2.9 * inch) / 4] * 4),
    ]
    out += [P(E(n), "small") for n in cf_notes]
    out += [P(f"{h}-day horizon (√{h} scaling of the 1-day figures, assumes independent returns)", "h2"),
            table(hrows, ["Method", "Confidence", f"{h}-day VaR", f"{h}-day CVaR", "VaR ($)"]),
            img(rc.return_distribution(rd.pa.path.returns, t[("Historical", 0.95)].var, t[("Historical", 0.99)].var))]
    return out


def contributions(rd: ReportData) -> list:
    c = rd.pa.contributions
    rows = [[t, pct(r.weight), pct(r.volatility), pct(r.marginal_contribution), pct(r.component_contribution, 2),
             pct(r.pct_contribution)] for t, r in c.table.sort_values("pct_contribution", ascending=False).iterrows()]
    out = h1(8) + [P(f"Euler decomposition of portfolio volatility using the historical (Ledoit–Wolf) covariance. "
                     f"Components sum exactly to the portfolio volatility of {pct(c.portfolio_volatility, 2)}; a negative "
                     f"contribution means the asset hedges the rest of the portfolio. {tag(RiskBasis.OBSERVED)}"),
                   table(rows, ["Asset", "Weight", "Volatility", "Marginal ∂σ/∂w", "Component", "% of risk"]),
                   img(rc.risk_contribution(c.table, RiskBasis.OBSERVED, "Weight vs share of portfolio risk (observed)"))]
    if rd.forward:
        frows = []
        for src, fr in rd.forward.items():
            e99 = next(e for e in fr.tail_risk if e.method == "Parametric (normal)" and e.confidence == 0.99)
            top = fr.contributions.table["pct_contribution"].idxmax()
            frows.append([src, pct(fr.contributions.portfolio_volatility), pct(e99.var, 2), pct(e99.cvar, 2),
                          f"{top} ({pct(fr.contributions.table.loc[top, 'pct_contribution'])})",
                          Paragraph(tag(fr.covariance.basis), S["cell"])])
        out += [P("Forward-looking risk under alternative volatility assumptions (Σ = D·R·D)", "h2"),
                P("Each asset's volatility is replaced by the stated estimate and combined with historical correlations. "
                  "VaR is zero-mean normal. Historical correlations may not hold in future and tend to rise in stress; the "
                  "chart shows the effect of pushing them towards 1."),
                table(frows, ["Volatility source", "Portfolio vol", "1-day VaR 99%", "1-day CVaR 99%",
                              "Largest contributor", "Basis"],
                      [1.15 * inch, 0.85 * inch, 0.9 * inch, 0.95 * inch, 1.2 * inch, FRAME_W - 5.05 * inch], align_right_from=1)]
        for src, err in rd.forward_errors.items():
            out.append(P(f"{E(src)}: not included — {E(err)}.", "small"))
        base = rd.forward.get("Historical 60d") or next(iter(rd.forward.values()))
        out.append(img(rc.correlation_stress(base.stress_profile), FRAME_W * 0.85))
    return out


def vol_forecasts(rd: ReportData) -> list:
    out = h1(9)
    fb = rd.fb
    if fb is None:
        return out + [P("Volatility forecasts were not computed for this report, so GARCH and LSTM results are not "
                        "included. Generate the report with forecasts enabled to include them.")]
    lt = fb.latest
    rows = [[n, pct(r["Historical 60d"]), pct(r["GARCH(1,1)"]), f"{pct(r['GARCH 10%'])}–{pct(r['GARCH 90%'])}",
             pct(r["LSTM"]), f"{pct(r['LSTM 10%'])}–{pct(r['LSTM 90%'])}"] for n, r in lt.iterrows()]
    out += [P(f"Annualised volatility forecasts over the next {fb.horizon_days} trading days as of "
              f"{date(lt['As of'].iloc[0])}. Historical 60-day volatility is {tag(RiskBasis.OBSERVED)}; GARCH and LSTM are "
              f"{tag(RiskBasis.FORECAST)} with 10–90% intervals (GARCH: simulation; LSTM: conformally calibrated)."),
            table(rows, ["Series", "Historical 60d", "GARCH(1,1)", "GARCH 10–90%", "LSTM", "LSTM 10–90%"])]
    for n, e in fb.failures.items():
        out.append(P(f"{E(n)}: forecasting failed — {E(e)}", "small"))
    cmp = pd.DataFrame({"Historical 60d": lt["Historical 60d"], "GARCH(1,1)": lt["GARCH(1,1)"], "LSTM": lt["LSTM"]}).astype(float)
    cmp["Historical 60d basis"], cmp["GARCH(1,1) basis"], cmp["LSTM basis"] = RiskBasis.OBSERVED, RiskBasis.FORECAST, RiskBasis.FORECAST
    cmp["GARCH(1,1) lo"], cmp["GARCH(1,1) hi"] = lt["GARCH 10%"].astype(float), lt["GARCH 90%"].astype(float)
    cmp["LSTM lo"], cmp["LSTM hi"] = lt["LSTM 10%"].astype(float), lt["LSTM 90%"].astype(float)
    if rd.ib is not None:
        for t, e in rd.ib.estimates.items():
            if t in cmp.index and e.value is not None:
                lab = "Implied (options)" if e.source is IVSource.OPTIONS_CHAIN else "Implied proxy (estimate)"
                cmp.loc[t, lab] = e.value
                cmp.loc[t, f"{lab} basis"] = e.basis
    out.append(img(rc.vol_comparison(cmp, fb.horizon_days)))
    if "Portfolio" in fb.studies:
        s = fb.studies["Portfolio"]
        out.append(img(rc.forecast_test(s.forecasts, "Portfolio", fb.horizon_days)))
        g = s.garch_dev_fit
        out.append(P(f"Portfolio GARCH(1,1) (development fit, {g.dist} innovations): α = {g.params['alpha[1]']:.3f}, "
                     f"β = {g.params['beta[1]']:.3f}, persistence {g.persistence:.3f}, half-life {g.half_life_days:.0f} "
                     f"trading days, long-run volatility {pct(g.unconditional_vol)}. LSTM best epoch {s.lstm.best_epoch} of "
                     f"{len(s.lstm.history)}; conformal adjustment {s.lstm.conformal_adjustment:+.3f} (log-volatility units).",
                     "small"))
    return out


def implied(rd: ReportData) -> list:
    out = h1(10)
    ib = rd.ib
    if ib is None:
        return out + [P("Market-implied data were not loaded for this report.")]
    out.append(P("Implied volatility is <b>risk-neutral</b>: it includes a premium for volatility risk and usually exceeds "
                 "subsequently realised volatility. A gap between implied and forecast volatility is not, by itself, "
                 "evidence that options are mispriced."))
    if ib.vix_summary is not None:
        v = ib.vix_summary
        out += [P("VIX", "h2"), P(f"{E(VIX_DESCRIPTION)} {tag(RiskBasis.MARKET_IMPLIED)}"),
                table([["Latest", f"{pct(v.latest)} on {date(v.latest_date)}"],
                       ["Change over 1 / 5 / 21 days", " / ".join(f"{x * 100:+.1f} pts" for x in (v.change_1d, v.change_5d, v.change_21d))],
                       ["20 / 60 / 252-day averages", f"{pct(v.mean_20d)} / {pct(v.mean_60d)} / {pct(v.mean_252d)}"],
                       ["Percentile since " + f"{v.history_start:%Y}", f"{v.percentile_full:.0%}"],
                       ["Percentile over the last 5 years", f"{v.percentile_5y:.0%}"]],
                      ["Measure", "Value"], [2.6 * inch, FRAME_W - 2.6 * inch])]
        if ib.vrp is not None:
            vr = ib.vrp
            out.append(P(f"Since {vr.frame.index[0]:%Y}, VIX exceeded the S&amp;P 500 volatility realised over the following "
                         f"{vr.horizon_days} trading days on {vr.share_vix_above:.0%} of days, by {vr.mean_spread * 100:.1f} "
                         f"volatility points on average (median {vr.median_spread * 100:.1f}) — the variance risk premium "
                         f"(ex-post comparison)."))
        out.append(img(rc.vix_history(ib.vix_frame, ib.vrp.frame if ib.vrp is not None else None)))
    rows = []
    for t, e in ib.estimates.items():
        ch = ib.chains.get(t)
        rows.append([t, pct(e.value), Paragraph(tag(e.basis), S["cell"]),
                     "yes" if e.reliable else "limited",
                     f"{int(ch.quotes.valid.sum()):,} / {len(ch.quotes):,}" if ch is not None else "—"])
    out += [P("Option-implied volatility by asset", "h2"),
            P(f"At-the-money implied volatility interpolated to the {rd.mb.settings.horizon_days}-trading-day horizon, "
              "computed from bid/ask mid prices (Black-76 on put-call-parity forwards) after liquidity filters. Proxy "
              "estimates are labelled and are not market data."),
            table(rows, ["Asset", "Implied vol", "Basis", "Reliable", "Valid / total quotes"],
                  [0.8 * inch, 0.9 * inch, 2.6 * inch, 0.8 * inch, FRAME_W - 5.1 * inch])]
    drows = []
    for d in ib.disagreement:
        imp_label = next((k for k in d.estimates if k.startswith("Implied")), None)
        drows.append([d.ticker, pct(d.estimates.get("Historical")), pct(d.estimates.get("GARCH")), pct(d.estimates.get("LSTM")),
                      pct(d.estimates.get(imp_label)) + (" (proxy)" if imp_label and "proxy" in imp_label else ""),
                      num(d.dispersion, 3), d.band, num(d.implied_to_forecast_ratio)])
    out += [P("Volatility Disagreement Indicator", "h2"), P(f"<b>{E(VDI_DISCLAIMER)}</b>"),
            table(drows, ["Asset", "Historical", "GARCH", "LSTM", "Implied", "Dispersion", "Band", "IV / forecast"]),
            P("Dispersion = standard deviation of the logs of the available estimates (bands fixed in advance: Low < 0.10 ≤ "
              "Moderate < 0.25 ≤ High). The implied-to-forecast ratio is computed only from actual option-implied "
              "volatility, never from proxies.", "small")]
    return out


def stress(rd: ReportData) -> list:
    price = [r for r in rd.scenarios if r.portfolio_return is not None]
    regime = [r for r in rd.scenarios if r.stressed_risk is not None]
    out = h1(11) + [P(f"<b>Hypothetical scenarios — not predictions.</b> Each result shows what would happen under the "
                      f"stated assumptions. {tag(RiskBasis.HYPOTHETICAL)}"),
                    table([[r.name, spct(r.portfolio_return, 2), money(-r.asset_pnl.sum()), money(r.portfolio_value_after)]
                           for r in price], ["Price-shock scenario", "Portfolio return", "Loss", "Value after"],
                          [2.6 * inch] + [(FRAME_W - 2.6 * inch) / 3] * 3),
                    Spacer(1, 6),
                    table([[r.name, pct(r.baseline_risk.volatility), pct(r.stressed_risk.volatility),
                            pct(r.baseline_risk.var_99_1d, 2), pct(r.stressed_risk.var_99_1d, 2)] for r in regime],
                          ["Risk-regime scenario", "Vol base", "Vol stressed", "VaR 99% base", "VaR 99% stressed"],
                          [2.6 * inch] + [(FRAME_W - 2.6 * inch) / 4] * 4),
                    img(rc.scenarios(rd.scenarios))]
    seen, notes = set(), []
    for r in rd.scenarios:
        for a in r.assumptions:
            if a not in seen:
                seen.add(a)
                notes.append(E(a))
    return out + [P("Assumptions", "h2")] + bullets(notes)


def evaluation(rd: ReportData) -> list:
    bb = rd.bb
    out = h1(12) + [P("Portfolio strategies: walk-forward backtest", "h2"), P(" ".join(E(n) for n in bb.notes[:3])),
                    img(rc.backtest(bb.results, bb.test_start))]
    cols = ["CAGR", "Annualised volatility", "Sharpe ratio", "Max drawdown", "Annual turnover (one-way)", "Annual cost drag"]
    for title, m in (("Locked test window", bb.metrics_test), ("Development window", bb.metrics_development)):
        out += [P(f"{title} ({date(m['Start'].iloc[0])} – {date(m['End'].iloc[0])})", "h2"),
                table([[n] + [num(r[c]) if c == "Sharpe ratio" else pct(r[c], 2 if "cost" in c else 1) for c in cols]
                       for n, r in m.iterrows()],
                      ["Strategy", "CAGR", "Volatility", "Sharpe", "Max DD", "Turnover p.a.", "Cost p.a."],
                      [1.7 * inch] + [(FRAME_W - 1.7 * inch) / 6] * 6)]
    out.append(P(" ".join(E(n) for n in bb.notes[3:]), "small"))
    fb = rd.fb
    out.append(P("Volatility forecasts: out-of-sample accuracy", "h2"))
    if fb is None:
        return out + [P("Volatility forecasts were not computed for this report.")]
    rows = []
    for n, s in fb.studies.items():
        m = s.metrics_all
        dm = s.dm_tests.loc["LSTM vs GARCH(1,1)"]
        better = "LSTM" if dm["QLIKE: mean loss diff"] < 0 else "GARCH"
        rows.append([n, m["QLIKE"].idxmin(), m["RMSE"].idxmin(), pct(m.loc["GARCH(1,1)", "Interval coverage"]),
                     pct(m.loc["LSTM", "Interval coverage"]),
                     f"{dm['QLIKE: p-value']:.3f} ({better} lower loss)"])
    out += [P("Each series is evaluated on its locked test window (never used for fitting, tuning or calibration). "
              "QLIKE is the preferred loss for volatility forecasts; coverage should be near the nominal 80%."),
            table(rows, ["Series", "Lowest QLIKE", "Lowest RMSE", "GARCH coverage", "LSTM coverage",
                         "LSTM vs GARCH DM p-value"],
                  [0.8 * inch, 1.05 * inch, 1.05 * inch, 0.9 * inch, 0.9 * inch, FRAME_W - 4.7 * inch])]
    if "Portfolio" in fb.studies:
        s = fb.studies["Portfolio"]
        for title, m in (("Portfolio — all forecast dates", s.metrics_all),
                         (f"Portfolio — non-overlapping subsample (every {s.horizon}th date)", s.metrics_non_overlapping)):
            out += [P(f"{title} (n = {int(m['n'].iloc[0])})", "h2"),
                    table([[n, num(r.MAE, 4), num(r.RMSE, 4), num(r.Bias, 4), num(r.QLIKE, 3),
                            pct(r["Directional accuracy"]), pct(r.get("Interval coverage"))] for n, r in m.iterrows()],
                          ["Model", "MAE", "RMSE", "Bias", "QLIKE", "Direction", "Coverage"])]
        out.append(P(f"Consecutive {s.horizon}-day targets overlap by {s.horizon - 1} days, so errors are autocorrelated and "
                     f"the effective sample is about one observation per horizon. Diebold–Mariano tests use a HAC variance "
                     f"with {s.horizon - 1} lags and the Harvey–Leybourne–Newbold correction.", "small"))
    return out


def methodology(rd: ReportData) -> list:
    s = rd.mb.settings
    items = [
        "<b>Conventions.</b> 252 trading days per year; volatility annualised by √252 and mean returns by ×252 (arithmetic). "
        "Simple returns aggregate across assets; log returns are used for volatility modelling. Losses are positive numbers.",
        f"<b>Data.</b> Split- and dividend-adjusted daily closes; assets aligned on common trading dates with no filling of "
        f"missing prices; bars for an unfinished trading day are excluded. Prices quoted in other currencies are "
        f"converted to {config.BASE_CURRENCY} with daily exchange rates (unhedged). History: {s.history_years} years "
        f"({s.forecast_history_years} years for volatility models).",
        "<b>Optimisation.</b> Ledoit–Wolf covariance; arithmetic mean returns; long-only, fully invested; maximum Sharpe "
        "(excess return / volatility), minimum variance, custom bounds, efficient frontier via minimum variance at target returns.",
        "<b>Risk.</b> Historical, parametric normal, Cornish–Fisher (withheld where the expansion is invalid) and Student-t Monte "
        "Carlo VaR/CVaR; multi-day figures scaled by √h. Euler risk contributions sum exactly to portfolio volatility.",
        "<b>Forward-looking covariance.</b> Σ = D·R·D: per-asset volatility estimates with historical correlations, all at the "
        "same horizon; the result inherits the least-observed input's basis; correlation stress R → (1−λ)R + λ11ᵀ.",
        f"<b>Volatility target.</b> Annualised realised volatility over the next {s.horizon_days} trading days, "
        f"√(252/h · Σ r²) of daily log returns. Forecasts at t use data up to t only.",
        "<b>GARCH(1,1).</b> Normal or Student-t innovations chosen by BIC on development data; expanding-window refits every "
        "21 days; 10–90% intervals from 500 simulated paths (shock uncertainty only).",
        "<b>LSTM.</b> One layer of 32 units on 60-day sequences of backward-looking features; chronological train/validation/"
        "calibration split with purge gaps; training-only scaling; quantile outputs with conformal calibration; early "
        "stopping and checkpointing; hyper-parameters fixed in advance.",
        "<b>Evaluation.</b> Locked final test window (last ~2 years); MAE, RMSE, bias, QLIKE, directional accuracy, interval "
        "coverage; non-overlapping subsample; Diebold–Mariano with HAC variance. Walk-forward strategy backtest with monthly "
        f"rebalancing, {s.backtest_lookback}-day estimation window and {s.cost_bps:g} bps costs on traded notional.",
        "<b>Market-implied volatility.</b> VIX (S&amp;P 500, 30 calendar days, risk-neutral); option-chain IV from mid quotes "
        "after liquidity filters, interpolated in total variance to the horizon; proxy = asset realised vol × (VIX / S&amp;P "
        "realised vol) where no usable options exist.",
        "<b>Stress tests.</b> Market moves via historical betas; rate shocks via estimated sensitivity to the 10-year yield; "
        "single-asset and custom shocks; volatility and correlation regimes.",
        "The full methodology, with formulas and the tests that verify them, is in docs/METHODOLOGY.md.",
    ]
    return h1(13) + bullets(items)


def limitations(rd: ReportData) -> list:
    mb, q = rd.mb, rd.mb.quality
    out = h1(14) + [P("Data quality", "h2")]
    if q.issues:
        out.append(table([[i.ticker or "—", i.severity.value, Paragraph(E(i.message), S["cell"])] for i in q.issues],
                         ["Asset", "Severity", "Issue"], [0.8 * inch, 0.8 * inch, FRAME_W - 1.6 * inch], align_right_from=9))
    else:
        out.append(P("No price-data problems were detected for the selected assets and window."))
    out.append(P(f"Aligned sample {date(q.aligned_start)} – {date(q.aligned_end)}: {q.n_aligned_observations:,} trading "
                 f"days; {q.rows_dropped_in_alignment} date(s) dropped because at least one asset had no price (never filled)."))
    prov = [[t, E(p.source), f"{p.fetched_at:%Y-%m-%d %H:%M} UTC", date(p.first_date), date(p.last_date),
             E(p.currency or "—") + (f" → {config.BASE_CURRENCY} ({E(p.fx_ticker)})" if p.fx_ticker else "")]
            for t, p in mb.price_data.provenance.items()]
    out += [P("Series provenance", "h2"), table(prov, ["Series", "Source", "Downloaded", "First obs.", "Last obs.", "Currency"],
                                                [0.7 * inch, 1.45 * inch, 1.25 * inch, 0.85 * inch, 0.85 * inch,
                                                 FRAME_W - 5.1 * inch], align_right_from=9)]
    for e in mb.errors:
        out.append(P(f"<b>Note:</b> {E(e)}", "small"))
    if rd.ib is not None and rd.ib.chains:
        ex = pd.concat({t: c.exclusions for t, c in rd.ib.chains.items()}, axis=1).fillna(0).astype(int)
        ex["Total"] = ex.sum(axis=1)
        out += [P("Option quotes excluded by the liquidity filters", "h2"),
                table([[E(r)] + [f"{v:,}" for v in row] for r, row in ex.sort_values("Total", ascending=False).iterrows()],
                      ["Reason"] + list(ex.columns), [1.9 * inch] + [(FRAME_W - 1.9 * inch) / len(ex.columns)] * len(ex.columns))]
    model = []
    if any(_bad(e.var) for e in rd.pa.tail_risk if e.method == "Cornish-Fisher"):
        model.append("Cornish–Fisher VaR was withheld: the sample's skew and kurtosis are outside the range where the "
                     "expansion is a valid quantile function.")
    model.append(f"Historical 99% one-day VaR rests on about {int(round(len(rd.pa.path.returns) * 0.01))} tail observations.")
    if rd.fb is not None:
        under = [n for n, s in rd.fb.studies.items() if s.metrics_all.loc["LSTM", "Interval coverage"] < 0.70]
        if under:
            model.append(f"LSTM intervals under-covered (below 70% against 80% nominal) for {', '.join(under)}: the calibration "
                         f"period's volatility regime differed from the test window's, which conformal calibration cannot "
                         f"guarantee against for time series.")
        early = [n for n, s in rd.fb.studies.items() if s.lstm.best_epoch < 10]
        if early:
            model.append(f"The LSTM stopped training early (best epoch below 10) for {', '.join(early)}, so it adds little "
                         f"beyond volatility persistence for those series.")
        model.append("GARCH intervals include shock uncertainty but not parameter-estimation uncertainty.")
    if rd.ib is not None:
        n_proxy = sum(e.source is IVSource.PROXY for e in rd.ib.estimates.values())
        if n_proxy:
            model.append(f"{n_proxy} asset(s) use an implied-volatility proxy (an estimate, not market data).")
        model.append("Option-chain implied volatilities are a snapshot at one point in time; Black-76 is a European model "
                     "applied to American options (smallest error for the out-of-the-money options used).")
    if mb.foreign_assets:
        model.append(f"{', '.join(mb.foreign_assets)} were converted to {config.BASE_CURRENCY} unhedged, so their figures "
                     f"include exchange-rate risk. Exchange rates are not fixed at each exchange's closing time, which adds "
                     f"some noise to daily returns and correlations.")
    fb_notes = [n for r in rd.bb.results.values() for n in r.notes]
    model += [E(n) for n in fb_notes]
    model += [
        "Mean–variance results depend on expected-return estimates whose standard errors are several percentage points a year.",
        "Correlations are estimated from history and are unstable; they tend to rise in market stress.",
        "√h scaling and normal or Student-t VaR models are approximations; volatility clusters.",
        "Realised volatility is itself a noisy estimate of true volatility, which limits forecast evaluation.",
    ]
    return out + [P("Model limitations observed in this analysis", "h2")] + bullets(model)


def disclaimers(rd: ReportData) -> list:
    items = [
        "Prices come from a free data source (Yahoo Finance via yfinance) and may be delayed, revised or incomplete. Adjusted "
        "histories are revised retroactively when dividends or splits occur.",
        "Survivorship bias: the asset list was chosen today from assets that still exist; delisted assets are largely absent.",
        "Single-stock implied volatility is available only as a current snapshot, so it has no history in this report.",
        "Backtest and forecast results are specific to the periods shown; two-year test windows are short.",
        "Hypothetical scenarios are not predictions. Forecasts are uncertain and are shown with intervals where feasible.",
        "Implied volatility is risk-neutral and contains risk premia; it is not an unbiased forecast of realised volatility.",
        "This report is generated automatically for analytical and educational purposes. It is not investment advice, an "
        "offer or a recommendation to buy or sell any security.",
    ]
    return h1(15) + bullets(items)


def generate_pdf(rd: ReportData) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=0.8 * inch,
                            bottomMargin=0.85 * inch, title="Portfolio Risk Report", author="Portfolio Intelligence",
                            subject=f"Portfolio risk report generated {rd.generated_at:%Y-%m-%d %H:%M} UTC")
    story = []
    for part in (cover, executive_summary, composition, performance, optimisation, frontier, historical_risk, var_cvar,
                 contributions, vol_forecasts, implied, stress, evaluation, methodology, limitations, disclaimers):
        story += part(rd)
    meta = {"generated": f"Generated {rd.generated_at:%d %b %Y %H:%M} UTC · data to {rd.mb.as_of:%d %b %Y}"}
    doc.build(story, canvasmaker=lambda *a, **k: NumberedCanvas(*a, report_meta=meta, **k))
    return buf.getvalue()
