"""Reusable UI elements: CSS, basis badges, KPI cards, notes, data-quality panel, formatting."""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from core.types import DataQualityReport, RiskBasis, Severity
from visualization import theme as T

CSS = f"""
<style>
.block-container {{ padding-top: 3.6rem; max-width: 1320px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
h3 {{ font-size: 0.95rem !important; font-family: {T.FONT} !important; font-weight: 600 !important; color: {T.TEXT}; }}
[data-testid="stMetricValue"] {{ font-size: clamp(1.15rem, 1.55vw, 1.6rem); font-weight: 400; font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }}
[data-testid="stMetricLabel"] p {{ color: {T.MUTED}; font-size: 0.8rem; white-space: normal; line-height: 1.25; }}
[data-testid="stMetricLabel"] {{ min-height: 2.4rem; align-items: flex-start; }}
[data-testid="stMetricLabel"] div {{ overflow: visible; white-space: normal; text-overflow: clip; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ background: {T.SURFACE}; }}
/* cards in the same row share one height */
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:has(> [class*="st-key-pi-card"]) {{ flex: 1 1 auto; }}
[class*="st-key-pi-card"] {{ height: 100%; background: {T.SURFACE}; }}

/* page header and sections */
.pi-eyebrow {{ color: {T.ACCENT}; font-size: 0.75rem; letter-spacing: 0.18em; text-transform: uppercase; font-weight: 600; }}
.pi-title {{ font-family: {T.DISPLAY_FONT}; font-weight: 400; font-size: 2.0rem; line-height: 1.12; text-transform: uppercase;
             letter-spacing: -0.01em; margin: 0.35rem 0 0.6rem 0; color: {T.TEXT}; }}
.pi-title .dim, .pi-sec .dim {{ color: {T.MUTED}; }}
/* Real heading elements (h1/h2) styled by our classes, not by Streamlit's heading defaults. */
h1.pi-title, h2.pi-sec-h {{ padding: 0 !important; scroll-margin-top: 4rem; }}
/* Streamlit styles h1 more specifically than a class; restate the title's look so it wins. */
.stMarkdown h1.pi-title {{ font-family: {T.DISPLAY_FONT} !important; font-weight: 400 !important; font-size: 2.0rem !important;
  line-height: 1.12 !important; letter-spacing: -0.01em !important; text-transform: uppercase; margin: 0.35rem 0 0.6rem 0 !important; }}
.stMarkdown .pi-hero h1.pi-title {{ font-size: 2.6rem !important; }}
@media (max-width: 760px) {{
  .stMarkdown h1.pi-title {{ font-size: 1.5rem !important; }}
  .stMarkdown .pi-hero h1.pi-title {{ font-size: 1.8rem !important; }}
}}
h2.pi-sec-h {{ margin: 0 !important; font-size: inherit !important; font-weight: inherit !important; line-height: inherit; }}
[data-testid="stHeadingWithActionElements"]:has(> .pi-title) [data-testid="stHeaderActionElements"],
[data-testid="stHeadingWithActionElements"]:has(.pi-sec-h) [data-testid="stHeaderActionElements"],
h1.pi-title [data-testid="stHeaderActionElements"], h2.pi-sec-h [data-testid="stHeaderActionElements"] {{ display: none !important; }}

/* Orange buttons: dark text (about 7:1) instead of white on orange (2.9:1). */
[data-testid^="stBaseButton-primary"], [data-testid^="stBaseButton-primary"] p,
[data-testid^="stBaseButton-primary"] span {{ color: {T.BG} !important; }}

/* A clear keyboard focus ring on links and buttons, including the sidebar navigation. */
a:focus-visible, button:focus-visible, [role="button"]:focus-visible, summary:focus-visible {{
  outline: 2px solid {T.ACCENT} !important; outline-offset: 2px; border-radius: 6px; }}

/* Sidebar footnote: its top border is the divider, with equal space above and below the line. */
.pi-side-note {{ border-top: 1px solid {T.BORDER}; margin-top: 10px; padding-top: 25px;
  color: {T.MUTED}; font-size: 0.875rem; line-height: 1.5; }}

/* Text read by screen readers but not shown (chart descriptions). */
.pi-sr {{ position: absolute !important; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden;
          clip: rect(0 0 0 0); white-space: nowrap; border: 0; }}
.pi-sub {{ color: {T.MUTED}; font-size: 0.95rem; max-width: 760px; margin-bottom: 1rem; }}
.pi-sec {{ display: flex; gap: 1.6rem; align-items: baseline; margin: 2.6rem 0 0.9rem 0;
           padding-top: 1.4rem; border-top: 1px solid {T.BORDER}; }}
.pi-sec .num {{ color: {T.MUTED}; font-size: 0.85rem; min-width: 2.2rem; font-variant-numeric: tabular-nums; }}
.pi-sec .txt {{ font-family: {T.DISPLAY_FONT}; font-weight: 400; font-size: 1.15rem; text-transform: uppercase;
                letter-spacing: 0.01em; color: {T.TEXT}; display: flex; align-items: center; flex-wrap: wrap;
                gap: 0.4rem 0.75rem; line-height: 1.2; }}
.pi-sec .txt .pi-badge {{ font-family: {T.FONT}; text-transform: none; letter-spacing: 0.02em; margin: 0; }}
.pi-sec .cap {{ color: {T.MUTED}; font-size: 0.88rem; margin-top: 0.35rem; font-family: {T.FONT}; text-transform: none; }}

/* hero */
.pi-hero {{ position: relative; overflow: hidden; border-radius: 28px; padding: 3.2rem 2.6rem 2.4rem 2.6rem;
            background:
              radial-gradient(120% 140% at 100% 0%, rgba(255,106,26,0.55) 0%, rgba(184,58,14,0.35) 22%, rgba(11,11,12,0) 52%),
              radial-gradient(80% 90% at 0% 110%, rgba(255,106,26,0.18) 0%, rgba(11,11,12,0) 55%),
              {T.SURFACE};
            border: 1px solid {T.BORDER}; margin-bottom: 1.2rem; }}
.pi-hero::after {{ content: ""; position: absolute; right: -30%; top: -70%; width: 62%; height: 190%;
                   border-radius: 50%; border: 1px solid rgba(255,138,61,0.22); transform: rotate(-18deg); pointer-events: none; }}
.pi-hero .pi-title {{ font-size: 2.6rem; max-width: 760px; }}
.pi-hero .row {{ display: flex; flex-wrap: wrap; gap: 0.7rem; margin-top: 1.4rem; position: relative; z-index: 1; }}
.pi-pill {{ display: inline-flex; flex-direction: column; padding: 0.55rem 1.1rem; border-radius: 16px; background: {T.TEXT};
            color: #0B0B0C; line-height: 1.1; }}
.pi-pill b {{ font-size: 1.15rem; font-weight: 600; }}
.pi-pill span {{ font-size: 0.72rem; color: #4A4A50; }}
.pi-pill.ghost {{ background: rgba(255,255,255,0.04); color: {T.TEXT}; border: 1px solid {T.BORDER}; }}
.pi-pill.ghost span {{ color: #C4C4CA; }}

/* badges and notes */
.pi-badge {{ display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.72rem;
            font-weight: 600; letter-spacing: 0.02em; border: 1px solid; margin-right: 6px; white-space: nowrap; }}
.pi-note {{ color: {T.MUTED}; font-size: 0.82rem; line-height: 1.45; }}
.pi-kpi-basis {{ margin-top: -0.35rem; }}

@media (max-width: 760px) {{
  .pi-hero {{ padding: 2rem 1.3rem; }}
  .pi-hero .pi-title {{ font-size: 1.8rem; }}
  .pi-title {{ font-size: 1.5rem; }}
  .pi-sec {{ gap: 0.8rem; }}
}}
</style>
"""


def inject_css() -> None:
    st.session_state["_pi_card"] = 0
    st.markdown(CSS, unsafe_allow_html=True)


def card():
    """Bordered card that stretches to the height of the tallest card in its row."""
    n = st.session_state.get("_pi_card", 0) + 1
    st.session_state["_pi_card"] = n
    return st.container(border=True, key=f"pi-card-{n}")


# --- formatting --------------------------------------------------------------------------------

def _bad(x) -> bool:
    return x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))


def pct(x, d: int = 1) -> str:
    return "—" if _bad(x) else f"{x:.{d}%}"


def signed_pct(x, d: int = 1) -> str:
    return "—" if _bad(x) else f"{x:+.{d}%}"


def money(x, d: int = 0) -> str:
    return "—" if _bad(x) else f"${x:,.{d}f}"


def num(x, d: int = 2) -> str:
    return "—" if _bad(x) else f"{x:.{d}f}"


# --- basis labelling ------------------------------------------------------------------------------

def badge(basis: RiskBasis | None, text: str | None = None) -> str:
    if basis is None:
        return f'<span class="pi-badge" style="color:{T.MUTED};border-color:{T.BORDER}">Unavailable</span>'
    c = T.BASIS_COLORS[basis]
    style = f"color:{c};border-color:{c}55;background:{c}14"
    if basis is RiskBasis.IMPLIED_PROXY:
        style += ";border-style:dashed"
    return f'<span class="pi-badge" style="{style}" title="{basis.label}">{text or T.BASIS_SHORT[basis]}</span>'


def basis_line(*bases: RiskBasis, note: str = "") -> None:
    st.markdown(" ".join(badge(b) for b in bases) + (f'<span class="pi-note">{note}</span>' if note else ""),
                unsafe_allow_html=True)


def basis_legend() -> None:
    items = " ".join(f"{badge(b)}<span class='pi-note' style='margin-right:14px'>{d}</span>" for b, d in (
        (RiskBasis.OBSERVED, "from historical data"),
        (RiskBasis.FORECAST, "model output"),
        (RiskBasis.MARKET_IMPLIED, "from option prices"),
        (RiskBasis.IMPLIED_PROXY, "estimate, not market data"),
        (RiskBasis.HYPOTHETICAL, "stress scenario, not a prediction"),
    ))
    st.markdown(items, unsafe_allow_html=True)


def _two_tone(dim: str, bright: str) -> str:
    return f'<span class="dim">{dim}</span> {bright}' if dim else bright


def header(title: str, subtitle: str = "", dim: str = "", eyebrow: str = "Portfolio Intelligence") -> None:
    """Page title: optional muted lead-in (``dim``) followed by the bright ``title``."""
    st.session_state["_pi_section"] = 0
    st.markdown(f'<div class="pi-eyebrow">{eyebrow}</div><h1 class="pi-title">{_two_tone(dim, title)}</h1>'
                + (f'<div class="pi-sub">{subtitle}</div>' if subtitle else ""), unsafe_allow_html=True)


def section(bright: str, dim: str = "", caption: str = "", basis: RiskBasis | None = None) -> None:
    """Numbered section heading (/01, /02 …) with a two-tone uppercase title and, optionally,
    the basis that applies to every figure in the section."""
    n = st.session_state.get("_pi_section", 0) + 1
    st.session_state["_pi_section"] = n
    tag = badge(basis, basis.label) if basis is not None else ""
    st.markdown(f'<div class="pi-sec"><div class="num" aria-hidden="true">/{n:02d}</div><div><div class="txt">'
                f'<h2 class="pi-sec-h">{_two_tone(dim, bright)}</h2>{tag}</div>'
                + (f'<div class="cap">{caption}</div>' if caption else "") + "</div></div>", unsafe_allow_html=True)


def hero(title: str, dim: str, subtitle: str, pills: list[tuple[str, str, bool]]) -> None:
    """Dashboard hero. ``pills``: (value, label, solid)."""
    p = "".join(f'<div class="pi-pill{"" if solid else " ghost"}"><b>{v}</b><span>{lab}</span></div>'
                for v, lab, solid in pills)
    st.session_state["_pi_section"] = 0
    st.markdown(f'<div class="pi-hero"><div class="pi-eyebrow">Dashboard</div>'
                f'<h1 class="pi-title">{_two_tone(dim, title)}</h1><div class="pi-sub">{subtitle}</div>'
                f'<div class="row">{p}</div></div>', unsafe_allow_html=True)


def kpi(label: str, value: str, basis: RiskBasis | None = None, help: str | None = None,
        delta: str | None = None) -> None:
    with card():
        st.metric(label, value, delta=delta, help=help, delta_color="off")
        # Observed is the default basis; it is stated once per section rather than on every card.
        if basis is not None and basis is not RiskBasis.OBSERVED:
            st.markdown(f'<div class="pi-kpi-basis">{badge(basis)}</div>', unsafe_allow_html=True)


def notes(items, title: str = "Assumptions and notes", expanded: bool = False) -> None:
    items = [i for i in items if i]
    if not items:
        return
    with st.expander(title, expanded=expanded):
        for i in items:
            st.markdown(f"- {i}")


def chart_description(fig) -> str:
    """A plain-text summary of a Plotly figure for screen readers: title, axes and series."""
    lay = fig.layout
    title = (lay.title.text or "Chart") if lay.title else "Chart"
    parts = [f"Chart: {title}."]
    x = lay.xaxis.title.text if lay.xaxis and lay.xaxis.title else None
    y = lay.yaxis.title.text if lay.yaxis and lay.yaxis.title else None
    if x:
        parts.append(f"Horizontal axis: {x}.")
    if y:
        parts.append(f"Vertical axis: {y}.")
    names = list(dict.fromkeys(t.name for t in fig.data if getattr(t, "name", None)))
    if names:
        parts.append("Series: " + ", ".join(names[:12]) + ("." if len(names) <= 12 else ", and more."))
    elif fig.data and getattr(fig.data[0], "labels", None) is not None:
        parts.append("Segments: " + ", ".join(map(str, list(fig.data[0].labels)[:12])) + ".")
    return " ".join(parts)


def chart(fig, description: str | None = None) -> None:
    """Interactive chart plus a screen-reader description (the figure itself is drawn as SVG)."""
    import html
    text = description or chart_description(fig)
    st.markdown(f'<p class="pi-sr">{html.escape(text)}</p>', unsafe_allow_html=True)
    st.plotly_chart(fig, width="stretch", theme=None, config={"displaylogo": False})


def quality_panel(report: DataQualityReport, expanded: bool = False) -> None:
    errs, warns = report.errors, report.warnings
    infos = [i for i in report.issues if i.severity is Severity.INFO]
    label = f"Data quality — {len(errs)} error(s), {len(warns)} warning(s)"
    with st.expander(label, expanded=expanded or bool(errs)):
        if not report.issues:
            st.markdown("No data problems detected.")
        for i in errs:
            st.error(i.message, icon=":material/block:")
        for i in warns:
            st.warning(i.message, icon=":material/warning:")
        for i in infos:
            st.info(i.message, icon=":material/info:")
        if report.aligned_start is not None:
            st.caption(f"Aligned sample: {report.aligned_start.date()} – {report.aligned_end.date()}, "
                       f"{report.n_aligned_observations:,} trading days. Missing prices are never filled.")


def sidebar_portfolio(slot, tickers, weights, pending: bool = False) -> None:
    """Draw the sidebar's 'Current portfolio' block into ``slot`` (replacing what was there).
    ``pending`` marks weights that are edited in the Builder but not applied yet."""
    with slot.container():
        st.markdown('<div class="pi-eyebrow" style="font-size:0.68rem;margin-bottom:10px">Current portfolio</div>',
                    unsafe_allow_html=True)
        st.caption(", ".join(f"{t} {w:.0%}" for t, w in zip(tickers, weights) if t))
        if pending:
            st.markdown(f'<div class="pi-note" style="color:{T.ACCENT}">Edited, not applied yet. Click '
                        f'<b>Apply portfolio</b> to analyse it.</div>', unsafe_allow_html=True)


def pct_columns(df: pd.DataFrame, cols) -> dict:
    return {c: st.column_config.NumberColumn(c, format="percent") for c in cols if c in df}
