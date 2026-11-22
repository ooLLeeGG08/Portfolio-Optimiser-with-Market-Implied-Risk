"""Design tokens and the Plotly template shared by every interactive chart."""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

from core.types import RiskBasis

BG = "#0B0B0C"
SURFACE = "#141415"
SURFACE_2 = "#1B1B1D"
BORDER = "#2A2A2D"
GRID = "#1E1E21"
TEXT = "#F2F2F2"
MUTED = "#8C8C93"
ACCENT = "#FF6A1A"  # brand accent: UI chrome and the portfolio's own series — never a basis colour
ACCENT_SOFT = "#FF8A3D"
ACCENT_DEEP = "#B83A0E"
POSITIVE = "#4CC38A"
NEGATIVE = "#FF4D6D"
NEUTRAL_BAR = "#3A3A3F"

# Categorical palette for assets and strategies: orange family plus neutrals. It avoids the basis
# hues below, so colour never suggests a figure is observed, forecast, implied, proxy or hypothetical.
SERIES = ["#FF6A1A", "#F2F2F2", "#FFA463", "#8C8C93", "#C9460F", "#FFD2B0", "#5E5E64",
          "#E88A4E", "#B8B8BE", "#7A2E0B"]

BASIS_COLORS = {
    RiskBasis.OBSERVED: "#D9D9DE",
    RiskBasis.FORECAST: "#8E9BFF",
    RiskBasis.MARKET_IMPLIED: "#35C9B0",
    RiskBasis.IMPLIED_PROXY: "#F2D04B",
    RiskBasis.HYPOTHETICAL: "#FF4D6D",
}
BASIS_SHORT = {
    RiskBasis.OBSERVED: "Observed",
    RiskBasis.FORECAST: "Forecast",
    RiskBasis.MARKET_IMPLIED: "Market-implied",
    RiskBasis.IMPLIED_PROXY: "Proxy estimate",
    RiskBasis.HYPOTHETICAL: "Hypothetical",
}

FONT = "Manrope, -apple-system, Segoe UI, Roboto, sans-serif"
DISPLAY_FONT = "Unbounded, Manrope, sans-serif"

pio.templates["portfolio_optimizer"] = go.layout.Template(layout=go.Layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family=FONT, color=TEXT, size=12),
    title=dict(font=dict(size=13, color=TEXT, family=FONT, weight=600), x=0, xanchor="left", y=0.97),
    colorway=SERIES,
    xaxis=dict(automargin=True, gridcolor=GRID, zerolinecolor=BORDER, linecolor=BORDER, title_font=dict(color=MUTED, size=12),
               tickfont=dict(color=MUTED)),
    yaxis=dict(automargin=True, gridcolor=GRID, zerolinecolor=BORDER, linecolor=BORDER, title_font=dict(color=MUTED, size=12),
               tickfont=dict(color=MUTED)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED, size=11), orientation="h",
                yanchor="bottom", y=1.0, xanchor="right", x=1),
    hoverlabel=dict(bgcolor=SURFACE_2, bordercolor=BORDER, font=dict(family=FONT, color=TEXT)),
    margin=dict(l=12, r=12, t=64, b=16),
))
pio.templates.default = "portfolio_optimizer"
