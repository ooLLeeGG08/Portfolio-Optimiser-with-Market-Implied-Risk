"""Portfolio Intelligence — Streamlit entry point.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Portfolio Intelligence", page_icon=":material/insights:",
                   layout="wide", initial_sidebar_state="auto")

import config  # noqa: E402
from ui import components as ui  # noqa: E402
from ui import state  # noqa: E402

ui.inject_css()
st.logo(str(config.PROJECT_ROOT / "assets" / "logo.svg"), size="large",
        icon_image=str(config.PROJECT_ROOT / "assets" / "icon.svg"))

pages = {
    "Overview": [
        st.Page("views/dashboard.py", title="Dashboard", icon=":material/space_dashboard:", default=True),
        st.Page("views/builder.py", title="Portfolio Builder", icon=":material/tune:"),
    ],
    "Analysis": [
        st.Page("views/optimization.py", title="Portfolio Optimisation", icon=":material/scatter_plot:"),
        st.Page("views/forecasting.py", title="Volatility Forecasting", icon=":material/timeline:"),
        st.Page("views/implied.py", title="Market-Implied Volatility", icon=":material/candlestick_chart:"),
        st.Page("views/risk.py", title="Risk Analytics", icon=":material/shield:"),
        st.Page("views/scenarios.py", title="Scenario Analysis", icon=":material/bolt:"),
        st.Page("views/evaluation.py", title="Model Evaluation", icon=":material/fact_check:"),
    ],
    "Output": [
        st.Page("views/report.py", title="PDF Report", icon=":material/picture_as_pdf:"),
    ],
}
nav = st.navigation(pages)

with st.sidebar:
    s = state.settings()
    # A slot the Portfolio Builder redraws live while weights are being edited.
    slot = st.empty()
    st.session_state["_sidebar_portfolio_slot"] = slot
    ui.sidebar_portfolio(slot, s.tickers, s.weights)
    st.markdown('<div class="pi-side-note">Data: Yahoo Finance via yfinance (free, may be delayed). '
                'Not investment advice.</div>', unsafe_allow_html=True)

nav.run()
