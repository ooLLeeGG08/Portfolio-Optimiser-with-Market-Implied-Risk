import streamlit as st

from app_core import pipeline as pl
from reporting.pdf_generator import SECTIONS, generate_pdf
from ui import components as ui
from ui import state

mb = state.market()
pa = state.portfolio()
ui.header("report", dim="PDF risk",
          subtitle="A research-style PDF generated from this portfolio's live results, with observation dates, "
                   "methodology, model limitations and data quality.")

fb, ib = state.forecasts(), state.implied(fetch=False)
ui.section("contents", "Report")
left, right = st.columns([3, 2])
with left:
    st.markdown("\n".join(f"{i:02d}. {t}" for i, t in enumerate(SECTIONS, 1)))
with right:
    with st.container(border=True):
        st.markdown("**Include**")
        inc_fc = st.checkbox("GARCH and LSTM volatility forecasts", value=True,
                             help="Computed now if needed (about 2 minutes for 7 assets).")
        inc_iv = st.checkbox("Market-implied volatility (VIX and option chains)", value=True,
                             help="Downloaded now if needed (about 25 seconds).")
        st.caption(f"Forecasts: {'ready' if fb is not None else 'not computed yet'} · "
                   f"Option data: {'ready' if ib is not None else 'not loaded yet'}")
        go = st.button("Generate PDF report", type="primary", icon=":material/picture_as_pdf:", width="stretch")

if go:
    if inc_fc and fb is None:
        fb = state.run_forecasts()
    if inc_iv:
        ib = state.implied(fetch=True)
    with st.spinner("Building report…"):
        oa = state.optimization(0.0, 0.4, "max_sharpe", None)
        rd = pl.assemble_report(mb, pa, oa, state.backtests(), fb if inc_fc else None, ib if inc_iv else None)
        st.session_state["report_pdf"] = (generate_pdf(rd), rd.generated_at)

if "report_pdf" in st.session_state:
    pdf, at = st.session_state["report_pdf"]
    st.success(f"Report generated {at:%d %b %Y %H:%M} UTC ({len(pdf) // 1024:,} KB).", icon=":material/check:")
    st.download_button("Download PDF", data=pdf, file_name=f"portfolio_risk_report_{at:%Y%m%d_%H%M}.pdf",
                       mime="application/pdf", icon=":material/download:", type="primary")
    st.caption("The report reflects the portfolio and data at the time it was generated. Regenerate after changing "
               "the portfolio.")
