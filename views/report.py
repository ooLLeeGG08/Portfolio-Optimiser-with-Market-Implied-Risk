import streamlit as st

import config

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
                             help=f"Computed now if needed ({config.forecast_runtime_text(len(mb.spec.tickers))}).")
        inc_iv = st.checkbox("Market-implied volatility (VIX and option chains)", value=True,
                             help=f"Downloaded now if needed (about {config.OPTION_FETCH_SECONDS} seconds).")
        st.caption(f"Forecasts: {'ready' if fb is not None else 'not computed yet'} · "
                   f"Option data: {'ready' if ib is not None else 'not loaded yet'}")
        go = st.button("Generate PDF report", type="primary", icon=":material/picture_as_pdf:", width="stretch")

if go:
    # Record the request in the URL and end this run first, so a reconnect during the long job
    # continues it by itself (the browser only receives URL changes when a run ends).
    state.mark_pending("rp", f"{int(inc_fc)}{int(inc_iv)}")
    st.rerun()
job = state.pending("rp")
if job is not None:
    inc_fc, inc_iv = job[:1] == "1", job[1:2] == "1"
    st.info("Building the report. If the connection resets, it continues by itself.", icon=":material/autorenew:")
    if inc_fc and fb is None:
        fb = state.run_forecasts()
    if inc_iv:
        ib = state.implied(fetch=True)
    with st.spinner("Building report…"):
        oa = state.optimization(0.0, 0.4, "max_sharpe", None)
        rd = pl.assemble_report(mb, pa, oa, state.backtests(), fb if inc_fc else None, ib if inc_iv else None)
        st.session_state["report_pdf"] = (generate_pdf(rd), rd.generated_at)
    state.clear_pending("rp")

if "report_pdf" in st.session_state:
    pdf, at = st.session_state["report_pdf"]
    st.success(f"Report generated {at:%d %b %Y %H:%M} UTC ({len(pdf) // 1024:,} KB).", icon=":material/check:")
    # The PDF travels inside the page (a data link), so the download never needs a second request to
    # the server. On serverless hosting that request can reach another instance, which does not hold
    # the file and answers "File not found" (Streamlit's download_button works that way).
    ui.download_link(pdf, f"portfolio_risk_report_{at:%Y%m%d_%H%M}.pdf", "application/pdf", "Download PDF")
    st.caption("The report reflects the portfolio and data at the time it was generated. Regenerate after changing "
               "the portfolio.")
