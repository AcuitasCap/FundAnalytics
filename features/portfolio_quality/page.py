"""Portfolio Quality page selectors and orchestration."""

import datetime as dt

import streamlit as st

from core.dates import month_year_to_last_day
from .compute import QUARTILE_MODE, ROC_MODE, compute_quality_analysis
from .data import fetch_portfolio_raw, fetch_quality_bucket_rows, load_stock_roe_roce
from .display import display_quality_analysis


def _period_selector():
    st.subheader("Select period (March / September portfolios only)")
    years = list(range(dt.date.today().year - 15, dt.date.today().year + 1))
    months = [3, 9]
    left, right = st.columns(2)
    with left:
        start_year = st.selectbox("Start year", years, index=0, key="pq_start_year")
        start_month = st.selectbox("Start month", months, index=0, key="pq_start_month",
                                   format_func=lambda month: "Mar" if month == 3 else "Sep")
    with right:
        end_year = st.selectbox("End year", years, index=len(years)-1, key="pq_end_year")
        end_month = st.selectbox("End month", months, index=1, key="pq_end_month",
                                 format_func=lambda month: "Mar" if month == 3 else "Sep")
    return month_year_to_last_day(start_year, start_month), month_year_to_last_day(end_year, end_month)


def portfolio_quality_page():
    """Compatibility entry point for callers of the former standalone page."""
    from features.fundamental_attributes.page import fundamental_attributes_page

    fundamental_attributes_page()


def render_quality_section(shared_context):
    """Reuse existing Quality mathematics with the page's common fund selection."""
    context = dict(shared_context)
    selected_ids = context["selected_fund_ids"]
    focus_id = context["focus_fund_id"]
    signature = (tuple(sorted(selected_ids)), focus_id)
    mode = st.radio("What do you want to analyse?", [ROC_MODE, QUARTILE_MODE],
                    horizontal=True, key="pq_view_mode")
    with st.form("fa_quality_controls"):
        start_date, end_date = _period_selector()
        if mode == ROC_MODE:
            context["segment_choice"] = st.radio("Show metrics for:", ["Financials", "Non-financials", "Total"], horizontal=True, key="pq_segment")
            context["comparison_mode"] = st.radio("Compare focus fund against:", ["Universe median", "Individual funds"], horizontal=True, key="pq_comparison_mode")
        else:
            st.markdown("**Analysis:** Quality quartile exposures for the focus fund (total domestic equities).")
            context["segment_choice"] = "Total"
        run = st.form_submit_button("Update quality", type="primary")
    if run:
        st.session_state.pop("fa_quality_result", None)
        if start_date > end_date:
            st.error("Start date must be on or before end date.")
            return
        try:
            with st.spinner("Computing portfolio quality..."):
                portfolio_data = fetch_portfolio_raw(selected_ids, start_date, end_date)
                roe_roce_data = load_stock_roe_roce()
                bucket_data = fetch_quality_bucket_rows(focus_id, start_date, end_date) if mode == QUARTILE_MODE else None
                result = compute_quality_analysis(mode, portfolio_data, roe_roce_data, bucket_data, context)
            st.session_state["fa_quality_result"] = {"signature": signature, "result": result, "mode": mode,
                                                     "context": context, "start": start_date, "end": end_date}
        except Exception as error:
            st.error(f"Unable to load portfolio quality: {error}")
            return
    saved = st.session_state.get("fa_quality_result")
    if saved and saved["signature"] == signature and saved["mode"] == mode:
        st.caption(f"Displayed results: {saved['start']:%b %Y} to {saved['end']:%b %Y}")
        display_quality_analysis(saved["mode"], saved["result"], saved["context"])
    else:
        st.info("Choose settings and click **Update quality**.")
