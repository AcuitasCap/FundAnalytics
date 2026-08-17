"""Portfolio Quality page selectors and orchestration."""

import datetime as dt

import streamlit as st

from core.dates import month_year_to_last_day
from core.fund_catalog import fetch_categories, fetch_funds_for_categories
from core.navigation import home_button
from .compute import QUARTILE_MODE, ROC_MODE, compute_quality_analysis
from .data import fetch_portfolio_raw, fetch_quality_bucket_rows, load_stock_roe_roce
from .display import display_quality_analysis


def _category_and_fund_selector():
    categories = fetch_categories()
    if not categories:
        st.warning("No categories found in fund_master.")
        return [], [], None, None
    st.subheader("1. Select categories")
    selected_categories = []
    cols = st.columns(min(4, len(categories)))
    for index, category in enumerate(categories):
        if cols[index % len(cols)].checkbox(category, value=False, key=f"pq_cat_{category}"):
            selected_categories.append(category)
    if not selected_categories:
        st.info("Please select at least one category.")
        return [], [], None, None
    st.subheader("2. Select funds (universe)")
    funds = fetch_funds_for_categories(selected_categories)
    if funds.empty:
        st.warning("No funds found for the selected categories.")
        return [], [], None, None
    options = {f"{row['fund_name']} ({row['category_name']})": row["fund_id"] for _, row in funds.iterrows()}
    labels = sorted(options)
    all_option = "All funds in selected categories"
    selected = st.multiselect("Select funds for analysis", [all_option] + labels,
                              default=[all_option], key="pq_funds_multiselect")
    selected_labels = labels if all_option in selected else [label for label in selected if label != all_option]
    if not selected_labels:
        st.info("Please select at least one fund.")
        return [], [], None, None
    st.subheader("3. Focus fund")
    focus_label = st.selectbox("Focus fund", selected_labels, index=0, key="pq_focus_fund")
    return [options[label] for label in selected_labels], selected_labels, options[focus_label], focus_label


def _period_selector():
    st.subheader("4. Select period (March / September portfolios only)")
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
    home_button()
    st.header("Portfolio quality – return on capital & quality buckets")
    selected_ids, selected_labels, focus_id, focus_label = _category_and_fund_selector()
    if not selected_ids:
        return
    with st.form("pq_filters"):
        start_date, end_date = _period_selector()
        st.subheader("5. Choose analysis")
        mode = st.radio("What do you want to analyse?", [ROC_MODE, QUARTILE_MODE], horizontal=True, key="pq_view_mode")
        apply_filters = st.form_submit_button("Update", type="primary")
    applied = st.session_state.get("pq_filters_applied", False)
    if apply_filters:
        applied = True
        st.session_state["pq_filters_applied"] = True
    if start_date > end_date:
        st.error("Start date must be earlier than end date.")
        return
    if not applied:
        st.info("Set period and analysis mode, then click **Update** above.")
        return
    context = {"selected_fund_ids": selected_ids, "selected_fund_labels": selected_labels,
               "focus_fund_id": focus_id, "focus_fund_label": focus_label}
    if mode == ROC_MODE:
        with st.form("pq_mode1"):
            st.subheader("6. Segment")
            context["segment_choice"] = st.radio("Show metrics for:", ["Financials", "Non-financials", "Total"], horizontal=True, key="pq_segment")
            st.subheader("7. Comparison mode")
            context["comparison_mode"] = st.radio("Compare focus fund against:", ["Universe median", "Individual funds"], horizontal=True, key="pq_comparison_mode")
            run = st.form_submit_button("Show results", type="primary")
    else:
        with st.form("pq_mode2"):
            st.markdown("**Analysis:** Quality quartile exposures for the focus fund (total domestic equities).")
            run = st.form_submit_button("Show results", type="primary")
        context["segment_choice"] = "Total"
    if not run:
        return
    with st.spinner("Computing portfolio fundamentals..." if mode == ROC_MODE else f"Computing quality bucket exposures for {focus_label}..."):
        portfolio_data = fetch_portfolio_raw(selected_ids, start_date, end_date)
        roe_roce_data = load_stock_roe_roce()
        bucket_data = fetch_quality_bucket_rows(focus_id, start_date, end_date) if mode == QUARTILE_MODE else None
        result = compute_quality_analysis(mode, portfolio_data, roe_roce_data, bucket_data, context)
    display_quality_analysis(mode, result, context)
