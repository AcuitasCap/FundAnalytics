"""Portfolio explorer selectors and orchestration."""

import datetime as dt

import streamlit as st

from core.dates import month_year_to_last_day
from core.fund_catalog import fetch_categories, fetch_funds_for_categories
from core.navigation import home_button
from .compute import compute_active_share
from .data import fetch_multi_fund_portfolios, fetch_portfolio_view_data
from .display import display_portfolio_result


def _select_categories(key_prefix: str):
    categories = fetch_categories()
    if not categories:
        st.warning("No categories found.")
        return []
    selected = []
    columns = st.columns(min(4, len(categories)))
    for index, category in enumerate(categories):
        if columns[index % len(columns)].checkbox(category, value=False, key=f"{key_prefix}_cat_{category}"):
            selected.append(category)
    return selected


def _period_selector(key_prefix: str):
    current_year = dt.date.today().year
    years, months = list(range(current_year - 15, current_year + 1)), list(range(1, 13))
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    left, right = st.columns(2)
    with left:
        start_year = st.selectbox("Start year", years, index=0, key=f"{key_prefix}_start_year")
        start_month = st.selectbox("Start month", months, index=0, key=f"{key_prefix}_start_month", format_func=lambda month: names[month - 1])
    with right:
        end_year = st.selectbox("End year", years, index=len(years) - 1, key=f"{key_prefix}_end_year")
        end_month = st.selectbox("End month", months, index=11, key=f"{key_prefix}_end_month", format_func=lambda month: names[month - 1])
    return month_year_to_last_day(start_year, start_month), month_year_to_last_day(end_year, end_month)


def _portfolio_view():
    st.subheader("1. Select categories")
    categories = _select_categories("port")
    if not categories:
        st.info("Select at least one category to continue.")
        return
    funds = fetch_funds_for_categories(categories)
    if funds.empty:
        st.warning("No funds found for the selected categories.")
        return
    st.subheader("2. Select fund")
    category_column = "category_name" if "category_name" in funds.columns else "category"
    options = {f"{row['fund_name']} ({row[category_column]})": row["fund_id"] for _, row in funds.iterrows()}
    label = st.selectbox("Fund", list(options))
    fund_id = int(options[label])
    st.subheader("3. Select period")
    start_date, end_date = _period_selector("port")
    if start_date > end_date:
        st.error("Start date must be earlier than end date.")
        return
    st.subheader("4. Frequency")
    frequency = st.radio("Aggregation", ["Monthly", "Quarterly", "Yearly"], horizontal=True)
    if st.button("Show portfolio"):
        with st.spinner("Loading portfolio..."):
            result = fetch_portfolio_view_data(fund_id, start_date, end_date)
        display_portfolio_result("View portfolio", result, {"fund_id": fund_id, "start_date": start_date, "end_date": end_date, "frequency": frequency})


def _active_share():
    st.subheader("Active share between two portfolios")
    st.subheader("1. Select categories")
    categories = _select_categories("as")
    if not categories:
        st.info("Select at least one category to continue.")
        return
    funds = fetch_funds_for_categories(categories)
    if funds.empty:
        st.warning("No funds found for the selected categories.")
        return
    category_column = "category_name" if "category_name" in funds.columns else "category"
    labels = [f"{row['fund_name']} ({row[category_column]})" for _, row in funds.iterrows()]
    label_to_id = {f"{row['fund_name']} ({row[category_column]})": row["fund_id"] for _, row in funds.iterrows()}
    st.subheader("2. Select funds for each portfolio")
    left, right = st.columns(2)
    with left:
        st.markdown("**Portfolio A – Funds**")
        selected_a = st.multiselect("Funds A", labels, default=[], key="as_funds_A")
    with right:
        st.markdown("**Portfolio B – Funds**")
        selected_b = st.multiselect("Funds B", labels, default=[], key="as_funds_B")
    if not selected_a or not selected_b:
        st.info("Select at least one fund for each portfolio to continue.")
        return
    st.subheader("3. Set fund proportions (%) in each portfolio")
    proportions_a, proportions_b = {}, {}
    left, right = st.columns(2)
    for column, labels_for_portfolio, target, title, prefix in ((left, selected_a, proportions_a, "Portfolio A composition", "asA"), (right, selected_b, proportions_b, "Portfolio B composition", "asB")):
        with column:
            st.markdown(f"**{title}**")
            for label in labels_for_portfolio:
                fund_id = label_to_id[label]
                name_column, input_column = st.columns([4, 1])
                with name_column:
                    st.markdown(f'<div style="white-space: nowrap; overflow-x: auto; text-overflow: clip; width: 100%;" title="{label}">{label}</div>', unsafe_allow_html=True)
                with input_column:
                    target[fund_id] = st.number_input("", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key=f"{prefix}_{fund_id}")
    sum_a, sum_b = sum(proportions_a.values()), sum(proportions_b.values())
    left, right = st.columns(2)
    left.markdown(f"**Total A: {sum_a:.1f}%**")
    right.markdown(f"**Total B: {sum_b:.1f}%**")
    st.subheader("4. Select period")
    start_date, end_date = _period_selector("as")
    if start_date > end_date:
        st.error("Start date must be earlier than end date.")
        return
    st.subheader("5. Frequency")
    frequency = st.radio("Aggregation", ["Monthly", "Quarterly", "Yearly"], horizontal=True, key="as_freq")
    if st.button("Calculate active share"):
        if abs(sum_a - 100.0) > 0.01:
            st.error(f"Portfolio A proportions must sum to 100. Currently: {sum_a:.1f}%")
            return
        if abs(sum_b - 100.0) > 0.01:
            st.error(f"Portfolio B proportions must sum to 100. Currently: {sum_b:.1f}%")
            return
        with st.spinner("Calculating active share..."):
            data = fetch_multi_fund_portfolios(list(set([label_to_id[label] for label in selected_a + selected_b])), start_date, end_date, frequency)
            if data.empty:
                st.warning("No portfolio data found for the selected funds and period.")
                return
            result = compute_active_share(data, {fund_id: value / 100.0 for fund_id, value in proportions_a.items()}, {fund_id: value / 100.0 for fund_id, value in proportions_b.items()})
        display_portfolio_result("Active share", result, {})


def portfolio_page():
    home_button()
    st.header("Portfolio explorer")
    if st.selectbox("Mode", ["View portfolio", "Active share"]) == "View portfolio":
        _portfolio_view()
    else:
        _active_share()
