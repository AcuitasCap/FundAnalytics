"""Portfolio Valuations selectors and live-analysis orchestration."""

import datetime as dt

import streamlit as st

from core.dates import month_year_to_last_day
from core.fund_catalog import fetch_categories, fetch_funds_for_categories
from core.navigation import home_button
from .compute import build_focus_peer_series, compute_fund_valuation_ranges
from .data import cached_portfolio_valuation_inputs
from .display import display_anchor_months, display_valuation_ranges


def portfolio_valuations_page():
    home_button()
    st.header("Portfolio valuations")

    categories = fetch_categories()
    if not categories:
        st.warning("No categories found in fundlab.category.")
        return

    st.subheader("1. Select categories")
    selected_categories = []
    cols = st.columns(min(4, len(categories)))
    for index, category in enumerate(categories):
        if cols[index % len(cols)].checkbox(category, value=False, key=f"pv_cat_{category}"):
            selected_categories.append(category)
    if not selected_categories:
        st.info("Please select at least one category.")
        return

    st.subheader("2. Select funds")
    funds_df = fetch_funds_for_categories(selected_categories)
    if funds_df.empty:
        st.warning("No funds found for selected categories.")
        return

    fund_options = {
        f"{row['fund_name']} ({row['category_name']})": int(row["fund_id"])
        for _, row in funds_df.iterrows()
    }
    selected_raw = st.multiselect(
        "Funds",
        options=["All", *fund_options],
        default=[],
        key="pv_funds_multiselect",
    )
    selected_labels = list(fund_options) if "All" in selected_raw else [x for x in selected_raw if x != "All"]
    selected_ids = [fund_options[label] for label in selected_labels]
    if not selected_ids:
        st.info("Please select at least one fund.")
        return
    if len(selected_ids) < 2:
        st.info("Please select at least two funds so the focus fund can be compared with a peer-set.")
        return

    st.subheader("3. Valuation settings")
    focus_label = st.selectbox("Focus fund", selected_labels, index=0, key="pv_focus_fund")
    focus_id = fund_options[focus_label]

    st.subheader("4. Period and valuation options")
    current_year = dt.date.today().year
    years = list(range(current_year - 15, current_year + 1))
    months = list(range(1, 13))

    with st.form("pv_controls"):
        start_col, end_col = st.columns(2)
        with start_col:
            start_year = st.selectbox("Valuation start year", years, index=0, key="pv_val_start_year")
            start_month = st.selectbox(
                "Valuation start month",
                months,
                index=0,
                key="pv_val_start_month",
                format_func=lambda month: dt.date(2000, month, 1).strftime("%b"),
            )
        with end_col:
            end_year = st.selectbox("Valuation end year", years, index=len(years) - 1, key="pv_val_end_year")
            end_month = st.selectbox(
                "Valuation end month",
                months,
                index=dt.date.today().month - 1,
                key="pv_val_end_month",
                format_func=lambda month: dt.date(2000, month, 1).strftime("%b"),
            )

        mode = st.radio(
            "Valuation mode",
            ["Valuations of historical portfolios", "Historical valuations of current portfolio"],
            horizontal=False,
            key="pv_val_mode",
        )
        segment = st.radio(
            "Segment for valuations",
            ["Financials", "Non-financials", "Total"],
            horizontal=True,
            key="pv_val_segment",
        )
        metric = st.radio(
            "Valuation metric",
            ["P/S", "P/B", "P/E"],
            horizontal=True,
            key="pv_val_metric",
        )
        valuation_range = st.radio(
            "Valuation range",
            ["Full valuation range", "Valuation range of top 50% stocks"],
            horizontal=True,
            key="pv_val_range",
        )
        run_analysis = st.form_submit_button("Update valuations", type="primary")

    if not run_analysis:
        st.info("Adjust filters above and click **Update valuations** to see results.")
        return

    start_date = month_year_to_last_day(start_year, start_month)
    end_date = month_year_to_last_day(end_year, end_month)
    if start_date > end_date:
        st.error("Valuation start date must be earlier than end date.")
        return

    st.subheader("5. Valuation ranges")
    try:
        holdings, multiples = cached_portfolio_valuation_inputs(
            tuple(sorted(selected_ids)), start_date, end_date, mode
        )
        loaded_fund_ids = set(holdings["fund_id"].unique()) if not holdings.empty else set()
        missing_fund_ids = [fund_id for fund_id in selected_ids if fund_id not in loaded_fund_ids]
        if missing_fund_ids:
            missing_names = [
                label for label, fund_id in fund_options.items() if fund_id in missing_fund_ids
            ]
            st.warning("No eligible domestic-equity portfolio was found for: " + ", ".join(missing_names))
        ranges = compute_fund_valuation_ranges(
            holdings,
            multiples,
            start_date=start_date,
            end_date=end_date,
            mode=mode,
            segment=segment,
            metric=metric,
            valuation_range=valuation_range,
            minimum_stocks=5,
        )
        series = build_focus_peer_series(ranges, focus_id)
    except ValueError as error:
        st.error(str(error))
        return
    except Exception as error:
        st.error(f"Error while loading valuations: {error}")
        return

    if ranges.empty:
        st.info("No valuation data available for the selected filters.")
        return

    fund_names = {fund_id: label for label, fund_id in fund_options.items() if fund_id in selected_ids}
    if mode == "Historical valuations of current portfolio":
        display_anchor_months(holdings, fund_names, end_date)
    display_valuation_ranges(
        series,
        ranges,
        metric=metric,
        focus_fund_id=focus_id,
        fund_names=fund_names,
    )
