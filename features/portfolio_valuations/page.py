"""Portfolio Valuations selectors and orchestration."""

import datetime as dt

import streamlit as st

from core.dates import month_year_to_last_day
from core.fund_catalog import fetch_categories, fetch_funds_for_categories
from core.navigation import home_button
from .data import cached_portfolio_exposures_timeseries, cached_portfolio_valuations_cube
from .display import display_exposure_diagnostics, display_valuation_time_series


def portfolio_valuations_page():
    home_button()
    st.header("Portfolio valuations")

    # ------------------------------------------------------------
    # 1) Category selector (outside form)
    # ------------------------------------------------------------
    categories = fetch_categories()
    if not categories:
        st.warning("No categories found in fundlab.category.")
        return

    st.subheader("1. Select categories")
    selected_categories = []
    cols = st.columns(min(4, len(categories)))
    for i, cat in enumerate(categories):
        col = cols[i % len(cols)]
        if col.checkbox(cat, value=False, key=f"pv_cat_{cat}"):
            selected_categories.append(cat)

    if not selected_categories:
        st.info("Please select at least one category.")
        return

    # ------------------------------------------------------------
    # 2) Fund multi-select (outside form)
    # ------------------------------------------------------------
    st.subheader("2. Select funds")

    funds_df = fetch_funds_for_categories(selected_categories)
    if funds_df.empty:
        st.warning("No funds found for selected categories.")
        return

    fund_options = {
        f"{row['fund_name']} ({row['category_name']})": int(row["fund_id"])
        for _, row in funds_df.iterrows()
    }

    all_option = "All"
    multiselect_options = [all_option] + list(fund_options.keys())

    selected_raw_labels = st.multiselect(
        "Funds",
        options=multiselect_options,
        default=[],
        key="pv_funds_multiselect",
    )

    if all_option in selected_raw_labels:
        selected_fund_labels = list(fund_options.keys())
    else:
        selected_fund_labels = [label for label in selected_raw_labels if label != all_option]

    selected_fund_ids = [fund_options[label] for label in selected_fund_labels]

    if not selected_fund_ids:
        st.info("Please select at least one fund.")
        return

    # ------------------------------------------------------------
    # 3) Focus fund (outside form; should NOT trigger DB fetch)
    # ------------------------------------------------------------
    st.subheader("3. Valuation settings")

    focus_fund_label = st.selectbox(
        "Focus fund",
        options=selected_fund_labels,
        index=0,
        key="pv_focus_fund",
    )
    focus_fund_id = fund_options[focus_fund_label]

    # ------------------------------------------------------------
    # 4) Period + mode + segment + metric in form
    # ------------------------------------------------------------
    st.subheader("4. Period and valuation options")

    current_year = dt.date.today().year
    years_val = list(range(current_year - 15, current_year + 1))
    months_val = list(range(1, 13))

    def month_name(m: int) -> str:
        return dt.date(2000, m, 1).strftime("%b")

    with st.form("pv_controls"):
        colv1, colv2 = st.columns(2)
        with colv1:
            val_start_year = st.selectbox(
                "Valuation start year",
                options=years_val,
                index=0,
                key="pv_val_start_year",
            )
            val_start_month = st.selectbox(
                "Valuation start month",
                options=months_val,
                index=0,
                key="pv_val_start_month",
                format_func=month_name,
            )
        with colv2:
            val_end_year = st.selectbox(
                "Valuation end year",
                options=years_val,
                index=len(years_val) - 1,
                key="pv_val_end_year",
            )
            val_end_month = st.selectbox(
                "Valuation end month",
                options=months_val,
                index=dt.date.today().month - 1,
                key="pv_val_end_month",
                format_func=month_name,
            )

        val_start_date = month_year_to_last_day(val_start_year, val_start_month)
        val_end_date = month_year_to_last_day(val_end_year, val_end_month)

        val_mode = st.radio(
            "Valuation mode",
            options=[
                "Valuations of historical portfolios",
                "Historical valuations of current portfolio",
            ],
            horizontal=False,
            key="pv_val_mode",
        )

        val_segment = st.radio(
            "Segment for valuations",
            options=["Financials", "Non-financials", "Total"],
            horizontal=True,
            key="pv_val_segment",
        )

        val_metric = st.radio(
            "Valuation metric",
            options=["P/S", "P/B", "P/E"],
            horizontal=True,
            key="pv_val_metric",
        )

        val_agg = st.radio(
            "Aggregation",
            options=["Weighted average multiple", "Median multiple"],
            horizontal=True,
            key="pv_val_agg",
        )


        run_charts = st.form_submit_button("Update valuations", type="primary")

    if not run_charts:
        st.info("Adjust filters above and click **Update valuations** to see results.")
        return

    if val_start_date > val_end_date:
        st.error("Valuation start date must be earlier than end date.")
        return

    # ------------------------------------------------------------
    # 5) Load cached cube ONCE per (fund_ids, period, mode)
    # ------------------------------------------------------------
    st.subheader("5. Valuation time series")

    try:
        df_cube = cached_portfolio_valuations_cube(
            fund_ids=selected_fund_ids,
            start_date=val_start_date,
            end_date=val_end_date,
            mode=val_mode,
            agg_choice=val_agg,
        )
    except ValueError as ve:
        st.error(str(ve))
        return
    except Exception as e:
        st.error(f"Error while loading valuations: {e}")
        return

    if not display_valuation_time_series(
        df_cube,
        focus_fund_id=focus_fund_id,
        segment=val_segment,
        metric=val_metric,
    ):
        return

    # ------------------------------------------------------------
    # 6) Additional exposure charts (optimized base-cache)
    # ------------------------------------------------------------
    st.subheader("6. Additional diagnostics (exposure %)")

    try:
        df_exp = cached_portfolio_exposures_timeseries(
            fund_ids=selected_fund_ids,
            focus_fund_id=focus_fund_id,
            start_date=val_start_date,
            end_date=val_end_date,
            segment_choice=val_segment,
            metric_choice=val_metric,
            mode=val_mode,
        )
    except Exception as e:
        st.error(f"Error while computing exposure diagnostics: {e}")
        return

    display_exposure_diagnostics(df_exp)
