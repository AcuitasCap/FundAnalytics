import datetime as dt

import pandas as pd
import streamlit as st

from core.dates import to_month_end
from core.fund_catalog import fetch_categories, fetch_funds_for_categories
from core.navigation import home_button
from features.fund_attribution.data import _load_attrib_raw_window
from features.fund_attribution.compute import _compute_attribution
from features.fund_attribution.display import display_attribution_results

def fund_attribution_page():
    home_button()
    st.title("Fund attribution")

    # 1) Category
    categories = fetch_categories()
    if not categories:
        st.warning("No categories found.")
        return

    cat = st.radio("1. Select category", categories, horizontal=True, key="fa_category")

    # 2) Focus fund
    funds_df = fetch_funds_for_categories([cat])
    if funds_df.empty:
        st.warning("No funds found for this category.")
        return

    focus_label = st.selectbox(
        "2. Select focus fund",
        options=funds_df["fund_name"].tolist(),
        key="fa_focus_fund",
    )
    focus_id = int(funds_df.loc[funds_df["fund_name"] == focus_label, "fund_id"].iloc[0])

    # 3) Benchmark mode (3 options)
    bench_mode = st.radio(
        "3. Attribution benchmark mode",
        options=["Vs NIFTY 50", "Vs NIFTY 500", "Like-for-like"],
        horizontal=True,
        key="fa_bench_mode",
    )
    universe_mode = st.radio(
        "4. Attribution holdings universe",
        options=["Domestic equities only", "Full holdings"],
        horizontal=True,
        key="fa_universe_mode",
    )

    # 5) Start/end period below radio
    with st.form("fa_form"):
        today = dt.date.today()
        years = list(range(today.year - 15, today.year + 1))
        months = list(range(1, 13))

        def _mname(m: int) -> str:
            return dt.date(2000, m, 1).strftime("%b")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            start_year = st.selectbox("5. Start year", years, index=years.index(today.year - 3) if (today.year - 3) in years else 0, key="fa_start_year")
        with c2:
            start_month = st.selectbox("Start month", months, index=today.month - 1, format_func=_mname, key="fa_start_month")
        with c3:
            end_year = st.selectbox("6. End year", years, index=len(years) - 1, key="fa_end_year")
        with c4:
            end_month = st.selectbox("End month", months, index=today.month - 1, format_func=_mname, key="fa_end_month")

        lookback = st.selectbox("Lookback window (years)", options=[5, 10, 15], index=1, key="fa_lookback")
        lens = st.radio(
            "7. Decomposition lens (domestic equities only)",
            options=["Earnings (P/E)", "Sales (P/S)", "Book (P/B)"],
            horizontal=True,
            key="fa_lens",
        )
        dom_triangulation_mode = st.radio(
            "Invalid fundamentals handling (triangulation)",
            options=["Fallback to price growth", "Exclude invalid from weighted average"],
            horizontal=True,
            key="fa_dom_triangulation_mode",
        )
        submit = st.form_submit_button("Submit", type="primary")

    if not submit:
        st.info("Select inputs above and click Submit.")
        return

    start_me = to_month_end(dt.date(int(start_year), int(start_month), 1))
    end_me = to_month_end(dt.date(int(end_year), int(end_month), 1))
    if start_me > end_me:
        st.error("Start period must be before end period.")
        return

    # 10-year cache anchored to end period
    with st.spinner("Loading fund history (cached) ..."):
        raw = _load_attrib_raw_window(focus_id, end_me, lookback_years=int(lookback), data_version="v1")

    if raw.get("weight_scale") not in (None, 1.0):
        st.caption(f"Note: holding_weight scale detected as {raw['weight_scale']:.6f} and normalised accordingly.")


    if raw["holdings"].empty:
        st.warning("No holdings data found for this fund in the selected window.")
        return

    if start_me < raw["window_start"]:
        st.warning(f"Start period {start_me} is older than cached window start {raw['window_start']}. Increase lookback window.")
        return

    with st.spinner("Computing attribution ..."):
        raw["dom_triangulation_mode"] = dom_triangulation_mode
        stock_df, hit_df, cat_df, diag = _compute_attribution(
            raw, start_me, end_me, bench_mode, lens=lens, universe_mode=universe_mode
        )


    if "error" in diag:
        st.error(diag["error"])
        return

    display_attribution_results(stock_df, hit_df, cat_df, diag, bench_mode)
