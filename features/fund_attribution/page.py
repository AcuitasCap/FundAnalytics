import datetime as dt
import logging

import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from core.dates import to_month_end
from core.fund_catalog import fetch_categories, fetch_funds_for_categories
from core.navigation import home_button
from features.fund_attribution.data import _load_attrib_raw_window, load_attrib_fundamental_endpoints
from features.fund_attribution.compute import _compute_attribution
from features.fund_attribution.display import display_attribution_results
from features.fund_attribution.fundamentals import build_fundamental_growth_table


logger = logging.getLogger(__name__)

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
        options=["Full holdings", "Domestic equities only"],
        horizontal=True,
        key="fa_universe_mode",
        help="Full holdings includes cash and other reported assets. Domestic equities only rescales the domestic stocks to 100% at each rebalance.",
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
        return_basis = st.radio(
            "Stock return basis",
            options=["Price only", "Price and dividends"],
            horizontal=True,
            key="fa_return_basis",
        )
        submit = st.form_submit_button("Submit", type="primary")

    start_me = to_month_end(dt.date(int(start_year), int(start_month), 1))
    end_me = to_month_end(dt.date(int(end_year), int(end_month), 1))
    compute_args = (start_me, end_me, bench_mode, universe_mode, return_basis == "Price and dividends")
    settings = (focus_id, lookback, *compute_args)
    state_key = "fa_attribution_run"
    revision_key = "fa_attribution_editor_revision"

    if submit:
        st.session_state.pop(state_key, None)
        if start_me >= end_me:
            st.error("Start period must be before end period.")
            return

        with st.spinner("Loading fund history ..."):
            raw = _load_attrib_raw_window(focus_id, end_me, lookback_years=int(lookback))

        if raw["holdings"].empty:
            st.warning("No holdings data found for this fund in the selected window.")
            return
        if start_me < raw["window_start"]:
            st.warning(f"Start period {start_me} is older than the available window. Increase the lookback window.")
            return

        with st.spinner("Computing attribution ..."):
            original = _compute_attribution(raw, *compute_args)
        if "error" in original[3]:
            st.error(original[3]["error"])
            return

        windows = original[3]["fundamental_windows"]
        endpoints = tuple(sorted({
            (window["ace_code"], window[boundary])
            for window in windows if window["ace_code"]
            for boundary in ("first_month", "last_month")
        }))
        try:
            fundamentals = build_fundamental_growth_table(windows, load_attrib_fundamental_endpoints(endpoints))
        except (SQLAlchemyError, ValueError):
            logger.exception("Unable to load stock fundamentals for attribution")
            fundamentals = None
        st.session_state[state_key] = {
            "settings": settings, "raw": raw, "original": original, "scenario": None, "selected": (),
            "fundamentals": fundamentals,
        }
        st.session_state[revision_key] = st.session_state.get(revision_key, 0) + 1

    run = st.session_state.get(state_key)
    if run is None or run["settings"] != settings or "fundamentals" not in run:
        st.info("Select inputs above and click Submit.")
        return

    scale = run["raw"].get("weight_scale")
    if scale not in (None, 1.0):
        st.caption(f"Holding weights were normalized using a scale of {scale:.6f}.")

    results = run["scenario"] or run["original"]
    original_summary = run["original"][3]["portfolio_return_summary_df"] if run["scenario"] else None
    selection = display_attribution_results(
        *results, bench_mode, selected_keys=run["selected"], original_summary=original_summary,
        editor_key=f"fa_attribution_stock_selection_{st.session_state[revision_key]}",
        fundamentals=run["fundamentals"],
    )
    if selection is None:
        return

    scenario = None
    if selection:
        with st.spinner("Recalculating attribution ..."):
            scenario = _compute_attribution(run["raw"], *compute_args, benchmark_stock_keys=selection)
        if "error" in scenario[3]:
            st.error(scenario[3]["error"])
            return

    st.session_state[state_key] = {**run, "scenario": scenario, "selected": selection}
    st.session_state[revision_key] += 1
    st.rerun()
