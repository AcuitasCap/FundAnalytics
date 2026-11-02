"""Shared fund controls followed by quality, valuations and Sales/PAT growth."""

import datetime as dt

import streamlit as st

from core.dates import month_year_to_last_day
from core.fund_catalog import fetch_categories, fetch_funds_for_categories
from core.navigation import home_button
from features.portfolio_quality.page import render_quality_section
from features.portfolio_valuations.display import display_anchor_months
from . import COMPARATORS, CURRENT, HISTORICAL
from .compute import compute_attribute_comparison
from .data import cached_attribute_inputs
from .display import display_median_comparison


def _save_shortlist(category_ids):
    picks = st.session_state.get("fa_fund_picker", [])
    chosen = [value for value in picks if value != "All funds in selected categories"]
    if "All funds in selected categories" in picks:
        chosen = sorted(set(chosen) | set(category_ids))
    st.session_state["fa_shortlisted_ids"] = chosen


def _save_focus():
    st.session_state["fa_saved_focus"] = st.session_state.get("fa_focus_picker")


def _reset_funds():
    st.session_state["fa_shortlisted_ids"] = []
    st.session_state["fa_saved_focus"] = None


def shared_fund_selector():
    categories = fetch_categories()
    if not categories:
        st.warning("No fund categories are available.")
        return None
    st.subheader("Categories, peer shortlist and focus fund")
    columns = st.columns(min(4, len(categories)))
    chosen_categories = [category for index, category in enumerate(categories)
                         if columns[index % len(columns)].checkbox(category, key=f"fa_category_{category}")]
    # The complete catalog keeps shortlisted funds available when categories change.
    funds = fetch_funds_for_categories(categories)
    names = {int(row.fund_id): f"{row.fund_name} ({row.category_name})" for row in funds.itertuples()}
    category_ids = sorted(int(row.fund_id) for row in funds.itertuples() if row.category_name in chosen_categories)
    saved = [fund_id for fund_id in st.session_state.get("fa_shortlisted_ids", []) if fund_id in names]
    st.session_state["fa_shortlisted_ids"] = saved
    options = sorted(set(category_ids) | set(saved), key=lambda fund_id: names[fund_id])
    st.session_state["fa_fund_picker"] = saved
    st.multiselect("Peer shortlist (includes focus fund)", ["All funds in selected categories", *options],
                   format_func=lambda value: names.get(value, value), key="fa_fund_picker",
                   on_change=_save_shortlist, args=(category_ids,))
    selected = st.session_state["fa_shortlisted_ids"]
    if not selected:
        st.info("Select categories and shortlist at least one fund. These selections apply to all three sections.")
        return None
    saved_focus = st.session_state.get("fa_saved_focus")
    if saved_focus not in selected:
        saved_focus = selected[0]
    st.session_state["fa_saved_focus"] = saved_focus
    st.session_state["fa_focus_picker"] = saved_focus
    focus_id = st.selectbox("Focus fund", selected, format_func=names.get, key="fa_focus_picker", on_change=_save_focus)
    st.button("Reset fund selection", key="fa_reset", on_click=_reset_funds)
    st.caption("The shortlist is shared by all sections. Peers means the other shortlisted funds. "
               "Shortlisted funds remain selected when categories change.")
    return {"selected_fund_ids": selected, "selected_fund_labels": [names[fund_id] for fund_id in selected],
            "focus_fund_id": focus_id, "focus_fund_label": names[focus_id],
            "fund_names": {fund_id: names[fund_id] for fund_id in selected}}


def period_selector(prefix, quarterly=False):
    today = dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30))).date()
    years = list(range(today.year - 15, today.year + 1))
    months = [3, 6, 9, 12] if quarterly else list(range(1, 13))
    end_month = max((month for month in months if month <= today.month), default=months[0])
    left, right = st.columns(2)
    with left:
        start_year = st.selectbox("Start year", years, index=0, key=f"{prefix}_start_year")
        start_month = st.selectbox("Start month", months, key=f"{prefix}_start_month",
                                   format_func=lambda month: dt.date(2000, month, 1).strftime("%b"))
    with right:
        end_year = st.selectbox("End year", years, index=len(years)-1, key=f"{prefix}_end_year")
        end_month = st.selectbox("End month", months, index=months.index(end_month), key=f"{prefix}_end_month",
                                 format_func=lambda month: dt.date(2000, month, 1).strftime("%b"))
    return month_year_to_last_day(start_year, start_month), month_year_to_last_day(end_year, end_month)


def _render_comparison_section(context, kind):
    valuation = kind == "valuation"
    prefix = "fa_valuation" if valuation else "fa_growth"
    signature = (tuple(sorted(context["selected_fund_ids"])), context["focus_fund_id"])
    with st.form(f"{prefix}_controls"):
        start, end = period_selector(prefix, quarterly=not valuation)
        mode = st.radio("Valuation mode", [HISTORICAL, CURRENT], key=f"{prefix}_mode") if valuation else HISTORICAL
        segment = st.radio("Segment", ["Financials", "Non-financials", "Total"], horizontal=True, key=f"{prefix}_segment")
        metric = st.radio("Valuation metric" if valuation else "3-year growth metric",
                          ["P/S", "P/B", "P/E"] if valuation else ["Sales", "PAT"],
                          horizontal=True, key=f"{prefix}_metric")
        selected_range = st.radio("Valuation range", ["Full portfolio", "Top 50% of portfolio"],
                                  horizontal=True, key=f"{prefix}_range") if valuation else "Full portfolio"
        compare = st.radio("Compare against", COMPARATORS, horizontal=True, key=f"{prefix}_compare")
        run = st.form_submit_button("Update valuations" if valuation else "Update growth", type="primary")
    if valuation:
        st.caption("Top 50% ranks the focus fund's domestic-equity holdings by reported portfolio weight, "
                   "includes the stock taking cumulative weight above 50%, then applies the segment filter. "
                   "Comparators always use their full eligible range.")
    else:
        st.caption("Growth is annualized TTM growth versus three years earlier. Four consecutive quarters "
                   "are required at both ends, with positive TTM values. Each portfolio uses the quarter "
                   "ending strictly before its holding date. Every recorded portfolio date inside the "
                   "selected bounds is retained, including monthly holdings.")
    result_key = f"{prefix}_result"
    if run:
        st.session_state.pop(result_key, None)
        if start > end:
            st.error("Start date must be on or before end date.")
            return
        ids = context["selected_fund_ids"] if compare == "Peers" else [context["focus_fund_id"]]
        try:
            with st.spinner("Loading fundamental attributes..."):
                holdings, members, facts = cached_attribute_inputs(tuple(sorted(ids)), start, end, mode, compare, kind)
                series, diagnostics = compute_attribute_comparison(
                    holdings, members, facts, start_date=start, end_date=end, mode=mode,
                    focus_id=context["focus_fund_id"], focus_label=context["focus_fund_label"],
                    compare=compare, segment=segment, kind=kind, metric=metric,
                    top_half=valuation and selected_range == "Top 50% of portfolio")
            st.session_state[result_key] = {"signature": signature, "series": series, "diagnostics": diagnostics,
                                            "holdings": holdings, "start": start, "end": end, "mode": mode,
                                            "compare": compare, "metric": metric, "segment": segment,
                                            "range": selected_range}
        except Exception as error:
            st.error(f"Unable to load {kind} analysis: {error}")
            return
    saved = st.session_state.get(result_key)
    if not saved or saved["signature"] != signature:
        st.info("Choose settings and click **Update valuations**." if valuation
                else "Choose settings and click **Update growth**.")
        return
    st.caption(f"Displayed results: {saved['start']:%b %Y} to {saved['end']:%b %Y} · "
               f"{saved['segment']} · {saved['metric']} · {saved['compare']}" +
               (f" · {saved['range']} · {saved['mode']}" if valuation else ""))
    if valuation and saved["mode"] == CURRENT:
        display_anchor_months(saved["holdings"], context["fund_names"], saved["end"])
    display_median_comparison(saved["series"], saved["diagnostics"], focus_label=context["focus_fund_label"],
                              compare=saved["compare"], metric=saved["metric"], kind=kind)


def fundamental_attributes_page():
    home_button()
    st.header("Fundamental attributes")
    try:
        context = shared_fund_selector()
    except Exception as error:
        st.error(f"Unable to load the fund catalog: {error}")
        return
    if not context:
        return
    st.divider()
    st.header("1. Portfolio quality")
    render_quality_section(context)
    st.divider()
    st.header("2. Valuations")
    _render_comparison_section(context, "valuation")
    st.divider()
    st.header("3. Sales / PAT growth")
    _render_comparison_section(context, "growth")
