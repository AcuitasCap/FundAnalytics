"""Fund manager tenure page controls and orchestration."""

import streamlit as st

from core.fund_catalog import fetch_categories, fetch_funds_for_categories
from core.navigation import home_button
from features.fund_manager_tenure.compute import compute_tenure_filter, prepare_focus_fund_timeline
from features.fund_manager_tenure.data import fetch_fund_manager_tenure
from features.fund_manager_tenure.display import display_tenure_filter, display_tenure_history, display_underlying_rows


def fund_manager_tenure_page() -> None:
    home_button()
    st.subheader("Fund manager tenure")
    categories = fetch_categories()
    if not categories:
        st.error("No categories available."); return
    defaults = {"fm_selected_categories": [], "fm_categories_submitted": False, "fm_mode": "Single fund manager history", "fm_min_tenure": 5.0}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    with st.form("fm_category_form"):
        st.subheader("1. Select categories")
        selected = []
        columns = st.columns(min(4, len(categories)))
        for index, category in enumerate(categories):
            with columns[index % len(columns)]:
                if st.checkbox(category, value=category in st.session_state["fm_selected_categories"], key=f"fm_cat_{category}"):
                    selected.append(category)
        st.subheader("2. Choose mode")
        mode = st.radio("View", ["Single fund manager history", "Filter funds based on tenure"], index=0 if st.session_state["fm_mode"] == "Single fund manager history" else 1, key="fm_mode_radio")
        if mode == "Filter funds based on tenure":
            min_tenure = st.number_input("Minimum required tenure (years)", min_value=0.0, step=0.5, value=float(st.session_state["fm_min_tenure"]), key="fm_min_tenure_input")
        submitted = st.form_submit_button("Submit")
    if submitted:
        st.session_state.update(fm_selected_categories=selected, fm_categories_submitted=True, fm_mode=st.session_state["fm_mode_radio"])
        if st.session_state["fm_mode"] == "Filter funds based on tenure": st.session_state["fm_min_tenure"] = float(min_tenure)
        st.session_state.pop("fm_focus_fund_name", None)
    if not st.session_state["fm_categories_submitted"]: return
    selected_categories = st.session_state["fm_selected_categories"]
    if not selected_categories:
        st.warning("Please select at least one category."); return
    funds = fetch_funds_for_categories(selected_categories)
    if funds.empty:
        st.info("No funds found for selected categories."); return
    tenure = fetch_fund_manager_tenure(funds["fund_id"].astype(int).tolist())
    if tenure.empty:
        st.info("No fund manager tenure data available."); return
    if st.session_state["fm_mode"] == "Filter funds based on tenure":
        minimum_years = float(st.session_state.get("fm_min_tenure", 5.0))
        display_tenure_filter(compute_tenure_filter(tenure, funds, minimum_years), minimum_years)
        return
    names = funds["fund_name"].tolist(); ids = dict(zip(funds["fund_name"], funds["fund_id"]))
    st.subheader("3. Select focus fund")
    st.session_state.setdefault("fm_focus_fund_name", names[0])
    if st.session_state["fm_focus_fund_name"] not in names: st.session_state["fm_focus_fund_name"] = names[0]
    focus = st.selectbox("Focus fund", names, index=names.index(st.session_state["fm_focus_fund_name"]), key="fm_focus_fund_selectbox")
    st.session_state["fm_focus_fund_name"] = focus
    timeline, current, last_update = prepare_focus_fund_timeline(tenure, int(ids[focus]))
    if timeline.empty:
        st.info("No tenure rows available for the selected focus fund."); return
    st.markdown(f"### Fund manager history - **{focus}**")
    display_tenure_history(timeline, current, last_update)
    display_underlying_rows(timeline)
