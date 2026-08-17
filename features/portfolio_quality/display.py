"""Streamlit rendering for Portfolio Quality results."""

import altair as alt
import pandas as pd
import streamlit as st

from .compute import QUARTILE_MODE, ROC_MODE


def display_quality_analysis(mode: str, result: pd.DataFrame, context: dict) -> None:
    """Render the selected Quality analysis from its computed result table."""
    if mode == ROC_MODE:
        if result.empty:
            st.warning("No fundamentals could be computed (check data availability).")
            return
        if context["focus_fund_id"] not in result["fund_id"].tolist():
            st.warning("No data for the focus fund in the selected period.")
            return
        result = result.copy()
        result["month_end"] = pd.to_datetime(result["month_end"])
        st.subheader("Return on capital (5-period median RoE / RoCE)")
        chart = alt.Chart(result).mark_line(point=True).encode(
            x=alt.X("month_end:T", title="Period", axis=alt.Axis(format="%b %Y")),
            y=alt.Y("metric:Q", title="Portfolio metric (%)"),
            color=alt.Color("fund_name:N", title="Fund"),
            tooltip=[alt.Tooltip("fund_name:N", title="Fund"),
                     alt.Tooltip("month_end:T", title="Period", format="%b %Y"),
                     alt.Tooltip("metric:Q", title="Metric", format=".1f")],
        ).properties(height=400)
        st.altair_chart(chart, use_container_width=True)
        pivot = result.pivot_table(index="fund_name", columns=result["month_end"].dt.strftime("%b %Y"),
                                   values="metric", aggfunc="first")
        st.dataframe(pivot.style.format("{:.1f}"), use_container_width=True)
        return
    if mode == QUARTILE_MODE:
        if result.empty:
            st.info("No quality bucket data available.")
            return
        st.subheader("7. Quality quartile exposures (Q1–Q4)")
        st.markdown(f"**Fund:** {context['focus_fund_label']}")
        chart_df = result.drop(index="Total", errors="ignore").T.reset_index().rename(columns={"index": "month_end"})
        chart_df["month_end"] = pd.to_datetime(chart_df["month_end"])
        chart_long = chart_df.melt("month_end", var_name="Quartile", value_name="Exposure")
        chart = alt.Chart(chart_long).mark_area().encode(
            x=alt.X("month_end:T", title="Period", axis=alt.Axis(format="%b %Y")),
            y=alt.Y("Exposure:Q", title="Exposure (% of domestic equities)"),
            color=alt.Color("Quartile:N", title="Quartile"),
            tooltip=[alt.Tooltip("month_end:T", title="Period", format="%b %Y"),
                     alt.Tooltip("Quartile:N", title="Quartile"),
                     alt.Tooltip("Exposure:Q", title="Exposure (%)", format=".1f")],
        ).properties(height=400)
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(result.style.format("{:.1f}"), use_container_width=True)
        return
    raise ValueError(f"Unsupported Quality analysis mode: {mode}")
