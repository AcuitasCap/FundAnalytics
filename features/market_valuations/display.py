"""Market valuation chart and explanatory labels."""

import altair as alt
import pandas as pd
import streamlit as st


def display_market_valuation_ranges(
    ranges: pd.DataFrame, *, cohort: str, segment: str, metric: str
) -> None:
    if ranges.empty:
        st.info("No period has at least five cohort stocks with a valid valuation multiple for this selection.")
        return

    chart_data = ranges.melt(
        id_vars="month_end",
        value_vars=["median", "p25", "p75", "historical_median"],
        var_name="Statistic",
        value_name="Multiple",
    )
    labels = {
        "median": "Median",
        "p25": "25th percentile",
        "p75": "75th percentile",
        "historical_median": "Historical median of medians",
    }
    chart_data["Statistic"] = chart_data["Statistic"].map(labels)
    lines = alt.Chart(chart_data).mark_line(strokeWidth=2).encode(
        x=alt.X("month_end:T", title="Month", axis=alt.Axis(format="%b %Y")),
        y=alt.Y("Multiple:Q", title=f"{metric} (x)", scale=alt.Scale(zero=False)),
        color=alt.Color(
            "Statistic:N",
            scale=alt.Scale(
                domain=list(labels.values()),
                range=["#1f77b4", "#93c5fd", "#64748b", "#d97706"],
            ),
        ),
        strokeDash=alt.StrokeDash(
            "Statistic:N",
            scale=alt.Scale(
                domain=list(labels.values()),
                range=[[1, 0], [4, 3], [4, 3], [7, 3]],
            ),
        ),
        tooltip=[
            alt.Tooltip("month_end:T", title="Month", format="%b %Y"),
            alt.Tooltip("Statistic:N"),
            alt.Tooltip("Multiple:Q", format=".2f"),
        ],
    )
    st.altair_chart(lines.properties(height=440), use_container_width=True)
    source_note = (
        "Nifty 50 membership is represented by the same-month domestic-equity holdings "
        "of Nippon India ETF Nifty 50 BeES; no holdings are carried forward."
        if cohort == "Nifty 50"
        else "Size classifications use the latest available ranking on or before each month."
    )
    st.caption(
        f"{cohort} · {segment} · {metric}. Monthly percentiles are unweighted across stocks "
        "with valid multiples. At least five stocks are required per month; "
        "the flat reference is the median of the displayed monthly medians. "
        + source_note
    )
