"""Median comparison charts and coverage notices for fundamental attributes."""

import altair as alt
import pandas as pd
import streamlit as st


def display_median_comparison(series, diagnostics, *, focus_label, compare, metric, kind):
    if series.empty:
        st.info("No focus-fund portfolio is available in the selected period.")
        return
    focus = series[series["series"] == focus_label]
    comparator = series[series["series"] == compare]
    missing_focus = int(focus["median"].isna().sum())
    missing_comparison = int(comparator["median"].isna().sum())
    if missing_focus:
        st.warning(f"The focus fund has {missing_focus} period(s) with fewer than five valid stocks; "
                   "their medians are unavailable.")
    if missing_comparison:
        st.warning(f"{compare} has no eligible comparison for {missing_comparison} focus-fund period(s). "
                   "Missing portfolios or fewer than five valid stocks per fund/cohort are not filled forward.")
    if focus["median"].notna().any():
        y_title = f"Median {metric} (x)" if kind == "valuation" else f"Median 3-year {metric} CAGR (%)"
        # Keep missing rows in the chart data so gaps are visible.
        chart = alt.Chart(series).mark_line(point=True, invalid="break-paths-show-domains").encode(
            x=alt.X("month_end:T", title="Portfolio period", axis=alt.Axis(format="%b %Y", labelAngle=-45)),
            y=alt.Y("median:Q", title=y_title, scale=alt.Scale(zero=False)),
            color=alt.Color("series:N", title="Series"),
            tooltip=[alt.Tooltip("series:N", title="Series"),
                     alt.Tooltip("month_end:T", title="Period", format="%b %Y"),
                     alt.Tooltip("median:Q", title=y_title, format=".2f"),
                     alt.Tooltip("valid_stock_count:Q", title="Valid stocks", format=".0f"),
                     alt.Tooltip("peer_fund_count:Q", title="Eligible peers", format=".0f")],
        ).properties(height=420)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No focus-fund period has at least five valid stocks for these settings.")
    st.caption("Stock medians are unweighted. Peers excludes the focus fund and uses the median of "
               "eligible peer-fund medians, each from its full portfolio. Market comparisons use "
               "all eligible stocks in the latest complete size-band snapshot on or before each period; "
               "Top 500 combines Large, Mid and Small.")
    with st.expander("Median data and coverage"):
        table = series.copy()
        table["month_end"] = pd.to_datetime(table["month_end"]).dt.strftime("%b %Y")
        st.dataframe(table.rename(columns={"month_end": "Period", "series": "Series", "median": "Median",
                                          "valid_stock_count": "Valid stocks", "peer_fund_count": "Eligible peers"}),
                     use_container_width=True, hide_index=True)
        st.dataframe(diagnostics, use_container_width=True, hide_index=True)
