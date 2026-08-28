"""Presentation helpers for live Portfolio Valuations ranges."""

import altair as alt
import pandas as pd
import streamlit as st


def display_anchor_months(
    holdings: pd.DataFrame,
    fund_names: dict[int, str],
    requested_end_date,
) -> None:
    """Disclose the actual current-portfolio anchor chosen for every fund."""
    if holdings.empty:
        return
    anchors = holdings[["fund_id", "anchor_month_end"]].drop_duplicates().copy()
    anchors["Fund"] = anchors["fund_id"].map(fund_names).fillna(anchors["fund_id"].astype(str))
    anchors["Portfolio used"] = pd.to_datetime(anchors["anchor_month_end"]).dt.strftime("%b %Y")
    anchors = anchors.sort_values("Fund")
    requested_period = pd.Period(requested_end_date, freq="M")
    fallback_count = int(
        (pd.to_datetime(anchors["anchor_month_end"]).dt.to_period("M") != requested_period).sum()
    )
    if fallback_count:
        st.warning(
            f"Exact end-period portfolios were unavailable for {fallback_count} selected fund(s). "
            "Their latest available portfolio on or before the end period was used."
        )
    with st.expander("Portfolio anchor months", expanded=fallback_count > 0):
        st.dataframe(anchors[["Fund", "Portfolio used"]], use_container_width=True, hide_index=True)


def display_valuation_ranges(
    series: pd.DataFrame,
    ranges: pd.DataFrame,
    *,
    metric: str,
    focus_fund_id: int,
    fund_names: dict[int, str],
) -> bool:
    """Render focus and peer interquartile bands, medians, and an audit table."""
    if series.empty:
        st.info("No valuation data available for the selected filters.")
        return False

    focus_rows = ranges[ranges["fund_id"] == focus_fund_id]
    insufficient_focus = int(focus_rows["median"].isna().sum()) if not focus_rows.empty else 0
    if insufficient_focus:
        st.warning(
            f"The focus fund has {insufficient_focus} period(s) with fewer than five valid stocks; "
            "no percentiles are shown for those periods."
        )

    valid_series = series.dropna(subset=["p25", "median", "p75"]).copy()
    if valid_series.empty:
        st.info("No period has at least five valid stock multiples for this selection.")
        return False
    available_series = set(valid_series["series"])
    if "Focus fund" not in available_series:
        st.error("The focus fund has no period with at least five valid stocks for this selection.")
        return False
    if "Peer-set" not in available_series:
        st.error("No peer fund has at least five valid stocks for the selected periods.")
        return False

    colors = alt.Scale(domain=["Focus fund", "Peer-set"], range=["#1f77b4", "#ff7f0e"])
    base = alt.Chart(valid_series).encode(
        x=alt.X("month_end:T", title="Period", axis=alt.Axis(format="%b %Y", labelAngle=-45)),
        color=alt.Color("series:N", title="Series", scale=colors),
    )
    bands = base.mark_area(opacity=0.16).encode(
        y=alt.Y("p25:Q", title=f"{metric} (x)"),
        y2="p75:Q",
        tooltip=[
            alt.Tooltip("month_end:T", title="Period", format="%b %Y"),
            alt.Tooltip("series:N", title="Series"),
            alt.Tooltip("p25:Q", title="25th percentile", format=".2f"),
            alt.Tooltip("median:Q", title="Median", format=".2f"),
            alt.Tooltip("p75:Q", title="75th percentile", format=".2f"),
            alt.Tooltip("peer_fund_count:Q", title="Peer funds", format=".0f"),
        ],
    )
    medians = base.mark_line(point=True, strokeWidth=2.5).encode(
        y=alt.Y("median:Q", title=f"{metric} (x)"),
        tooltip=[
            alt.Tooltip("month_end:T", title="Period", format="%b %Y"),
            alt.Tooltip("series:N", title="Series"),
            alt.Tooltip("p25:Q", title="25th percentile", format=".2f"),
            alt.Tooltip("median:Q", title="Median", format=".2f"),
            alt.Tooltip("p75:Q", title="75th percentile", format=".2f"),
        ],
    )
    st.altair_chart((bands + medians).properties(height=430), use_container_width=True)
    st.caption(
        "Bands show the unweighted 25th–75th percentile range. The peer-set lines are the "
        "median of each corresponding statistic across eligible peer funds."
    )

    st.subheader("Valuation range data")
    table = valid_series.copy()
    table["Period"] = pd.to_datetime(table["month_end"]).dt.strftime("%b %Y")
    table = table.rename(
        columns={
            "series": "Series",
            "p25": "25th percentile",
            "median": "Median",
            "p75": "75th percentile",
            "valid_stock_count": "Valid stocks",
            "peer_fund_count": "Eligible peer funds",
        }
    )
    st.dataframe(
        table[
            [
                "Period",
                "Series",
                "25th percentile",
                "Median",
                "75th percentile",
                "Valid stocks",
                "Eligible peer funds",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Coverage diagnostics"):
        diagnostics = ranges.copy()
        diagnostics["Period"] = pd.to_datetime(diagnostics["month_end"]).dt.strftime("%b %Y")
        diagnostics["Fund"] = diagnostics["fund_id"].map(fund_names).fillna(diagnostics["fund_id"].astype(str))
        st.dataframe(
            diagnostics[
                ["Period", "Fund", "selected_stock_count", "valid_stock_count", "coverage_pct"]
            ].rename(
                columns={
                    "selected_stock_count": "Selected stocks",
                    "valid_stock_count": "Valid stocks",
                    "coverage_pct": "Valid valuation weight (%)",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    return True
