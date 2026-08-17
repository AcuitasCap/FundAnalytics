"""Presentation for Portfolio Valuations results and diagnostics."""

import altair as alt
import pandas as pd
import streamlit as st


def display_valuation_time_series(
    df_cube: pd.DataFrame,
    *,
    focus_fund_id: int,
    segment: str,
    metric: str,
) -> bool:
    """Render the selected valuation series. Return whether diagnostics may continue."""
    if df_cube.empty:
        st.info("No valuation data available for the selected filters.")
        return False

    # Slice locally (no DB)
    df_slice = df_cube[
        (df_cube["segment"] == segment) &
        (df_cube["metric"] == metric)
    ].copy()

    if df_slice.empty:
        st.info("No valuation data after applying segment/metric filters.")
        return False

    # Focus vs median series locally
    focus = df_slice[df_slice["fund_id"] == focus_fund_id][["month_end", "value"]].copy()
    focus["series"] = "Focus fund"

    others = df_slice[df_slice["fund_id"] != focus_fund_id][["month_end", "value"]].copy()
    if others.empty:
        st.error("Need at least one other fund besides the focus fund to compute universe median.")
        return False

    median_others = others.groupby("month_end", as_index=False)["value"].median()
    median_others["series"] = "Universe median (others)"

    df_val = pd.concat([focus, median_others], ignore_index=True)
    df_val["month_end"] = pd.to_datetime(df_val["month_end"], errors="coerce")
    df_val = df_val.dropna(subset=["month_end"])
    df_val = df_val.sort_values(["month_end", "series"]).reset_index(drop=True)

    if df_val.empty:
        st.info("No valuation series to chart.")
        return False

    # Chart
    val_chart = (
        alt.Chart(df_val)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "month_end:T",
                title="Period",
                axis=alt.Axis(format="%b %Y", labelAngle=-45),
            ),
            y=alt.Y("value:Q", title=f"{metric} (x)"),
            color=alt.Color("series:N", title="Series"),
            tooltip=[
                alt.Tooltip("month_end:T", title="Period", format="%b %Y"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("value:Q", title=f"{metric} (x)", format=".2f"),
            ],
        )
        .properties(height=400)
    )
    st.altair_chart(val_chart, use_container_width=True)

    # ------------------------------------------------------------
    # 5B) Valuation data table (HORIZONTAL months; 2 rows)
    # ------------------------------------------------------------
    st.subheader("Valuation data")

    df_t = df_val.copy()
    df_t["month_label"] = df_t["month_end"].dt.strftime("%b %Y")

    month_order = (
        df_t[["month_end", "month_label"]]
        .drop_duplicates()
        .sort_values("month_end")
    )
    ordered_labels = month_order["month_label"].tolist()

    wide = (
        df_t.pivot_table(
            index="series",
            columns="month_label",
            values="value",
            aggfunc="first",
        )
        .reindex(columns=ordered_labels)
        .reindex(["Focus fund", "Universe median (others)"])
        .reset_index()
        .rename(columns={"series": "Series"})
    )

    st.dataframe(wide, use_container_width=True)
    return True


def display_exposure_diagnostics(df_exp: pd.DataFrame) -> None:
    """Render the existing exposure diagnostic charts without changing their specifications."""
    if df_exp.empty:
        st.info("No exposure diagnostics available for this selection.")
        return

    for metric_label in df_exp["metric"].dropna().unique().tolist():
        sub = df_exp[df_exp["metric"] == metric_label].copy()
        if sub.empty:
            continue

        ch = (
            alt.Chart(sub)
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    "month_end:T",
                    title="Period",
                    axis=alt.Axis(format="%b %Y", labelAngle=-45),
                ),
                y=alt.Y("value:Q", title=metric_label),
                color=alt.Color("series:N", title="Series"),
                tooltip=[
                    alt.Tooltip("month_end:T", title="Period", format="%b %Y"),
                    alt.Tooltip("series:N", title="Series"),
                    alt.Tooltip("value:Q", title=metric_label, format=".2f"),
                ],
            )
            .properties(height=250)
        )
        st.altair_chart(ch, use_container_width=True)
