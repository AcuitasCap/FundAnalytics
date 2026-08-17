"""Streamlit rendering for fund manager tenure."""

import pandas as pd
import streamlit as st


def _build_tenure_chart(timeline: pd.DataFrame):
    """Build a Plotly date-range timeline without changing the input frame."""
    import plotly.graph_objects as go

    chart_rows = timeline.copy()
    chart_rows["from_date"] = pd.to_datetime(chart_rows["from_date"])
    chart_rows["to_date_filled"] = pd.to_datetime(chart_rows["to_date_filled"])
    # Preserve one row per stint so separate assignments by the same manager
    # cannot overlap.  The earliest stint is shown at the top.
    chart_rows = chart_rows.sort_values(
        ["from_date", "to_date_filled", "fund_manager"]
    ).reset_index(drop=True)
    chart_rows["status"] = chart_rows["stint_is_current"].map(
        {True: "Current", False: "Prior"}
    )
    chart_rows["ypos"] = (len(chart_rows) - 1) - chart_rows.index
    chart_rows["timeline_row"] = [
        f"{manager} ({start:%b %Y})"
        for manager, start in zip(chart_rows["fund_manager"], chart_rows["from_date"])
    ]

    figure = go.Figure()
    colours = {"Prior": "#c7c7c7", "Current": "#1f77b4"}
    for row in chart_rows.itertuples(index=False):
        figure.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=row.from_date,
            x1=row.to_date_filled,
            y0=row.ypos - 0.32,
            y1=row.ypos + 0.32,
            line=dict(width=0),
            fillcolor=colours[row.status],
        )

    # Shapes provide the reliably visible date ranges.  Invisible midpoint
    # markers supply rich per-stint hover information without obscuring bars.
    chart_rows["midpoint"] = chart_rows["from_date"] + (
        chart_rows["to_date_filled"] - chart_rows["from_date"]
    ) / 2
    figure.add_trace(
        go.Scatter(
            x=chart_rows["midpoint"],
            y=chart_rows["ypos"],
            mode="markers",
            marker=dict(size=20, color="rgba(0,0,0,0)"),
            customdata=chart_rows[
                ["fund_manager", "from_date", "to_date_filled", "status"]
            ],
            hovertemplate=(
                "Fund manager: %{customdata[0]}<br>"
                "From: %{customdata[1]|%b-%Y}<br>"
                "To: %{customdata[2]|%b-%Y}<br>"
                "Status: %{customdata[3]}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    timeline_start = chart_rows["from_date"].min()
    timeline_end = chart_rows["to_date_filled"].max()
    # Layout shapes are not considered by Plotly's automatic date-axis range.
    # Give the rectangles a small margin, including for a single short stint.
    padding = max(
        pd.Timedelta(days=14), (timeline_end - timeline_start) * 0.03
    )

    return figure.update_layout(
        height=max(240, 45 * len(chart_rows)),
        margin=dict(l=10, r=10, t=20, b=10),
        legend_title_text=None,
        xaxis=dict(
            title=None,
            type="date",
            tickformat="%Y",
            range=[timeline_start - padding, timeline_end + padding],
        ),
        yaxis=dict(
            title=None,
            tickmode="array",
            tickvals=chart_rows["ypos"].tolist(),
            ticktext=chart_rows["timeline_row"].tolist(),
            range=[-0.7, len(chart_rows) - 0.3],
            zeroline=False,
        ),
    )


def display_tenure_history(
    timeline: pd.DataFrame, current: pd.DataFrame, last_update: pd.Timestamp | None
) -> None:
    """Render the tenure timeline and current-manager captions."""
    if timeline.empty:
        st.info("No fund manager tenure data available for this fund.")
        return

    st.plotly_chart(_build_tenure_chart(timeline), use_container_width=True)

    if not current.empty:
        parts = [
            f"{row.fund_manager} managing since the past {row.tenure_years:.1f} years"
            for row in current.itertuples(index=False)
        ]
        st.caption("Current fund manager: " + "; ".join(parts))
    if pd.notna(last_update):
        st.caption(
            "Last tenure update on Supabase: "
            + pd.to_datetime(last_update).strftime("%b-%Y")
        )
    else:
        st.caption(
            "Last tenure update on Supabase: Not available "
            "(no non-null 'to date' recorded)"
        )


def display_tenure_filter(result: pd.DataFrame, minimum_years: float) -> None:
    st.markdown(f"### Funds where current manager tenure is >= {minimum_years:.1f} years")
    if result.empty:
        st.info("No fund-manager pairs match the minimum tenure filter.")
    else:
        st.dataframe(result, use_container_width=True)


def display_underlying_rows(timeline: pd.DataFrame) -> None:
    with st.expander("Show underlying tenure rows"):
        show = timeline.copy()
        show["from_date"] = pd.to_datetime(show["from_date"]).dt.strftime("%b-%Y")
        show["to_date"] = pd.to_datetime(show["to_date"]).dt.strftime("%b-%Y")
        show.loc[show["to_date"].isna(), "to_date"] = "Current"
        st.dataframe(show[["fund_manager", "from_date", "to_date"]], use_container_width=True)
