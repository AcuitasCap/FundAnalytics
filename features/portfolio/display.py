"""Formatting and Streamlit rendering for the Portfolio explorer."""

import pandas as pd
import streamlit as st


def _apply_frequency_snapshots(data: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if data.empty:
        return data
    result = data.copy()
    result["month_end"] = pd.to_datetime(result["month_end"]).dt.to_period("M").dt.to_timestamp("M")
    if frequency == "Monthly":
        return result
    group = result["month_end"].dt.to_period("Q" if frequency == "Quarterly" else "Y")
    return result[result["month_end"].eq(result.groupby(group)["month_end"].transform("max"))].copy()


def _format_number(value) -> str:
    try:
        return "-" if pd.isna(value) or abs(float(value)) < 0.0001 else f"{float(value):.1f}"
    except Exception:
        return "-"


def _build_size_asset_allocation_pivot(holdings: pd.DataFrame, size_band: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if holdings is None or holdings.empty:
        return pd.DataFrame()
    data = _apply_frequency_snapshots(holdings, frequency)
    data["asset_type"] = data.get("asset_type", "").fillna("").astype(str)
    data["isin"] = data.get("isin", "").fillna("").astype(str)
    data["weight_pct"] = pd.to_numeric(data["weight_pct"], errors="coerce").fillna(0.0)
    bands = size_band.copy() if size_band is not None else pd.DataFrame()
    if not bands.empty:
        bands["month_end"] = pd.to_datetime(bands["month_end"]).dt.to_period("M").dt.to_timestamp("M")
        bands["isin"] = bands["isin"].fillna("").astype(str)
        bands = bands[bands["size_band"].isin(["Large", "Mid", "Small"])].sort_values(["isin", "month_end"])
        def merge_band(group):
            matching = bands[bands["isin"] == group.name][["month_end", "size_band"]]
            if matching.empty:
                group["size_band"] = ""
                return group
            return pd.merge_asof(group.sort_values("month_end"), matching, on="month_end", direction="backward")
        data = data.sort_values(["isin", "month_end"]).groupby("isin", group_keys=False).apply(merge_band)
    else:
        data["size_band"] = ""
    asset_type = data["asset_type"].str.strip()
    data["Allocation"] = asset_type.where(asset_type != "", other="Others")
    data.loc[asset_type.str.lower().eq("cash"), "Allocation"] = "Cash"
    data.loc[asset_type.isin(["Overseas Equities", "ADRs & GDRs"]), "Allocation"] = "Overseas Equities"
    domestic = asset_type.eq("Domestic Equities")
    bands = data.loc[domestic, "size_band"].astype(str).str.strip()
    data.loc[domestic, "Allocation"] = bands.where(bands.isin(["Large", "Mid", "Small"]), other="Unclassified")
    data["period"] = data["month_end"].dt.strftime("%b %Y")
    order = data[["period", "month_end"]].drop_duplicates().sort_values("month_end")["period"].tolist()
    pivot = data.groupby(["Allocation", "period"], as_index=False)["weight_pct"].sum().pivot(index="Allocation", columns="period", values="weight_pct").fillna(0.0).reindex(columns=order)
    preferred = ["Large", "Mid", "Small", "Overseas Equities", "Cash", "Unclassified", "Others"]
    return pivot.reindex(preferred + sorted(row for row in pivot.index if row not in preferred)).reset_index()


def _render_portfolio_view(result: tuple[pd.DataFrame, pd.DataFrame], context: dict) -> None:
    holdings, size_bands = result
    if holdings.empty:
        st.warning("No portfolio data found for this fund and period.")
        return
    frequency, fund_id = context["frequency"], context["fund_id"]
    holdings["month_end"] = pd.to_datetime(holdings["month_end"]).dt.to_period("M").dt.to_timestamp("M")
    holdings["weight_pct"] = pd.to_numeric(holdings["weight_pct"], errors="coerce").fillna(0.0)
    holdings["company_name"] = holdings["company_name"].fillna("").astype(str)
    snapshots = _apply_frequency_snapshots(holdings, frequency)
    if snapshots.empty:
        st.warning("No portfolio snapshots found after applying the selected frequency.")
        return
    snapshots["period"] = snapshots["month_end"].dt.strftime("%b %Y")
    period_order = snapshots[["period", "month_end"]].drop_duplicates().sort_values("month_end")["period"].tolist()
    if not period_order:
        st.warning("No periods found for the selected range/frequency.")
        return
    numeric = snapshots.pivot_table(index="company_name", columns="period", values="weight_pct", aggfunc="sum", fill_value=0.0).reindex(columns=period_order)
    numeric = numeric.sort_values(period_order[-1], ascending=False)
    numeric = pd.concat([numeric, pd.DataFrame([numeric.sum(axis=0)], index=["Total"])])
    holdings_numeric = numeric.reset_index().rename(columns={numeric.reset_index().columns[0]: "Instrument"})
    holdings_numeric.insert(0, "Sr. No", pd.Series(range(1, len(holdings_numeric) + 1), dtype="object"))
    holdings_numeric.loc[holdings_numeric["Instrument"].eq("Total"), "Sr. No"] = ""
    display = holdings_numeric.copy()
    for column in display.columns:
        if column not in ["Sr. No", "Instrument"]:
            display[column] = display[column].apply(_format_number)
    st.subheader("5. Portfolio holdings")
    st.caption(f"Rows: instruments · Columns: {frequency.lower()} snapshots from {period_order[0]} to {period_order[-1]}")
    st.dataframe(display, use_container_width=True, hide_index=True, height=520, column_config={"Sr. No": st.column_config.TextColumn("Sr. No", width="small"), "Instrument": st.column_config.TextColumn("Instrument", width="large")})
    csv = holdings_numeric.copy()
    for column in csv.columns:
        if column not in ["Sr. No", "Instrument"]:
            csv[column] = pd.to_numeric(csv[column], errors="coerce").fillna(0.0).round(1)
    st.download_button("Download holdings CSV", csv.to_csv(index=False).encode("utf-8"), f"holdings_{fund_id}_{period_order[0].replace(' ','_')}_to_{period_order[-1].replace(' ','_')}_{frequency.lower()}.csv", "text/csv", key="dl_holdings_csv")
    allocation = _build_size_asset_allocation_pivot(holdings, size_bands, frequency)
    st.subheader("6. Size / asset-type allocation")
    if allocation.empty:
        st.info("No allocation data available.")
        return
    numeric_columns = [column for column in allocation.columns if column != "Allocation"]
    allocation_numeric = allocation.copy()
    for column in numeric_columns:
        allocation_numeric[column] = pd.to_numeric(allocation_numeric[column], errors="coerce").fillna(0.0)
    allocation_numeric = pd.concat([allocation_numeric, pd.DataFrame([{"Allocation": "Total", **allocation_numeric[numeric_columns].sum().to_dict()}])], ignore_index=True)
    allocation_display = allocation_numeric.copy()
    for column in numeric_columns:
        allocation_display[column] = allocation_display[column].apply(_format_number)
    st.dataframe(allocation_display, use_container_width=True, hide_index=True, height=320, column_config={"Allocation": st.column_config.TextColumn("Allocation", width="large")})
    st.download_button("Download allocation CSV", allocation_numeric.to_csv(index=False).encode("utf-8"), f"allocation_{fund_id}_{context['start_date']}_to_{context['end_date']}_{frequency.lower()}.csv", "text/csv", key="dl_alloc_csv")


def _render_active_share(result: pd.DataFrame) -> None:
    if result.empty:
        st.warning("Could not compute active share for any period.")
        return
    chart_data = result.dropna(subset=["active_share_pct"])
    if chart_data.empty:
        st.warning("Active share is NaN for all periods.")
        return
    import altair as alt
    chart = alt.Chart(chart_data).mark_line(point=True).encode(x=alt.X("period_date:T", title="Period", axis=alt.Axis(format="%b %Y", labelAngle=-45)), y=alt.Y("active_share_pct:Q", title="Active share (%)"), tooltip=[alt.Tooltip("period_date:T", title="Period", format="%b %Y"), alt.Tooltip("active_share_pct:Q", title="Active share (%)", format=".1f")]).properties(height=300)
    st.subheader("6. Active share over time")
    st.altair_chart(chart, use_container_width=True)
    st.subheader("7. Active share table")
    horizontal = result[["period_label", "active_share_pct"]].set_index("period_label").T
    horizontal.index = ["Active share (%)"]
    st.dataframe(horizontal.style.format("{:.1f}"))


def display_portfolio_result(mode: str, result, context: dict) -> None:
    """Render either the view-portfolio result or Active Share result."""
    if mode == "View portfolio":
        _render_portfolio_view(result, context)
    else:
        _render_active_share(result)
