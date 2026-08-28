"""Housekeeping page: UI and orchestration for explicit maintenance jobs."""

import pandas as pd
import streamlit as st

from core.navigation import home_button
from . import jobs
from .display import show_exception_report, store_exception_report

def housekeeping_page():
    home_button()
    st.header("Housekeeping \u2013 Derived Tables")
    st.write("Run these steps after uploading new raw data (NAVs, portfolios, RoE/RoCE, prices) to refresh derived tables.")

    simple_jobs = [
        ("1. Recompute size bands (Large/Mid/Small)", jobs.recompute_size_bands, "Size bands updated."),
        ("2. Recompute 5-year median RoE/RoCE", jobs.recompute_quality_medians, "5-year medians updated."),
        ("3. Recompute quality quartiles (Q1\u2013Q4)", jobs.recompute_quality_quartiles, "Quality quartiles updated."),
        ("4. Refresh stock valuations", jobs.rebuild_stock_monthly_valuations, "Stock valuations updated."),
    ]
    for label, job, success in simple_jobs:
        if st.button(label):
            job()
            st.success(success)

    st.markdown("---")
    st.subheader("Precompute rolling returns")
    if st.button("5. Pre-compute and store 3Y and 1Y rolling returns"):
        st.session_state["rolling_refresh_summary"] = jobs.refresh_precomputed_rolling_returns()
    if "rolling_refresh_summary" in st.session_state:
        summary = st.session_state["rolling_refresh_summary"]
        st.write("Summary:", {key: summary.get(key) for key in ("windows_months", "fund_entities_seen", "benchmark_entities_seen", "fund_rows_candidate", "benchmark_rows_candidate", "fund_rows_inserted", "benchmark_rows_inserted", "fund_entities_updated", "benchmark_entities_updated")})
        inserted = int(summary.get("fund_rows_inserted", 0)) + int(summary.get("benchmark_rows_inserted", 0))
        if inserted:
            st.success(f"Inserted {inserted:,} new rolling-return rows without modifying existing data.")
        else:
            st.info("No new month-end NAV windows were available to append to the rolling-return tables.")

    st.markdown("---")
    st.subheader("Refresh stock dividend yields")
    scope = st.selectbox("Dividend yield refresh scope", ["All", "Last 3 years"], key="dividend_yield_scope")
    if st.button("6. Refresh stock dividend yields"):
        for suffix in ("excel_bytes", "excel_name", "exc_rows", "exceptions_preview"):
            st.session_state.pop(f"div_yield_{suffix}", None)
        summary, exceptions_df = jobs.refresh_stock_dividend_yields(scope=scope)
        st.session_state["div_yield_summary"] = summary
        store_exception_report("div_yield", summary, exceptions_df, "stock_dividend_yield_exceptions.xlsx")
    if "div_yield_summary" in st.session_state:
        summary = st.session_state["div_yield_summary"]
        st.write("Summary:", {key: summary.get(key) for key in ("scope", "date_from", "rows_aggregated", "updates_attempted", "updates_applied", "exceptions")})
    show_exception_report("div_yield", "\N{DOWNWARDS BLACK ARROW}\ufe0f Download dividend yield exception report (Excel)", "stock_dividend_yield_exceptions.xlsx")

    st.markdown("---")
    st.subheader("Refresh adjusted prices")
    if st.button("7. Refresh adjusted prices (adj_multiplier + adj_price)"):
        for suffix in ("exc_excel_bytes", "exc_excel_name", "exc_rows", "exceptions_preview", "summary"):
            st.session_state.pop(f"adj_price_{suffix}", None)
        summary, exceptions_df = jobs.refresh_adjusted_prices()
        st.session_state["adj_price_summary"] = summary
        # Preserve existing session-state naming for this report.
        if isinstance(exceptions_df, pd.DataFrame) and not exceptions_df.empty:
            from io import BytesIO
            xlsx_io = BytesIO()
            with pd.ExcelWriter(xlsx_io, engine="openpyxl") as writer:
                exceptions_df.to_excel(writer, sheet_name="exceptions", index=False)
                pd.DataFrame([summary]).to_excel(writer, sheet_name="summary", index=False)
            st.session_state["adj_price_exc_excel_bytes"] = xlsx_io.getvalue()
            st.session_state["adj_price_exc_excel_name"] = "adjusted_price_refresh_exceptions.xlsx"
            st.session_state["adj_price_exc_rows"] = int(len(exceptions_df))
            st.session_state["adj_price_exceptions_preview"] = exceptions_df
    if "adj_price_summary" in st.session_state:
        summary = st.session_state["adj_price_summary"]
        st.write("Summary:", {key: summary.get(key) for key in ("actions_total", "actions_mapped", "actions_unmapped", "unique_isins_actions", "price_rows_total", "price_rows_updated", "min_price_date", "max_price_date", "exceptions")})
    preview = st.session_state.get("adj_price_exceptions_preview")
    if isinstance(preview, pd.DataFrame) and not preview.empty:
        st.warning(f"Exceptions found: {len(preview):,}")
        st.dataframe(preview.head(200))
    if "adj_price_exc_excel_bytes" in st.session_state:
        st.download_button(label="\N{DOWNWARDS BLACK ARROW}\ufe0f Download adjusted price exception report (Excel)", data=st.session_state["adj_price_exc_excel_bytes"], file_name=st.session_state.get("adj_price_exc_excel_name", "adjusted_price_refresh_exceptions.xlsx"), mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="download_adjusted_price_exception_excel")
        st.caption(f"Exception rows: {st.session_state.get('adj_price_exc_rows', 0):,}")

    st.markdown("---")
    st.subheader("Upload precomputed stock valuations to DB")
    uploaded = st.file_uploader("Select a stock valuations Excel workbook (.xlsx)", type=["xlsx"], key="stock_val_upload")
    if uploaded is not None:
        st.write(f"Selected file: **{uploaded.name}**")
    if st.button("8. Upload this workbook to Supabase"):
        jobs.upload_stock_monthly_valuations_from_excel(uploaded)
