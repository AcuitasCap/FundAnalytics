"""Streamlit rendering helpers for Housekeeping results."""

from io import BytesIO
import pandas as pd
import streamlit as st

def show_fund_valuation_download():
    if "fund_valuations_csv_bytes" in st.session_state:
        st.download_button(label="\N{DOWNWARDS BLACK ARROW}\ufe0f Download fund_monthly_valuations CSV", data=st.session_state["fund_valuations_csv_bytes"], file_name=st.session_state.get("fund_valuations_csv_name", "fund_monthly_valuations_delta.csv"), mime="text/csv", key="download_fund_monthly_valuations_csv")
        st.caption(f"Rows: {st.session_state.get('fund_valuations_rows', 0):,}")

def store_exception_report(prefix, summary, exceptions_df, filename):
    if isinstance(exceptions_df, pd.DataFrame) and not exceptions_df.empty:
        xlsx_io = BytesIO()
        with pd.ExcelWriter(xlsx_io, engine="openpyxl") as writer:
            exceptions_df.to_excel(writer, sheet_name="exceptions", index=False)
            pd.DataFrame([summary]).to_excel(writer, sheet_name="summary", index=False)
        st.session_state[f"{prefix}_excel_bytes"] = xlsx_io.getvalue()
        st.session_state[f"{prefix}_excel_name"] = filename
        st.session_state[f"{prefix}_exc_rows"] = int(len(exceptions_df))
        st.session_state[f"{prefix}_exceptions_preview"] = exceptions_df

def show_exception_report(prefix, label, default_filename):
    preview = st.session_state.get(f"{prefix}_exceptions_preview")
    if isinstance(preview, pd.DataFrame) and not preview.empty:
        st.warning(f"Exceptions found: {len(preview):,}")
        st.dataframe(preview.head(200))
    if f"{prefix}_excel_bytes" in st.session_state:
        st.download_button(label=label, data=st.session_state[f"{prefix}_excel_bytes"], file_name=st.session_state.get(f"{prefix}_excel_name", default_filename), mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"download_{prefix}_excel")
        st.caption(f"Exception rows: {st.session_state.get(f'{prefix}_exc_rows', 0):,}")
