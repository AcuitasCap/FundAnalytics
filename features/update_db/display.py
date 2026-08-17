from __future__ import annotations

import pandas as pd
import streamlit as st

def show_expected_format(upload_type: str):
    if upload_type == "Fund NAVs":
        st.markdown("**Expected format (Fund NAV update.xlsx):**")
        sample = pd.DataFrame(
            {
                "Fund": ["ABC Flexi Cap Fund", "ABC Flexi Cap Fund"],
                "Date": ["2024-01-31", "2024-02-29"],
                "NAV": [145.32, 147.10],
                "Category": ["Flexi Cap Fund", "Flexi Cap Fund"],
                "Style": ["Growth", "Growth"],
            }
        )
        st.dataframe(sample)
    elif upload_type == "Benchmark NAVs":
        st.markdown("**Expected format (BM NAV update.xlsx):**")
        sample = pd.DataFrame(
            {
                "Benchmark": ["Nifty 50 TRI", "Nifty 50 TRI"],
                "Date": ["2024-01-31", "2024-02-29"],
                "NAV": [24857.23, 25210.11],
                "Category": ["Index", "Index"],
                "Style": ["Blend", "Blend"],
            }
        )
        st.dataframe(sample)
    elif upload_type == "Fund portfolios":
        st.markdown("**Expected format (Fund portfolios.xlsx):**")
        sample = pd.DataFrame(
            {
                "Scheme Name": ["ABC Flexi Cap Fund"],
                "Month-end (yyyymm)": [202401],
                "Instrument": ["Reliance Industries"],
                "Holding (%)": [8.12],
                "Asset type": ["Domestic Equities"],
                "ISIN": ["INE002A01018"],
            }
        )
        st.dataframe(sample)
        st.info(
            "Note: In 'Asset type', anything other than "
            "'Domestic Equities', 'Overseas Equities', 'Others Equities', "
            "and 'ADRs & GDRs' will be treated as cash and uploads with other values will be aborted."
        )
    elif upload_type == "Stock ISIN, industry, financial/non-financial":
        st.markdown("**Expected format (Stock industry.xlsx):**")
        sample = pd.DataFrame(
            {
                "ISIN": ["INE002A01018"],
                "Company name": ["Reliance Industries"],
                "Industry": ["Petroleum"],
                "Financial?": [False],
            }
        )
        st.dataframe(sample)
    elif upload_type == "Company RoE / RoCE":
        st.markdown("**Expected format (Stock RoE.xlsx):**")
        sample = pd.DataFrame(
            {
                "ISIN": ["INE002A01018"],
                "Company name": ["Reliance Industries"],
                "Year-end (YYYYMM)": [202303],
                "RoE": [10.2],
                "RoCE": [12.8],
            }
        )
        st.dataframe(sample)
    elif upload_type == "Stock prices and market cap":
        st.markdown("**Expected format (example):**")
        sample = pd.DataFrame(
            {
                "ISIN": ["INE002A01018", "INE002A01018"],
                "Date": ["31-01-2024", "29-02-2024"],  # dd-mm-yyyy
                "Market cap": [190000.0, 195000.0],
                "Stock price": [2450.5, 2501.0],
            }
        )
        st.dataframe(sample)
        st.info("Dates must be in dd-mm-yyyy format. Market cap and stock price must be numeric.")
    elif upload_type == "Company PAT (quarterly)":
        st.markdown(
            "- **3 columns**: ISIN, quarter-end in `YYYYMM`, adjusted PAT (absolute)\n"
            "- Example row: `INE123A01016 | 202403 | 125000000`"
            )
    elif upload_type == "Company sales (quarterly)":
            st.markdown(
                "- **3 columns**: ISIN, quarter-end in `YYYYMM`, sales (absolute)\n"
                "- Example row: `INE123A01016 | 202403 | 875000000`"
            )
    elif upload_type == "Company book value (annual)":
            st.markdown(
                "- **3 columns**: ISIN, year-end in `YYYYMM`, book value (absolute net worth)\n"
                "- Example row: `INE123A01016 | 202403 | 2150000000`"
            )
    elif upload_type == "Fund manager tenure":
        st.markdown("**Expected format (Fund manager tenure.xlsx):**")
        sample = pd.DataFrame(
            {
                "Fund name": ["ICICI Value Fund", "ICICI Value Fund"],
                "Inception date": ["2004-08-16", "2004-08-16"],
                "Fund manager": ["A. Manager", "B. Manager"],
                "From date": ["Jan-2020", "Apr-2023"],  # Mmm-YYYY
                "To date": ["Mar-2023", ""],            # blank => current
            }
        )
        st.dataframe(sample)
        st.info("From/To must be in Mmm-YYYY (e.g., Jan-2024). 'To date' can be blank for the current manager.")

def display_validation_success(summary: dict) -> None:
    st.success("Dry run successful. No critical format errors detected.")
    st.write("Summary:")
    st.json(summary)

def display_upload_success(summary: dict | None = None) -> None:
    st.success("✅ Upload completed successfully.")
    if summary is not None:
        st.write("Summary:")
        st.json(summary)
