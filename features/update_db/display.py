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
                "Scheme Name": ["ABC Flexi Cap Fund", "ABC Flexi Cap Fund"],
                "Month-end (yyyymm)": [202401, 202401],
                "Instrument": ["Reliance Industries", "Cash"],
                "Holding (%)": [8.12, 1.25],
                "Asset type": ["Domestic Equities", "Cash"],
                "ACE code": ["100325", ""],
            }
        )
        st.dataframe(sample)
        st.info(
            "ACE code is required for Domestic Equities and must exist in "
            "fundlab.stock_master_ace. Leave it blank for other asset types. "
            "Allowed asset types are Domestic Equities, Overseas Equities, "
            "Others Equities, ADRs & GDRs, and Cash. Tag other instruments as Cash "
            "in the source file."
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
    elif upload_type == "Update stock master":
        st.markdown("**Expected format (ACE stock master.xlsx):**")
        st.dataframe(pd.DataFrame({
            "ace_code": ["100325"],
            "company_name": ["Reliance Industries Ltd."],
            "industry": ["Petroleum"],
        }))
        st.info(
            "ACE codes should be stored as text in Excel. Existing industries inherit "
            "their financial classification; each new industry requires your confirmation."
        )
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
            "- **3 columns**: ACE code, quarter-end in `YYYYMM`, adjusted PAT (absolute)\n"
            "- Example row: `288853 | 202403 | 125000000`\n"
            "- Keep ACE codes as text in Excel so any leading zeros are preserved."
            )
    elif upload_type == "Company sales (quarterly)":
            st.markdown(
                "- **3 columns**: ACE code, quarter-end in `YYYYMM`, sales (absolute)\n"
                "- Example row: `288853 | 202403 | 875000000`\n"
                "- Keep ACE codes as text in Excel so any leading zeros are preserved."
            )
    elif upload_type == "Company book value (annual)":
            st.markdown(
                "- **3 columns**: ACE code, year-end in `YYYYMM`, book value (absolute net worth)\n"
                "- Example row: `288853 | 202403 | 2150000000`\n"
                "- Keep ACE codes as text in Excel so any leading zeros are preserved."
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

def display_missing_portfolio_ace_codes(report: pd.DataFrame) -> None:
    st.error(
        "Portfolio upload blocked. The ACE codes below are missing from "
        "fundlab.stock_master_ace. Update the ACE stock master before retrying; "
        "no portfolio rows were uploaded."
    )
    st.dataframe(
        report.rename(columns={
            "ace_code": "Missing ACE code",
            "rows_in_upload": "Holding rows",
            "example_instrument": "Example instrument",
        }),
        hide_index=True,
    )
    st.caption(
        "Use Update DB → Update stock master to add the missing ACE codes. "
        "The older 'Stock ISIN, industry, financial/non-financial' option still "
        "updates the legacy ISIN stock master."
    )


def display_company_name_ace_conflicts(report: pd.DataFrame) -> None:
    st.error(
        "Stock-master upload blocked. Resolve these company names that already map "
        "to different ACE codes; neither table was changed."
    )
    st.dataframe(report.rename(columns={
        "company_name": "Uploaded company name",
        "uploaded_ace_code": "Uploaded ACE code",
        "existing_name": "Existing company name",
        "existing_ace_code": "Existing ACE code",
    }), hide_index=True)


def display_validation_success(summary: dict) -> None:
    st.success("Dry run successful. No critical format errors detected.")
    st.write("Summary:")
    st.json(summary)

def display_upload_success(summary: dict | None = None) -> None:
    st.success("✅ Upload completed successfully.")
    if summary is not None:
        st.write("Summary:")
        st.json(summary)


def display_ace_stock_master_next_steps() -> None:
    st.warning(
        "**IMPORTANT NEXT STEP: Update price, market-cap, and fundamental data "
        "for any newly added stocks.**\n\n"
        "Populate `stock_price_ace`, quarterly sales/PAT in "
        "`stock_quarterly_financials_ace`, and annual book value in "
        "`stock_annual_bv_ace` before relying on their valuation results."
    )
