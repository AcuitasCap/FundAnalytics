"""
Fund Analytics Dashboard v3.13

What’s new vs v3.7:
- Adds "Relative CAGR vs Benchmark — 1Y / 3Y / 5Y / 7Y" table (as of P2P End month), green/red shading.
- Fixes syntax error in PDF section (balanced try/except; no stray blocks).
- Keeps all prior features: stacked selectors, rolling (start-domain vs end-domain), multi-fund charts,
  yearly tables (Funds as rows), misalignment highlight only on fund name cell, P2P table, PDF printing.

Notes:
- All dates are treated as month-ends.
- Strict FY uses March; CY uses December; missing endpoints => blank for that year.
- Rolling: Start selector = START-DOMAIN; End selector = END-DOMAIN.
"""

import re
import math
import io
from io import BytesIO
import datetime as dt   
import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
import sqlalchemy as sa
import altair as alt
from sqlalchemy.exc import SQLAlchemyError
import plotly.express as px
import plotly.graph_objects as go
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from datetime import datetime, date


def check_password():
    """Return True if the user entered the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["auth"]["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store plaintext
        else:
            st.session_state["password_correct"] = False

    # First run: ask for password
    if "password_correct" not in st.session_state:
        st.text_input(
            "Enter password", type="password",
            on_change=password_entered, key="password"
        )
        return False

    # Password was entered, but incorrect
    elif not st.session_state["password_correct"]:
        st.text_input(
            "Enter password", type="password",
            on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False

    # Password correct
    else:
        return True

if not check_password():
    st.stop()



def home_button():
    """Render a Home button that jumps back to the Home page ."""
    if st.button("🏠 Home"):
        st.session_state["page"] = "Home"
        st.rerun()


st.set_page_config(page_title="", layout="wide")

engine = create_engine(
    "postgresql+psycopg2://",
    connect_args={
        "host": st.secrets["pg"]["host"],
        "port": st.secrets["pg"]["port"],
        "user": st.secrets["pg"]["user"],
        "password": st.secrets["pg"]["password"],
        "dbname": st.secrets["pg"]["database"],
        "sslmode": "require",
    },
    pool_pre_ping=True,
)

def get_engine():
    return engine

##### HELPER FUNCTIONS RESIDE IN THIS BLOCK, <BEFORE>! THE PAGE FUNCTIONS #####

def month_year_to_last_day(year: int, month: int) -> dt.date:
    if month == 12:
        return dt.date(year, 12, 31)
    # first day of next month minus one day
    first_next = dt.date(year + (month // 12), ((month % 12) + 1), 1)
    return first_next - dt.timedelta(days=1)

def load_stock_roe_roce():
    engine = get_engine()
    query = """
        SELECT isin, year_end_date, roe, roce
        FROM fundlab.stock_roe_roce
        WHERE roe IS NOT NULL OR roce IS NOT NULL
        ORDER BY isin, year_end_date
    """
    df = pd.read_sql(query, engine)
    df["year_end_date"] = pd.to_datetime(df["year_end_date"]).dt.date

    # group by ISIN for fast lookup
    grouped = {isin: sub_df.sort_values("year_end_date").reset_index(drop=True)
               for isin, sub_df in df.groupby("isin")}
    return grouped


def map_headers(df: pd.DataFrame, mapping: dict, required: set):
    """
    df: raw DataFrame from Excel
    mapping: { logical_col: [alias1, alias2, ...] }
    required: set of logical columns that must be present

    Returns: new_df with canonical column names
    Raises: ValueError with a clear message if missing required columns
    """
    clean_cols = {c: c.strip().lower() for c in df.columns}
    col_map = {}  # physical -> logical

    for logical, aliases in mapping.items():
        found = None
        for phys, clean in clean_cols.items():
            if clean in aliases:
                found = phys
                break
        if found:
            col_map[found] = logical
        elif logical in required:
            raise ValueError(f"Missing required column for role '{logical}' (expected one of: {aliases})")

    # Apply renames
    df = df.rename(columns=col_map)

    # Ensure required logical columns exist
    missing_after = [col for col in required if col not in df.columns]
    if missing_after:
        raise ValueError(f"Missing required columns after mapping: {missing_after}")

    return df


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





# Update Supabase with fund NAVs
FUND_NAV_COLS = {
    "fund_name": ["fund", "scheme", "scheme name", "fund_name"],
    "nav_date": ["date", "nav date", "month_end", "month-end"],
    "nav_value": ["nav", "nav value"],
    "fund_category": ["category", "fund category"],
    "fund_style": ["style", "style name"],
}

def validate_fund_navs(df_raw: pd.DataFrame):
    required = {"fund_name", "nav_date", "nav_value"}
    df = map_headers(df_raw.copy(), FUND_NAV_COLS, required)

    # Parse dates
    df["nav_date"] = pd.to_datetime(df["nav_date"]).dt.date
    df["nav_value"] = pd.to_numeric(df["nav_value"], errors="coerce")

    if df["nav_value"].isna().any():
        raise ValueError("Some NAV values could not be parsed as numbers.")

    # In-file duplicate check
    dup_keys = df.groupby(["fund_name", "nav_date"]).size()
    dups = dup_keys[dup_keys > 1]
    if not dups.empty:
        raise ValueError(
            f"Duplicate (fund_name, nav_date) rows found in file: {len(dups)} duplicates."
        )

    summary = {
        "rows": int(len(df)),
        "unique_fund_dates": int(len(dup_keys)),
    }
    return df, summary

def debug_portfolio_valuation_point(
    fund_id: int,
    target_date: dt.date,
    segment_choice: str,
    metric_choice: str,
    mode: str,
):
    """
    Debug helper: for a single fund + month, print all guts of the valuation calc.

    mode:
      - 'Valuations of historical portfolios'
      - 'Historical valuations of current portfolio'
    """

    engine = get_engine()

    # Work at month-level (ignore day)
    target_period = pd.Period(target_date, freq="M")
    month_start = target_period.to_timestamp("M")  # last day-of-month as canonical date

    st.write(f"### Debug valuation point")
    st.write(f"Fund: {fund_id}, Month: {target_period} (canonical date {month_start.date()})")
    st.write(f"Segment: {segment_choice}, Metric: {metric_choice}, Mode: {mode}")

    # Metric mapping
    metric_map = {"P/S": "ps", "P/E": "pe", "P/B": "pb"}
    metric_key = metric_map.get(metric_choice)
    if metric_key is None:
        st.error(f"Unsupported metric_choice: {metric_choice}")
        return

    # ------------------------------------------------------------------
    # 1) Get holdings snapshot for this mode
    # ------------------------------------------------------------------
    if mode == "Valuations of historical portfolios":
        # Use actual historical holdings for that calendar month
        with engine.begin() as conn:
            h_sql = text(
                """
                SELECT
                    fp.fund_id,
                    fp.month_end,
                    fp.isin,
                    fp.holding_weight AS weight_pct,
                    fp.asset_type,
                    sm.is_financial
                FROM fundlab.fund_portfolio fp
                JOIN fundlab.stock_master sm
                  ON sm.isin = fp.isin
                WHERE fp.fund_id = :fund_id
                  AND fp.asset_type = 'Domestic Equities';
                """
            )
            holdings = pd.read_sql(h_sql, conn, params={"fund_id": fund_id})

        if holdings.empty:
            st.error("No historical holdings found for this fund.")
            return

        holdings["month_end"] = pd.to_datetime(holdings["month_end"], errors="coerce")
        holdings = holdings.dropna(subset=["month_end"])
        holdings["month_key"] = holdings["month_end"].dt.to_period("M")
        holdings = holdings[holdings["month_key"] == target_period].copy()

        if holdings.empty:
            st.error("No holdings for this fund in the selected month.")
            return

    elif mode == "Historical valuations of current portfolio":
        # Use latest snapshot <= target_date as anchor
        with engine.begin() as conn:
            anchor_sql = text(
                """
                SELECT MAX(month_end) AS anchor_month_end
                FROM fundlab.fund_portfolio
                WHERE fund_id = :fund_id
                  AND month_end <= :target_date;
                """
            )
            anchor = conn.execute(
                anchor_sql, {"fund_id": fund_id, "target_date": target_date}
            ).fetchone()
            if not anchor or anchor.anchor_month_end is None:
                st.error("No anchor holdings found on or before target_date.")
                return

            h_sql = text(
                """
                SELECT
                    fp.fund_id,
                    fp.month_end,
                    fp.isin,
                    fp.holding_weight AS weight_pct,
                    fp.asset_type,
                    sm.is_financial
                FROM fundlab.fund_portfolio fp
                JOIN fundlab.stock_master sm
                  ON sm.isin = fp.isin
                WHERE fp.fund_id = :fund_id
                  AND fp.month_end = :anchor_month_end
                  AND fp.asset_type = 'Domestic Equities';
                """
            )
            holdings = pd.read_sql(
                h_sql,
                conn,
                params={"fund_id": fund_id, "anchor_month_end": anchor.anchor_month_end},
            )

        if holdings.empty:
            st.error("No anchor holdings found.")
            return

        holdings["month_end"] = pd.to_datetime(holdings["month_end"], errors="coerce")
        holdings["month_key"] = target_period  # we will revalue at this month

    else:
        st.error(f"Unsupported mode: {mode}")
        return

    # Segment filter
    if segment_choice == "Financials":
        holdings = holdings[holdings["is_financial"].astype(bool)].copy()
    elif segment_choice == "Non-financials":
        holdings = holdings[~holdings["is_financial"].astype(bool)].copy()

    if holdings.empty:
        st.error("No holdings in this segment for the selected month.")
        return

    holdings["isin"] = holdings["isin"].astype(str).str.strip()
    holdings["weight_pct"] = pd.to_numeric(holdings["weight_pct"], errors="coerce")
    sum_w = holdings["weight_pct"].sum()
    if sum_w <= 0:
        st.error("Sum of weights is not positive.")
        return

    holdings["w_domestic"] = holdings["weight_pct"] / sum_w

    st.write("#### Step 1: Raw holdings (after segment filter & domestic rebase)")
    st.dataframe(
        holdings[
            ["isin", "weight_pct", "w_domestic", "is_financial"]
        ].reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # 2) Get stock-level multiples for that month (using month_key)
    # ------------------------------------------------------------------
    all_isins = sorted(holdings["isin"].unique().tolist())
    with engine.begin() as conn:
        v_sql = text(
            """
            SELECT
                isin,
                month_end,
                ps,
                pe,
                pb
            FROM fundlab.stock_monthly_valuations
            WHERE isin = ANY(:isins)
              AND month_end BETWEEN :start AND :end;
            """
        )
        # small buffer around the month just in case; we'll filter via period
        start = target_period.start_time.date()
        end = target_period.to_timestamp("M").date()
        vals = pd.read_sql(
            v_sql,
            conn,
            params={"isins": all_isins, "start": start, "end": end},
        )

    if vals.empty:
        st.error("No stock_monthly_valuations for these ISINs in this month window.")
        return

    vals["isin"] = vals["isin"].astype(str).str.strip()
    vals["month_end"] = pd.to_datetime(vals["month_end"], errors="coerce")
    vals = vals.dropna(subset=["month_end"])
    vals["month_key"] = vals["month_end"].dt.to_period("M")
    vals = vals[vals["month_key"] == target_period].copy()

    if vals.empty:
        st.error("No stock valuations matching this year-month.")
        return

    st.write("#### Step 2: Stock-level multiples in this month")
    st.dataframe(
        vals[["isin", "month_end", "ps", "pe", "pb"]].reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # 3) Merge holdings + multiples, compute weights used and metric
    # ------------------------------------------------------------------
    df = holdings.merge(
        vals[["isin", "month_key", "ps", "pe", "pb"]],
        on=["isin", "month_key"],
        how="left",
    )

    df["metric"] = pd.to_numeric(df[metric_key], errors="coerce")
    df = df[df["metric"] > 0].copy()
    if df.empty:
        st.error("After dropping invalid/<=0 metrics, no stocks remain.")
        return

    sum_w_valid = df["w_domestic"].sum()
    if sum_w_valid <= 0:
        st.error("Sum of w_domestic over valid stocks is not positive.")
        return

    df["w_norm"] = df["w_domestic"] / sum_w_valid
    df["w_times_metric"] = df["w_norm"] * df["metric"]

    st.write("#### Step 3: Effective weights and contributions")
    st.dataframe(
        df[
            ["isin", "w_domestic", "w_norm", "metric", "w_times_metric"]
        ].reset_index(drop=True)
    )

    fund_metric = df["w_times_metric"].sum()
    st.write(f"#### Result: Fund-level {metric_choice} for this point = **{fund_metric:.4f}**")

    # ------------------------------------------------------------------
    # 4) If historical portfolio mode, show stored DB value for comparison
    # ------------------------------------------------------------------
    if mode == "Valuations of historical portfolios":
        with engine.begin() as conn:
            db_sql = text(
                """
                SELECT ps, pe, pb
                FROM fundlab.fund_monthly_valuations
                WHERE fund_id = :fund_id
                  AND month_end = :month_end
                  AND segment = :segment;
                """
            )
            row = conn.execute(
                db_sql,
                {
                    "fund_id": fund_id,
                    "month_end": month_start.date(),
                    "segment": segment_choice,
                },
            ).fetchone()

        if row:
            stored_val = getattr(row, metric_key)
            st.write(
                f"Stored value in fund_monthly_valuations ({metric_choice}) "
                f"for this point: **{stored_val}**"
            )
        else:
            st.write(
                "No row found in fund_monthly_valuations for this fund/month/segment yet."
            )



def upload_fund_navs(df: pd.DataFrame):
    engine = get_engine()
    with engine.begin() as conn:
        # Ensure fund names exist
        funds = sorted(set(df["fund_name"]))
        if funds:
            conn.execute(
                text("""
                    INSERT INTO fundlab.fund (fund_name)
                    SELECT unnest(:names)
                    ON CONFLICT (fund_name) DO NOTHING
                """),
                {"names": funds},
            )

        # Upsert NAVs
        ins = text("""
            INSERT INTO fundlab.fund_nav (fund_id, nav_date, nav_value)
            SELECT f.fund_id, :d, :v FROM fundlab.fund f WHERE f.fund_name = :n
            ON CONFLICT (fund_id, nav_date) DO UPDATE
            SET nav_value = EXCLUDED.nav_value
        """)
        for _, r in df.iterrows():
            conn.execute(
                ins,
                {"n": r["fund_name"], "d": r["nav_date"], "v": float(r["nav_value"])},
            )


def find_conflicts_quarterly(df_clean: pd.DataFrame, table: str):
    """
    Check for (isin, period_end) conflicts in either PAT or sales upload.
    Returns a small list of conflicts if found.
    """
    if df_clean.empty:
        return []

    # Only test unique key pairs
    pairs = df_clean[["isin", "period_end"]].drop_duplicates()

    # Build VALUES block for up to first 5000 pairs (avoid giant SQL)
    max_rows = min(len(pairs), 5000)
    pairs = pairs.iloc[:max_rows]

    values_clause = ", ".join(
        f"(:isin{i}, :period_end{i})" for i in range(len(pairs))
    )

    params = {}
    for i, (_, row) in enumerate(pairs.iterrows()):
        params[f"isin{i}"] = row["isin"]
        params[f"period_end{i}"] = row["period_end"]

    query = text(
        f"""
        SELECT v.isin, v.period_end
        FROM (VALUES
            {values_clause}
        ) AS v(isin, period_end)
        JOIN fundlab.{table} t
        USING (isin, period_end)
        LIMIT 20;
        """
    )

    with engine.begin() as conn:
        rows = conn.execute(query, params).fetchall()

    return rows


def find_conflicts_annual(df_clean: pd.DataFrame):
    """Check (isin, year_end) conflicts for annual book value."""
    if df_clean.empty:
        return []

    pairs = df_clean[["isin", "year_end"]].drop_duplicates()
    max_rows = min(len(pairs), 5000)
    pairs = pairs.iloc[:max_rows]

    values_clause = ", ".join(
        f"(:isin{i}, :year_end{i})" for i in range(len(pairs))
    )

    params = {}
    for i, (_, row) in enumerate(pairs.iterrows()):
        params[f"isin{i}"] = row["isin"]
        params[f"year_end{i}"] = row["year_end"]

    query = text(
        f"""
        SELECT v.isin, v.year_end
        FROM (VALUES
            {values_clause}
        ) AS v(isin, year_end)
        JOIN fundlab.stock_annual_book_value b
        USING (isin, year_end)
        LIMIT 20;
        """
    )

    with engine.begin() as conn:
        rows = conn.execute(query, params).fetchall()

    return rows



#Update Supabase with benchmark NAVs

BENCH_NAV_COLS = {
    "benchmark_name": ["benchmark", "benchmark name", "index name"],
    "nav_date": ["date", "nav date", "month_end", "month-end"],
    "nav_value": ["nav", "nav value"],
    "bench_category": ["category", "index category"],
    "bench_style": ["style", "style name"],
}

def validate_bench_navs(df_raw: pd.DataFrame):
    required = {"benchmark_name", "nav_date", "nav_value"}
    df = map_headers(df_raw.copy(), BENCH_NAV_COLS, required)

    df["nav_date"] = pd.to_datetime(df["nav_date"]).dt.date
    df["nav_value"] = pd.to_numeric(df["nav_value"], errors="coerce")
    if df["nav_value"].isna().any():
        raise ValueError("Some NAV values could not be parsed as numbers.")

    dup_keys = df.groupby(["benchmark_name", "nav_date"]).size()
    dups = dup_keys[dup_keys > 1]
    if not dups.empty:
        raise ValueError(
            f"Duplicate (benchmark_name, nav_date) rows found in file: {len(dups)} duplicates."
        )

    summary = {
        "rows": int(len(df)),
        "unique_bench_dates": int(len(dup_keys)),
    }
    return df, summary

def upload_bench_navs(df: pd.DataFrame):
    engine = get_engine()
    with engine.begin() as conn:
        benches = sorted(set(df["benchmark_name"]))
        if benches:
            conn.execute(
                text("""
                    INSERT INTO fundlab.benchmark (benchmark_name)
                    SELECT unnest(:names)
                    ON CONFLICT (benchmark_name) DO NOTHING
                """),
                {"names": benches},
            )

        ins = text("""
            INSERT INTO fundlab.bench_nav (bench_id, nav_date, nav_value)
            SELECT b.bench_id, :d, :v FROM fundlab.benchmark b WHERE b.benchmark_name = :n
            ON CONFLICT (bench_id, nav_date) DO UPDATE
            SET nav_value = EXCLUDED.nav_value
        """)
        for _, r in df.iterrows():
            conn.execute(
                ins,
                {"n": r["benchmark_name"], "d": r["nav_date"], "v": float(r["nav_value"])},
            )


#Update Supabase with fund portfolios
PORT_COLS = {
    "scheme_name": ["scheme name", "fund", "scheme", "fund_name"],
    "month_end": ["month-end", "month end", "month-end (yyyymm)", "month-end (yyyymm)", "monthend", "month_end"],
    "instrument": ["instrument", "security", "stock name"],
    "holding_pct": ["holding (%)", "holding %", "weight (%)", "weight"],
    "asset_type": ["asset type", "asset_type"],
    "isin": ["isin"],
}

ALLOWED_EQUITY_ASSET_TYPES = {
    "domestic equities",
    "overseas equities",
    "others equities",
    "adrs & gdrs",
    "cash"
}

def validate_fund_portfolios(df_raw: pd.DataFrame):
    required = {"scheme_name", "month_end", "instrument", "holding_pct", "asset_type", "isin"}
    df = map_headers(df_raw.copy(), PORT_COLS, required)

    # Month-end as YYYYMM → last day of month
    df["month_end"] = pd.to_numeric(df["month_end"], errors="coerce").astype("Int64")
    if df["month_end"].isna().any():
        raise ValueError("Some 'Month-end' values could not be parsed as YYYYMM.")

    df["month_end"] = df["month_end"].astype(int).astype(str)
    df["month_end"] = pd.to_datetime(df["month_end"] + "01", format="%Y%m%d") + pd.offsets.MonthEnd(0)
    df["month_end"] = df["month_end"].dt.date

    df["holding_pct"] = pd.to_numeric(df["holding_pct"], errors="coerce")
    if df["holding_pct"].isna().any():
        raise ValueError("Some holding percentages could not be parsed as numbers.")

    # Asset type check
    invalid_asset_types = set(
        a for a in df["asset_type"].dropna().str.strip()
        if a.lower() not in ALLOWED_EQUITY_ASSET_TYPES
    )
    if invalid_asset_types:
        raise ValueError(
            "Invalid asset type(s) detected: "
            + ", ".join(sorted(invalid_asset_types))
            + ". Only Domestic Equities, Overseas Equities, Others Equities and ADRs & GDRs are allowed; others must be tagged as cash in the source file."
        )

    # In-file duplicate check
    dup_keys = df.groupby(["scheme_name", "month_end", "isin"]).size()
    dups = dup_keys[dup_keys > 1]
    if not dups.empty:
        raise ValueError(
            f"Duplicate (scheme_name, month_end, isin) rows found in file: {len(dups)} duplicates."
        )

    summary = {
        "rows": int(len(df)),
        "unique_scheme_month_isin": int(len(dup_keys)),
    }
    return df, summary

def upload_fund_portfolios(df: pd.DataFrame, batch_size: int = 10000):
    """
    Upload cleaned fund portfolio data into fundlab.fund_portfolio in batches.

    Expected canonical columns in df:
      - scheme_name
      - month_end      (python date)
      - holding_pct    (0–100)
      - asset_type
      - isin
      - instrument     (used for instrument_name in DB)
    """
    engine = get_engine()
    with engine.begin() as conn:
        # 1) Ensure funds exist
        schemes = sorted(set(df["scheme_name"]))
        if schemes:
            conn.execute(
                text("""
                    INSERT INTO fundlab.fund (fund_name)
                    SELECT unnest(:names)
                    ON CONFLICT (fund_name) DO NOTHING
                """),
                {"names": schemes},
            )

        # 2) Batched insert into fund_portfolio
        n = len(df)
        if n == 0:
            return

        insert_sql = text("""
            INSERT INTO fundlab.fund_portfolio (
                fund_id,
                month_end,
                instrument_name,
                holding_weight,
                asset_type,
                isin
            )
            SELECT
                f.fund_id,
                t.month_end,
                t.instrument_name,
                t.holding_weight,
                t.asset_type,
                t.isin
            FROM (
                SELECT
                    unnest(:scheme_names)      AS scheme_name,
                    unnest(:month_ends)        AS month_end,
                    unnest(:instrument_names)  AS instrument_name,
                    unnest(:weights)           AS holding_weight,
                    unnest(:asset_types)       AS asset_type,
                    unnest(:isins)             AS isin
            ) t
            JOIN fundlab.fund f
              ON f.fund_name = t.scheme_name
            ON CONFLICT (fund_id, month_end, instrument_name, asset_type, holding_weight)
            DO NOTHING
        """)

        # Process in chunks
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            chunk = df.iloc[start:end]

            params = {
                "scheme_names":   list(chunk["scheme_name"].astype(str)),
                "month_ends":     list(chunk["month_end"]),        # python date → date[]
                "instrument_names": list(chunk["instrument"].astype(str)),
                "weights":        [float(x) for x in chunk["holding_pct"]],
                "asset_types":    list(chunk["asset_type"].astype(str)),
                "isins":          list(chunk["isin"].astype(str)),
            }

            conn.execute(insert_sql, params)



#Update Supabase with Stock ISIN, industry, financial/non-financial
STOCK_MASTER_COLS = {
    "isin": ["isin"],
    "company_name": ["company name", "company_name", "name"],
    "industry": ["industry", "sector"],
    "financial_flag": ["financial?", "financial", "is_financial"],
}

FINANCIAL_INDUSTRIES = {
    "finance - stock broking",
    "finance - housing",
    "finance - nbfc",
    "finance - asset management",
    "finance - investment",
    "bank - public",
    "finance - others",
    "bank - private",
    "insurance",
    "finance term lending",
    "fintech",
}

def validate_stock_master(df_raw: pd.DataFrame):
    required = {"isin", "company_name", "industry"}
    df = map_headers(df_raw.copy(), STOCK_MASTER_COLS, required)

    df["isin"] = df["isin"].astype(str).str.strip()
    df["company_name"] = df["company_name"].astype(str).str.strip()
    df["industry"] = df["industry"].astype(str).str.strip()

    # Force financial flag based on industry
    df["is_financial"] = df["industry"].str.lower().isin(FINANCIAL_INDUSTRIES)

    dup_keys = df.groupby("isin").size()
    dups = dup_keys[dup_keys > 1]
    if not dups.empty:
        raise ValueError(f"Duplicate ISINs found in file: {len(dups)} duplicates.")

    summary = {
        "rows": int(len(df)),
        "unique_isins": int(len(dup_keys)),
        "financial_true": int(df["is_financial"].sum()),
    }
    return df, summary

def upload_stock_master(df: pd.DataFrame):
    engine = get_engine()
    with engine.begin() as conn:
        ins = text("""
            INSERT INTO fundlab.stock_master (isin, company_name, industry, is_financial)
            VALUES (:isin, :name, :industry, :fin)
            ON CONFLICT (isin) DO UPDATE
            SET company_name = EXCLUDED.company_name,
                industry = EXCLUDED.industry,
                is_financial = EXCLUDED.is_financial
        """)
        for _, r in df.iterrows():
            conn.execute(
                ins,
                {
                    "isin": r["isin"],
                    "name": r["company_name"],
                    "industry": r.get("industry"),
                    "fin": bool(r["is_financial"]),
                },
            )


#Update Supabase with Company RoE / RoCE
ROE_ROCE_COLS = {
    "isin": ["isin"],
    "company_name": ["company name", "company_name", "name"],
    "year_end": ["year-end (yyyymm)", "year end (yyyymm)", "year_end", "yearend", "yyyymm"],
    "roe": ["roe"],
    "roce": ["roce"],
}

def validate_roe_roce(df_raw: pd.DataFrame):
    required = {"isin", "year_end"}
    df = map_headers(df_raw.copy(), ROE_ROCE_COLS, required)

    df["isin"] = df["isin"].astype(str).str.strip()

    df["year_end"] = pd.to_numeric(df["year_end"], errors="coerce").astype("Int64")
    if df["year_end"].isna().any():
        raise ValueError("Some year-end values could not be parsed as YYYYMM.")

    # Convert YYYYMM to date (last day of month)
    df["year_end"] = df["year_end"].astype(int).astype(str)
    df["year_end_date"] = pd.to_datetime(df["year_end"] + "01", format="%Y%m%d") + pd.offsets.MonthEnd(0)
    df["year_end_date"] = df["year_end_date"].dt.date

    for col in ("roe", "roce"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = pd.NA

    dup_keys = df.groupby(["isin", "year_end_date"]).size()
    dups = dup_keys[dup_keys > 1]
    if not dups.empty:
        raise ValueError(f"Duplicate (isin, year_end_date) rows found in file: {len(dups)} duplicates.")

    summary = {
        "rows": int(len(df)),
        "unique_isin_year": int(len(dup_keys)),
    }
    return df, summary

def upload_roe_roce(df: pd.DataFrame):
    engine = get_engine()
    with engine.begin() as conn:
        ins = text("""
            INSERT INTO fundlab.stock_roe_roce (isin, year_end_date, roe, roce, company_name)
            VALUES (:isin, :d, :roe, :roce, :name)
            ON CONFLICT (isin, year_end_date) DO UPDATE
            SET roe = COALESCE(EXCLUDED.roe, stock_roe_roce.roe),
                roce = COALESCE(EXCLUDED.roce, stock_roe_roce.roce),
                company_name = COALESCE(EXCLUDED.company_name, stock_roe_roce.company_name)
        """)
        for _, r in df.iterrows():
            conn.execute(
                ins,
                {
                    "isin": r["isin"],
                    "d": r["year_end_date"],
                    "roe": None if pd.isna(r.get("roe")) else float(r["roe"]),
                    "roce": None if pd.isna(r.get("roce")) else float(r["roce"]),
                    "name": r.get("company_name"),
                },
            )


def _parse_period_to_month_end(series: pd.Series) -> pd.Series:
    """
    Convert a column containing quarter/year ends to month-end Timestamps.

    Handles:
    - Excel dates / datetime
    - Numeric or string 'YYYYMM'
    - Numeric or string 'YYYYMMDD'
    """
    # Case 1: already datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        dt = pd.to_datetime(series)
        return dt + pd.offsets.MonthEnd(0)

    s = series.astype(str).str.strip()

    # Keep only the numeric part (in case Excel did something funky)
    s_digits = s.str.extract(r"(\d+)", expand=False)

    # Length 6 => YYYYMM, length 8 => YYYYMMDD
    len6 = s_digits.str.len() == 6
    len8 = s_digits.str.len() == 8

    if not (len6 | len8).all():
        bad = s[~(len6 | len8)].unique()[:10]
        raise ValueError(
            "Invalid period values found (expected YYYYMM or YYYYMMDD). "
            f"Sample invalid values: {bad}"
        )

    # Parse 6-digit as YYYYMM01, 8-digit as YYYYMMDD
    dt = pd.to_datetime(
        np.where(len6, s_digits + "01", s_digits),
        format="%Y%m%d",
        errors="raise",
    )

    # Snap to month-end
    return dt + pd.offsets.MonthEnd(0)



def validate_quarterly_pat(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Expect: Col A = ISIN, Col B = quarter end (YYYYMM), Col C = adjusted PAT (absolute).
    """
    if df_raw.shape[1] < 3:
        raise ValueError("Expected at least 3 columns: ISIN, quarter_end_YYYYMM, PAT.")

    df = df_raw.iloc[:, :3].copy()
    df.columns = ["isin", "yyyymm", "pat"]

    df["isin"] = df["isin"].astype(str).str.strip().str.upper()
    df = df[df["isin"] != ""].copy()

    df["period_end"] = _parse_period_to_month_end(df["yyyymm"])

    # Calendar year & quarter (good enough for TTM calcs)
    df["fiscal_year"] = df["period_end"].dt.year.astype(int)
    df["fiscal_quarter"] = ((df["period_end"].dt.month - 1) // 3 + 1).astype(int)

    df["pat"] = pd.to_numeric(df["pat"], errors="coerce")
    df = df.dropna(subset=["pat"])

    # Check for duplicates within the upload file
    dup_mask = df.duplicated(subset=["isin", "period_end"], keep=False)
    if dup_mask.any():
        dup_rows = df.loc[dup_mask, ["isin", "period_end"]].drop_duplicates().head(20)
        raise ValueError(
            "Duplicate rows found for the same ISIN + period_end in the PAT file. "
            f"Examples:\n{dup_rows}"
        )

    df_clean = df[["isin", "period_end", "fiscal_year", "fiscal_quarter", "pat"]].copy()

    summary = {
        "rows_raw": int(len(df_raw)),
        "rows_clean": int(len(df_clean)),
        "min_period_end": str(df_clean["period_end"].min().date())
        if not df_clean.empty
        else None,
        "max_period_end": str(df_clean["period_end"].max().date())
        if not df_clean.empty
        else None,
    }
    return df_clean, summary


def validate_quarterly_sales(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Expect: Col A = ISIN, Col B = quarter end (YYYYMM), Col C = sales (absolute).
    """
    if df_raw.shape[1] < 3:
        raise ValueError("Expected at least 3 columns: ISIN, quarter_end_YYYYMM, sales.")

    df = df_raw.iloc[:, :3].copy()
    df.columns = ["isin", "yyyymm", "sales"]

    df["isin"] = df["isin"].astype(str).str.strip().str.upper()
    df = df[df["isin"] != ""].copy()

    df["period_end"] = _parse_period_to_month_end(df["yyyymm"])

    df["fiscal_year"] = df["period_end"].dt.year.astype(int)
    df["fiscal_quarter"] = ((df["period_end"].dt.month - 1) // 3 + 1).astype(int)

    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df = df.dropna(subset=["sales"])

    dup_mask = df.duplicated(subset=["isin", "period_end"], keep=False)
    if dup_mask.any():
        dup_rows = df.loc[dup_mask, ["isin", "period_end"]].drop_duplicates().head(20)
        raise ValueError(
            "Duplicate rows found for the same ISIN + period_end in the sales file. "
            f"Examples:\n{dup_rows}"
        )

    df_clean = df[["isin", "period_end", "fiscal_year", "fiscal_quarter", "sales"]].copy()

    summary = {
        "rows_raw": int(len(df_raw)),
        "rows_clean": int(len(df_clean)),
        "min_period_end": str(df_clean["period_end"].min().date())
        if not df_clean.empty
        else None,
        "max_period_end": str(df_clean["period_end"].max().date())
        if not df_clean.empty
        else None,
    }
    return df_clean, summary


def validate_annual_book_value(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Expect: Col A = ISIN, Col B = year-end (YYYYMM), Col C = book value (absolute net worth).
    """
    if df_raw.shape[1] < 3:
        raise ValueError("Expected at least 3 columns: ISIN, year_end_YYYYMM, book_value.")

    df = df_raw.iloc[:, :3].copy()
    df.columns = ["isin", "yyyymm", "book_value"]

    df["isin"] = df["isin"].astype(str).str.strip().str.upper()
    df = df[df["isin"] != ""].copy()

    df["year_end"] = _parse_period_to_month_end(df["yyyymm"])
    df["fiscal_year"] = df["year_end"].dt.year.astype(int)

    df["book_value"] = pd.to_numeric(df["book_value"], errors="coerce")
    df = df.dropna(subset=["book_value"])

    dup_mask = df.duplicated(subset=["isin", "year_end"], keep=False)
    if dup_mask.any():
        dup_rows = df.loc[dup_mask, ["isin", "year_end"]].drop_duplicates().head(20)
        raise ValueError(
            "Duplicate rows found for the same ISIN + year_end in the book value file. "
            f"Examples:\n{dup_rows}"
        )

    df_clean = df[["isin", "year_end", "fiscal_year", "book_value"]].copy()

    summary = {
        "rows_raw": int(len(df_raw)),
        "rows_clean": int(len(df_clean)),
        "min_year_end": str(df_clean["year_end"].min().date())
        if not df_clean.empty
        else None,
        "max_year_end": str(df_clean["year_end"].max().date())
        if not df_clean.empty
        else None,
    }
    return df_clean, summary



def upload_stock_prices_mc(df: pd.DataFrame, batch_size: int = 10000):
    """
    Upload stock prices + market cap into fundlab.stock_price in batches.

    Expected canonical columns in df:
      - isin
      - price_date  (python date)
      - market_cap
      - price

    Primary key / unique constraint: (isin, price_date, market_cap, price)
    ON CONFLICT DO NOTHING to avoid duplicate uploads.
    """
    engine = get_engine()
    with engine.begin() as conn:
        n = len(df)
        if n == 0:
            return

        # IMPORTANT: ensure you have this unique constraint in Supabase:
        # ALTER TABLE fundlab.stock_price
        # ADD CONSTRAINT stock_price_unq UNIQUE (isin, price_date, market_cap, price);
        insert_sql = text("""
            INSERT INTO fundlab.stock_price (
                isin,
                price_date,
                market_cap,
                price
            )
            SELECT
                unnest(:isins)       AS isin,
                unnest(:dates)       AS price_date,
                unnest(:mcaps)       AS market_cap,
                unnest(:prices)      AS price
            ON CONFLICT (isin, price_date, market_cap, price)
            DO NOTHING
        """)

        # Process in chunks
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            chunk = df.iloc[start:end]

            params = {
                "isins":  list(chunk["isin"].astype(str)),
                "dates":  list(chunk["price_date"]),                     # python date → date[]
                "mcaps":  [float(x) for x in chunk["market_cap"]],
                "prices": [float(x) for x in chunk["price"]],
            }

            conn.execute(insert_sql, params)




def upload_quarterly_pat(df_clean: pd.DataFrame) -> None:
    """
    Bulk upload quarterly PAT into fundlab.stock_quarterly_financials.

    Behaviour:
    - If (isin, period_end) does NOT exist → INSERT a new row with PAT.
    - If (isin, period_end) already exists → UPDATE PAT (and fiscal_year/quarter)
      for that row, leaving sales untouched.
    - Uses set-based INSERT/UPDATE via unnest for speed.
    """

    if df_clean.empty:
        return

    # Remove duplicates within the file
    df_clean = df_clean.drop_duplicates(subset=["isin", "period_end"]).copy()

    n = len(df_clean)
    if n == 0:
        return

    BATCH_SIZE = 10_000

    engine = get_engine()
    progress_bar = st.progress(0.0)
    progress_text = st.empty()

    with engine.begin() as conn:
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            chunk = df_clean.iloc[start:end].copy()

            # 1) Find which (isin, period_end) already exist in DB
            keys = chunk[["isin", "period_end"]].drop_duplicates()
            existing_sql = text(
                """
                SELECT v.isin, v.period_end
                FROM (
                    SELECT
                        unnest(:isins)       AS isin,
                        unnest(:period_ends) AS period_end
                ) v
                JOIN fundlab.stock_quarterly_financials q
                  ON q.isin = v.isin
                 AND q.period_end = v.period_end
                 AND q.is_consolidated = TRUE
                """
            )
            key_params = {
                "isins":       list(keys["isin"].astype(str)),
                "period_ends": list(keys["period_end"]),
            }
            existing_rows = conn.execute(existing_sql, key_params).fetchall()
            existing_set = {(r.isin, r.period_end) for r in existing_rows}

            # 2) Split into new vs existing rows
            chunk["key"] = list(zip(chunk["isin"], chunk["period_end"]))
            mask_existing = chunk["key"].isin(existing_set)

            new_rows = chunk[~mask_existing].copy()
            upd_rows = chunk[mask_existing].copy()

            # 3) Insert new rows
            if not new_rows.empty:
                insert_sql = text(
                    """
                    INSERT INTO fundlab.stock_quarterly_financials (
                        isin,
                        period_end,
                        fiscal_year,
                        fiscal_quarter,
                        pat
                    )
                    SELECT
                        t.isin,
                        t.period_end,
                        t.fiscal_year,
                        t.fiscal_quarter,
                        t.pat
                    FROM (
                        SELECT
                            unnest(:isins)           AS isin,
                            unnest(:period_ends)     AS period_end,
                            unnest(:fiscal_years)    AS fiscal_year,
                            unnest(:fiscal_quarters) AS fiscal_quarter,
                            unnest(:pats)            AS pat
                    ) t
                    """
                )
                params_ins = {
                    "isins":           list(new_rows["isin"].astype(str)),
                    "period_ends":     list(new_rows["period_end"]),
                    "fiscal_years":    list(new_rows["fiscal_year"].astype(int)),
                    "fiscal_quarters": list(new_rows["fiscal_quarter"].astype(int)),
                    "pats":            [float(x) for x in new_rows["pat"]],
                }
                conn.execute(insert_sql, params_ins)

            # 4) Update existing rows (PAT only)
            if not upd_rows.empty:
                update_sql = text(
                    """
                    UPDATE fundlab.stock_quarterly_financials q
                    SET
                        pat          = v.pat,
                        fiscal_year  = v.fiscal_year,
                        fiscal_quarter = v.fiscal_quarter,
                        updated_at   = NOW()
                    FROM (
                        SELECT
                            unnest(:isins)           AS isin,
                            unnest(:period_ends)     AS period_end,
                            unnest(:fiscal_years)    AS fiscal_year,
                            unnest(:fiscal_quarters) AS fiscal_quarter,
                            unnest(:pats)            AS pat
                    ) v
                    WHERE q.isin = v.isin
                      AND q.period_end = v.period_end
                      AND q.is_consolidated = TRUE
                    """
                )
                params_upd = {
                    "isins":           list(upd_rows["isin"].astype(str)),
                    "period_ends":     list(upd_rows["period_end"]),
                    "fiscal_years":    list(upd_rows["fiscal_year"].astype(int)),
                    "fiscal_quarters": list(upd_rows["fiscal_quarter"].astype(int)),
                    "pats":            [float(x) for x in upd_rows["pat"]],
                }
                conn.execute(update_sql, params_upd)

            # Progress
            frac = end / n
            progress_bar.progress(frac)
            progress_text.text(f"Uploading PAT… {end} / {n} rows processed")

    progress_text.text(f"PAT upload complete: {n} rows processed (inserts + updates).")


def upload_quarterly_sales(df_clean: pd.DataFrame) -> None:
    """
    Bulk upload quarterly sales into fundlab.stock_quarterly_financials.

    Behaviour:
    - If (isin, period_end) does NOT exist → INSERT a new row with sales.
    - If (isin, period_end) already exists → UPDATE sales for that row,
      leaving PAT untouched.
    """

    if df_clean.empty:
        return

    df_clean = df_clean.drop_duplicates(subset=["isin", "period_end"]).copy()

    n = len(df_clean)
    if n == 0:
        return

    BATCH_SIZE = 10_000

    engine = get_engine()
    progress_bar = st.progress(0.0)
    progress_text = st.empty()

    with engine.begin() as conn:
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            chunk = df_clean.iloc[start:end].copy()

            # 1) Which keys already exist?
            keys = chunk[["isin", "period_end"]].drop_duplicates()
            existing_sql = text(
                """
                SELECT v.isin, v.period_end
                FROM (
                    SELECT
                        unnest(:isins)       AS isin,
                        unnest(:period_ends) AS period_end
                ) v
                JOIN fundlab.stock_quarterly_financials q
                  ON q.isin = v.isin
                 AND q.period_end = v.period_end
                 AND q.is_consolidated = TRUE
                """
            )
            key_params = {
                "isins":       list(keys["isin"].astype(str)),
                "period_ends": list(keys["period_end"]),
            }
            existing_rows = conn.execute(existing_sql, key_params).fetchall()
            existing_set = {(r.isin, r.period_end) for r in existing_rows}

            # 2) Split into new vs existing
            chunk["key"] = list(zip(chunk["isin"], chunk["period_end"]))
            mask_existing = chunk["key"].isin(existing_set)

            new_rows = chunk[~mask_existing].copy()
            upd_rows = chunk[mask_existing].copy()

            # 3) Insert new rows
            if not new_rows.empty:
                insert_sql = text(
                    """
                    INSERT INTO fundlab.stock_quarterly_financials (
                        isin,
                        period_end,
                        fiscal_year,
                        fiscal_quarter,
                        sales
                    )
                    SELECT
                        t.isin,
                        t.period_end,
                        t.fiscal_year,
                        t.fiscal_quarter,
                        t.sales
                    FROM (
                        SELECT
                            unnest(:isins)           AS isin,
                            unnest(:period_ends)     AS period_end,
                            unnest(:fiscal_years)    AS fiscal_year,
                            unnest(:fiscal_quarters) AS fiscal_quarter,
                            unnest(:sales_vals)      AS sales
                    ) t
                    """
                )
                params_ins = {
                    "isins":           list(new_rows["isin"].astype(str)),
                    "period_ends":     list(new_rows["period_end"]),
                    "fiscal_years":    list(new_rows["fiscal_year"].astype(int)),
                    "fiscal_quarters": list(new_rows["fiscal_quarter"].astype(int)),
                    "sales_vals":      [float(x) for x in new_rows["sales"]],
                }
                conn.execute(insert_sql, params_ins)

            # 4) Update existing rows (sales only)
            if not upd_rows.empty:
                update_sql = text(
                    """
                    UPDATE fundlab.stock_quarterly_financials q
                    SET
                        sales        = v.sales,
                        fiscal_year  = v.fiscal_year,
                        fiscal_quarter = v.fiscal_quarter,
                        updated_at   = NOW()
                    FROM (
                        SELECT
                            unnest(:isins)           AS isin,
                            unnest(:period_ends)     AS period_end,
                            unnest(:fiscal_years)    AS fiscal_year,
                            unnest(:fiscal_quarters) AS fiscal_quarter,
                            unnest(:sales_vals)      AS sales
                    ) v
                    WHERE q.isin = v.isin
                      AND q.period_end = v.period_end
                      AND q.is_consolidated = TRUE
                    """
                )
                params_upd = {
                    "isins":           list(upd_rows["isin"].astype(str)),
                    "period_ends":     list(upd_rows["period_end"]),
                    "fiscal_years":    list(upd_rows["fiscal_year"].astype(int)),
                    "fiscal_quarters": list(upd_rows["fiscal_quarter"].astype(int)),
                    "sales_vals":      [float(x) for x in upd_rows["sales"]],
                }
                conn.execute(update_sql, params_upd)

            frac = end / n
            progress_bar.progress(frac)
            progress_text.text(f"Uploading sales… {end} / {n} rows processed")

    progress_text.text(f"Sales upload complete: {n} rows processed (inserts + updates).")



def upload_annual_book_value(df_clean: pd.DataFrame) -> None:
    if df_clean.empty:
        return

    df_clean = df_clean.drop_duplicates(subset=["isin", "year_end"])

    conflicts = find_conflicts_annual(df_clean)
    if conflicts:
        sample = "; ".join([f"{r.isin} @ {r.year_end.date()}" for r in conflicts])
        raise ValueError(
            "Duplicate book value data detected: These ISIN + year-end rows already exist.\n"
            f"Examples: {sample}\n\n"
            "Upload aborted. No rows inserted."
        )

    n = len(df_clean)
    BATCH_SIZE = 10000
    if n == 0:
        return

    insert_sql = text(
        """
        INSERT INTO fundlab.stock_annual_book_value (
            isin,
            year_end,
            fiscal_year,
            book_value
        )
        SELECT
            t.isin,
            t.year_end,
            t.fiscal_year,
            t.book_value
        FROM (
            SELECT
                unnest(:isins)        AS isin,
                unnest(:year_ends)    AS year_end,
                unnest(:fiscal_years) AS fiscal_year,
                unnest(:bvals)        AS book_value
        ) t
        """
    )

    progress_bar = st.progress(0.0)
    progress_text = st.empty()

    with engine.begin() as conn:
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            chunk = df_clean.iloc[start:end]

            params = {
                "isins":        list(chunk["isin"].astype(str)),
                "year_ends":    list(chunk["year_end"]),
                "fiscal_years": list(chunk["fiscal_year"].astype(int)),
                "bvals":        [float(x) for x in chunk["book_value"]],
            }

            frac = (end / n)
            progress_bar.progress(frac)
            progress_text.text(f"Inserting book value… {end} / {n}")

            conn.execute(insert_sql, params)

    progress_text.text(f"Annual book value insert complete: {n} rows.")


# -----------------------------
# Fund manager tenure (upload)
# -----------------------------

FM_TENURE_COLS = {
    "fund_name": ["fund name", "fund", "scheme", "scheme name", "fund_name"],
    "inception_date": ["inception date", "inception", "start date"],
    "fund_manager": ["fund manager", "manager", "fm", "portfolio manager"],
    "from_period": ["from date", "from", "from_period", "from period"],
    "to_period": ["to date", "to", "to_period", "to period"],
}

def _parse_month_year(series: pd.Series, colname: str, allow_blank: bool) -> pd.Series:
    """
    Accepts:
    - Excel dates / pandas Timestamps (e.g., 2024-05-31)
    - Strings like 'May-2024', 'May 2024', '2024-05', etc.

    Returns:
    - pandas Timestamp normalized to month (day=1)
    """

    # If pandas already parsed it as datetime → accept directly
    if pd.api.types.is_datetime64_any_dtype(series):
        dt = series.copy()
    else:
        s = series.astype(str).str.strip()
        blanks = s.eq("") | s.str.lower().isin(["nan", "nat", "none", "null"])

        if not allow_blank and blanks.any():
            raise ValueError(f"Blank values in '{colname}'. Expected month-year.")

        dt = pd.to_datetime(
            s.where(~blanks, pd.NA),
            errors="coerce",
            dayfirst=True,
        )

        bad = (~blanks) & dt.isna()
        if bad.any():
            sample = series[bad].head(10).tolist()
            raise ValueError(
                f"Invalid '{colname}' values. Expected month/year (e.g. Jan-2024) or Excel date. "
                f"Sample: {sample}"
            )

    # Normalize to MONTH only (canonical)
    dt = dt.dt.to_period("M").dt.to_timestamp()

    return dt

def render_sticky_first_col_table(df: pd.DataFrame, height_px: int = 520, freeze_cols: int = 1):
    import streamlit as st
    import streamlit.components.v1 as components

    if df is None or df.empty:
        st.info("No data to display.")
        return

    # Ensure freeze_cols sane
    freeze_cols = max(1, int(freeze_cols))

    table_html = df.to_html(index=False, escape=False)

    # Build CSS that makes first N columns sticky
    sticky_cols_css = []
    left_px = 0
    # Column widths vary; use cumulative left offsets by assigning a fixed min-width per column.
    # This is the only robust approach in pure CSS without JS measurement.
    # You can tweak col_width_px if needed.
    col_width_px = 180

    for i in range(1, freeze_cols + 1):
        sticky_cols_css.append(f"""
        .sticky-wrap th:nth-child({i}),
        .sticky-wrap td:nth-child({i}) {{
          position: sticky;
          left: {left_px}px;
          background: #ffffff;
          z-index: 4;
          border-right: 1px solid rgba(49, 51, 63, 0.15);
          min-width: {col_width_px}px;
          max-width: {col_width_px}px;
        }}
        """)
        left_px += col_width_px

    # Ensure header cells are above body cells
    sticky_cols_css.append(f"""
    .sticky-wrap thead th {{
      z-index: 6;
    }}
    """)

    css = f"""
    <style>
      .sticky-wrap {{
        max-height: {height_px}px;
        overflow: auto;
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 6px;
      }}
      .sticky-wrap table {{
        border-collapse: collapse;
        width: max-content;
        min-width: 100%;
        font-size: 13px;
      }}
      .sticky-wrap th, .sticky-wrap td {{
        padding: 6px 10px;
        border-bottom: 1px solid rgba(49, 51, 63, 0.15);
        white-space: nowrap;
      }}
      .sticky-wrap thead th {{
        position: sticky;
        top: 0;
        background: #ffffff;
        border-bottom: 1px solid rgba(49, 51, 63, 0.25);
      }}
      {''.join(sticky_cols_css)}
    </style>
    """

    html = css + f"<div class='sticky-wrap'>{table_html}</div>"
    components.html(html, height=height_px + 70, scrolling=True)


def build_size_asset_allocation_pivot(
    holdings: pd.DataFrame,
    size_band: pd.DataFrame,
    freq: str,
    period_col: str = "month_end",
):
    """
    Allocation table:
    - Domestic Equities split into Large/Mid/Small using stock_size_band (isin + band_date)
    - stock_size_band is only tagged in Jun/Dec; we forward-fill per ISIN using last known tag <= month_end
      (Jun applies through Nov; Dec applies through May).
    - Other asset types shown via asset_type (with light normalization)
    Output: Allocation | <periods...> with values = sum(weight_pct OR holding_weight)
    """

    if holdings is None or holdings.empty:
        return pd.DataFrame()

    h = holdings.copy()
    h[period_col] = pd.to_datetime(h[period_col]).dt.to_period("M").dt.to_timestamp("M")
    h["asset_type"] = h.get("asset_type", "").fillna("").astype(str)
    h["isin"] = h.get("isin", "").fillna("").astype(str)

    # Determine weight column
    if "weight_pct" in h.columns:
        wcol = "weight_pct"
    elif "holding_weight" in h.columns:
        wcol = "holding_weight"
    else:
        raise ValueError("Holdings must contain 'weight_pct' or 'holding_weight'.")

    h[wcol] = pd.to_numeric(h[wcol], errors="coerce").fillna(0.0)

    # Snapshot selection
    if freq == "Quarterly":
        grp = h[period_col].dt.to_period("Q")
        last_me = h.groupby(grp)[period_col].transform("max")
        h = h[h[period_col].eq(last_me)].copy()
    elif freq == "Yearly":
        grp = h[period_col].dt.to_period("Y")
        last_me = h.groupby(grp)[period_col].transform("max")
        h = h[h[period_col].eq(last_me)].copy()

    if h.empty:
        return pd.DataFrame()

    # Prepare size bands and apply carry-forward via merge_asof per ISIN
    sb = size_band.copy() if size_band is not None else pd.DataFrame()
    if not sb.empty:
        # Accept either month_end alias or raw band_date
        if "month_end" not in sb.columns and "band_date" in sb.columns:
            sb = sb.rename(columns={"band_date": "month_end"})

        sb["month_end"] = pd.to_datetime(sb["month_end"]).dt.to_period("M").dt.to_timestamp("M")
        sb["isin"] = sb["isin"].fillna("").astype(str)
        sb["size_band"] = sb["size_band"].fillna("").astype(str)

        # Only keep valid bands
        sb = sb[sb["size_band"].isin(["Large", "Mid", "Small"])].copy()

        # Sort for merge_asof
        sb = sb.sort_values(["isin", "month_end"])
        h = h.sort_values(["isin", period_col])

        # merge_asof needs a single key; do it per ISIN with groupby apply (still fast enough here)
        def _asof_merge(g):
            isin = g.name
            sb_i = sb[sb["isin"] == isin][["month_end", "size_band"]]
            if sb_i.empty:
                g["size_band"] = ""
                return g
            out = pd.merge_asof(
                g.sort_values(period_col),
                sb_i.sort_values("month_end"),
                left_on=period_col,
                right_on="month_end",
                direction="backward",
            )
            out.drop(columns=["month_end_y"], inplace=True, errors="ignore")
            out.rename(columns={period_col: "month_end", "month_end_x": "month_end"}, inplace=True)
            # Ensure column name uniform
            if "month_end" not in out.columns:
                out["month_end"] = g[period_col].values
            return out

        h = h.groupby("isin", group_keys=False).apply(_asof_merge)
        # restore standard period_col name if altered
        if "month_end" not in h.columns:
            h["month_end"] = pd.to_datetime(h[period_col]).dt.to_period("M").dt.to_timestamp("M")
        period_col = "month_end"
    else:
        h["size_band"] = ""

    at = h["asset_type"].str.strip()

    # Default Allocation = asset_type / Others
    h["Allocation"] = at.where(at != "", other="Others")
    h.loc[at.str.lower().eq("cash"), "Allocation"] = "Cash"
    h.loc[at.isin(["Overseas Equities", "ADRs & GDRs"]), "Allocation"] = "Overseas Equities"

    is_dom = at.eq("Domestic Equities")
    band = h.loc[is_dom, "size_band"].astype(str).str.strip()
    h.loc[is_dom, "Allocation"] = band.where(band.isin(["Large", "Mid", "Small"]), other="Unclassified")

    h["period"] = pd.to_datetime(h[period_col]).dt.strftime("%b %Y")

    period_order = (
        h[["period", period_col]]
        .drop_duplicates()
        .sort_values(period_col)["period"]
        .tolist()
    )

    piv = (
        h.groupby(["Allocation", "period"], as_index=False)[wcol]
        .sum()
        .pivot(index="Allocation", columns="period", values=wcol)
        .fillna(0.0)
    )

    piv = piv.reindex(columns=period_order)

    preferred = ["Large", "Mid", "Small", "Overseas Equities", "Cash", "Unclassified", "Others"]
    others = [r for r in piv.index.tolist() if r not in preferred]
    piv = piv.reindex(preferred + sorted(others))

    return piv.reset_index()



def validate_fund_manager_tenure(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = {"fund_name", "fund_manager", "from_period"}  # to_period optional, inception optional
    df = map_headers(df_raw, FM_TENURE_COLS, required=required)

    # Clean strings
    df["fund_name"] = df["fund_name"].astype(str).str.strip()
    df["fund_manager"] = df["fund_manager"].astype(str).str.strip()

    if df["fund_name"].eq("").any():
        raise ValueError("Blank fund name(s) found.")
    if df["fund_manager"].eq("").any():
        raise ValueError("Blank fund manager(s) found.")

    # inception_date: permissive
    if "inception_date" in df.columns:
        df["inception_date"] = pd.to_datetime(df["inception_date"], errors="coerce").dt.date
    else:
        df["inception_date"] = pd.NaT

    # Parse Mmm-YYYY
    df["from_ts"] = _parse_month_year(df["from_period"], "from date", allow_blank=False)

    if "to_period" in df.columns:
        df["to_ts"] = _parse_month_year(df["to_period"], "to date", allow_blank=True)
    else:
        df["to_ts"] = pd.NaT


    # Logical check
    bad_range = df["to_ts"].notna() & (df["from_ts"] > df["to_ts"])
    if bad_range.any():
        bad_rows = df.loc[bad_range, ["fund_name", "fund_manager", "from_period", "to_period"]].head(10)
        raise ValueError(f"Found rows where from date > to date. Sample:\n{bad_rows}")

    # Store canonical month-date (day=1); keep original Mmm-YYYY columns for reference if needed
    df["from_date"] = df["from_ts"].dt.date
    df["to_date"] = df["to_ts"].dt.date  # will be NaT->NaN; upload will convert to None

    # Drop helper cols
    df = df.drop(columns=[c for c in ["from_ts", "to_ts"] if c in df.columns])

    # Duplicate row check within upload
    dedup_key = ["fund_name", "fund_manager", "from_date", "to_date"]
    dupes = df.duplicated(subset=dedup_key, keep=False)
    if dupes.any():
        sample = df.loc[dupes, dedup_key].head(10)
        raise ValueError(f"Duplicate tenure rows detected in upload. Sample:\n{sample}")

    # Determine missing funds in master (for rename/new-fund flow)
    engine = get_engine()
    upload_funds = sorted(df["fund_name"].unique().tolist())
    existing = set(pd.read_sql("select fund_name from fundlab.fund", engine)["fund_name"].astype(str).str.strip())
    missing_funds = sorted([f for f in upload_funds if f not in existing])

    summary = {
        "rows": int(len(df)),
        "funds_in_upload": int(len(upload_funds)),
        "missing_funds": missing_funds,
        "notes": "from/to parsed as Mmm-YYYY; stored as canonical month-date (day=1). to_date blank => current.",
    }
    return df.reset_index(drop=True), summary

def _category_name_to_id(engine) -> dict:
    df = pd.read_sql(
        """
        SELECT category_id, category_name
        FROM fundlab.category
        WHERE LOWER(category_name) <> 'portfolio'
        """,
        engine,
    )
    return dict(zip(df["category_name"], df["category_id"]))

def upload_fund_manager_tenure(df_clean: pd.DataFrame, resolutions: dict) -> None:
    """
    resolutions: dict keyed by NEW fund_name:
      {
        "ICICI Value Fund": {"old_name": "ICICI Value Discovery", "category_name": None},
        "Brand New Fund":   {"old_name": "", "category_name": "Flexi Cap Fund"},
      }
    """
    if df_clean.empty:
        return

    engine = get_engine()

    # Fresh master snapshots for validation
    fund_master = pd.read_sql("select fund_id, fund_name from fundlab.fund", engine)
    existing_names = set(fund_master["fund_name"].astype(str).str.strip())

    cat_map = _category_name_to_id(engine)
    category_options = set(cat_map.keys())

    with engine.begin() as conn:
        # 1) Apply rename / new-fund creation
        for new_name, info in (resolutions or {}).items():
            old_name = (info.get("old_name") or "").strip()
            category_name = info.get("category_name")

            if old_name:
                if old_name not in existing_names:
                    raise ValueError(f"Old fund name '{old_name}' not found in fund master for rename to '{new_name}'.")
                # Rename (fund_id unchanged)
                conn.execute(
                    text("UPDATE fundlab.fund SET fund_name = :new_name WHERE fund_name = :old_name"),
                    {"new_name": new_name, "old_name": old_name},
                )
                existing_names.discard(old_name)
                existing_names.add(new_name)
            else:
                if not category_name:
                    raise ValueError(f"Category is required to create new fund '{new_name}'.")
                if category_name not in category_options:
                    raise ValueError(f"Unknown category '{category_name}' for new fund '{new_name}'.")
                conn.execute(
                    text("""
                        INSERT INTO fundlab.fund (fund_name, category_id)
                        VALUES (:fund_name, :category_id)
                    """),
                    {"fund_name": new_name, "category_id": int(cat_map[category_name])},
                )
                existing_names.add(new_name)

        # 2) Resolve fund_id for all funds in this upload (post rename/insert)
        upload_funds = sorted(df_clean["fund_name"].unique().tolist())
        fund_rows = conn.execute(
            text("SELECT fund_id, fund_name FROM fundlab.fund WHERE fund_name = ANY(:names)"),
            {"names": upload_funds},
        ).fetchall()

        name_to_id = {r.fund_name: r.fund_id for r in fund_rows}
        missing_after = [n for n in upload_funds if n not in name_to_id]
        if missing_after:
            raise ValueError(f"Funds still missing in master after resolution: {missing_after}")

        df = df_clean.copy()
        df["fund_id"] = df["fund_name"].map(name_to_id).astype(int)

        affected_ids = sorted(df["fund_id"].unique().tolist())

        # 3) Delete existing tenure rows for affected funds
        conn.execute(
            text("DELETE FROM fundlab.fund_manager_tenure WHERE fund_id = ANY(:fund_ids)"),
            {"fund_ids": affected_ids},
        )

        # 4) Insert tenure rows (batch)
        # Convert NaN to None for to_date
        to_dates = []
        for x in df["to_date"].tolist():
            if pd.isna(x):
                to_dates.append(None)
            else:
                to_dates.append(x)

        insert_sql = text("""
            INSERT INTO fundlab.fund_manager_tenure
                (fund_id, inception_date, fund_manager, from_date, to_date)
            SELECT
                unnest(:fund_ids)        AS fund_id,
                unnest(:inceptions)      AS inception_date,
                unnest(:managers)        AS fund_manager,
                unnest(:from_dates)      AS from_date,
                unnest(:to_dates)        AS to_date
        """)

        conn.execute(
            insert_sql,
            {
                "fund_ids": list(df["fund_id"].astype(int)),
                "inceptions": [None if pd.isna(x) else x for x in df["inception_date"].tolist()],
                "managers": list(df["fund_manager"].astype(str)),
                "from_dates": list(df["from_date"]),
                "to_dates": to_dates,
            },
        )



def upload_stock_monthly_valuations_from_excel(uploaded_file, batch_size: int = 2000):
    """
    Upload a single precomputed stock valuations Excel workbook (.xlsx)
    into fundlab.stock_monthly_valuations using batched executemany.

    Assumptions:
    - fundlab.stock_monthly_valuations exists with columns:
        isin text,
        month_end date,
        ttm_sales numeric,
        ttm_pat numeric,
        book_value numeric,
        ps numeric,
        pe numeric,
        pb numeric
      and is currently empty (or we don't care about duplicates yet).
    - The Excel was generated by rebuild_stock_monthly_valuations and
      has columns: isin, month_end, ttm_sales, ttm_pat,
                   book_value, ps, pe, pb
    """

    if uploaded_file is None:
        st.warning("Please select an Excel file first.")
        return

    # ------------------------------------------------------------------
    # 1) Read Excel
    # ------------------------------------------------------------------
    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Could not read Excel file: {e}")
        return

    if df.empty:
        st.warning("The uploaded workbook has no rows.")
        return

    # Normalise column names
    df.columns = [str(c).strip().lower() for c in df.columns]

    required_cols = [
        "isin",
        "month_end",
        "ttm_sales",
        "ttm_pat",
        "book_value",
        "ps",
        "pe",
        "pb",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(
            "Uploaded workbook is missing required columns: "
            + ", ".join(missing)
        )
        st.write("Columns found:", list(df.columns))
        return

    # Keep only required columns in correct order
    df = df[required_cols].copy()

    # ------------------------------------------------------------------
    # 2) Type cleanup
    # ------------------------------------------------------------------
    df["isin"] = df["isin"].astype(str).str.strip()

    df["month_end"] = pd.to_datetime(
        df["month_end"], errors="coerce"
    ).dt.date

    for col in ["ttm_sales", "ttm_pat", "book_value", "ps", "pe", "pb"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows without isin or month_end
    df = df.dropna(subset=["isin", "month_end"]).reset_index(drop=True)

    n_rows = len(df)
    if n_rows == 0:
        st.warning("No valid rows (isin + month_end) to upload after cleaning.")
        return

    st.write(f"Preparing to upload **{n_rows:,}** valuation rows…")

    # Replace NaNs with None so psycopg2 sends NULLs
    df = df.where(pd.notnull(df), None)

    # ------------------------------------------------------------------
    # 3) Prepare insert statement
    # ------------------------------------------------------------------
    insert_sql = text(
        """
        INSERT INTO fundlab.stock_monthly_valuations (
            isin,
            month_end,
            ttm_sales,
            ttm_pat,
            book_value,
            ps,
            pe,
            pb
        )
        VALUES (
            :isin,
            :month_end,
            :ttm_sales,
            :ttm_pat,
            :book_value,
            :ps,
            :pe,
            :pb
        );
        """
    )

    engine = get_engine()
    progress = st.progress(0.0, text="Uploading stock valuations…")
    inserted = 0

    try:
        with engine.begin() as conn:
            for start_idx in range(0, n_rows, batch_size):
                end_idx = min(start_idx + batch_size, n_rows)
                chunk = df.iloc[start_idx:end_idx]

                params = []
                for _, r in chunk.iterrows():
                    params.append(
                        {
                            "isin": r["isin"],
                            "month_end": r["month_end"],
                            "ttm_sales": r["ttm_sales"],
                            "ttm_pat": r["ttm_pat"],
                            "book_value": r["book_value"],
                            "ps": r["ps"],
                            "pe": r["pe"],
                            "pb": r["pb"],
                        }
                    )

                if params:
                    conn.execute(insert_sql, params)
                    inserted += len(params)
                    progress.progress(
                        inserted / n_rows,
                        text=(
                            f"Uploading stock valuations… "
                            f"{inserted:,} / {n_rows:,} rows"
                        ),
                    )

    except Exception as e:
        st.error("❌ Error while inserting into stock_monthly_valuations.")
        st.write("Python exception:", repr(e))
        orig = getattr(e, "orig", None)
        if orig is not None:
            st.write("DBAPI .orig:", repr(orig))
        return

    progress.progress(1.0, text="Upload complete.")
    st.success(f"✅ Uploaded {inserted:,} rows into fundlab.stock_monthly_valuations.")


# Recompute Large / mid /small labels based on market cap at each end-June and end-December
def recompute_size_bands(batch_size: int = 10000):
    """
    For each end-June and end-December in stock_price, rank by market_cap
    and assign size bands: top 100 Large, next 150 Mid, rest Small.
    Upserts into fundlab.stock_size_band.
    """
    engine = get_engine()
    with engine.begin() as conn:
        prices = pd.read_sql(
            """
            SELECT isin, price_date, market_cap
            FROM fundlab.stock_price
            WHERE EXTRACT(MONTH FROM price_date) IN (6, 12)
            ORDER BY price_date, market_cap DESC
            """,
            conn,
        )

    if prices.empty:
        st.warning("No stock_price data for June/December month-ends.")
        return

    # Rank by market cap within each date
    prices["rank_by_mcap"] = (
        prices.groupby("price_date")["market_cap"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    def band_for_rank(r):
        if r <= 100:
            return "Large"
        elif r <= 250:
            return "Mid"
        else:
            return "Small"

    prices["size_band"] = prices["rank_by_mcap"].apply(band_for_rank)
    prices = prices.rename(columns={"price_date": "band_date"})

    # Batched upsert
    insert_sql = text("""
        INSERT INTO fundlab.stock_size_band (
            isin,
            band_date,
            size_band,
            rank_by_mcap,
            market_cap
        )
        SELECT
            unnest(:isins),
            unnest(:band_dates),
            unnest(:bands),
            unnest(:ranks),
            unnest(:mcaps)
        ON CONFLICT (isin, band_date)
        DO UPDATE
        SET size_band    = EXCLUDED.size_band,
            rank_by_mcap = EXCLUDED.rank_by_mcap,
            market_cap   = EXCLUDED.market_cap
    """)

    with engine.begin() as conn:
        n = len(prices)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            chunk = prices.iloc[start:end]
            params = {
                "isins":      list(chunk["isin"].astype(str)),
                "band_dates": list(chunk["band_date"]),
                "bands":      list(chunk["size_band"].astype(str)),
                "ranks":      list(chunk["rank_by_mcap"].astype(int)),
                "mcaps":      [float(x) for x in chunk["market_cap"]],
            }
            conn.execute(insert_sql, params)


#Compute 5 year (RoE/RoCE) medians for each stock at each month_end
def recompute_quality_medians(batch_size: int = 10000):
    """
    For each stock and each month_end, compute median of last 5 RoE/RoCE
    observations (by year_end_date <= month_end) and store in
    fundlab.stock_quality_median.
    """
    engine = get_engine()
    with engine.begin() as conn:
        roe = pd.read_sql(
            """
            SELECT isin, year_end_date, roe, roce
            FROM fundlab.stock_roe_roce
            ORDER BY isin, year_end_date
            """,
            conn,
        )
        months = pd.read_sql(
            """
            SELECT DISTINCT month_end
            FROM fundlab.fund_portfolio
            ORDER BY month_end
            """,
            conn,
        )

    if roe.empty or months.empty:
        st.warning("RoE/RoCE or portfolio month-end data is empty.")
        return

    months_list = list(months["month_end"])
    rows = []

    grouped = roe.groupby("isin", sort=False)
    for isin, g in grouped:
        g = g.sort_values("year_end_date")
        for m in months_list:
            hist = g[g["year_end_date"] <= m].tail(5)
            if hist.empty:
                continue
            rows.append(
                {
                    "isin": isin,
                    "month_end": m,
                    "median_roe_5y": hist["roe"].median(skipna=True),
                    "median_roce_5y": hist["roce"].median(skipna=True),
                }
            )

    if not rows:
        st.warning("No 5-year medians could be computed.")
        return

    med_df = pd.DataFrame(rows)

    insert_sql = text("""
        INSERT INTO fundlab.stock_quality_median (
            isin,
            month_end,
            median_roe_5y,
            median_roce_5y
        )
        SELECT
            unnest(:isins),
            unnest(:month_ends),
            unnest(:med_roe),
            unnest(:med_roce)
        ON CONFLICT (isin, month_end)
        DO UPDATE
        SET median_roe_5y  = EXCLUDED.median_roe_5y,
            median_roce_5y = EXCLUDED.median_roce_5y
    """)

    engine = get_engine()
    with engine.begin() as conn:
        n = len(med_df)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            chunk = med_df.iloc[start:end]
            params = {
                "isins":      list(chunk["isin"].astype(str)),
                "month_ends": list(chunk["month_end"]),
                "med_roe":    [None if pd.isna(x) else float(x) for x in chunk["median_roe_5y"]],
                "med_roce":   [None if pd.isna(x) else float(x) for x in chunk["median_roce_5y"]],
            }
            conn.execute(insert_sql, params)

#Assign Q1–Q4 quality quartiles based on 5y median RoE/RoCE within (is_financial, size_band) buckets
def recompute_quality_quartiles(batch_size: int = 10000):
    """
    For each month_end, label each stock as Q1–Q4 within its
    (is_financial, size_band) bucket based on 5y median RoE/RoCE.
    """
    engine = get_engine()
    with engine.begin() as conn:
        med = pd.read_sql(
            """
            SELECT isin, month_end, median_roe_5y, median_roce_5y
            FROM fundlab.stock_quality_median
            ORDER BY isin, month_end
            """,
            conn,
        )
        size = pd.read_sql(
            """
            SELECT isin, band_date, size_band
            FROM fundlab.stock_size_band
            ORDER BY isin, band_date
            """,
            conn,
        )
        master = pd.read_sql(
            """
            SELECT isin, is_financial
            FROM fundlab.stock_master
            """,
            conn,
        )

    if med.empty or size.empty:
        st.warning("Quality medians or size bands are empty.")
        return

    # Map size_band to each (isin, month_end) using last band_date <= month_end
    months = med["month_end"].sort_values().unique()
    size_records = []

    for isin, g in size.groupby("isin", sort=False):
        g = g.sort_values("band_date")
        for m in months:
            sub = g[g["band_date"] <= m]
            if sub.empty:
                continue
            band = sub.iloc[-1]["size_band"]
            size_records.append(
                {"isin": isin, "month_end": m, "size_band": band}
            )

    size_for_month = pd.DataFrame(size_records)

    full = (
        med.merge(size_for_month, on=["isin", "month_end"], how="inner")
           .merge(master, on="isin", how="left")
    )
    full["is_financial"] = full["is_financial"].fillna(False)

    rows = []

    for m, g_m in full.groupby("month_end", sort=False):
        for fin_flag in (True, False):
            g_f = g_m[g_m["is_financial"] == fin_flag]
            for band in ("Large", "Mid", "Small"):
                g_b = g_f[g_f["size_band"] == band]
                if g_b.empty:
                    continue

                if fin_flag:
                    g_b = g_b.assign(quality_metric=g_b["median_roe_5y"])
                else:
                    g_b = g_b.assign(quality_metric=g_b["median_roce_5y"])

                g_b = g_b[~g_b["quality_metric"].isna()]
                if g_b.empty:
                    continue

                # Assign quartiles: Q1 highest quality
                try:
                    # Use rank so ties handled deterministically
                    rank_series = g_b["quality_metric"].rank(
                        method="first", ascending=False
                    )
                    q_labels = pd.qcut(
                        rank_series,
                        4,
                        labels=["Q1", "Q2", "Q3", "Q4"],
                    )
                except ValueError:
                    # Too few stocks for qcut; manual thresholds
                    ranks = g_b["quality_metric"].rank(
                        method="first", ascending=False
                    )
                    n = len(ranks)

                    def q_of_rank(r):
                        if r <= 0.25 * n:
                            return "Q1"
                        elif r <= 0.5 * n:
                            return "Q2"
                        elif r <= 0.75 * n:
                            return "Q3"
                        else:
                            return "Q4"

                    q_labels = ranks.apply(q_of_rank)

                for isin_val, qm, qlab in zip(
                    g_b["isin"], g_b["quality_metric"], q_labels
                ):
                    rows.append(
                        {
                            "isin": isin_val,
                            "month_end": m,
                            "size_band": band,
                            "is_financial": fin_flag,
                            "quality_metric": float(qm),
                            "quality_quartile": str(qlab),
                        }
                    )

    if not rows:
        st.warning("No quartile labels could be computed.")
        return

    quart_df = pd.DataFrame(rows)

    insert_sql = text("""
        INSERT INTO fundlab.stock_quality_quartile (
            isin,
            month_end,
            size_band,
            is_financial,
            quality_metric,
            quality_quartile
        )
        SELECT
            unnest(:isins),
            unnest(:month_ends),
            unnest(:bands),
            unnest(:fin_flags),
            unnest(:metrics),
            unnest(:quartiles)
        ON CONFLICT (isin, month_end)
        DO UPDATE
        SET size_band        = EXCLUDED.size_band,
            is_financial     = EXCLUDED.is_financial,
            quality_metric   = EXCLUDED.quality_metric,
            quality_quartile = EXCLUDED.quality_quartile
    """)

    with engine.begin() as conn:
        n = len(quart_df)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            chunk = quart_df.iloc[start:end]
            params = {
                "isins":      list(chunk["isin"].astype(str)),
                "month_ends": list(chunk["month_end"]),
                "bands":      list(chunk["size_band"].astype(str)),
                "fin_flags":  list(chunk["is_financial"].astype(bool)),
                "metrics":    [float(x) for x in chunk["quality_metric"]],
                "quartiles":  list(chunk["quality_quartile"].astype(str)),
            }
            conn.execute(insert_sql, params)


def compute_quality_bucket_exposure(fund_id: int, month_ends: list[date]) -> pd.DataFrame:
    """
    Returns a DataFrame with index Q1..Q4 and columns per month_end (Mmm YYYY),
    values = % of domestic equity weight in that quality bucket for the given fund.
    """
    if not month_ends:
        return pd.DataFrame()

    start = min(month_ends)
    end = max(month_ends)

    engine = get_engine()
    with engine.connect() as conn:
        query = text(
            """
            SELECT
                fp.month_end,
                fp.holding_weight,
                sq.quality_quartile
            FROM fundlab.fund_portfolio fp
            JOIN fundlab.stock_quality_quartile sq
              ON sq.isin = fp.isin
             AND sq.month_end = fp.month_end
            WHERE fp.fund_id = :fid
              AND fp.month_end BETWEEN :d_start AND :d_end
              AND fp.asset_type = 'Domestic Equities'
            """
        )

        df = pd.read_sql(
            query,
            conn,
            params={"fid": fund_id, "d_start": start, "d_end": end},
        )

    if df.empty:
        return pd.DataFrame()

    # Rebase domestic equity weights to 100% per month
    df["holding_weight"] = df["holding_weight"].astype(float)
    totals = df.groupby("month_end")["holding_weight"].transform("sum")
    df["re_based_weight"] = df["holding_weight"] / totals * 100.0

    # Aggregate by quartile and month
    pivot = (
        df.groupby(["quality_quartile", "month_end"])["re_based_weight"]
        .sum()
        .unstack("month_end")
        .reindex(index=["Q1", "Q2", "Q3", "Q4"])
    )

    # Use raw holding_weight WITHOUT re-basing - uncomment to see in code. But comment the above block when doing so
    
    #df["holding_weight"] = df["holding_weight"].astype(float)

    #pivot = (
    #    df.groupby(["quality_quartile", "month_end"])["holding_weight"]
    #    .sum()
    #    .unstack("month_end")
    #    .reindex(index=["Q1", "Q2", "Q3", "Q4"])
    #)'''




    if pivot is None or pivot.empty:
        return pd.DataFrame()

    # Pretty month labels
    pivot.columns = [pd.to_datetime(col).strftime("%b %Y") for col in pivot.columns]

    # Add a 'Total' row so user can see sum of exposures
    total_row = pivot.sum(axis=0).to_frame().T
    total_row.index = ["Total"]

    pivot = pd.concat([pivot, total_row])

    return pivot


def rebuild_stock_monthly_valuations(
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
    year_block: int = 5,
):
    """
    Housekeeping job (full-history export to a single Excel file):

    1) Reads monthly market cap from fundlab.stock_price
    2) Attaches TTM sales / PAT (from stock_quarterly_financials, consolidated)
    3) Attaches latest annual book value (from stock_annual_book_value, consolidated)
    4) Computes stock-level P/S, P/E, P/B with:
          - only positive denominators
          - ratios capped to abs(value) <= 1000 (else NULL)
    5) Does NOT write to Supabase.
       Instead, produces a single Excel file with 5-year sheets
       for manual upload.

    Output columns (aligned to fundlab.stock_monthly_valuations):
      isin, month_end, ttm_sales, ttm_pat, book_value, ps, pe, pb
    """
    engine = get_engine()

    # --------------------------------------------------------------
    # 0) Auto-derive start/end if not given, from stock_price
    # --------------------------------------------------------------
    if start_date is None or end_date is None:
        with engine.begin() as conn:
            rng = conn.execute(
                text(
                    """
                    SELECT
                        MIN(price_date)::date AS min_d,
                        MAX(price_date)::date AS max_d
                    FROM fundlab.stock_price;
                    """
                )
            ).fetchone()

        if not rng or rng.min_d is None or rng.max_d is None:
            st.warning("No data in fundlab.stock_price to infer date range.")
            return

        if start_date is None:
            start_date = rng.min_d
        if end_date is None:
            end_date = rng.max_d

    st.write(f"Using price date range: {start_date} → {end_date}")

    # --------------------------------------------------------------
    # 1) Fetch monthly price / market cap
    # --------------------------------------------------------------
    with engine.begin() as conn:
        price_sql = text(
            """
            SELECT
                isin,
                price_date::date AS month_end,
                market_cap
            FROM fundlab.stock_price
            WHERE price_date BETWEEN :start_date AND :end_date
              AND market_cap IS NOT NULL;
            """
        )
        prices = pd.read_sql(
            price_sql,
            conn,
            params={"start_date": start_date, "end_date": end_date},
        )

    if prices.empty:
        st.info("No stock_price data found in the given period for stock valuations.")
        return

    # Clean up price data
    prices["isin"] = prices["isin"].astype(str).str.strip()
    prices["month_end"] = pd.to_datetime(
        prices["month_end"], errors="coerce"
    ).dt.normalize()
    prices["market_cap"] = pd.to_numeric(prices["market_cap"], errors="coerce")

    prices = prices.dropna(subset=["isin", "month_end", "market_cap"])
    if prices.empty:
        st.info("After cleaning, no usable price/market_cap rows remain.")
        return

    base = prices.copy()

    all_isins = sorted(base["isin"].unique().tolist())
    min_month = base["month_end"].min()
    max_month = base["month_end"].max()

    st.write(f"Distinct ISINs in price data: {len(all_isins)}")
    st.write(
        f"Month_end range in price data: {min_month.date()} → {max_month.date()}"
    )

    # --------------------------------------------------------------
    # 2) Fetch quarterly & annual fundamentals once
    # --------------------------------------------------------------
    with engine.begin() as conn:
        # Quarterly sales & PAT
        q_sql = text(
            """
            SELECT isin, period_end, sales, pat
            FROM fundlab.stock_quarterly_financials
            WHERE isin = ANY(:isins)
              AND is_consolidated = TRUE
              AND period_end <= :end_date
            ORDER BY isin, period_end;
            """
        )
        qdf = pd.read_sql(
            q_sql,
            conn,
            params={"isins": all_isins, "end_date": max_month},
        )

        # Annual book value
        bv_sql = text(
            """
            SELECT isin, year_end, book_value
            FROM fundlab.stock_annual_book_value
            WHERE isin = ANY(:isins)
              AND is_consolidated = TRUE
              AND year_end <= :end_date
            ORDER BY isin, year_end;
            """
        )
        bvdf = pd.read_sql(
            bv_sql,
            conn,
            params={"isins": all_isins, "end_date": max_month},
        )

    # ---- Quarterly data / TTM ----
    if not qdf.empty:
        qdf["isin"] = qdf["isin"].astype(str).str.strip()
        qdf["period_end"] = pd.to_datetime(
            qdf["period_end"], errors="coerce"
        ).dt.normalize()
        qdf["sales"] = pd.to_numeric(qdf["sales"], errors="coerce")
        qdf["pat"] = pd.to_numeric(qdf["pat"], errors="coerce")

        qdf = qdf.dropna(subset=["isin", "period_end"])
        qdf = qdf.sort_values(["isin", "period_end"])

        qdf["ttm_sales"] = (
            qdf.groupby("isin")["sales"]
            .rolling(window=4, min_periods=4)
            .sum()
            .reset_index(level=0, drop=True)
        )
        qdf["ttm_pat"] = (
            qdf.groupby("isin")["pat"]
            .rolling(window=4, min_periods=4)
            .sum()
            .reset_index(level=0, drop=True)
        )

        qdf_ttm = qdf[["isin", "period_end", "ttm_sales", "ttm_pat"]].dropna(
            how="all", subset=["ttm_sales", "ttm_pat"]
        )
    else:
        qdf_ttm = pd.DataFrame(columns=["isin", "period_end", "ttm_sales", "ttm_pat"])

    # ---- Annual BV ----
    if not bvdf.empty:
        bvdf["isin"] = bvdf["isin"].astype(str).str.strip()
        bvdf["year_end"] = pd.to_datetime(
            bvdf["year_end"], errors="coerce"
        ).dt.normalize()
        bvdf["book_value"] = pd.to_numeric(bvdf["book_value"], errors="coerce")
        bvdf = bvdf.dropna(subset=["isin", "year_end"])
        bvdf = bvdf.sort_values(["isin", "year_end"]).reset_index(drop=True)
    else:
        bvdf = pd.DataFrame(columns=["isin", "year_end", "book_value"])

    # --------------------------------------------------------------
    # 3) Attach TTM sales / PAT and BV to (isin, month_end)
    # --------------------------------------------------------------
    base = base.reset_index(drop=True)
    base["ttm_sales"] = np.nan
    base["ttm_pat"] = np.nan
    base["book_value"] = np.nan

    # ---- TTM sales/PAT via manual "as-of" alignment ----
    if not qdf_ttm.empty:
        qdf_ttm = qdf_ttm.sort_values(["isin", "period_end"]).reset_index(drop=True)

        for isin, sub_ttm in qdf_ttm.groupby("isin"):
            mask = base["isin"] == isin
            if not mask.any():
                continue

            idx = base.index[mask]
            h_dates = base.loc[idx, "month_end"].values.astype("datetime64[ns]")
            q_dates = sub_ttm["period_end"].values.astype("datetime64[ns]")

            pos = np.searchsorted(q_dates, h_dates, side="right") - 1
            valid = pos >= 0
            if not np.any(valid):
                continue

            aligned_sales = np.full(h_dates.shape, np.nan)
            aligned_pat = np.full(h_dates.shape, np.nan)
            aligned_sales[valid] = sub_ttm["ttm_sales"].values[pos[valid]]
            aligned_pat[valid] = sub_ttm["ttm_pat"].values[pos[valid]]

            base.loc[idx, "ttm_sales"] = aligned_sales
            base.loc[idx, "ttm_pat"] = aligned_pat

    # ---- Book value via manual "as-of" alignment ----
    if not bvdf.empty:
        for isin, sub_bv in bvdf.groupby("isin"):
            mask = base["isin"] == isin
            if not mask.any():
                continue

            idx = base.index[mask]
            h_dates = base.loc[idx, "month_end"].values.astype("datetime64[ns]")
            b_dates = sub_bv["year_end"].values.astype("datetime64[ns]")

            pos = np.searchsorted(b_dates, h_dates, side="right") - 1
            valid = pos >= 0
            if not np.any(valid):
                continue

            aligned_bv = np.full(h_dates.shape, np.nan)
            aligned_bv[valid] = sub_bv["book_value"].values[pos[valid]]

            base.loc[idx, "book_value"] = aligned_bv

    # --------------------------------------------------------------
    # 4) Compute P/S, P/E, P/B at stock level with cap of 1000
    # --------------------------------------------------------------
    df = base.copy()

    mc = pd.to_numeric(df["market_cap"], errors="coerce")
    ttm_sales = pd.to_numeric(df["ttm_sales"], errors="coerce")
    ttm_pat = pd.to_numeric(df["ttm_pat"], errors="coerce")
    bv = pd.to_numeric(df["book_value"], errors="coerce")

    ps_raw = np.where((mc > 0) & (ttm_sales > 0), mc / ttm_sales, np.nan)
    pe_raw = np.where((mc > 0) & (ttm_pat > 0), mc / ttm_pat, np.nan)
    pb_raw = np.where((mc > 0) & (bv > 0), mc / bv, np.nan)

    cap = 1000.0
    df["ps"] = np.where(np.abs(ps_raw) <= cap, ps_raw, np.nan)
    df["pe"] = np.where(np.abs(pe_raw) <= cap, pe_raw, np.nan)
    df["pb"] = np.where(np.abs(pb_raw) <= cap, pb_raw, np.nan)

    valuations_df = df[
        ["isin", "month_end", "ttm_sales", "ttm_pat", "book_value", "ps", "pe", "pb"]
    ].copy()

    valuations_df["isin"] = valuations_df["isin"].astype(str).str.strip()
    valuations_df["month_end"] = pd.to_datetime(
        valuations_df["month_end"], errors="coerce"
    ).dt.date

    valuations_df = valuations_df.dropna(subset=["isin", "month_end"])
    valuations_df = valuations_df.sort_values(["month_end", "isin"]).reset_index(
        drop=True
    )

    n_total = len(valuations_df)
    if n_total == 0:
        st.info("No rows to export after processing.")
        return

    st.write(f"Total valuation rows to export: {n_total:,}")

    valuations_df = valuations_df.replace({np.inf: np.nan, -np.inf: np.nan})

    # --------------------------------------------------------------
    # 5) Create ONE Excel file with 5-year sheets + single download
    # --------------------------------------------------------------
    years = valuations_df["month_end"].apply(lambda d: d.year)
    min_year = int(years.min())
    max_year = int(years.max())

    st.write(
        f"Preparing Excel workbook with sheets in {year_block}-year blocks "
        f"from {min_year} to {max_year}..."
    )

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for block_start_year in range(min_year, max_year + 1, year_block):
            block_end_year = min(block_start_year + year_block - 1, max_year)

            block_mask = valuations_df["month_end"].between(
                dt.date(block_start_year, 1, 1),
                dt.date(block_end_year, 12, 31),
            )
            block_df = valuations_df.loc[block_mask].copy()
            if block_df.empty:
                continue

            sheet_name = f"{block_start_year}_{block_end_year}"
            st.write(
                f"Sheet {sheet_name}: {len(block_df):,} rows"
            )

            block_df[
                ["isin", "month_end", "ttm_sales", "ttm_pat", "book_value", "ps", "pe", "pb"]
            ].to_excel(writer, sheet_name=sheet_name, index=False)

    buffer.seek(0)

    st.download_button(
        label="Download ALL stock valuations (multi-sheet Excel)",
        data=buffer,
        file_name="stock_monthly_valuations_all_years.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.success("Valuation Excel workbook ready. Download and upload to Supabase as needed.")

def rebuild_fund_monthly_valuations(
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
    fund_ids: list[int] | None = None,
):
    """
    Housekeeping job (CSV version):

    1) Figures out the date range to process:
         - If start_date / end_date are given, use them.
         - Else infer from fundlab.fund_portfolio.
    2) Pulls Domestic Equity holdings in that period (optionally for selected funds),
       joined with stock_master to get is_financial.
    3) Rebases holding_weight within (fund_id, month_end) so Domestic Equities = 100%.
    4) Joins to fundlab.stock_monthly_valuations on (isin, month_end).
    5) For each (fund_id, month_end, segment = Total/Financials/Non-financials)
       computes weighted P/S, P/E, P/B as:
          - Filter to ps/pe/pb > 0 and not null.
          - Weights = rebased domestic weights within the segment, normalized
            over stocks with valid metric values.
    6) Before computing, it looks at fundlab.fund_monthly_valuations and
       **skips any (fund_id, month_end, segment) that already exist**.
    7) Instead of writing to Supabase, it exposes a CSV download with only
       the *new* rows to be uploaded manually via the Supabase UI.

    Notes:
      - This avoids all UNNEST / executemany issues with large arrays.
      - You can periodically run this, download the CSV, and import it into
        fundlab.fund_monthly_valuations (using Supabase UI's CSV import).
    """

    engine = get_engine()

    # --------------------------------------------------------------
    # 0) Derive date range from fund_portfolio if needed
    # --------------------------------------------------------------
    with engine.begin() as conn:
        if start_date is None or end_date is None:
            if fund_ids:
                rng = conn.execute(
                    text(
                        """
                        SELECT
                            MIN(month_end)::date AS min_d,
                            MAX(month_end)::date AS max_d
                        FROM fundlab.fund_portfolio
                        WHERE fund_id = ANY(:fund_ids)
                        """
                    ),
                    {"fund_ids": fund_ids},
                ).fetchone()
            else:
                rng = conn.execute(
                    text(
                        """
                        SELECT
                            MIN(month_end)::date AS min_d,
                            MAX(month_end)::date AS max_d
                        FROM fundlab.fund_portfolio
                        """
                    )
                ).fetchone()

            if not rng or rng.min_d is None or rng.max_d is None:
                st.warning("No data in fundlab.fund_portfolio to infer date range.")
                return

            if start_date is None:
                start_date = rng.min_d
            if end_date is None:
                end_date = rng.max_d

    # Just for safety: enforce date order
    if start_date > end_date:
        st.error(f"Invalid date range: start_date {start_date} > end_date {end_date}.")
        return

    # --------------------------------------------------------------
    # 1) Fetch holdings (Domestic Equities only) for the period
    # --------------------------------------------------------------
    with engine.begin() as conn:
        if fund_ids:
            holdings_sql = text(
                """
                SELECT
                    fp.fund_id,
                    fp.month_end,
                    fp.isin,
                    fp.holding_weight AS weight_pct,
                    fp.asset_type,
                    sm.is_financial
                FROM fundlab.fund_portfolio fp
                JOIN fundlab.stock_master sm
                  ON fp.isin = sm.isin
                WHERE fp.month_end BETWEEN :start_date AND :end_date
                  AND fp.asset_type = 'Domestic Equities'
                  AND fp.fund_id = ANY(:fund_ids)
                ORDER BY fp.fund_id, fp.month_end, fp.isin
                """
            )
            holdings = pd.read_sql(
                holdings_sql,
                conn,
                params={
                    "start_date": start_date,
                    "end_date": end_date,
                    "fund_ids": fund_ids,
                },
            )
        else:
            holdings_sql = text(
                """
                SELECT
                    fp.fund_id,
                    fp.month_end,
                    fp.isin,
                    fp.holding_weight AS weight_pct,
                    fp.asset_type,
                    sm.is_financial
                FROM fundlab.fund_portfolio fp
                JOIN fundlab.stock_master sm
                  ON fp.isin = sm.isin
                WHERE fp.month_end BETWEEN :start_date AND :end_date
                  AND fp.asset_type = 'Domestic Equities'
                ORDER BY fp.fund_id, fp.month_end, fp.isin
                """
            )
            holdings = pd.read_sql(
                holdings_sql,
                conn,
                params={
                    "start_date": start_date,
                    "end_date": end_date,
                },
            )

    if holdings.empty:
        st.info("No fund_portfolio holdings found in the given period.")
        return

    # Clean holdings
    holdings["isin"] = holdings["isin"].astype(str).str.strip()
    holdings["month_end"] = pd.to_datetime(
        holdings["month_end"], errors="coerce"
    ).dt.normalize()
    holdings["weight_pct"] = pd.to_numeric(holdings["weight_pct"], errors="coerce")
    holdings = holdings.dropna(subset=["fund_id", "isin", "month_end", "weight_pct"])

    if holdings.empty:
        st.info("No usable holdings after cleaning.")
        return

    # --------------------------------------------------------------
    # 2) Rebase weights within (fund_id, month_end) to Domestic = 100
    # --------------------------------------------------------------
    sum_w = holdings.groupby(["fund_id", "month_end"])["weight_pct"].transform("sum")
    holdings = holdings[sum_w > 0].copy()
    if holdings.empty:
        st.info("No holdings with positive weight sums.")
        return

    holdings["w_domestic"] = holdings["weight_pct"] / sum_w

    # --------------------------------------------------------------
    # 3) Pull stock valuations only for relevant ISINs and dates
    # --------------------------------------------------------------
    all_isins = sorted(holdings["isin"].unique().tolist())
    min_month = holdings["month_end"].min()
    max_month = holdings["month_end"].max()

    with engine.begin() as conn:
        val_sql = text(
            """
            SELECT
                isin,
                month_end,
                ttm_sales,
                ttm_pat,
                book_value,
                ps,
                pe,
                pb
            FROM fundlab.stock_monthly_valuations
            WHERE isin = ANY(:isins)
              AND month_end BETWEEN :start_date AND :end_date
            """
        )
        vals = pd.read_sql(
            val_sql,
            conn,
            params={
                "isins": all_isins,
                "start_date": min_month,
                "end_date": max_month,
            },
        )

    if vals.empty:
        st.info("No stock_monthly_valuations rows found for the given holdings.")
        return

    vals["isin"] = vals["isin"].astype(str).str.strip()
    vals["month_end"] = pd.to_datetime(vals["month_end"], errors="coerce").dt.normalize()

    # Merge valuations into holdings
    
    # --------------------------------------------------------------
    # 3) Pull stock valuations only for relevant ISINs and dates
    #    (match by year-month, ignore day)
    # --------------------------------------------------------------
    all_isins = sorted(holdings["isin"].unique().tolist())
    min_month = holdings["month_end"].min()
    max_month = holdings["month_end"].max()

    with engine.begin() as conn:
        val_sql = text(
            """
            SELECT
                isin,
                month_end,
                ttm_sales,
                ttm_pat,
                book_value,
                ps,
                pe,
                pb
            FROM fundlab.stock_monthly_valuations
            WHERE isin = ANY(:isins)
              AND month_end BETWEEN :start_date AND :end_date
            """
        )
        vals = pd.read_sql(
            val_sql,
            conn,
            params={
                "isins": all_isins,
                "start_date": min_month,
                "end_date": max_month,
            },
        )

    if vals.empty:
        st.info("No stock_monthly_valuations rows found for the given holdings.")
        return

    # Clean & build month_key (year-month) on both sides
    holdings["isin"] = holdings["isin"].astype(str).str.strip()
    holdings["month_end"] = pd.to_datetime(holdings["month_end"], errors="coerce")
    holdings = holdings.dropna(subset=["month_end"])
    holdings["month_key"] = holdings["month_end"].dt.to_period("M")

    vals["isin"] = vals["isin"].astype(str).str.strip()
    vals["month_end"] = pd.to_datetime(vals["month_end"], errors="coerce")
    vals = vals.dropna(subset=["month_end"])
    vals["month_key"] = vals["month_end"].dt.to_period("M")

    # Merge on (isin, month_key) instead of exact date
    df = holdings.merge(
        vals.drop(columns=["month_end"]),   # drop its date; keep month_key
        on=["isin", "month_key"],
        how="left",
        suffixes=("", "_val"),
    )

    if df[["ps", "pe", "pb"]].isna().all(axis=None):
        st.info("All merged valuations are NaN after month_key merge; nothing to write.")
        return

    # Use canonical month_end = last day of month for output
    df["month_end"] = df["month_key"].dt.to_timestamp("M").dt.date



    # If no valuations at all, nothing to do
    if df[["ps", "pe", "pb"]].isna().all(axis=None):
        st.info("All merged valuations are NaN; nothing to write.")
        return

    # --------------------------------------------------------------
    # 4) Load existing fund_monthly_valuations to support incremental runs
    # --------------------------------------------------------------
    with engine.begin() as conn:
        if fund_ids:
            existing_sql = text(
                """
                SELECT fund_id, month_end, segment
                FROM fundlab.fund_monthly_valuations
                WHERE month_end BETWEEN :start_date AND :end_date
                  AND fund_id = ANY(:fund_ids)
                """
            )
            existing_rows = conn.execute(
                existing_sql,
                {"start_date": start_date, "end_date": end_date, "fund_ids": fund_ids},
            ).fetchall()
        else:
            existing_sql = text(
                """
                SELECT fund_id, month_end, segment
                FROM fundlab.fund_monthly_valuations
                WHERE month_end BETWEEN :start_date AND :end_date
                """
            )
            existing_rows = conn.execute(
                existing_sql,
                {"start_date": start_date, "end_date": end_date},
            ).fetchall()

    existing_keys: set[tuple[int, dt.date, str]] = set()
    for r in existing_rows:
        # r.month_end should already be a date; keep as-is
        existing_keys.add((int(r.fund_id), r.month_end, str(r.segment)))

    # --------------------------------------------------------------
    # 5) Compute fund-level valuations: P/S, P/E, P/B by segment
    # --------------------------------------------------------------
    df["is_financial"] = df["is_financial"].astype(bool)

    segments = ["Total", "Financials", "Non-financials"]
    records: list[dict] = []

    grouped = df.groupby(["fund_id", "month_end"], sort=True)
    n_groups = len(grouped)

    progress = st.progress(0)
    status_placeholder = st.empty()

    if n_groups == 0:
        st.info("No (fund, month_end) groups after merge; nothing to compute.")
        return

    for i, ((fund_id, month_end), grp) in enumerate(grouped, start=1):
        grp = grp.copy()

        # convert month_end to python date for key comparisons
        if isinstance(month_end, pd.Timestamp):
            month_end_date = month_end.date()
        else:
            # should be datetime.date already
            month_end_date = month_end

        # Helper to compute weighted avg of any given metric column name
        def _weighted_metric(seg_grp: pd.DataFrame, col_name: str) -> tuple[float, int, float]:
            g = seg_grp.copy()
            g[col_name] = pd.to_numeric(g[col_name], errors="coerce")
            g = g[g[col_name] > 0].copy()
            if g.empty:
                return (np.nan, 0, 0.0)

            w = pd.to_numeric(g["w_domestic"], errors="coerce").fillna(0.0)
            sum_w = float(w.sum())
            if sum_w <= 0:
                return (np.nan, 0, 0.0)

            w_rebased = w / sum_w
            metric = g[col_name].to_numpy()
            value = float((w_rebased * metric).sum())
            return (value, len(g), sum_w)

        for seg in segments:
            key = (int(fund_id), month_end_date, seg)

            # Skip if this combination already exists in Supabase
            if key in existing_keys:
                continue

            if seg == "Financials":
                seg_grp = grp[grp["is_financial"]].copy()
            elif seg == "Non-financials":
                seg_grp = grp[~grp["is_financial"]].copy()
            else:  # 'Total'
                seg_grp = grp.copy()

            if seg_grp.empty:
                continue

            total_weight_seg = float(seg_grp["w_domestic"].sum())

            ps_val, ps_count, _ = _weighted_metric(seg_grp, "ps")
            pe_val, pe_count, _ = _weighted_metric(seg_grp, "pe")
            pb_val, pb_count, _ = _weighted_metric(seg_grp, "pb")

            if np.isnan(ps_val) and np.isnan(pe_val) and np.isnan(pb_val):
                # No usable metrics for this segment
                continue

            stock_count = max(ps_count, pe_count, pb_count)

            records.append(
                {
                    "fund_id": int(fund_id),
                    "month_end": month_end_date,
                    "segment": seg,
                    "ps": None if np.isnan(ps_val) else float(ps_val),
                    "pe": None if np.isnan(pe_val) else float(pe_val),
                    "pb": None if np.isnan(pb_val) else float(pb_val),
                    "stock_count": int(stock_count),
                    "total_weight": total_weight_seg,
                    "notes": None,
                }
            )

        # Update progress bar
        if n_groups > 0:
            pct = int(i * 100 / n_groups)
            progress.progress(min(pct, 100))
            if i % 50 == 0 or i == n_groups:
                status_placeholder.text(f"Processed {i} / {n_groups} fund-month groups")

    progress.empty()
    status_placeholder.empty()

    if not records:
        st.success(
            f"No *new* fund_monthly_valuations rows needed between "
            f"{start_date} and {end_date}. Table already up to date for this range."
        )
        return

    df_out = pd.DataFrame.from_records(records)
    df_out.sort_values(["fund_id", "month_end", "segment"], inplace=True)

    st.write(
        f"Prepared **{len(df_out)}** new (fund_id, month_end, segment) valuation rows "
        f"between {start_date} and {end_date} that do *not* yet exist in "
        f"`fundlab.fund_monthly_valuations`."
    )

    # Show a small preview
    st.dataframe(df_out.head(50))

    # Offer CSV download for manual Supabase import
    csv_bytes = df_out.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download fund_monthly_valuations CSV (new rows only)",
        data=csv_bytes,
        file_name="fund_monthly_valuations_delta.csv",
        mime="text/csv",
    )

    st.info(
        "Upload this CSV into Supabase (fundlab.fund_monthly_valuations). "
        "Because we only included rows that don't already exist for this date range, "
        "it will act as an incremental update."
    )


def compute_portfolio_valuations_timeseries(
    fund_ids: list[int],
    focus_fund_id: int,
    start_date: dt.date,
    end_date: dt.date,
    segment_choice: str,
    metric_choice: str,
    mode: str,
) -> pd.DataFrame:
    """
    Compute valuation time series for:
      - the focus fund, and
      - the median of other selected funds (excluding the focus fund).

    metric_choice: 'P/S', 'P/B', 'P/E'
    segment_choice: 'Financials', 'Non-financials', 'Total'
    mode:
      - 'Valuations of historical portfolios'
      - 'Historical valuations of current portfolio'

    Returns DataFrame with columns:
      month_end (datetime64[ns]),
      series ('Focus fund' / 'Universe median (others)'),
      value (float)
    """

    if not fund_ids or focus_fund_id not in fund_ids:
        raise ValueError("Focus fund must be among selected funds.")

    other_fund_ids = [fid for fid in fund_ids if fid != focus_fund_id]
    if not other_fund_ids:
        raise ValueError("Need at least one other fund to compute universe median.")

    # Map UI selections to DB column names
    metric_map = {
        "P/S": "ps",
        "P/E": "pe",
        "P/B": "pb",
    }
    metric_key = metric_map.get(metric_choice)
    if metric_key is None:
        raise ValueError(f"Unsupported metric_choice: {metric_choice}")

    # Segment labels in DB and UI are already aligned:
    # 'Financials', 'Non-financials', 'Total'
    segment_value = segment_choice

    engine = get_engine()

    # ------------------------------------------------------------------
    # MODE 1: Valuations of historical portfolios
    # ------------------------------------------------------------------
    if mode == "Valuations of historical portfolios":
        # Here we DO NOT recompute from holdings; we trust and use the
        # precomputed values stored in fundlab.fund_monthly_valuations,
        # which were built using weighted-average-of-stock-multiples.
        with engine.begin() as conn:
            sql = text(
                """
                SELECT
                    fund_id,
                    month_end,
                    segment,
                    ps,
                    pe,
                    pb
                FROM fundlab.fund_monthly_valuations
                WHERE fund_id = ANY(:fund_ids)
                  AND month_end BETWEEN :start_date AND :end_date
                  AND segment = :segment;
                """
            )
            df = pd.read_sql(
                sql,
                conn,
                params={
                    "fund_ids": fund_ids,
                    "start_date": start_date,
                    "end_date": end_date,
                    "segment": segment_value,
                },
            )

        if df.empty:
            return pd.DataFrame(columns=["month_end", "series", "value"])

        # Pick the chosen metric column
        df["value"] = pd.to_numeric(df[metric_key], errors="coerce")
        df = df.dropna(subset=["value"])
        if df.empty:
            return pd.DataFrame(columns=["month_end", "series", "value"])

        # Ensure month_end is a proper datetime
        df["month_end"] = pd.to_datetime(df["month_end"], errors="coerce").dt.normalize()
        df = df.dropna(subset=["month_end"])
        if df.empty:
            return pd.DataFrame(columns=["month_end", "series", "value"])

        # Build focus series
        focus = df[df["fund_id"] == focus_fund_id].copy()
        if focus.empty:
            return pd.DataFrame(columns=["month_end", "series", "value"])
        focus = focus[["month_end", "value"]]
        focus["series"] = "Focus fund"

        # Build universe-median (others) series
        others = df[df["fund_id"].isin(other_fund_ids)].copy()
        if others.empty:
            return pd.DataFrame(columns=["month_end", "series", "value"])

        median_others = (
            others.groupby("month_end", as_index=False)["value"]
            .median()
            .rename(columns={"value": "value_median"})
        )
        median_others["series"] = "Universe median (others)"
        median_others = median_others.rename(columns={"value_median": "value"})[
            ["month_end", "value", "series"]
        ]

        df_chart = pd.concat([focus, median_others], ignore_index=True)
        df_chart = df_chart.sort_values(["month_end", "series"]).reset_index(drop=True)
        return df_chart

    # ------------------------------------------------------------------
    # MODE 2: Historical valuations of current portfolio
    # ------------------------------------------------------------------
    if mode != "Historical valuations of current portfolio":
        raise ValueError(f"Unsupported mode: {mode}")

    engine = get_engine()

    # Build month-level grid (as Periods) for the requested range
    month_periods = pd.period_range(start=start_date, end=end_date, freq="M")
    if len(month_periods) == 0:
        return pd.DataFrame(columns=["month_end", "series", "value"])

    with engine.begin() as conn:
        # 1) Anchor portfolios: latest month_end <= end_date for each fund
        anchor_sql = text(
            """
            SELECT fund_id, MAX(month_end) AS anchor_month_end
            FROM fundlab.fund_portfolio
            WHERE fund_id = ANY(:fund_ids)
              AND month_end <= :end_date
            GROUP BY fund_id;
            """
        )
        anchors = pd.read_sql(
            anchor_sql, conn, params={"fund_ids": fund_ids, "end_date": end_date}
        )

        if anchors.empty:
            raise ValueError("No portfolio data found on or before the valuation end date.")

        holdings_sql = text(
            """
            SELECT
                fp.fund_id,
                fp.month_end,
                fp.isin,
                fp.holding_weight AS weight_pct,
                fp.asset_type,
                sm.is_financial
            FROM fundlab.fund_portfolio fp
            JOIN fundlab.stock_master sm
              ON sm.isin = fp.isin
            JOIN (
                SELECT fund_id, MAX(month_end) AS anchor_month_end
                FROM fundlab.fund_portfolio
                WHERE fund_id = ANY(:fund_ids)
                  AND month_end <= :end_date
                GROUP BY fund_id
            ) a
              ON a.fund_id = fp.fund_id
             AND a.anchor_month_end = fp.month_end
            WHERE fp.asset_type = 'Domestic Equities'
            ORDER BY fp.fund_id, fp.month_end, fp.isin;
            """
        )
        anchor_holdings = pd.read_sql(
            holdings_sql, conn, params={"fund_ids": fund_ids, "end_date": end_date}
        )

    if anchor_holdings.empty:
        return pd.DataFrame(columns=["month_end", "series", "value"])

    # Clean holdings
    anchor_holdings["isin"] = anchor_holdings["isin"].astype(str).str.strip()
    anchor_holdings["month_end"] = pd.to_datetime(
        anchor_holdings["month_end"], errors="coerce"
    ).dt.normalize()

    # Segment filter at anchor level
    if segment_choice == "Financials":
        anchor_holdings = anchor_holdings[anchor_holdings["is_financial"].astype(bool)].copy()
    elif segment_choice == "Non-financials":
        anchor_holdings = anchor_holdings[~anchor_holdings["is_financial"].astype(bool)].copy()

    if anchor_holdings.empty:
        return pd.DataFrame(columns=["month_end", "series", "value"])

    # Rebase weights within each (fund_id, anchor_month) so Domestic = 100%
    anchor_holdings["weight_pct"] = pd.to_numeric(
        anchor_holdings["weight_pct"], errors="coerce"
    )
    sum_w = anchor_holdings.groupby(["fund_id", "month_end"])["weight_pct"].transform("sum")
    anchor_holdings = anchor_holdings[sum_w > 0].copy()
    if anchor_holdings.empty:
        return pd.DataFrame(columns=["month_end", "series", "value"])

    anchor_holdings["w_domestic"] = anchor_holdings["weight_pct"] / sum_w

    base_holdings = anchor_holdings[["fund_id", "isin", "w_domestic"]].copy()
    base_holdings["isin"] = base_holdings["isin"].astype(str).str.strip()

    all_isins = sorted(base_holdings["isin"].dropna().unique().tolist())
    if not all_isins:
        return pd.DataFrame(columns=["month_end", "series", "value"])

    # Fetch stock-level multiples for these ISINs over the full date window
    with engine.begin() as conn:
        vals_sql = text(
            """
            SELECT
                isin,
                month_end,
                ps,
                pe,
                pb
            FROM fundlab.stock_monthly_valuations
            WHERE isin = ANY(:isins)
              AND month_end BETWEEN :start_date AND :end_date;
            """
        )
        vals = pd.read_sql(
            vals_sql,
            conn,
            params={
                "isins": all_isins,
                "start_date": start_date,
                "end_date": end_date,
            },
        )

    if vals.empty:
        return pd.DataFrame(columns=["month_end", "series", "value"])

    vals["isin"] = vals["isin"].astype(str).str.strip()
    vals["month_end"] = pd.to_datetime(vals["month_end"], errors="coerce")
    vals = vals.dropna(subset=["month_end"])
    # Month key on valuations side
    vals["month_key"] = vals["month_end"].dt.to_period("M")

    # Restrict to requested month_periods
    vals = vals[vals["month_key"].isin(month_periods)].copy()
    if vals.empty:
        return pd.DataFrame(columns=["month_end", "series", "value"])

    # Build cross-join: (fund_id, isin, w_domestic) × month_key
    months_df = pd.DataFrame({"month_key": month_periods})
    base_holdings["_key"] = 1
    months_df["_key"] = 1
    combo = base_holdings.merge(months_df, on="_key").drop(columns=["_key"])

    # Join stock-level multiples on (isin, month_key)
    df = combo.merge(
        vals[["isin", "month_key", "ps", "pe", "pb"]],
        on=["isin", "month_key"],
        how="left",
    )

    # Select metric
    metric_map = {"P/S": "ps", "P/E": "pe", "P/B": "pb"}
    metric_key = metric_map.get(metric_choice)
    if metric_key is None:
        raise ValueError(f"Unsupported metric_choice: {metric_choice}")

    df["metric"] = pd.to_numeric(df[metric_key], errors="coerce")
    df = df[df["metric"] > 0].copy()
    if df.empty:
        return pd.DataFrame(columns=["month_end", "series", "value"])

    # Weighted average of stock-level multiples within (fund_id, month_key)
    def _agg_weighted_metric(group: pd.DataFrame) -> float | None:
        w = pd.to_numeric(group["w_domestic"], errors="coerce").fillna(0.0)
        if (w <= 0).all():
            return None
        sum_w = float(w.sum())
        if sum_w <= 0:
            return None
        w_norm = w / sum_w
        m = group["metric"].to_numpy()
        return float((w_norm * m).sum())

    agg_list = []
    for (fid, m_key), grp in df.groupby(["fund_id", "month_key"], sort=True):
        val = _agg_weighted_metric(grp)
        if val is None or np.isnan(val):
            continue
        # canonical month_end = last day of month
        month_end = m_key.to_timestamp("M")
        agg_list.append({"fund_id": int(fid), "month_end": month_end, "value": val})

    if not agg_list:
        return pd.DataFrame(columns=["month_end", "series", "value"])

    agg = pd.DataFrame(agg_list)
    agg["month_end"] = pd.to_datetime(agg["month_end"], errors="coerce").dt.normalize()
    agg = agg.dropna(subset=["month_end", "value"])

    if agg.empty:
        return pd.DataFrame(columns=["month_end", "series", "value"])

    # Focus fund vs universe median (others)
    focus = agg[agg["fund_id"] == focus_fund_id].copy()
    others = agg[agg["fund_id"].isin(other_fund_ids)].copy()
    if focus.empty or others.empty:
        return pd.DataFrame(columns=["month_end", "series", "value"])

    focus = focus[["month_end", "value"]]
    focus["series"] = "Focus fund"

    median_others = (
        others.groupby("month_end", as_index=False)["value"]
        .median()
        .rename(columns={"value": "value_median"})
    )
    median_others["series"] = "Universe median (others)"
    median_others = median_others.rename(columns={"value_median": "value"})[
        ["month_end", "value", "series"]
    ]

    df_chart = pd.concat([focus, median_others], ignore_index=True)
    df_chart = df_chart.sort_values(["month_end", "series"]).reset_index(drop=True)
    return df_chart


@st.cache_data(ttl=1800, show_spinner=False)
def cached_portfolio_valuations_timeseries(
    fund_ids,
    focus_fund_id,
    start_date,
    end_date,
    segment_choice,
    metric_choice,
    mode,
):
    """
    Cached wrapper for compute_portfolio_valuations_timeseries.

    Caches results for 30 minutes for a given combination of:
    - fund_ids (universe)
    - focus_fund_id
    - start_date / end_date
    - segment_choice
    - metric_choice
    - mode
    """
    return compute_portfolio_valuations_timeseries(
        fund_ids=fund_ids,
        focus_fund_id=focus_fund_id,
        start_date=start_date,
        end_date=end_date,
        segment_choice=segment_choice,
        metric_choice=metric_choice,
        mode=mode,
    )



def get_median_metric_for_stock(roe_roce_dict, isin, eval_date, is_financial):
    sub = roe_roce_dict.get(isin)
    if sub is None or sub.empty:
        return 0.0

    # take rows strictly before eval_date
    mask = sub["year_end_date"] < eval_date
    sub = sub.loc[mask]
    if sub.empty:
        return 0.0

    # last 5 periods
    sub = sub.tail(5)

    if is_financial:
        metric_series = sub["roe"]
    else:
        metric_series = sub["roce"]

    metric_series = metric_series.dropna()
    if metric_series.empty:
        return 0.0

    return float(metric_series.median())


def fetch_categories():
    engine = get_engine()
    # Pull distinct category names
    query = """
        SELECT DISTINCT category_name
        FROM fundlab.category
        WHERE LOWER(category_name) <> 'portfolio'
        ORDER BY category_name;
    """
    df = pd.read_sql(query, engine)
    if df.empty:
        return []
    return df["category_name"].dropna().tolist()



def fetch_funds_for_categories(categories):
    if not categories:
        return pd.DataFrame(columns=["fund_id", "fund_name", "category_name"])

    engine = get_engine()
    query = text("""
        SELECT f.fund_id,
               f.fund_name,
               c.category_name
        FROM fundlab.fund f
        JOIN fundlab.category c
          ON f.category_id = c.category_id
        WHERE c.category_name = ANY(:cats)
        ORDER BY f.fund_name;
    """)
    df = pd.read_sql(query, engine, params={"cats": categories})
    return df


# -----------------------------
# Fund manager tenure (dashboard)
# -----------------------------

@st.cache_data(ttl=3600)
def fetch_funds_by_categories(category_names: list[str]) -> pd.DataFrame:
    if not category_names:
        return pd.DataFrame()

    engine = get_engine()
    query = """
        SELECT f.fund_id, f.fund_name, c.category_name
        FROM fundlab.fund f
        JOIN fundlab.category c
          ON f.category_id = c.category_id
        WHERE c.category_name = ANY(%(cats)s)
        ORDER BY f.fund_name
    """
    return pd.read_sql(query, engine, params={"cats": category_names})


@st.cache_data(ttl=3600)
def fetch_fund_manager_tenure(fund_ids: list[int]) -> pd.DataFrame:
    if not fund_ids:
        return pd.DataFrame()

    engine = get_engine()
    query = """
        SELECT
            fund_id,
            fund_manager,
            from_date,
            to_date
        FROM fundlab.fund_manager_tenure
        WHERE fund_id = ANY(%(fund_ids)s)
        ORDER BY fund_id, from_date
    """
    return pd.read_sql(query, engine, params={"fund_ids": fund_ids})



def fetch_portfolio_raw(fund_ids, start_date, end_date):
    if not fund_ids:
        return pd.DataFrame()

    engine = get_engine()
    query = text("""
        SELECT
            fp.fund_id,
            fm.fund_name,
            fp.month_end,
            fp.isin,
            fp.holding_weight AS weight_pct,
            fp.asset_type,
            sm.is_financial
        FROM fundlab.fund_portfolio fp
        JOIN fundlab.fund fm
          ON fp.fund_id = fm.fund_id
        JOIN fundlab.stock_master sm
          ON fp.isin = sm.isin
        WHERE fp.fund_id = ANY(:fund_ids)
          AND fp.month_end BETWEEN :start_date AND :end_date
          AND EXTRACT(MONTH FROM fp.month_end) IN (3, 9)
        ORDER BY fp.fund_id, fp.month_end, fp.isin
    """)

    df = pd.read_sql(query, engine, params={
        "fund_ids": fund_ids,
        "start_date": start_date,
        "end_date": end_date
    })
    if df.empty:
        return df

    df["month_end"] = pd.to_datetime(df["month_end"]).dt.date
    return df


def compute_portfolio_fundamentals(df_portfolio, roe_roce_dict, segment_choice):
    """
    df_portfolio: columns [fund_id, fund_name, month_end, isin, weight_pct, asset_type, is_financial]
    roe_roce_dict: dict[isin] -> df of roe/roce history
    segment_choice: "Financials" | "Non-financials" | "Total"
    """

    if df_portfolio.empty:
        return pd.DataFrame(columns=["fund_id", "fund_name", "month_end", "metric"])

    # Only Domestic Equities
    df = df_portfolio[df_portfolio["asset_type"] == "Domestic Equities"].copy()
    if df.empty:
        return pd.DataFrame(columns=["fund_id", "fund_name", "month_end", "metric"])

    # Compute domestic-equity-rebased weights (sum to 1 per fund, date)
    df["weight_pct"] = df["weight_pct"].astype(float)
    group_keys = ["fund_id", "fund_name", "month_end"]

    domestic_totals = df.groupby(group_keys)["weight_pct"].transform("sum")
    df["dom_weight"] = np.where(domestic_totals > 0, df["weight_pct"] / domestic_totals, 0.0)

    # Attach the 5-point median metric per stock
    metrics = []
    for idx, row in df.iterrows():
        m = get_median_metric_for_stock(
            roe_roce_dict,
            row["isin"],
            row["month_end"],
            bool(row["is_financial"])
        )
        metrics.append(m)
    df["stock_metric"] = metrics   # this is RoE for financials, RoCE for non-financials

    # Now compute portfolio metric for each segment choice
    records = []
    for (fund_id, fund_name, month_end), sub in df.groupby(group_keys):
        if segment_choice == "Financials":
            sub_seg = sub[sub["is_financial"] == True].copy()
            if sub_seg.empty:
                metric = np.nan
            else:
                seg_total = sub_seg["dom_weight"].sum()
                sub_seg["seg_weight"] = np.where(seg_total > 0, sub_seg["dom_weight"] / seg_total, 0.0)
                sub_seg["metric_contrib"] = sub_seg["seg_weight"] * sub_seg["stock_metric"].fillna(0.0)
                metric = sub_seg["metric_contrib"].sum()

        elif segment_choice == "Non-financials":
            sub_seg = sub[sub["is_financial"] == False].copy()
            if sub_seg.empty:
                metric = np.nan
            else:
                seg_total = sub_seg["dom_weight"].sum()
                sub_seg["seg_weight"] = np.where(seg_total > 0, sub_seg["dom_weight"] / seg_total, 0.0)
                sub_seg["metric_contrib"] = sub_seg["seg_weight"] * sub_seg["stock_metric"].fillna(0.0)
                metric = sub_seg["metric_contrib"].sum()

        else:  # "Total"
            sub_seg = sub.copy()
            sub_seg["metric_contrib"] = sub_seg["dom_weight"] * sub_seg["stock_metric"].fillna(0.0)
            metric = sub_seg["metric_contrib"].sum()

        records.append({
            "fund_id": fund_id,
            "fund_name": fund_name,
            "month_end": month_end,
            "metric": metric
        })

    result = pd.DataFrame.from_records(records)
    result = result.sort_values(["fund_name", "month_end"])
    return result

# --- CACHED HEAVY HELPERS FOR QUALITY PAGE Start ---

@st.cache_data(ttl=1800, show_spinner=False)
def get_portfolio_fundamentals_cached(
    fund_ids: list[int],
    start_date: date,
    end_date: date,
    segment_choice: str,
) -> pd.DataFrame:
    """
    Wrapper around load_stock_roe_roce + fetch_portfolio_raw + compute_portfolio_fundamentals,
    cached for 30 minutes.
    """
    if not fund_ids:
        return pd.DataFrame(columns=["fund_id", "fund_name", "month_end", "metric"])

    roe_roce_dict = load_stock_roe_roce()
    df_portfolio = fetch_portfolio_raw(fund_ids, start_date, end_date)
    if df_portfolio.empty:
        return pd.DataFrame(columns=["fund_id", "fund_name", "month_end", "metric"])

    df_result = compute_portfolio_fundamentals(
        df_portfolio,
        roe_roce_dict,
        segment_choice,
    )
    return df_result


@st.cache_data(ttl=1800, show_spinner=False)
def compute_quality_bucket_exposure_cached(
    fund_id: int,
    month_ends: list[date],
) -> pd.DataFrame:
    """
    Cached wrapper around compute_quality_bucket_exposure.
    """
    return compute_quality_bucket_exposure(fund_id, month_ends)

# --- CACHED HEAVY HELPERS FOR QUALITY PAGE end ---

# --- SHARED SELECTORS FOR QUALITY PAGE  Start---

def quality_category_and_fund_selector():
    """
    1) Category selector (checkboxes)
    2) Fund multi-select (from selected categories)
    3) Focus fund single-select (from selected funds)

    Returns:
        selected_fund_ids: list[int]
        selected_fund_labels: list[str]
        focus_fund_id: int
        focus_fund_label: str
        fund_options: dict[label -> fund_id]
    """
    categories = fetch_categories()
    if not categories:
        st.warning("No categories found in fund_master.")
        return [], [], None, None, {}

    st.subheader("1. Select categories")
    selected_categories = []
    cols = st.columns(min(4, len(categories)))
    for i, cat in enumerate(categories):
        col = cols[i % len(cols)]
        if col.checkbox(cat, value=False, key=f"pq_cat_{cat}"):
            selected_categories.append(cat)

    if not selected_categories:
        st.info("Please select at least one category.")
        return [], [], None, None, {}

    # 2) Fund multi-select
    st.subheader("2. Select funds (universe)")
    df_funds = fetch_funds_for_categories(selected_categories)
    if df_funds.empty:
        st.warning("No funds found for the selected categories.")
        return [], [], None, None, {}

    fund_options = {
        f"{row['fund_name']} ({row['category_name']})": row["fund_id"]
        for _, row in df_funds.iterrows()
    }
    fund_labels = sorted(fund_options.keys())

    all_option = "All funds in selected categories"
    raw_labels = [all_option] + fund_labels

    selected_raw_labels = st.multiselect(
        "Select funds for analysis",
        options=raw_labels,
        default=[all_option],
        key="pq_funds_multiselect",
    )

    if all_option in selected_raw_labels:
        selected_fund_labels = fund_labels
    else:
        selected_fund_labels = [lbl for lbl in selected_raw_labels if lbl != all_option]

    if not selected_fund_labels:
        st.info("Please select at least one fund.")
        return [], [], None, None, {}

    selected_fund_ids = [fund_options[label] for label in selected_fund_labels]

    # 3) Focus fund
    st.subheader("3. Focus fund")
    focus_fund_label = st.selectbox(
        "Focus fund",
        options=selected_fund_labels,
        index=0,
        key="pq_focus_fund",
    )
    focus_fund_id = fund_options[focus_fund_label]

    return selected_fund_ids, selected_fund_labels, focus_fund_id, focus_fund_label, fund_options

def quality_period_selector():
    """
    Period selector (March / September portfolios).
    Returns:
        start_date: date
        end_date: date
    """
    st.subheader("4. Select period (March / September portfolios only)")

    current_year = dt.date.today().year
    years = list(range(current_year - 15, current_year + 1))
    month_options = [3, 9]  # Mar, Sep

    col1, col2 = st.columns(2)
    with col1:
        start_year = st.selectbox(
            "Start year",
            options=years,
            index=0,
            key="pq_start_year",
        )
        start_month = st.selectbox(
            "Start month",
            options=month_options,
            index=0,
            key="pq_start_month",
            format_func=lambda m: "Mar" if m == 3 else "Sep",
        )
    with col2:
        end_year = st.selectbox(
            "End year",
            options=years,
            index=len(years) - 1,
            key="pq_end_year",
        )
        end_month = st.selectbox(
            "End month",
            options=month_options,
            index=1,
            key="pq_end_month",
            format_func=lambda m: "Mar" if m == 3 else "Sep",
        )

    start_date = month_year_to_last_day(start_year, start_month)
    end_date = month_year_to_last_day(end_year, end_month)

    if start_date > end_date:
        st.error("Start date must be earlier than end date.")
        st.stop()

    return start_date, end_date


# --- SHARED SELECTORS FOR QUALITY PAGE  End---


# --- Split quality page into SECTION 1: RoE/RoCE-like return on capital vs peers / universe and 2. Q1–Q4 exposures --- START ---

def render_quality_roc_section(
    selected_fund_ids: list[int],
    selected_fund_labels: list[str],
    focus_fund_id: int,
    focus_fund_label: str,
    start_date: date,
    end_date: date,
    segment_choice: str,
    comparison_mode: str,  # "Universe median" or "Individual funds"
):
    with st.spinner("Computing portfolio fundamentals..."):
        df_result = get_portfolio_fundamentals_cached(
            selected_fund_ids,
            start_date,
            end_date,
            segment_choice,
        )

    if df_result.empty:
        st.warning("No fundamentals could be computed (check data availability).")
        return

    df_result["month_end"] = pd.to_datetime(df_result["month_end"])

    focus_df = df_result[df_result["fund_id"] == focus_fund_id].copy()
    if focus_df.empty:
        st.warning("No data for the focus fund in the selected period.")
        return

    other_ids = [fid for fid in selected_fund_ids if fid != focus_fund_id]

    if comparison_mode == "Universe median" and other_ids:
        others = df_result[df_result["fund_id"].isin(other_ids)].copy()
        if others.empty:
            st.warning("No data for peer funds; falling back to focus fund only.")
            df_chart = focus_df.copy()
        else:
            median_others = (
                others.groupby("month_end")["metric"]
                .median()
                .reset_index()
                .rename(columns={"metric": "metric_median"})
            )
            median_others["fund_name"] = "Universe median (others)"
            median_others["fund_id"] = -1
            median_others = median_others.rename(columns={"metric_median": "metric"})
            df_chart = pd.concat([focus_df, median_others], ignore_index=True)
    else:
        df_chart = df_result[df_result["fund_id"].isin(selected_fund_ids)].copy()

    df_chart["month_end"] = pd.to_datetime(df_chart["month_end"])

    st.subheader("Return on capital (5-period median RoE / RoCE)")
    chart = (
        alt.Chart(df_chart)
        .mark_line(point=True)
        .encode(
            x=alt.X("month_end:T", title="Period", axis=alt.Axis(format="%b %Y")),
            y=alt.Y("metric:Q", title="Portfolio metric (%)"),
            color=alt.Color("fund_name:N", title="Fund"),
            tooltip=[
                alt.Tooltip("fund_name:N", title="Fund"),
                alt.Tooltip("month_end:T", title="Period", format="%b %Y"),
                alt.Tooltip("metric:Q", title="Metric", format=".1f"),
            ],
        )
        .properties(height=400)
    )
    st.altair_chart(chart, use_container_width=True)

    df_pivot = df_chart.pivot_table(
        index="fund_name",
        columns=df_chart["month_end"].dt.strftime("%b %Y"),
        values="metric",
        aggfunc="first",
    )
    st.dataframe(df_pivot.style.format("{:.1f}"), use_container_width=True)


# --- SECTION 2: Quality quartile exposures (Q1–Q4) ---

def render_quality_quartiles_section(
    selected_fund_ids: list[int],
    selected_fund_labels: list[str],
    focus_fund_id: int,
    focus_fund_label: str,
    start_date: date,
    end_date: date,
    segment_choice: str,          # currently unused, kept for signature compatibility
    fund_options: dict[str, int],
):
    st.subheader("7. Quality quartile exposures (Q1–Q4)")

    if not selected_fund_ids:
        st.info("No funds selected.")
        return

    # Use the TOP-LEVEL focus fund directly – no extra selector here
    quality_fund_id = focus_fund_id
    quality_fund_label = focus_fund_label

    # Reuse portfolio fundamentals to get consistent month_ends universe
    with st.spinner(
        f"Preparing periods for quartile exposures for {quality_fund_label}..."
    ):
        df_result = get_portfolio_fundamentals_cached(
            selected_fund_ids,
            start_date,
            end_date,
            segment_choice,   # quartiles ignore it, but fundamentals don’t mind
        )

    if df_result.empty:
        st.info("No data available to compute Q1–Q4 quality bucket exposures.")
        return

    df_result["month_end"] = pd.to_datetime(df_result["month_end"])
    all_month_ends = sorted(df_result["month_end"].dt.date.unique())
    if not all_month_ends:
        st.info("No periods found to compute quality buckets.")
        return

    # Filter month_ends where THIS focus fund actually has data
    fund_months = sorted(
        df_result[df_result["fund_id"] == quality_fund_id]["month_end"].dt.date.unique()
    )
    if not fund_months:
        st.info(
            f"No periods found for {quality_fund_label} to compute quality buckets."
        )
        return

    with st.spinner(
        f"Computing quality bucket exposures for {quality_fund_label}..."
    ):
        quality_table = compute_quality_bucket_exposure_cached(
            quality_fund_id,
            fund_months,
        )

    if quality_table.empty:
        st.info("No quality bucket data available.")
        return

    # Title with fund name
    st.markdown(f"**Fund:** {quality_fund_label}")

    # Chart: stack Q1–Q4 over time (ignore 'Total' row for chart)
    table_for_chart = quality_table.drop(index="Total", errors="ignore")
    chart_df = table_for_chart.T.reset_index().rename(columns={"index": "month_end"})
    chart_df["month_end"] = pd.to_datetime(chart_df["month_end"])

    chart_long = chart_df.melt(
        "month_end",
        var_name="Quartile",
        value_name="Exposure",
    )

    chart_q = (
        alt.Chart(chart_long)
        .mark_area()
        .encode(
            x=alt.X(
                "month_end:T",
                title="Period",
                axis=alt.Axis(format="%b %Y"),
            ),
            y=alt.Y(
                "Exposure:Q",
                title="Exposure (% of domestic equities)",
            ),
            color=alt.Color("Quartile:N", title="Quartile"),
            tooltip=[
                alt.Tooltip("month_end:T", title="Period", format="%b %Y"),
                alt.Tooltip("Quartile:N", title="Quartile"),
                alt.Tooltip("Exposure:Q", title="Exposure (%)", format=".1f"),
            ],
        )
        .properties(height=400)
    )

    st.altair_chart(chart_q, use_container_width=True)

    # Raw table (including 'Total' row)
    st.dataframe(
        quality_table.style.format("{:.1f}"),
        use_container_width=True,
    )


# --- Split quality page into SECTION 1: RoE/RoCE-like return on capital vs peers / universe and 2. Q1–Q4 exposures --- END ---


def fetch_fund_portfolio_timeseries(fund_id, start_date, end_date, freq):
    """
    Fetch portfolio holdings for a single fund between start_date and end_date.

    freq: "Monthly", "Quarterly", "Yearly"
    """
    engine = get_engine()
    query = text("""
        SELECT
            fp.month_end,
            fp.isin,
            fp.holding_weight AS weight_pct,
            sm.company_name
        FROM fundlab.fund_portfolio fp
        JOIN fundlab.stock_master sm
          ON fp.isin = sm.isin
        WHERE fp.fund_id = :fund_id
          AND fp.month_end BETWEEN :start_date AND :end_date
        ORDER BY fp.month_end, sm.company_name;
    """)

    df = pd.read_sql(
        query,
        engine,
        params={
            "fund_id": fund_id,
            "start_date": start_date,
            "end_date": end_date,
        },
    )

    if df.empty:
        return df

    df["month_end"] = pd.to_datetime(df["month_end"]).dt.date

    # Apply frequency filter
    if freq == "Quarterly":
        df = df[df["month_end"].apply(lambda d: d.month in (3, 6, 9, 12))]
    elif freq == "Yearly":
        # Use the same month as end_date for yearly snapshots (e.g. every Mar or every Sep)
        end_month = end_date.month
        df = df[df["month_end"].apply(lambda d: d.month == end_month)]

    return df

# helper for active share calculations
def fetch_multi_fund_portfolios(fund_ids, start_date, end_date, freq):
    """
    Fetch portfolio holdings for multiple funds between start_date and end_date.

    Returns: DataFrame with columns:
      fund_id, month_end, isin, weight_pct
    freq: "Monthly", "Quarterly", "Yearly"
    """
    if not fund_ids:
        return pd.DataFrame()

    engine = get_engine()
    query = text("""
        SELECT
            fp.fund_id,
            fp.month_end,
            fp.isin,
            fp.holding_weight AS weight_pct
        FROM fundlab.fund_portfolio fp
        WHERE fp.fund_id = ANY(:fund_ids)
          AND fp.month_end BETWEEN :start_date AND :end_date
        ORDER BY fp.fund_id, fp.month_end, fp.isin;
    """)

    df = pd.read_sql(
        query,
        engine,
        params={
            "fund_ids": fund_ids,
            "start_date": start_date,
            "end_date": end_date,
        },
    )

    if df.empty:
        return df

    df["month_end"] = pd.to_datetime(df["month_end"]).dt.date

    # Apply frequency filter
    if freq == "Quarterly":
        df = df[df["month_end"].apply(lambda d: d.month in (3, 6, 9, 12))]
    elif freq == "Yearly":
        # Use the same month as end_date for yearly snapshots (e.g. every Mar or every Sep)
        end_month = end_date.month
        df = df[df["month_end"].apply(lambda d: d.month == end_month)]

    return df


# Build composite portfolio to compute active share
def build_composite_portfolio(df_all, fund_prop_dict, date):
    """
    Build a composite portfolio for a given date.

    df_all: DataFrame with columns [fund_id, month_end, isin, weight_pct]
    fund_prop_dict: {fund_id: normalized weight (sum to 1.0)}
    date: datetime.date

    Returns: Series indexed by isin with weights summing to 1.0, or None if no data.
    """
    # Filter to selected funds and the given date
    fund_ids = list(fund_prop_dict.keys())
    sub = df_all[(df_all["fund_id"].isin(fund_ids)) & (df_all["month_end"] == date)].copy()

    if sub.empty:
        return None

    sub["fund_prop"] = sub["fund_id"].map(fund_prop_dict).astype(float)
    # raw allocation: fund weight * fund_prop
    sub["alloc"] = sub["weight_pct"].astype(float) * sub["fund_prop"]

    grouped = sub.groupby("isin")["alloc"].sum()
    total = grouped.sum()
    if total == 0:
        return None

    weights = grouped / total
    return weights

# Compute active share time series between two composite portfolios
def compute_active_share_series(df_all, fund_props_A, fund_props_B):
    """
    df_all: DataFrame with [fund_id, month_end, isin, weight_pct]
    fund_props_A/B: {fund_id: normalized weight (sum to 1.0)}

    Returns: DataFrame with columns:
      period_date, period_label, active_share_pct
    """
    if df_all.empty:
        return pd.DataFrame(columns=["period_date", "period_label", "active_share_pct"])

    dates = sorted(df_all["month_end"].unique())

    records = []
    for dt_ in dates:
        wA = build_composite_portfolio(df_all, fund_props_A, dt_)
        wB = build_composite_portfolio(df_all, fund_props_B, dt_)

        if wA is None or wB is None:
            active_share_pct = np.nan
        else:
            # union of ISINs
            union_isins = wA.index.union(wB.index)
            overlap = 0.0
            for isin in union_isins:
                wa = float(wA.get(isin, 0.0))
                wb = float(wB.get(isin, 0.0))
                overlap += min(wa, wb)

            active_share = 1.0 - overlap  # weights are in fractions (0–1)
            active_share_pct = active_share * 100.0

        records.append(
            {
                "period_date": pd.to_datetime(dt_),
                "period_label": pd.to_datetime(dt_).strftime("%b %Y"),
                "active_share_pct": active_share_pct,
            }
        )

    df = pd.DataFrame.from_records(records)
    df = df.sort_values("period_date")
    return df
 


@st.cache_data(show_spinner="Loading precomputed rolling returns (funds)...")
def load_fund_rolling(window_months: int, fund_names, start=None, end=None):
    """
    Returns a DataFrame with columns: ['asof_date', 'fund_name', 'rolling_cagr']
    for the given window and list of fund names.
    """
    if not fund_names:
        return pd.DataFrame([])

    # Ensure list
    fund_names = list(fund_names)

    date_filter = ""
    params = {"window": window_months, "fund_names": tuple(fund_names)}
    if start is not None:
        date_filter += " AND r.asof_date >= :start"
        params["start"] = start
    if end is not None:
        date_filter += " AND r.asof_date <= :end"
        params["end"] = end

    query = f"""
        SELECT
            f.fund_name,
            r.asof_date,
            r.rolling_cagr
        FROM fundlab.fund_rolling_return r
        JOIN fundlab.fund f
          ON f.fund_id = r.fund_id
        WHERE r.window_months = :window
          AND f.fund_name IN :fund_names
          {date_filter}
        ORDER BY r.asof_date, f.fund_name;
    """

    with engine.begin() as conn:
        df = pd.read_sql(text(query), conn, params=params, parse_dates=["asof_date"])
    return df


@st.cache_data(show_spinner="Loading precomputed rolling returns (benchmark)...")
def load_bench_rolling(window_months: int, bench_name: str, start=None, end=None):
    if bench_name is None:
        return pd.DataFrame([])

    date_filter = ""
    params = {"window": window_months, "bench_name": bench_name}
    if start is not None:
        date_filter += " AND r.asof_date >= :start"
        params["start"] = start
    if end is not None:
        date_filter += " AND r.asof_date <= :end"
        params["end"] = end

    query = f"""
        SELECT
            b.bench_name,
            r.asof_date,
            r.rolling_cagr
        FROM fundlab.bench_rolling_return r
        JOIN fundlab.benchmark b
          ON b.bench_id = r.bench_id
        WHERE r.window_months = :window
          AND b.bench_name = :bench_name
          {date_filter}
        ORDER BY r.asof_date;
    """

    with engine.begin() as conn:
        df = pd.read_sql(text(query), conn, params=params, parse_dates=["asof_date"])
    return df


@st.cache_data(ttl=3600, show_spinner="Loading fund NAVs from database...")
def load_funds_from_db():
    query = """
        SELECT
            f.fund_name      AS "Fund name",
            n.nav_date::date AS "month-end",
            n.nav_value::float AS "NAV",
            c.category_name  AS "market-cap",
            s.style_name     AS "style"
        FROM fundlab.fund_nav n
        JOIN fundlab.fund f
          ON f.fund_id = n.fund_id
        LEFT JOIN fundlab.category c
          ON c.category_id = f.category_id
        LEFT JOIN fundlab.style s
          ON s.style_id = f.style_id
        ORDER BY "Fund name", "month-end";
    """
    with engine.begin() as conn:
        df = pd.read_sql(query, conn, parse_dates=["month-end"])
    return df


@st.cache_data(show_spinner="Loading benchmark NAVs from database...")
def load_bench_from_db():
    query = """
        SELECT
            b.bench_name     AS "benchmark_name",
            n.nav_date::date AS "month-end",
            n.nav_value::float AS "NAV",
            c.category_name  AS "category_type",
            s.style_name     AS "category_value"
        FROM fundlab.bench_nav n
        JOIN fundlab.benchmark b
          ON b.bench_id = n.bench_id
        LEFT JOIN fundlab.category c
          ON c.category_id = b.category_id
        LEFT JOIN fundlab.style s
          ON s.style_id = b.style_id
        ORDER BY "benchmark_name", "month-end";
    """
    with engine.begin() as conn:
        df = pd.read_sql(query, conn, parse_dates=["month-end"])
    return df

# Cache stock master ISINs for validation when uploading market cap and price data
@st.cache_data
def load_stock_master_isins():
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql("SELECT isin FROM fundlab.stock_master", conn)
    return set(df["isin"].astype(str))



def to_eom(series):
    """Coerce any datetime-like series to month-end timestamps."""
    if series is None:
        return pd.Series(dtype="datetime64[ns]")
    s = pd.to_datetime(series, errors="coerce")
    return s.dt.to_period("M").dt.to_timestamp("M")

def coerce_num(series):
    """Coerce to numeric (NAV columns), invalid -> NaN."""
    return pd.to_numeric(series, errors="coerce")


# === Helper functions for FY/CY bounds with custom month-year domain ===
def _fy_year_for(date_eom: pd.Timestamp) -> int:
    return date_eom.year + 1 if date_eom.month >= 4 else date_eom.year

def _fy_bounds(y_label: int):
    start = pd.Timestamp(year=y_label-1, month=4, day=1).to_period('M').to_timestamp('M')
    end   = pd.Timestamp(year=y_label,   month=3, day=1).to_period('M').to_timestamp('M')
    return start, end

def _cy_bounds(y_label: int):
    start = pd.Timestamp(year=y_label, month=1, day=1).to_period('M').to_timestamp('M')
    end   = pd.Timestamp(year=y_label, month=12, day=1).to_period('M').to_timestamp('M')
    return start, end

def yearly_returns_with_custom_domain(
    nav_series: pd.Series,
    start_domain: pd.Timestamp,
    end_domain: pd.Timestamp,
    fy: bool = True
) -> pd.Series:
    """
    Yearly returns anchored to FY (Apr–Mar) or CY (Jan–Dec),
    trimmed to [start_domain, end_domain].
    IMPORTANT: Uses the *previous month-end* of the start as the baseline.
    Example: Full FY2020 (Apr 2019–Mar 2020) => ret = NAV[Mar 2020] / NAV[Mar 2019] - 1
    Partial: start=Jan 2015 => baseline = Dec 2014.
    """
    # Normalize index strictly to EOM and deduplicate by last value
    s = nav_series.dropna().sort_index()
    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp("M")
    s = s.groupby(level=0).last()

    # Normalize domain to EOM
    start_domain = pd.Timestamp(start_domain).to_period("M").to_timestamp("M")
    end_domain   = pd.Timestamp(end_domain).to_period("M").to_timestamp("M")

    if s.empty or end_domain <= start_domain:
        return pd.Series(dtype=float)

    # Helpers
    def fy_year_for(eom: pd.Timestamp) -> int:
        return eom.year + 1 if eom.month >= 4 else eom.year

    def fy_bounds(y_label: int):
        a = pd.Timestamp(year=y_label-1, month=4, day=1).to_period("M").to_timestamp("M")
        b = pd.Timestamp(year=y_label,   month=3, day=1).to_period("M").to_timestamp("M")
        return a, b

    def cy_bounds(y_label: int):
        a = pd.Timestamp(year=y_label, month=1,  day=1).to_period("M").to_timestamp("M")
        b = pd.Timestamp(year=y_label, month=12, day=1).to_period("M").to_timestamp("M")
        return a, b

    # Label range intersecting the domain
    if fy:
        first_label = fy_year_for(start_domain)
        last_label  = fy_year_for(end_domain)
    else:
        first_label = start_domain.year
        last_label  = end_domain.year

    labels = range(first_label, last_label + 1)
    out_vals, out_idx = [], []

    for y in labels:
        yr_start, yr_end = fy_bounds(y) if fy else cy_bounds(y)

        # Trim to domain
        a = max(yr_start, start_domain)     # first included month-end
        b = min(yr_end,   end_domain)       # last included month-end
        if b <= a:
            continue

        # Baseline is the previous month-end of 'a'
        baseline = (a.to_period("M") - 1).to_timestamp("M")

        # Need both baseline and b to exist
        if baseline not in s.index or b not in s.index:
            out_vals.append(np.nan)
        else:
            try:
                ret = s.loc[b] / s.loc[baseline] - 1.0
            except Exception:
                ret = np.nan
            out_vals.append(ret)

        # Label (show the trimmed range only if partial)
        full = (a == yr_start) and (b == yr_end)
        if fy:
            lbl = f"FY{y}" if full else f"FY{y} ({a:%b %Y}–{b:%b %Y})"
        else:
            lbl = f"{y}" if full else f"{y} ({a:%b %Y}–{b:%b %Y})"
        out_idx.append(lbl)

    return pd.Series(out_vals, index=out_idx, dtype=float)




# ------------------------ Parsing & Cleaning ------------------------
def parse_month_end_cell(x):
    """Parse many date formats; coerce to END-OF-MONTH Timestamp. Return Timestamp or None."""
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None

    # Try YYYY-MM-DD (/)
    if re.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$", s):
        dt = pd.to_datetime(s, errors="coerce", dayfirst=False)
        return dt.to_period("M").to_timestamp("M") if pd.notna(dt) else None

    # Try DD-MM-YYYY (/)
    if re.match(r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$", s):
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
        return dt.to_period("M").to_timestamp("M") if pd.notna(dt) else None

    # Heuristics
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True, infer_datetime_format=True)
    if pd.isna(dt):
        dt = pd.to_datetime(s, errors="coerce", dayfirst=False, infer_datetime_format=True)
    if pd.notna(dt):
        return dt.to_period("M").to_timestamp("M")
    return None


def smart_to_month_end(series: pd.Series) -> pd.Series:
    return series.astype(str).map(parse_month_end_cell)


def clean_nav_series(s: pd.Series) -> pd.Series:
    """Strip NBSP/spaces/commas & convert bracketed negatives, then numeric."""
    s = (
        s.astype(str)
         .str.replace("\u00A0", "", regex=False)
         .str.replace(" ", "", regex=False)
         .str.replace(",", "", regex=False)
         .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    )
    return pd.to_numeric(s, errors="coerce")


@st.cache_data
def _clean_funds(df_raw: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "Fund name":"fund","fund name":"fund","fund":"fund",
        "month-end":"date","month_end":"date","date":"date",
        "NAV":"nav","nav":"nav",
        "market-cap":"market_cap","market cap":"market_cap","market_cap":"market_cap",
        "style":"style"
    }
    fixed = {}
    for k,v in rename.items():
        for c in df_raw.columns:
            if c.lower() == k.lower():
                fixed[c] = v
    df = df_raw.rename(columns=fixed)
    needed = {"fund","date","nav","market_cap","style"}
    if not needed.issubset(df.columns):
        raise ValueError("Funds CSV must contain: Fund name, month-end, NAV, market-cap, style")

    df["date"] = smart_to_month_end(df["date"])
    df["nav"] = clean_nav_series(df["nav"])
    df = df.dropna(subset=["fund","date","nav"]).copy()
    df["market_cap"] = df["market_cap"].astype(str)
    df["style"] = df["style"].astype(str)

    df = (
        df.sort_values(["fund","date"])
          .groupby(["fund","date","market_cap","style"], as_index=False)
          .last()
          .sort_values(["fund","date"])
          .reset_index(drop=True)
    )
    return df


@st.cache_data
def _clean_bench(df_raw: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "benchmark_name":"benchmark_name","benchmark":"benchmark_name","name":"benchmark_name",
        "month-end":"date","month_end":"date","date":"date",
        "NAV":"nav","nav":"nav",
        "category_type":"category_type","type":"category_type",
        "category_value":"category_value","value":"category_value"
    }
    fixed = {}
    for k,v in rename.items():
        for c in df_raw.columns:
            if c.lower() == k.lower():
                fixed[c] = v
    df = df_raw.rename(columns=fixed)
    needed = {"benchmark_name","date","nav","category_type","category_value"}
    if not needed.issubset(df.columns):
        raise ValueError("Benchmarks CSV must contain: benchmark_name, month-end, NAV, category_type, category_value")

    df["date"] = smart_to_month_end(df["date"])
    df["nav"] = clean_nav_series(df["nav"])
    df = df.dropna(subset=["benchmark_name","date","nav"]).copy()
    df["category_type"] = df["category_type"].astype(str)
    df["category_value"] = df["category_value"].astype(str)

    df = (
        df.sort_values(["benchmark_name","date"])
          .groupby(["benchmark_name","date","category_type","category_value"], as_index=False)
          .last()
          .sort_values(["benchmark_name","date"])
          .reset_index(drop=True)
    )
    return df


# ------------------------ Metrics ------------------------
def trailing_cagr(series: pd.Series, months: int) -> pd.Series:
    """Trailing CAGR at each END date. Returns decimals (not %)."""
    s = series.sort_index()
    if len(s) < months + 1:
        return pd.Series(index=s.index, dtype=float)
    shifted = s.shift(months)
    with np.errstate(divide="ignore", invalid="ignore"):
        cagr = (s / shifted) ** (12.0 / months) - 1.0
    return cagr.replace([np.inf, -np.inf], np.nan)


def window_label_series(end_idx: pd.Index, months: int) -> pd.Series:
    """Labels 'Jun 2018–2021' from END index and window length."""
    start = (pd.to_datetime(end_idx) - pd.DateOffset(months=months)).to_period("M").to_timestamp("M")
    end = pd.to_datetime(end_idx)
    return pd.Series([f"{s:%b %Y}–{e:%Y}" for s,e in zip(start, end)], index=end_idx)


def combine_peer_rolling(funds_df, selected_funds, months, exclude=None):
    """
    Build a 'peer average' rolling series for the selected funds, excluding the focus fund.

    For each fund:
      - take its monthly NAV series
      - compute the N-month rolling CAGR: (NAV_t / NAV_{t-N})**(12/N) - 1
    Then:
      - at each date, peer average = simple arithmetic mean of those CAGRs
    """

    # Detect column names in the cleaned dataframe
    fund_candidates = ["Fund", "Fund name", "fund", "fund_name"]
    date_candidates = ["date", "Date", "month-end", "nav_date"]
    nav_candidates = ["nav", "NAV", "nav_value"]

    def pick(col_list):
        for c in col_list:
            if c in funds_df.columns:
                return c
        raise KeyError(f"None of {col_list} found in funds_df columns: {list(funds_df.columns)}")

    fund_col = pick(fund_candidates)
    date_col = pick(date_candidates)
    nav_col = pick(nav_candidates)

    # Work on a trimmed, clean copy
    df = funds_df[[fund_col, date_col, nav_col]].dropna(subset=[fund_col, date_col, nav_col]).copy()
    df[date_col] = pd.to_datetime(df[date_col])

    res = {}
    for f in selected_funds:
        if f == exclude:
            continue

        # One NAV time series per peer fund
        s = (
            df.loc[df[fund_col] == f, [date_col, nav_col]]
              .drop_duplicates(subset=[date_col])
              .set_index(date_col)[nav_col]
              .sort_index()
        )

        if s.empty:
            continue

        # ✅ Correct rolling CAGR: no extra rolling, no double compounding
        r = (s / s.shift(months)) ** (12.0 / months) - 1.0

        res[f] = r

    if not res:
        return None

    # Simple arithmetic average across peer funds at each date
    peer = pd.concat(res, axis=1).mean(axis=1)
    peer.name = "Peer avg"
    return peer




def pick_benchmark(bench_df, mode, sel_caps, sel_styles):
    """Pick benchmark: Portfolio(Nifty) > style-aligned > mcap-aligned > any portfolio."""
    if bench_df is None or bench_df.empty:
        return None, None
    b = bench_df.copy()
    if mode == "Portfolio (Nifty)":
        cand = b[b["category_type"].str.lower()=="portfolio"]
        if not cand.empty:
            if (cand["category_value"].str.lower()=="nifty").any():
                cand = cand[cand["category_value"].str.lower()=="nifty"]
            nm = cand["benchmark_name"].iloc[0]
            ser = cand[cand["benchmark_name"]==nm].set_index("date")["nav"].sort_index()
            return nm, ser
        return None, None
    for sty in sel_styles:
        cand = b[(b["category_type"].str.lower()=="style") & (b["category_value"].str.lower()==str(sty).lower())]
        if not cand.empty:
            nm = cand["benchmark_name"].iloc[0]
            ser = cand[cand["benchmark_name"]==nm].set_index("date")["nav"].sort_index()
            return nm, ser
    for cap in sel_caps:
        cand = b[(b["category_type"].str.lower()=="market-cap") & (b["category_value"].str.lower()==str(cap).lower())]
        if not cand.empty:
            nm = cand["benchmark_name"].iloc[0]
            ser = cand[cand["benchmark_name"]==nm].set_index("date")["nav"].sort_index()
            return nm, ser
    cand = b[b["category_type"].str.lower()=="portfolio"]
    if not cand.empty:
        nm = cand["benchmark_name"].iloc[0]
        ser = cand[cand["benchmark_name"]==nm].set_index("date")["nav"].sort_index()
        return nm, ser
    return None, None


# ------ Yearly returns (strict) ------
def yearly_returns_strict(nav_series: pd.Series, fy=True) -> pd.Series:
    """Strict endpoints: FY uses March; CY uses December."""
    s = nav_series.dropna().sort_index()
    if s.empty:
        return pd.Series(dtype=float)
    if fy:
        march = s[s.index.month == 3]
        if march.empty:
            return pd.Series(dtype=float)
        fy_nav = march.copy()
        fy_nav.index = [f"FY{dt.year}" for dt in fy_nav.index]
        return fy_nav.pct_change()
    else:
        dec = s[s.index.month == 12]
        if dec.empty:
            return pd.Series(dtype=float)
        cy_nav = dec.copy()
        cy_nav.index = [f"{dt.year}" for dt in cy_nav.index]
        return cy_nav.pct_change()


def build_actual_returns_table(funds_df: pd.DataFrame, fund_list: list, bench_ser: pd.Series | None, use_fy=True):
    """Returns (funds_df: rows=funds, cols=years), (bench_row: Series of years)."""
    rows = {}
    for f in fund_list:
        s = funds_df.loc[funds_df["Fund name"] == f, ["date","nav"]].drop_duplicates("date").set_index("date")["nav"]
        rows[f] = yearly_returns_strict(s, fy=use_fy)
    df_f = pd.DataFrame(rows).T
    bench_row = None
    if bench_ser is not None and not bench_ser.empty:
        bench_row = yearly_returns_strict(bench_ser, fy=use_fy).rename("Benchmark")
    return df_f, bench_row


def _sort_year_labels(labels: list, fy=True) -> list:
    key = (lambda lbl: int(str(lbl).replace("FY",""))) if fy else (lambda lbl: int(lbl))
    return sorted(labels, key=key)


def filter_year_range(actual_df: pd.DataFrame, bench_row: pd.Series | None, start_label: str, end_label: str, fy=True):
    """Slice safely by the UNION of labels across funds + benchmark to avoid KeyErrors."""
    uni = set(actual_df.columns.tolist())
    if bench_row is not None and not bench_row.empty:
        uni |= set(bench_row.index.tolist())
    all_labels = _sort_year_labels(list(uni), fy=fy)
    if not all_labels:
        return actual_df, bench_row, []

    if start_label not in all_labels:
        start_label = all_labels[0]
    if end_label not in all_labels:
        end_label = all_labels[-1]

    si, ei = all_labels.index(start_label), all_labels.index(end_label)
    sel = all_labels[min(si,ei):max(si,ei)+1]
    df2 = actual_df.reindex(columns=sel)
    br = bench_row[sel] if bench_row is not None and not bench_row.empty else bench_row
    return df2, br, sel


# ------------------------ Rolling helpers ------------------------
def apply_window_mask(r: pd.Series, months: int, start_domain, end_domain) -> pd.Series:
    """Keep windows where (END - months) >= start_domain AND END <= end_domain."""
    if r.empty:
        return r
    end_idx = r.index
    start_idx = (pd.to_datetime(end_idx) - pd.DateOffset(months=months)).to_period("M").to_timestamp("M")
    mask = pd.Series(True, index=end_idx)
    if start_domain is not None:
        mask &= (start_idx >= start_domain)
    if end_domain is not None:
        mask &= (end_idx <= end_domain)
    return r[mask]


def make_rolling_df(funds_df, selected_funds, focus_fund, bench_ser, months, start_domain, end_domain):
    """
    Build a DataFrame for rolling charts using precomputed rolling_cagr from Supabase.

    Output columns:
        - focus_fund
        - other selected funds (peers)
        - 'Peer avg'
        - benchmark (if bench_ser provided)

    Index:
        - 'Window' (i.e. asof_date = END of rolling window)

    IMPORTANT:
    We filter windows so that:
        start_of_window >= start_domain
        and
        end_of_window   <= end_domain

    where:
        start_of_window = asof_date - months
    """

    # ----------------------------
    # 1) Load precomputed FUND rolling returns for all selected funds (NO date filter here)
    # ----------------------------
    fund_roll = load_fund_rolling(
        window_months=months,
        fund_names=selected_funds,
        start=None,
        end=None,
    )
    if fund_roll.empty:
        return pd.DataFrame()

    # Pivot → wide matrix: rows = dates (asof_date), columns = fund_name
    wide = (
        fund_roll.pivot_table(
            index="asof_date",
            columns="fund_name",
            values="rolling_cagr",
            aggfunc="first",
        )
        .sort_index()
    )

    # ----------------------------
    # 2) Apply START/END domain filter at the *window* level
    # ----------------------------
    end_idx = pd.to_datetime(wide.index)
    start_idx = (end_idx - pd.DateOffset(months=months)).to_period("M").to_timestamp("M")

    mask = pd.Series(True, index=end_idx)

    if start_domain is not None:
        mask &= (start_idx >= start_domain)
    if end_domain is not None:
        mask &= (end_idx <= end_domain)

    wide = wide.loc[mask]
    if wide.empty:
        return pd.DataFrame()

    # ----------------------------
    # 3) Add BENCHMARK if provided
    # ----------------------------
    bench_label = None
    if bench_ser is not None and hasattr(bench_ser, "name"):
        bench_label = bench_ser.name

    if bench_label:
        bench_roll = load_bench_rolling(
            window_months=months,
            bench_name=bench_label,
            start=None,
            end=None,
        )
        if not bench_roll.empty:
            bench_series = (
                bench_roll
                .set_index("asof_date")["rolling_cagr"]
                .sort_index()
                .reindex(wide.index)  # align with fund windows
            )
            wide[bench_label] = bench_series

    # ----------------------------
    # 4) Compute PEER AVERAGE (exclude focus + benchmark)
    # ----------------------------
    cols_excluding_bench = [
        c for c in wide.columns
        if c != bench_label
    ]

    peer_cols = [
        c for c in cols_excluding_bench
        if c != focus_fund
    ]

    if peer_cols:
        wide["Peer avg"] = wide[peer_cols].mean(axis=1)
    else:
        if "Peer avg" in wide.columns:
            wide = wide.drop(columns=["Peer avg"])

    # 🔹 Convert decimals (0.25) → percents (25.0) for plotting & stats
    wide = wide * 100.0

    wide.index.name = "Window"
    return wide



def make_multi_fund_rolling(funds_df, selected_funds, months, start_domain, end_domain):
    """
    Build a wide DataFrame of precomputed rolling CAGRs for multiple funds,
    for the '3Y Rolling — Multiple Selected Funds' and
    '1Y Rolling — Multiple Selected Funds' charts.

    Uses precomputed fundlab.fund_rolling_return and applies the SAME
    start/end-domain window logic as make_rolling_df.
    """

    if not selected_funds:
        return pd.DataFrame()

    roll = load_fund_rolling(
        window_months=months,
        fund_names=selected_funds,
        start=None,
        end=None,
    )
    if roll.empty:
        return pd.DataFrame()

    wide = roll.pivot_table(
        index="asof_date",
        columns="fund_name",
        values="rolling_cagr",
        aggfunc="first",
    ).sort_index()

    end_idx = pd.to_datetime(wide.index)
    start_idx = (end_idx - pd.DateOffset(months=months)).to_period("M").to_timestamp("M")

    mask = pd.Series(True, index=end_idx)
    if start_domain is not None:
        mask &= (start_idx >= start_domain)
    if end_domain is not None:
        mask &= (end_idx <= end_domain)

    wide = wide.loc[mask]
    if wide.empty:
        return pd.DataFrame()

    # 🔹 Convert decimals (0.25) → percents (25.0) for plotting & stats
    wide = wide * 100.0
    
    wide.index.name = "Window"
    return wide


# Backward-compat wrapper: old name → new implementation
def make_multi_fund_rolling_df(funds_df, selected_funds, months, start_domain, end_domain):
    return make_multi_fund_rolling(funds_df, selected_funds, months, start_domain, end_domain)



def plot_rolling(df, months, focus_name, bench_label, chart_height=560, include_cols=None):
    """
    Plot rolling CAGR for:
      - focus fund (black)
      - peer average (blue, column name: 'Peer avg')
      - benchmark (red, using bench_label)

    df is expected to have index as dates (from make_rolling_df) and columns including
    focus_name, optionally 'Peer avg', and optionally a benchmark column named bench_label
    (or 'Benchmark' in legacy cases).
    """
    if df.empty:
        return None

    # Build the "Window" label from index and reset index
    labels = window_label_series(df.index, months)
    plot_df = df.copy()
    plot_df["Window"] = labels.values
    plot_df = plot_df.reset_index(drop=True)

    # Determine actual benchmark column name in the DataFrame
    bench_col = None
    if bench_label:
        if bench_label in plot_df.columns:
            bench_col = bench_label
        elif "Benchmark" in plot_df.columns:
            # Legacy case: rename 'Benchmark' column to the provided bench_label
            plot_df = plot_df.rename(columns={"Benchmark": bench_label})
            bench_col = bench_label

    # Decide which columns to plot
    default_cols = []
    if focus_name in plot_df.columns:
        default_cols.append(focus_name)
    if "Peer avg" in plot_df.columns:
        default_cols.append("Peer avg")
    if bench_col:
        default_cols.append(bench_col)

    if include_cols:
        ycols = [c for c in include_cols if c in plot_df.columns]
    else:
        ycols = default_cols

    if not ycols:
        return None

    # Color mapping:
    #   - Focus fund: black
    #   - Benchmark: red
    #   - Peer avg: blue
    #   - Any other series: assigned from a qualitative palette
    palette = (
        px.colors.qualitative.Dark24
        + px.colors.qualitative.Alphabet
        + px.colors.qualitative.Bold
    )
    color_map = {}

    # Focus fund
    if focus_name in ycols:
        color_map[focus_name] = "#000000"  # black

    # Benchmark
    if bench_col and bench_col in ycols:
        color_map[bench_col] = "#d62728"  # red

    # Peer avg
    if "Peer avg" in ycols:
        color_map["Peer avg"] = "#1f77b4"  # blue

    # Assign colors for any remaining series
    palette_idx = 0
    for col in ycols:
        if col not in color_map:
            color_map[col] = palette[palette_idx % len(palette)]
            palette_idx += 1

    fig = px.line(
        plot_df,
        x="Window",
        y=ycols,
        labels={"value": "Return (%)", "Window": "Rolling window (start–end)"},
        title=f"{months // 12}Y Rolling CAGR",
        color_discrete_map=color_map,
    )

    # Thicker line for focus fund
    for tr in fig.data:
        if tr.name == focus_name:
            tr.update(line=dict(width=4))
        else:
            tr.update(line=dict(width=3))

    fig.update_layout(
        height=chart_height,
        margin=dict(l=40, r=40, t=60, b=80),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
        ),
        hovermode="x unified",
    )

    n = len(plot_df["Window"])
    tickvals = (
        plot_df["Window"].tolist()
        if n <= 12
        else plot_df["Window"].tolist()[:: math.ceil(n / 12)]
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=tickvals,
        ticktext=tickvals,
        tickangle=-20,
        tickfont=dict(size=12),
        categoryorder="array",
        categoryarray=plot_df["Window"].tolist(),
        showgrid=True,
    )
    fig.update_yaxes(tickformat=".2f", ticksuffix="%", showgrid=True)

    return fig



def plot_multi_fund_rolling(df, months, focus_name=None, chart_height=560):
    """
    Plot rolling CAGR for multiple funds (no benchmark here).

    - Each fund gets a distinct color from a large qualitative palette.
    - Optional focus fund is highlighted with a thicker black line.
    """

    if df.empty:
        return None

    labels = window_label_series(df.index, months)
    plot_df = df.copy()
    plot_df["Window"] = labels.values
    plot_df = plot_df.reset_index(drop=True)

    series_cols = [c for c in plot_df.columns if c != "Window"]
    if not series_cols:
        return None

    palette = (
        px.colors.qualitative.Dark24
        + px.colors.qualitative.Alphabet
        + px.colors.qualitative.Bold
    )

    color_map = {}

    # Focus fund (if any): black
    if focus_name and focus_name in series_cols:
        color_map[focus_name] = "#000000"

    # Assign colors for all other series
    palette_idx = 0
    for col in series_cols:
        if col not in color_map:
            color_map[col] = palette[palette_idx % len(palette)]
            palette_idx += 1

    fig = px.line(
        plot_df,
        x="Window",
        y=series_cols,
        labels={"value": "Return (%)", "Window": "Rolling window (start–end)"},
        title=f"{months // 12}Y Rolling CAGR — Multiple funds",
        color_discrete_map=color_map,
    )

    # Thicker line for focus fund, if present
    for tr in fig.data:
        if focus_name and tr.name == focus_name:
            tr.update(line=dict(width=4))
        else:
            tr.update(line=dict(width=3))

    fig.update_layout(
        height=chart_height,
        margin=dict(l=40, r=40, t=60, b=80),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
        ),
        hovermode="x unified",
    )

    n = len(plot_df["Window"])
    tickvals = (
        plot_df["Window"].tolist()
        if n <= 12
        else plot_df["Window"].tolist()[:: math.ceil(n / 12)]
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=tickvals,
        ticktext=tickvals,
        tickangle=-20,
        tickfont=dict(size=12),
        categoryorder="array",
        categoryarray=plot_df["Window"].tolist(),
        showgrid=True,
    )
    fig.update_yaxes(tickformat=".2f", ticksuffix="%", showgrid=True)

    return fig


def rolling_outperf_stats(df: pd.DataFrame, focus_name: str, bench_label: str | None = None):
    """
    Compute stats of focus fund *relative to benchmark* over rolling windows.

    df columns are expected to be in PERCENT units (e.g. 12.3 for 12.3%),
    consistent with what we plot.

    We detect the benchmark column as:
        - bench_label, if provided and present in df; else
        - 'Benchmark' if present; else
        - return None.
    """

    if df is None or df.empty:
        return None

    if focus_name not in df.columns:
        return None

    # Determine benchmark column
    bcol = None
    if bench_label and bench_label in df.columns:
        bcol = bench_label
    elif "Benchmark" in df.columns:
        bcol = "Benchmark"
    else:
        return None  # no usable benchmark column

    # Extract series as decimals
    f = df[focus_name] 
    b = df[bcol] 

    op = (f - b).dropna()
    if op.empty:
        return None

    return pd.DataFrame({
        "windows": [int(op.notna().count())],
        "median (ppt)": [float(np.nanmedian(op))],
        "mean   (ppt)": [float(np.nanmean(op))],
        "min    (ppt)": [float(np.nanmin(op))],
        "max    (ppt)": [float(np.nanmax(op))],
        "prob. of outperformance": [float((op > 0).mean() * 100.0)],
    })


def _read_any(uploaded_file):
    """
    Read CSV or Excel upload into a DataFrame.
    Returns a pandas.DataFrame or raises an Exception for the caller to handle.
    """
    import pandas as pd

    if uploaded_file is None:
        raise ValueError("No file provided")

    name = (uploaded_file.name or "").lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        elif name.endswith(".xlsx") or name.endswith(".xls"):
            # requires: pip install openpyxl
            return pd.read_excel(uploaded_file, engine="openpyxl")
        else:
            raise ValueError(f"Unsupported file type: {uploaded_file.name}")
    except Exception as e:
        raise RuntimeError(f"Error reading {uploaded_file.name}: {e}")


def prepare_focus_fund_timeline(df: pd.DataFrame, focus_fund_id: int) -> pd.DataFrame:
    d = df[df["fund_id"] == focus_fund_id].copy()
    if d.empty:
        return d

    d["from_date"] = pd.to_datetime(d["from_date"])
    d["to_date"] = pd.to_datetime(d["to_date"])
    d["to_date_filled"] = d["to_date"].fillna(pd.Timestamp.today())

    # Oldest tenure at top, latest at bottom
    d = d.sort_values(["from_date", "to_date_filled", "fund_manager"]).reset_index(drop=True)
    d["ypos"] = (len(d) - 1) - d.index

    # Identify current manager(s)
    if d["to_date"].isna().any():
        d["is_current"] = d["to_date"].isna()
    else:
        d["is_current"] = d["to_date_filled"] == d["to_date_filled"].max()

    return d


def render_fund_manager_tenure_chart(df: pd.DataFrame):
    if df.empty:
        st.info("No fund manager tenure data available for this fund.")
        return

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("from_date:T", title=None, axis=alt.Axis(format="%Y")),
            x2="to_date_filled:T",
            y=alt.Y("ypos:O", axis=None),
            color=alt.condition(
                alt.datum.is_current,
                alt.value("#1f77b4"),   # highlighted current manager
                alt.value("#c7c7c7"),
            ),
            tooltip=[
                alt.Tooltip("fund_manager:N", title="Fund manager"),
                alt.Tooltip("from_date:T", title="From"),
                alt.Tooltip("to_date:T", title="To"),
            ],
        )
    )

    labels = (
        alt.Chart(df)
        .mark_text(dx=0, dy=0, color="black")
        .transform_calculate(
            mid="datetime((datum.from_date.getTime() + datum.to_date_filled.getTime())/2)"
        )
        .encode(
            x=alt.X("mid:T"),
            y=alt.Y("ypos:O"),
            text="fund_manager:N",
        )
    )

    st.altair_chart(
        (chart + labels).properties(height=max(200, 35 * len(df))),
        use_container_width=True,
    )


def render_current_manager_summary(df: pd.DataFrame):
    cur = df[df["is_current"]].copy()
    if cur.empty:
        return

    today = pd.Timestamp.today()
    cur["years"] = (today - cur["from_date"]).dt.days / 365.25

    parts = [
        f"{r.fund_manager} managing since the past {r.years:.1f} years"
        for r in cur.itertuples(index=False)
    ]
    st.caption("Current fund manager: " + "; ".join(parts))



# ------------------------ Inputs ------------------------
# Accept CSV or Excel for both uploads - Commented out the excel upload functionality to upload directly from Supabase - 21 Nov 2025
# funds_file = st.file_uploader("Upload Funds file (CSV or Excel)", type=["csv", "xlsx", "xls"], key="funds")
# bench_file = st.file_uploader("Upload Benchmarks file (CSV or Excel) — optional but recommended", type=["csv", "xlsx", "xls"], key="bench")

# if funds_file is None:
#    st.info("Upload your files to proceed.")
#    st.stop()

# try:
#     raw_funds_df = _read_any(funds_file)
#     funds_df = _clean_funds(raw_funds_df)
# except Exception as e:
#     st.error(str(e)); st.stop()

# bench_df = None
# if bench_file is not None:
#     try:
#         raw_bench_df = _read_any(bench_file)
#         bench_df = _clean_bench(raw_bench_df)
#     except Exception as e:
#         st.error(str(e)); st.stop()


# 🔄 Load from PostgreSQL instead of file upload
def performance_page():
    home_button()
    raw_funds_df = load_funds_from_db()
    if raw_funds_df.empty:
        st.error("No fund NAV data found in database.")
        st.stop()

    funds_df = _clean_funds(raw_funds_df.copy())

    bench_df = None
    try:
        raw_bench_df = load_bench_from_db()
        if not raw_bench_df.empty:
            bench_df = _clean_bench(raw_bench_df.copy())
    except Exception as e:
        st.warning(f"Could not load benchmark data from DB: {e}")
        bench_df = None

    # Optional: small status line at the top of the app
    # Small status line at the top of the app
    # Small status line at the top of the app – robust to different cleaned column names
    fund_candidates = ["Fund", "Fund name", "fund", "fund_name"]
    fund_col = next((c for c in fund_candidates if c in funds_df.columns), None)

    date_candidates = ["month-end", "Date", "date", "nav_date"]
    date_col = next((c for c in date_candidates if c in funds_df.columns), None)

    # Latest date string
    latest_str = "N/A"
    if date_col is not None:
        try:
            latest_date = funds_df[date_col].max()
            latest_str = latest_date.strftime("%d-%b-%Y")
        except Exception:
            latest_str = str(latest_date)

    # Number of funds
    if fund_col is not None:
        num_funds = funds_df[fund_col].nunique()
    else:
        num_funds = "N/A"

    st.caption(
        f"Data source: Supabase · Funds: {num_funds} · Latest NAV date: {latest_str}"
    )






    # ------------------------ Selectors (stacked with divider) ------------------------
    def checkbox_group(title: str, options: list, key_prefix: str) -> list:
        st.markdown(f"**{title}**")
        cols = st.columns(min(4, max(1, len(options))))
        chosen = []
        for i, opt in enumerate(options):
            with cols[i % len(cols)]:
                if st.checkbox(opt, value=False, key=f"{key_prefix}_{opt}"):
                    chosen.append(opt)
        return chosen


    all_caps = sorted(funds_df["market_cap"].dropna().unique().tolist())
    # all_styles = sorted(funds_df["style"].dropna().unique().tolist())

    # Market-cap (checkbox group)
    caps = checkbox_group("Market-cap (tick multiple as needed)", all_caps, "cap")
    st.divider()
    # Style removed in favour of benchmark selector
    # styles = checkbox_group("Style (tick multiple as needed)", all_styles, "sty")

    # Require at least one market-cap
    if not caps:
        st.warning("Tick at least one Market-cap to continue.")
        st.stop()

    # Filter funds by selected market-caps
    filtered = funds_df[funds_df["market_cap"].isin(caps)].copy()

    # ---------- Benchmark selector (checkboxes, similar to Market-cap) ----------
    if bench_df is not None and not bench_df.empty:
        bench_names = sorted(bench_df["benchmark_name"].dropna().unique().tolist())
    else:
        bench_names = []

    if not bench_names:
        st.warning("No benchmarks found in database.")
        bench_selected = []
        bench_label = None
        bench_ser = None
    else:
        #st.markdown("**Benchmarks (tick multiple as needed)**")
        bench_selected = checkbox_group("Benchmarks (tick multiple as needed)", bench_names, "bench")

        # Primary benchmark for analytics = first ticked benchmark
        if bench_selected:
            bench_label = bench_selected[0]
            bmask = bench_df["benchmark_name"] == bench_label
            bench_ser = (
                bench_df.loc[bmask, ["date", "nav"]]
                .drop_duplicates("date")
                .set_index("date")["nav"]
                .sort_index()
            )
            bench_ser.name = bench_label
        else:
            bench_label = None
            bench_ser = None





    if not caps:
        st.warning("Tick at least one Market-cap to continue.")
        st.stop()

    # filtered = funds_df[(funds_df["market_cap"].isin(caps)) & (funds_df["style"].isin(styles))]
    # Ensure all dates are month-end timestamps

    # --- EOM + numeric normalization for funds data ---
    filtered["date"] = to_eom(filtered["date"])
    filtered["nav"]  = coerce_num(filtered["nav"])

    # If the same fund/date appears multiple times, keep the last
    filtered = (filtered
                .sort_values(["fund", "date"])
                .drop_duplicates(subset=["fund", "date"], keep="last"))


    fund_options = sorted(filtered["fund"].unique().tolist())

    st.markdown("**Funds (multi-select)**")
    funds_with_all = ["ALL"] + fund_options
    funds_selected = st.multiselect("Choose funds (include 'ALL' for all in list)", options=funds_with_all, default=[])
    if any(str(f).upper()=="ALL" for f in funds_selected):
        funds_selected = fund_options
    if not funds_selected:
        st.warning("Select at least one fund."); st.stop()

    focus_fund = st.selectbox("Focus fund (vs. peers)", options=["-- none --"] + funds_selected, index=0)
    if focus_fund == "-- none --":
        st.warning("Pick a Focus fund to compute Peers Avg (we exclude the focus from peers).")
        st.stop()


    ## Removing radio button selector between Portfolio and Category - 24 Nov 2025

    # bench_mode = st.radio("Benchmark", options=["Portfolio (Nifty)","Category"], index=0, horizontal=True)
    # bench_name, bench_ser = pick_benchmark(bench_df, bench_mode, caps, styles)
    # bench_label = bench_name if (bench_ser is not None and not bench_ser.empty) else "Benchmark"
    # if bench_label != "Benchmark":
    #     st.caption(f"Using **Benchmark:** {bench_label}")
    # else:
    #     st.warning("No matching benchmark series found. Upload a Benchmarks CSV or adjust the toggle/filters.")

        # ------------------------ Performance Analysis: Rolling / Yearly / P2P ------------------------
    st.header("Performance analysis – rolling, yearly, and P2P")

    # Shared Month + Year pickers for all analysis modes
    def eom(y: int, m: int) -> pd.Timestamp:
        return pd.Timestamp(year=y, month=m, day=1).to_period("M").to_timestamp("M")

    date_years = sorted(pd.to_datetime(filtered["date"]).dt.year.unique().tolist())
    if not date_years:
        st.stop()
    min_y, max_y = min(date_years), max(date_years)

    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    m2n = {m: i + 1 for i, m in enumerate(months)}

    # ------------------------ Shared period + mode + Update (form) ------------------------
    with st.form("perf_filters"):
        cA1, cA2, cB1, cB2 = st.columns([1, 1, 1, 1])
        with cA1:
            start_month = st.selectbox(
                "Start month (start-domain)", months, index=0, key="rr_start_m"
            )
        with cA2:
            start_year = st.selectbox(
                "Start year  (start-domain)",
                list(range(min_y, max_y + 1)),
                index=0,
                key="rr_start_y",
            )
        with cB1:
            end_month = st.selectbox(
                "End month   (end-domain)",
                months,
                index=len(months) - 1,
                key="rr_end_m",
            )
        with cB2:
            end_year = st.selectbox(
                "End year    (end-domain)",
                list(range(min_y, max_y + 1)),
                index=len(range(min_y, max_y + 1)) - 1,
                key="rr_end_y",
            )

        start_domain = eom(int(start_year), m2n[start_month])
        end_domain = eom(int(end_year), m2n[end_month])

        analysis_mode = st.radio(
            "Analysis mode",
            [
                "Rolling returns vs. peer average and benchmarks",
                "Rolling returns - multiple funds",
                "Returns (Strict FY/CY endpoints)",
                "Point to point returns & relative return vs. benchmark (1/3/5/7 year CAGR)",
            ],
            index=0,
            key="perf_analysis_mode",
        )

        apply_filters = st.form_submit_button("Update", type="primary")

    # Track whether filters (period + mode) have been applied at least once
    filters_applied = st.session_state.get("perf_filters_applied", False)
    if apply_filters:
        filters_applied = True
        st.session_state["perf_filters_applied"] = True

    if end_domain <= start_domain:
        st.warning("End month must be after Start month.")
        return

    # If user has never clicked Update at least once, don’t run any heavy logic yet
    if not filters_applied:
        st.info("Adjust period and analysis mode above, then click **Update**.")
        return


    def window_ok(start_dt, end_dt, months: int) -> bool:
        return (start_dt + pd.DateOffset(months=months)) <= end_dt

    # This will collect rolling charts you mark "To print" – keep as before
    print_items = []

    # =========================================================================
    # MODE 1: Rolling returns vs. peer average and benchmarks (3Y & 1Y, focus)
    # =========================================================================
    if analysis_mode == "Rolling returns vs. peer average and benchmarks":
        st.subheader("3Y Rolling — Focus vs Peer avg vs Benchmark")

        series_opts_3 = [focus_fund, "Peer avg"] + ([bench_label] if bench_label else [])
        series_selected_3 = st.multiselect(
            "Show series (3Y)",
            options=series_opts_3,
            default=series_opts_3,
            key="series3",
        )

        if not window_ok(start_domain, end_domain, 36):
            st.info("Selected range too short for 3Y windows.")
        else:
            df3 = make_rolling_df(
                filtered,
                funds_selected,
                focus_fund,
                bench_ser,
                36,
                start_domain,
                end_domain,
            )
            fig3 = plot_rolling(
                df3,
                36,
                focus_fund,
                bench_label,
                chart_height=560,
                include_cols=series_selected_3,
            )
            if fig3 is None:
                st.info("Insufficient data or no series selected for 3Y rolling chart.")
            else:
                st.plotly_chart(fig3, use_container_width=True)
                stats3 = rolling_outperf_stats(df3, focus_fund, bench_label)
                st.subheader("3Y Rolling Outperformance Stats (Focus fund vs Benchmark)")
                st.dataframe(
                    stats3.round(2)
                    if stats3 is not None
                    else pd.DataFrame(
                        {"info": ["Not enough overlapping 3Y windows"]}
                    )
                )
                if st.checkbox("To print", key="print_fig3"):
                    print_items.append(("3Y Rolling — Focus/Peers/Benchmark", fig3))

        st.subheader("1Y Rolling — Focus vs Peer avg vs Benchmark")

        series_opts_1 = [focus_fund, "Peer avg"] + ([bench_label] if bench_label else [])
        series_selected_1 = st.multiselect(
            "Show series (1Y)",
            options=series_opts_1,
            default=series_opts_1,
            key="series1",
        )

        if not window_ok(start_domain, end_domain, 12):
            st.info("Selected range too short for 1Y windows.")
        else:
            df1 = make_rolling_df(
                filtered,
                funds_selected,
                focus_fund,
                bench_ser,
                12,
                start_domain,
                end_domain,
            )
            fig1 = plot_rolling(
                df1,
                12,
                focus_fund,
                bench_label,
                chart_height=560,
                include_cols=series_selected_1,
            )
            if fig1 is None:
                st.info("Insufficient data or no series selected for 1Y rolling chart.")
            else:
                st.plotly_chart(fig1, use_container_width=True)
                stats1 = rolling_outperf_stats(df1, focus_fund, bench_label)
                st.subheader("1Y Rolling Outperformance Stats (Focus fund vs Benchmark)")
                st.dataframe(
                    stats1.round(2)
                    if stats1 is not None
                    else pd.DataFrame(
                        {"info": ["Not enough overlapping 1Y windows"]}
                    )
                )
                if st.checkbox("To print", key="print_fig1"):
                    print_items.append(("1Y Rolling — Focus/Peers/Benchmark", fig1))

    # =========================================================================
    # MODE 2: Rolling returns – multiple funds (3Y & 1Y multi-fund charts)
    # =========================================================================
    elif analysis_mode == "Rolling returns - multiple funds":
        st.subheader("3Y Rolling — Multiple Selected Funds")
        mf3 = st.multiselect(
            "Pick funds to plot (3Y multi-fund)",
            options=funds_selected,
            default=[focus_fund],
            key="mf3",
        )
        if not window_ok(start_domain, end_domain, 36):
            st.info("Selected range too short for 3Y windows.")
        else:
            df3m = make_multi_fund_rolling_df(
                filtered, mf3, 36, start_domain, end_domain
            )
            fig3m = plot_multi_fund_rolling(
                df3m, 36, focus_name=focus_fund, chart_height=560
            )
            if fig3m is None:
                st.info("Not enough data to plot selected funds (3Y).")
            else:
                st.plotly_chart(fig3m, use_container_width=True)
                if st.checkbox("To print", key="print_fig3m"):
                    print_items.append(("3Y Rolling — Multiple funds", fig3m))

        st.subheader("1Y Rolling — Multiple Selected Funds")
        mf1 = st.multiselect(
            "Pick funds to plot (1Y multi-fund)",
            options=funds_selected,
            default=[focus_fund],
            key="mf1",
        )
        if not window_ok(start_domain, end_domain, 12):
            st.info("Selected range too short for 1Y windows.")
        else:
            df1m = make_multi_fund_rolling_df(
                filtered, mf1, 12, start_domain, end_domain
            )
            fig1m = plot_multi_fund_rolling(
                df1m, 12, focus_name=focus_fund, chart_height=560
            )
            if fig1m is None:
                st.info("Not enough data to plot selected funds (1Y).")
            else:
                st.plotly_chart(fig1m, use_container_width=True)
                if st.checkbox("To print", key="print_fig1m"):
                    print_items.append(("1Y Rolling — Multiple funds", fig1m))

    # =========================================================================
    # MODE 3: Returns (Strict FY/CY endpoints) – use shared start/end domain
    # =========================================================================
    elif analysis_mode == "Returns (Strict FY/CY endpoints)":
        st.header("Yearly Returns (Strict FY/CY endpoints)")

        yr_type_strict = st.radio(
            "Year type",
            options=["Financial (Apr–Mar)", "Calendar (Jan–Dec)"],
            index=0,
            horizontal=True,
            key="strict_year_type",
        )
        use_fy_strict = yr_type_strict.startswith("Financial")

        # Compute per-fund yearly returns with trimmed first/last years
        yr_rows = {}
        for f in funds_selected:
            s = (
                filtered.loc[filtered["fund"] == f, ["date", "nav"]]
                .drop_duplicates("date")
                .set_index("date")["nav"]
            )
            yr_rows[f] = yearly_returns_with_custom_domain(
                s, start_domain, end_domain, fy=use_fy_strict
            )
        yr_df = pd.DataFrame(yr_rows).T

        # Benchmark
        yr_bench = None
        if bench_ser is not None and not bench_ser.empty:
            yr_bench = yearly_returns_with_custom_domain(
                bench_ser, start_domain, end_domain, fy=use_fy_strict
            ).rename("Benchmark")

        if yr_df.empty:
            st.info("Not enough data to compute yearly returns for the selected funds.")
        else:
            # Align columns across funds and benchmark; keep original order
            cols_order = list(yr_df.columns)
            if yr_bench is not None and not yr_bench.empty:
                for c in yr_bench.index:
                    if c not in cols_order:
                        cols_order.append(c)
            yr_df = yr_df.reindex(columns=cols_order)

            # Actual table
            st.subheader("Actual yearly returns — Funds (rows) vs Years (columns)")

            disp_actual = (yr_df.loc[funds_selected, cols_order] * 100.0).copy()
            disp_actual.insert(0, "Fund", disp_actual.index)
            disp_actual = disp_actual.reset_index(drop=True)

            num_cols = [c for c in disp_actual.columns if c != "Fund"]
            disp_actual[num_cols] = disp_actual[num_cols].round(1)

            st.dataframe(
                disp_actual.style.format(
                    {c: "{:.1f}" for c in num_cols}, na_rep="—"
                ).set_table_styles(
                    [
                        {"selector": "table", "props": "table-layout:fixed"},
                        {
                            "selector": "th.col_heading",
                            "props": "white-space:normal; line-height:1.1; height:56px",
                        },
                    ]
                ),
                use_container_width=True,
            )
            p_act = st.checkbox("To print", key="print_actual_tbl")

            st.markdown(
                "<div style='height:12px;'></div>", unsafe_allow_html=True
            )

            # Benchmark table
            st.subheader(f"Benchmark yearly returns — {bench_label}")
            if yr_bench is None or yr_bench.empty:
                st.info("Benchmark not available.")
                bench_df_print = pd.DataFrame()
            else:
                bench_df_print = ((yr_bench[cols_order] * 100.0).to_frame().T).round(2)
                bench_df_print.index = [bench_label]
                bench_df_print = bench_df_print.reset_index().rename(
                    columns={"index": "Benchmark"}
                )
                st.dataframe(
                    bench_df_print.style.format(
                        {
                            c: "{:.1f}"
                            for c in bench_df_print.columns
                            if c != "Benchmark"
                        },
                        na_rep="—",
                    ).set_table_styles(
                        [
                            {"selector": "table", "props": "table-layout:fixed"},
                            {
                                "selector": "th.col_heading",
                                "props": "white-space:normal; line-height:1.1; height:56px",
                            },
                        ]
                    ),
                    use_container_width=True,
                )

            p_bench = st.checkbox("To print", key="print_bench_tbl")

            st.markdown(
                "<div style='height:12px;'></div>", unsafe_allow_html=True
            )

            # Relative (ppt) vs benchmark
            st.subheader(
                "Relative yearly returns (ppt) — Funds (rows) vs Years (columns)"
            )
            rel_df = pd.DataFrame()
            if yr_bench is not None and not yr_bench.empty:
                common = [c for c in cols_order if c in yr_bench.index]
                if common:
                    rel_df = (
                        yr_df.loc[funds_selected, common].subtract(
                            yr_bench[common], axis=1
                        )
                        * 100.0
                    ).round(2)

            if rel_df.empty:
                st.info("Not enough overlap to compute relative returns.")
                disp_rel = pd.DataFrame()
            else:
                disp_rel = rel_df.copy()
                disp_rel.insert(0, "Fund", disp_rel.index)
                disp_rel = disp_rel.reset_index(drop=True)

                def rel_colors(df):
                    styles = pd.DataFrame("", index=df.index, columns=df.columns)
                    for c in df.columns:
                        if c == "Fund":
                            continue
                        styles[c] = df[c].apply(
                            lambda v: "background-color:#e6f4ea;color:#0b8043"
                            if pd.notna(v) and v > 0
                            else "background-color:#fdecea;color:#a50e0e"
                            if pd.notna(v) and v < 0
                            else ""
                        )
                    return styles

                st.write(
                    disp_rel.style.apply(rel_colors, axis=None)
                    .format(
                        {c: "{:.1f}" for c in disp_rel.columns if c != "Fund"},
                        na_rep="—",
                    )
                    .set_table_styles(
                        [
                            {"selector": "table", "props": "table-layout:fixed"},
                            {
                                "selector": "th.col_heading",
                                "props": "white-space:normal; line-height:1.1; height:56px",
                            },
                        ]
                    )
                )

            p_rel = st.checkbox("To print", key="print_rel_tbl")

    # =========================================================================
    # MODE 4: Point to point returns & relative vs benchmark (1/3/5/7Y)
    # =========================================================================
    else:
        st.header("Point-to-Point (P2P) Returns — Custom period CAGR")

        p2p_start = start_domain
        p2p_end = end_domain

        def months_between(a: pd.Timestamp, b: pd.Timestamp) -> int:
            return (b.year - a.year) * 12 + (b.month - a.month)

        def series_cagr_between(
            s: pd.Series, start_eom: pd.Timestamp, end_eom: pd.Timestamp
        ):
            """Exact month-end CAGR (decimal) between two month-ends."""
            s = s.dropna().sort_index()
            if (
                start_eom not in s.index
                or end_eom not in s.index
                or end_eom <= start_eom
            ):
                return np.nan
            m = months_between(start_eom, end_eom)
            if m <= 0:
                return np.nan
            try:
                return (s.loc[end_eom] / s.loc[start_eom]) ** (12.0 / m) - 1.0
            except Exception:
                return np.nan

        if p2p_end <= p2p_start:
            st.warning("P2P End must be after Start.")
        else:
            rows = []
            for f in funds_selected:
                s = (
                    filtered.loc[filtered["fund"] == f, ["date", "nav"]]
                    .drop_duplicates("date")
                    .set_index("date")["nav"]
                )
                val = series_cagr_between(s, p2p_start, p2p_end)
                rows.append(
                    {
                        "Fund": f,
                        "Start": f"{p2p_start:%b %Y}",
                        "End": f"{p2p_end:%b %Y}",
                        "Months": months_between(p2p_start, p2p_end),
                        "CAGR %": None if np.isnan(val) else round(val * 100.0, 2),
                    }
                )
            if bench_ser is not None and not bench_ser.empty:
                bval = series_cagr_between(bench_ser, p2p_start, p2p_end)
                rows.append(
                    {
                        "Fund": bench_label,
                        "Start": f"{p2p_start:%b %Y}",
                        "End": f"{p2p_end:%b %Y}",
                        "Months": months_between(p2p_start, p2p_end),
                        "CAGR %": None if np.isnan(bval) else round(bval * 100.0, 2),
                    }
                )

            p2p_df = pd.DataFrame(rows)
            for col in list(p2p_df.columns):
                if isinstance(col, str) and col.endswith("CAGR %"):
                    p2p_df[col] = pd.to_numeric(p2p_df[col], errors="coerce").round(1)
            if "CAGR %" in p2p_df.columns:
                p2p_df["CAGR %"] = pd.to_numeric(
                    p2p_df["CAGR %"], errors="coerce"
                ).round(1)
                p2p_df = p2p_df.sort_values(by="CAGR %", ascending=False)

            st.dataframe(
                p2p_df.style.format(
                    {c: "{:.1f}%" for c in p2p_df.columns if c.endswith("CAGR %")},
                    na_rep="—",
                ),
                use_container_width=True,
            )
            p_p2p = st.checkbox("To print", key="print_p2p_tbl")

            # ------------------------ Relative Multi-Horizon vs Benchmark ------------------------
            st.subheader(
                "Relative CAGR vs Benchmark — 1Y / 3Y / 5Y / 7Y (as of P2P end month)"
            )

            def end_aligned_cagr(
                series: pd.Series, end_eom: pd.Timestamp, months: int
            ) -> float:
                s = series.dropna().sort_index()
                if end_eom not in s.index:
                    return np.nan
                start_eom = (
                    end_eom - pd.DateOffset(months=months)
                ).to_period("M").to_timestamp("M")
                if start_eom not in s.index or end_eom <= start_eom:
                    return np.nan
                try:
                    return (s.loc[end_eom] / s.loc[start_eom]) ** (
                        12.0 / months
                    ) - 1.0
                except Exception:
                    return np.nan

            horizons = [(12, "1Y"), (36, "3Y"), (60, "5Y"), (84, "7Y")]
            bench_cagrs = {}
            if bench_ser is not None and not bench_ser.empty:
                for m, lbl in horizons:
                    bench_cagrs[lbl] = end_aligned_cagr(bench_ser, p2p_end, m)
            else:
                bench_cagrs = {lbl: np.nan for _, lbl in horizons}

            rel_rows = []
            for f in funds_selected:
                s = (
                    filtered.loc[filtered["fund"] == f, ["date", "nav"]]
                    .drop_duplicates("date")
                    .set_index("date")["nav"]
                )
                row = {"Fund": f}
                for m, lbl in horizons:
                    fc = end_aligned_cagr(s, p2p_end, m)
                    bc = bench_cagrs.get(lbl, np.nan)
                    row[lbl] = (
                        None
                        if (np.isnan(fc) or np.isnan(bc))
                        else round((fc - bc) * 100.0, 2)
                    )
                rel_rows.append(row)

            rel_mh_df = pd.DataFrame(rel_rows).set_index("Fund")

            def style_rel_mh(df: pd.DataFrame):
                df2 = df.copy()
                for c in df2.columns:
                    if c != "Fund" and not pd.api.types.is_numeric_dtype(df2[c]):
                        try:
                            tmp = pd.to_numeric(df2[c], errors="coerce")
                            if tmp.notna().any():
                                df2[c] = tmp
                        except Exception:
                            pass

                num_cols = [
                    c
                    for c in df2.columns
                    if c != "Fund" and pd.api.types.is_numeric_dtype(df2[c])
                ]

                def rel_colors(dfin: pd.DataFrame):
                    styles = pd.DataFrame("", index=dfin.index, columns=dfin.columns)
                    for c in num_cols:
                        styles[c] = dfin[c].apply(
                            lambda v: "background-color:#e6f4ea;color:#0b8043"
                            if pd.notna(v) and v > 0
                            else "background-color:#fdecea;color:#a50e0e"
                            if pd.notna(v) and v < 0
                            else ""
                        )
                    return styles

                sty = df2.style.apply(rel_colors, axis=None)
                fmt_map = {c: "{:.2f}" for c in num_cols}
                sty = sty.format(fmt_map, na_rep="—")
                sty = sty.set_table_styles(
                    [
                        {"selector": "table", "props": "table-layout:fixed"},
                        {
                            "selector": "th.col_heading",
                            "props": "white-space:normal; line-height:1.1; height:56px",
                        },
                    ]
                )
                return sty

            if rel_mh_df.empty:
                st.info("Not enough data to compute relative multi-horizon returns.")
                rel_mh_disp = pd.DataFrame()
            else:
                st.dataframe(style_rel_mh(rel_mh_df), use_container_width=True)
                rel_mh_disp = rel_mh_df.copy()

            p_relmh = st.checkbox("To print", key="print_rel_mh")



    # ------------------------ Printing to PDF ------------------------
    st.markdown("---")

    def df_to_table_figure(df: pd.DataFrame, title: str, fill=None):
        """Render a DataFrame to a Plotly table figure with optional cell fills."""
        if df.empty:
            return None
        df_print = df.copy()
        df_print = df_print.reset_index()  # first column becomes visible index
        headers = list(df_print.columns)
        cells = [df_print[c].astype(object).astype(str).tolist() for c in headers]

        # Fills: either single color or column-wise list of lists
        if fill is None:
            cell_fill = "white"
        else:
            # Expecting fill to be a list of lists per column; if given as rows, transpose-ish:
            if isinstance(fill, list) and fill and isinstance(fill[0], list):
                # fill currently built as [index_col, col1, col2, ...] -> matches our cells already
                cell_fill = fill
            else:
                cell_fill = "white"

        fig = go.Figure(data=[go.Table(
            header=dict(values=headers, fill_color="#f0f0f0", align="left", font=dict(size=12, color="black")),
            cells=dict(values=cells, align="left", fill_color=cell_fill, font=dict(size=11, color="black"))
        )])
        fig.update_layout(title=title, template="plotly_white", margin=dict(l=20,r=20,t=60,b=20), height=560)
        return fig

    def build_rel_fill(df_with_fund_col: pd.DataFrame, fund_col="Fund", misaligned=None):
        """Green/red/white fill matrix for relative tables. Expects Fund column present."""
        misaligned = set(misaligned or [])
        # Ensure the Fund column exists as a data column
        if fund_col not in df_with_fund_col.columns:
            tmp = df_with_fund_col.copy()
            tmp.insert(0, fund_col, tmp.index)
            df_with_fund_col = tmp

        headers = df_with_fund_col.columns.tolist()
        fills = []
        # Build fill per column to match df_to_table_figure() cells order
        for c in headers:
            col_fill = []
            for v in df_with_fund_col[c].tolist():
                if c == fund_col:
                    col_fill.append("#FFF59D" if (v in misaligned) else "white")
                else:
                    if v is None or (isinstance(v, str) and v.strip()==""):
                        col_fill.append("white")
                    else:
                        try:
                            fv = float(v)
                            if fv > 0:   col_fill.append("#e6f4ea")  # light green
                            elif fv < 0: col_fill.append("#fdecea")  # light red
                            else:        col_fill.append("white")
                        except Exception:
                            col_fill.append("white")
            fills.append(col_fill)
        return fills

    if st.button("Print charts"):
        if not print_items:
            st.warning("Nothing selected. Tick “To print” under the charts/tables you want.")
        else:
            try:
                pdf_bytes = BytesIO()
                c = canvas.Canvas(pdf_bytes, pagesize=landscape(A4))
                W, H = landscape(A4)

                # Title page
                c.setFont("Helvetica-Bold", 18)
                c.drawString(40, H - 40, f"Performance charts & metrics - {focus_fund}")
                c.setFont("Helvetica", 10)
                c.drawString(40, H - 58, f"Benchmark: {bench_label}")
                c.showPage()

                # Pre-selected plotly figures (rolling charts, etc.)
                for caption, fig in print_items:
                    c.setFont("Helvetica-Bold", 14)
                    c.drawString(40, H - 40, caption)
                    png = fig.to_image(format="png", scale=2)  # kaleido
                    img = ImageReader(BytesIO(png))
                    max_w, max_h = W - 80, H - 100
                    iw, ih = img.getSize()
                    scale = min(max_w/iw, max_h/ih)
                    c.drawImage(img, 40 + (max_w - iw*scale)/2, 40, width=iw*scale, height=ih*scale)
                    c.showPage()

                # Yearly — Actual (Funds vs Years)
                if 'p_act' in locals() and p_act and 'disp_actual' in locals() and not disp_actual.empty:
                    figA = df_to_table_figure(disp_actual.round(2), "Actual yearly returns — Funds vs Years", fill="white")
                    if figA is not None:
                        png = figA.to_image(format="png", scale=2)
                        img = ImageReader(BytesIO(png))
                        c.setFont("Helvetica-Bold", 14)
                        c.drawString(40, H - 40, "Actual yearly returns — Funds vs Years")
                        max_w, max_h = W - 80, H - 100
                        iw, ih = img.getSize()
                        scale = min(max_w/iw, max_h/ih)
                        c.drawImage(img, 40, 40, width=iw*scale, height=ih*scale)
                        c.showPage()

                # Yearly — Benchmark
                if 'p_bmk' in locals() and p_bmk and 'bench_df_print' in locals() and not bench_df_print.empty:
                    figB = df_to_table_figure(bench_df_print.round(2), f"Benchmark yearly returns — {bench_label}", fill="white")
                    if figB is not None:
                        png = figB.to_image(format="png", scale=2)
                        img = ImageReader(BytesIO(png))
                        c.setFont("Helvetica-Bold", 14)
                        c.drawString(40, H - 40, f"Benchmark yearly returns — {bench_label}")
                        max_w, max_h = W - 80, H - 100
                        iw, ih = img.getSize()
                        scale = min(max_w/iw, max_h/ih)
                        c.drawImage(img, 40, 40, width=iw*scale, height=ih*scale)
                        c.showPage()

                # Yearly — Relative (ppt)
                if 'p_rel' in locals() and p_rel and 'disp_rel' in locals() and not disp_rel.empty:
                    fills_rel = build_rel_fill(disp_rel, fund_col="Fund", misaligned=locals().get('misaligned_funds', []))
                    figR = df_to_table_figure(disp_rel.round(2), "Relative yearly returns (ppt)", fill=fills_rel)
                    if figR is not None:
                        png = figR.to_image(format="png", scale=2)
                        img = ImageReader(BytesIO(png))
                        c.setFont("Helvetica-Bold", 14)
                        c.drawString(40, H - 40, "Relative yearly returns (ppt)")
                        max_w, max_h = W - 80, H - 100
                        iw, ih = img.getSize()
                        scale = min(max_w/iw, max_h/ih)
                        c.drawImage(img, 40, 40, width=iw*scale, height=ih*scale)
                        c.showPage()

                # P2P table
                if 'p_p2p' in locals() and p_p2p and 'p2p_df' in locals() and not p2p_df.empty:
                    figP = df_to_table_figure(p2p_df, "Point-to-Point (P2P) Returns", fill="white")
                    if figP is not None:
                        png = figP.to_image(format="png", scale=2)
                        img = ImageReader(BytesIO(png))
                        c.setFont("Helvetica-Bold", 14)
                        c.drawString(40, H - 40, "Point-to-Point (P2P) Returns")
                        max_w, max_h = W - 80, H - 100
                        iw, ih = img.getSize()
                        scale = min(max_w/iw, max_h/ih)
                        c.drawImage(img, 40, 40, width=iw*scale, height=ih*scale)
                        c.showPage()

                # Relative multi-horizon table
                if 'p_relmh' in locals() and p_relmh and 'rel_mh_disp' in locals() and not rel_mh_disp.empty:
                    # Build sign-based fills (no misalignment concept here)
                    dfmh = rel_mh_disp.copy()
                    dfmh = dfmh.reset_index()  # Fund becomes column
                    # Fill: index first (Fund), then 1Y/3Y/5Y/7Y
                    fills_mh = build_rel_fill(dfmh, fund_col="Fund", misaligned=None)
                    figMH = df_to_table_figure(dfmh.set_index("Fund"), "Relative CAGR vs Benchmark — 1Y / 3Y / 5Y / 7Y (ppt)", fill=fills_mh)
                    if figMH is not None:
                        png = figMH.to_image(format="png", scale=2)
                        img = ImageReader(BytesIO(png))
                        c.setFont("Helvetica-Bold", 14)
                        c.drawString(40, H - 40, "Relative CAGR vs Benchmark — 1Y / 3Y / 5Y / 7Y (ppt)")
                        max_w, max_h = W - 80, H - 100
                        iw, ih = img.getSize()
                        scale = min(max_w/iw, max_h/ih)
                        c.drawImage(img, 40, 40, width=iw*scale, height=ih*scale)
                        c.showPage()

                c.save()
                pdf_bytes.seek(0)
                st.download_button(
                    label="Download PDF",
                    file_name=f"Performance_{focus_fund.replace(' ','_')}.pdf",
                    mime="application/pdf",
                    data=pdf_bytes.getvalue(),
                )
            except Exception as e:
                st.error(f"PDF generation failed: {e}. Ensure 'kaleido' and 'reportlab' are installed.")


# New page: Portfolio quality split into two sections with conditionals to optimize performmance - START

def portfolio_quality_page():
    home_button()
    st.header("Portfolio quality – return on capital & quality buckets")

    # --------------------------------------------------
    # 1–3: Categories → Universe → Focus fund
    # (OUTSIDE the forms, dynamic & hierarchical)
    # --------------------------------------------------
    (
        selected_fund_ids,
        selected_fund_labels,
        focus_fund_id,
        focus_fund_label,
        fund_options,
    ) = quality_category_and_fund_selector()

    if not selected_fund_ids:
        # helper already shows guidance
        return

    # --------------------------------------------------
    # 4–6: Period + Analysis mode + first UPDATE
    # (INSIDE the first form)
    # --------------------------------------------------
    with st.form("pq_filters"):
        # 4) Period
        start_date, end_date = quality_period_selector()

        # 5) Analysis mode
        st.subheader("5. Choose analysis")
        view_mode = st.radio(
            "What do you want to analyse?",
            options=[
                "Return on capital vs peers / universe",
                "Quality quartile exposures (Q1–Q4)",
            ],
            horizontal=True,
            key="pq_view_mode",
        )

        # 6) First update button
        update_main = st.form_submit_button("Update", type="primary")

    # Track whether filters have been applied at least once
    filters_applied = st.session_state.get("pq_filters_applied", False)
    if update_main:
        filters_applied = True
        st.session_state["pq_filters_applied"] = True

    # Basic validation
    if start_date > end_date:
        st.error("Start date must be earlier than end date.")
        return

    # If user hasn’t applied filters (period + mode) at least once, stop here
    if not filters_applied:
        st.info("Set period and analysis mode, then click **Update** above.")
        return

    # --------------------------------------------------
    # SECOND STAGE: Mode-specific second form
    # Segment + comparison + Show results for Mode 1
    # Show results only for Mode 2
    # --------------------------------------------------

    # MODE 1: Return on capital vs peers / universe
    if view_mode == "Return on capital vs peers / universe":
        with st.form("pq_mode1"):
            # 7.1.a – Segment selector (inside form now)
            st.subheader("6. Segment")
            segment_choice = st.radio(
                "Show metrics for:",
                options=["Financials", "Non-financials", "Total"],
                horizontal=True,
                key="pq_segment",
            )

            # 7.1.b – Comparison mode (inside same form)
            st.subheader("7. Comparison mode")
            comparison_mode = st.radio(
                "Compare focus fund against:",
                options=["Universe median", "Individual funds"],
                horizontal=True,
                key="pq_comparison_mode",
            )

            # 8) Second update: Show results
            run_charts = st.form_submit_button("Show results", type="primary")

        if not run_charts:
            # User is still adjusting segment/comparison; don't compute yet
            return

        # Heavy section only runs after Show results
        render_quality_roc_section(
            selected_fund_ids=selected_fund_ids,
            selected_fund_labels=selected_fund_labels,
            focus_fund_id=focus_fund_id,
            focus_fund_label=focus_fund_label,
            start_date=start_date,
            end_date=end_date,
            segment_choice=segment_choice,
            comparison_mode=comparison_mode,
        )
        return

    # MODE 2: Quality quartile exposures (Q1–Q4)
    if view_mode == "Quality quartile exposures (Q1–Q4)":
        with st.form("pq_mode2"):
            st.markdown(
                "**Analysis:** Quality quartile exposures for the focus fund "
                "(total domestic equities)."
            )
            # 8) Second update: Show results (no extra selectors)
            run_charts = st.form_submit_button("Show results", type="primary")

        if not run_charts:
            # User hasn’t asked to see results yet
            return

        render_quality_quartiles_section(
            selected_fund_ids=selected_fund_ids,
            selected_fund_labels=selected_fund_labels,
            focus_fund_id=focus_fund_id,        # chosen at the top
            focus_fund_label=focus_fund_label,
            start_date=start_date,
            end_date=end_date,
            segment_choice="Total",             # quartiles are total domestic equities
            fund_options=fund_options,
        )
        return


# New page: Portfolio quality split into two sections with conditionals to optimize performmance - END


def portfolio_valuations_page():
    home_button()
    st.header("Portfolio valuations")

    # 1) Category selector (checkboxes)
    categories = fetch_categories()
    if not categories:
        st.warning("No categories found in fund_master.")
        return

    st.subheader("1. Select categories")
    selected_categories = []
    cols = st.columns(min(4, len(categories)))
    for i, cat in enumerate(categories):
        col = cols[i % len(cols)]
        if col.checkbox(cat, value=False, key=f"pv_cat_{cat}"):
            selected_categories.append(cat)

    if not selected_categories:
        st.info("Please select at least one category.")
        return

    # 2) Fund multi-select with "All" option
    st.subheader("2. Select funds")

    funds_df = fetch_funds_for_categories(selected_categories)
    if funds_df.empty:
        st.warning("No funds found for selected categories.")
        return

    fund_options = {
        f"{row['fund_name']} ({row['category_name']})": row["fund_id"]
        for _, row in funds_df.iterrows()
    }

    all_option = "All"
    multiselect_options = [all_option] + list(fund_options.keys())

    selected_raw_labels = st.multiselect(
        "Funds",
        options=multiselect_options,
        default=[],
        key="pv_funds_multiselect",
    )

    if all_option in selected_raw_labels:
        selected_fund_labels = list(fund_options.keys())
    else:
        selected_fund_labels = [
            label for label in selected_raw_labels if label != all_option
        ]

    selected_fund_ids = [fund_options[label] for label in selected_fund_labels]

    if not selected_fund_ids:
        st.info("Please select at least one fund.")
        return

    # 3) Portfolio valuations controls
    st.subheader("3. Valuation settings")

    # 3.a Focus fund: single-select from already selected funds
    focus_fund_label = st.selectbox(
        "Focus fund",
        options=selected_fund_labels,
        index=0,
        key="pv_focus_fund",
    )
    focus_fund_id = fund_options[focus_fund_label]

    current_year = dt.date.today().year
    years_val = list(range(current_year - 15, current_year + 1))
    months_val = list(range(1, 13))

    def month_name(m: int) -> str:
        return dt.date(2000, m, 1).strftime("%b")

    # === Form: period + mode + segment + metric + Update ===
    st.subheader("4. Period and valuation options")

    with st.form("pv_controls"):
        colv1, colv2 = st.columns(2)
        with colv1:
            val_start_year = st.selectbox(
                "Valuation start year",
                options=years_val,
                index=0,
                key="pv_val_start_year",
            )
            val_start_month = st.selectbox(
                "Valuation start month",
                options=months_val,
                index=0,
                key="pv_val_start_month",
                format_func=month_name,
            )
        with colv2:
            val_end_year = st.selectbox(
                "Valuation end year",
                options=years_val,
                index=len(years_val) - 1,
                key="pv_val_end_year",
            )
            val_end_month = st.selectbox(
                "Valuation end month",
                options=months_val,
                index=dt.date.today().month - 1,
                key="pv_val_end_month",
                format_func=month_name,
            )

        # Convert to month-end dates inside the form
        val_start_date = month_year_to_last_day(val_start_year, val_start_month)
        val_end_date = month_year_to_last_day(val_end_year, val_end_month)

        val_mode = st.radio(
            "Valuation mode",
            options=[
                "Valuations of historical portfolios",
                "Historical valuations of current portfolio",
            ],
            horizontal=False,
            key="pv_val_mode",
        )

        val_segment = st.radio(
            "Segment for valuations",
            options=["Financials", "Non-financials", "Total"],
            horizontal=True,
            key="pv_val_segment",
        )

        val_metric = st.radio(
            "Valuation metric",
            options=["P/S", "P/B", "P/E"],
            horizontal=True,
            key="pv_val_metric",
        )

        run_charts = st.form_submit_button("Update valuations", type="primary")

    # Do not compute anything until the user clicks Update
    if not run_charts:
        st.info("Adjust filters above and click **Update valuations** to see results.")
        return

    # Basic validation
    if val_start_date > val_end_date:
        st.error("Valuation start date must be earlier than end date.")
        return

    # 5) Compute valuations (precompute all combinations into cache)
    st.subheader("5. Valuation time series")

    # Define the universe of combinations we want to precompute
    modes_to_precompute = [
        "Valuations of historical portfolios",
        "Historical valuations of current portfolio",
    ]
    segments_to_precompute = ["Financials", "Non-financials", "Total"]
    metrics_to_precompute = ["P/S", "P/B", "P/E"]

    with st.spinner("Computing portfolio valuations (warming cache for all combinations)..."):
        # Warm the cache for all combinations for this universe + period
        for m in modes_to_precompute:
            for seg in segments_to_precompute:
                for met in metrics_to_precompute:
                    _ = cached_portfolio_valuations_timeseries(
                        fund_ids=selected_fund_ids,
                        focus_fund_id=focus_fund_id,
                        start_date=val_start_date,
                        end_date=val_end_date,
                        segment_choice=seg,
                        metric_choice=met,
                        mode=m,
                    )

        # Now fetch the specific combination the user asked for
        try:
            df_val = cached_portfolio_valuations_timeseries(
                fund_ids=selected_fund_ids,
                focus_fund_id=focus_fund_id,
                start_date=val_start_date,
                end_date=val_end_date,
                segment_choice=val_segment,
                metric_choice=val_metric,
                mode=val_mode,
            )
        except ValueError as ve:
            st.error(str(ve))
            return
        except Exception as e:
            st.error(f"Error while computing valuations: {e}")
            return

    if df_val.empty:
        st.info("No valuation data available for the selected filters.")
        return

    df_val["month_end"] = pd.to_datetime(df_val["month_end"])

    val_chart = (
        alt.Chart(df_val)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "month_end:T",
                title="Period",
                axis=alt.Axis(format="%b %Y", labelAngle=-45),
            ),
            y=alt.Y(
                "value:Q",
                title=f"{val_metric} (x)",
            ),
            color=alt.Color("series:N", title="Series"),
            tooltip=[
                alt.Tooltip("month_end:T", title="Period", format="%b %Y"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("value:Q", title=f"{val_metric} (x)", format=".2f"),
            ],
        )
        .properties(height=400)
    )
    st.altair_chart(val_chart, use_container_width=True)

def fund_manager_tenure_page():
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    home_button()
    st.subheader("Fund manager tenure")

    # -----------------------------
    # Nested helper: render chart (one row per manager, multiple stints)
    # Correct "current" logic:
    #   - If any open-ended (to_date NULL): current = those rows
    #   - Else: current = rows with latest RECORDED to_date
    # Ongoing assumption: current rows extend to today for display/tenure
    # -----------------------------
    def _render_tenure_chart(df_focus: pd.DataFrame):
        if df_focus.empty:
            st.info("No fund manager tenure data available for this fund.")
            return

        df = df_focus.copy()
        df["from_date"] = pd.to_datetime(df["from_date"])
        df["to_date"] = pd.to_datetime(df["to_date"])

        today = pd.Timestamp.today().normalize()

        # Base fill for charting convenience
        df["to_date_filled"] = df["to_date"].fillna(today)

        # Determine current rows + last update recorded
        has_open_ended = df["to_date"].isna().any()
        if has_open_ended:
            df["stint_is_current"] = df["to_date"].isna()
            last_update_recorded = df["to_date"].dropna().max()
            # open-ended already extends to today via fillna
        else:
            last_update_recorded = df["to_date"].dropna().max()
            df["stint_is_current"] = df["to_date"].eq(last_update_recorded)

            # Ongoing assumption: latest recorded stint continues to today
            df.loc[df["stint_is_current"], "to_date_filled"] = today

        # ---- Group by manager: one row per manager; split segments into current vs other ----
        grouped = []
        for mgr, g in df.groupby("fund_manager"):
            g = g.sort_values(["from_date", "to_date_filled"])

            segs_current = []
            segs_other = []

            for r in g.itertuples(index=False):
                start = r.from_date.to_pydatetime()
                end = r.to_date_filled.to_pydatetime()

                s = mdates.date2num(start)
                e = mdates.date2num(end)
                width = max(0.1, e - s)

                if bool(r.stint_is_current):
                    segs_current.append((s, width))
                else:
                    segs_other.append((s, width))

            grouped.append(
                {
                    "fund_manager": mgr,
                    "earliest": g["from_date"].min(),
                    "segs_current": segs_current,
                    "segs_other": segs_other,
                }
            )

        # Order rows: oldest manager at top
        grouped = sorted(grouped, key=lambda x: x["earliest"])

        fig_h = max(2.5, 0.55 * len(grouped))
        fig, ax = plt.subplots(figsize=(12, fig_h))

        height = 0.7
        highlight_color = "#1f77b4"
        muted_alpha = 0.25

        for idx, item in enumerate(grouped):
            y = (len(grouped) - 1) - idx

            if item["segs_other"]:
                ax.broken_barh(item["segs_other"], (y - height / 2, height), alpha=muted_alpha)

            if item["segs_current"]:
                ax.broken_barh(
                    item["segs_current"],
                    (y - height / 2, height),
                    facecolors=highlight_color,
                    alpha=1.0,
                )

            all_segs = (item["segs_other"] or []) + (item["segs_current"] or [])
            x_left = min(s for s, w in all_segs)
            ax.text(x_left, y, str(item["fund_manager"]), va="center", ha="left", fontsize=9)

        # X-axis formatting: always Mmm-YYYY
        locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        ax.set_yticks([])
        ax.set_ylim(-1, len(grouped))
        ax.set_xlabel("")
        ax.grid(axis="x", alpha=0.2)

        st.pyplot(fig, clear_figure=True)

        # ---- Current manager summary: include ALL current managers and their tenures ----
        cur = df[df["stint_is_current"]].copy()
        if not cur.empty:
            cur["tenure_years"] = (today - cur["from_date"]).dt.days / 365.25
            # If a manager has multiple current rows (rare), use earliest start for that manager
            cur_mgr = (
                cur.groupby("fund_manager", as_index=False)
                   .agg(from_date=("from_date", "min"), tenure_years=("tenure_years", "max"))
            )
            cur_mgr["tenure_years"] = cur_mgr["tenure_years"].round(1)
            # Sort by longer tenure first (optional)
            cur_mgr = cur_mgr.sort_values("tenure_years", ascending=False)
            parts = [f"{r.fund_manager} managing since the past {r.tenure_years:.1f} years" for r in cur_mgr.itertuples(index=False)]
            st.caption("Current fund manager: " + "; ".join(parts))

        # ---- Last tenure update on Supabase (latest recorded to_date) ----
        if pd.notna(last_update_recorded):
            st.caption("Last tenure update on Supabase: " + pd.to_datetime(last_update_recorded).strftime("%b-%Y"))
        else:
            st.caption("Last tenure update on Supabase: Not available (no non-null 'to date' recorded)")

    # -----------------------------
    # Step 1: Category selection + mode selection (single form)
    # -----------------------------
    categories = fetch_categories()
    if not categories:
        st.error("No categories available.")
        return

    if "fm_selected_categories" not in st.session_state:
        st.session_state["fm_selected_categories"] = []
    if "fm_categories_submitted" not in st.session_state:
        st.session_state["fm_categories_submitted"] = False
    if "fm_mode" not in st.session_state:
        st.session_state["fm_mode"] = "Single fund manager history"
    if "fm_min_tenure" not in st.session_state:
        st.session_state["fm_min_tenure"] = 5.0

    with st.form("fm_category_form"):
        st.subheader("1. Select categories")

        selected = []
        cols = st.columns(min(4, len(categories)))
        for i, cat in enumerate(categories):
            with cols[i % len(cols)]:
                key = f"fm_cat_{cat}"
                default_checked = cat in st.session_state["fm_selected_categories"]
                checked = st.checkbox(cat, value=default_checked, key=key)
                if checked:
                    selected.append(cat)

        st.subheader("2. Choose mode")
        mode = st.radio(
            "View",
            options=["Single fund manager history", "Filter funds based on tenure"],
            index=0 if st.session_state["fm_mode"] == "Single fund manager history" else 1,
            key="fm_mode_radio",
        )

        if mode == "Filter funds based on tenure":
            min_tenure = st.number_input(
                "Minimum required tenure (years)",
                min_value=0.0,
                step=0.5,
                value=float(st.session_state["fm_min_tenure"]),
                key="fm_min_tenure_input",
            )

        submitted = st.form_submit_button("Submit")

    if submitted:
        st.session_state["fm_selected_categories"] = selected
        st.session_state["fm_categories_submitted"] = True
        st.session_state["fm_mode"] = st.session_state["fm_mode_radio"]
        if st.session_state["fm_mode"] == "Filter funds based on tenure":
            st.session_state["fm_min_tenure"] = float(st.session_state["fm_min_tenure_input"])
        # reset focus fund when user re-submits
        st.session_state.pop("fm_focus_fund_name", None)

    if not st.session_state["fm_categories_submitted"]:
        return

    selected_categories = st.session_state["fm_selected_categories"]
    if not selected_categories:
        st.warning("Please select at least one category.")
        return

    # -----------------------------
    # Step 2: Pull universe + tenure (cached)
    # -----------------------------
    df_funds = fetch_funds_by_categories(selected_categories)
    if df_funds.empty:
        st.info("No funds found for selected categories.")
        return

    df_tenure = fetch_fund_manager_tenure(df_funds["fund_id"].astype(int).tolist())
    if df_tenure.empty:
        st.info("No fund manager tenure data available.")
        return

    # -----------------------------
    # MODE B: Filter funds based on tenure
    # Include ALL current managers with latest to_date (or NULL to_date if any)
    # -----------------------------
    if st.session_state["fm_mode"] == "Filter funds based on tenure":
        min_years = float(st.session_state.get("fm_min_tenure", 5.0))
        today = pd.Timestamp.today().normalize()

        d = df_tenure.copy()
        d["from_date"] = pd.to_datetime(d["from_date"])
        d["to_date"] = pd.to_datetime(d["to_date"])

        # Per-fund open-ended flag
        has_open = d.groupby("fund_id")["to_date"].apply(lambda s: s.isna().any())
        d = d.merge(has_open.rename("has_open_ended"), on="fund_id", how="left")

        # Latest recorded to_date per fund (ignoring NaT)
        latest_recorded_to = d.groupby("fund_id")["to_date"].transform(lambda s: s.dropna().max())
        d["latest_recorded_to"] = latest_recorded_to

        # Current rows per fund:
        d["is_current_row"] = False
        d.loc[d["has_open_ended"] & d["to_date"].isna(), "is_current_row"] = True
        d.loc[(~d["has_open_ended"]) & d["to_date"].eq(d["latest_recorded_to"]), "is_current_row"] = True

        cur = d[d["is_current_row"]].copy()
        if cur.empty:
            st.info("No current manager rows found.")
            return

        # Ongoing assumption: current rows continue to today
        cur["tenure_years"] = (today - cur["from_date"]).dt.days / 365.25
        cur = cur[cur["tenure_years"] >= min_years].copy()
        cur["tenure_years"] = cur["tenure_years"].round(1)

        # Attach fund_name and output 3 columns as requested
        cur = cur.merge(df_funds[["fund_id", "fund_name"]], on="fund_id", how="left")
        cur = cur.dropna(subset=["fund_name"])

        out = cur[["fund_name", "fund_manager", "tenure_years"]].copy()
        out = out.sort_values(["fund_name", "tenure_years"], ascending=[True, False]).reset_index(drop=True)

        st.markdown(f"### Funds where current manager tenure is ≥ {min_years:.1f} years")
        if out.empty:
            st.info("No fund-manager pairs match the minimum tenure filter.")
        else:
            st.dataframe(out, use_container_width=True)

        return

    # -----------------------------
    # MODE A: Single fund manager history (chart)
    # -----------------------------
    fund_names = df_funds["fund_name"].tolist()
    fund_name_to_id = dict(zip(df_funds["fund_name"], df_funds["fund_id"]))

    st.subheader("3. Select focus fund")

    if "fm_focus_fund_name" not in st.session_state:
        st.session_state["fm_focus_fund_name"] = fund_names[0]
    if st.session_state["fm_focus_fund_name"] not in fund_names:
        st.session_state["fm_focus_fund_name"] = fund_names[0]

    focus_fund = st.selectbox(
        "Focus fund",
        options=fund_names,
        index=fund_names.index(st.session_state["fm_focus_fund_name"]),
        key="fm_focus_fund_selectbox",
    )
    st.session_state["fm_focus_fund_name"] = focus_fund

    focus_fund_id = int(fund_name_to_id[focus_fund])
    df_focus = df_tenure[df_tenure["fund_id"] == focus_fund_id].copy()

    if df_focus.empty:
        st.info("No tenure rows available for the selected focus fund.")
        return

    st.markdown(f"### Fund manager history – **{focus_fund}**")
    _render_tenure_chart(df_focus)

    with st.expander("Show underlying tenure rows"):
        show = df_focus.copy()
        show["from_date"] = pd.to_datetime(show["from_date"]).dt.strftime("%b-%Y")
        show["to_date"] = pd.to_datetime(show["to_date"]).dt.strftime("%b-%Y")
        show.loc[show["to_date"].isna(), "to_date"] = "Current"
        st.dataframe(show[["fund_manager", "from_date", "to_date"]], use_container_width=True)



def portfolio_page():
    home_button()
    st.header("Portfolio explorer")

    # Mode selector
    mode = st.selectbox("Mode", ["View portfolio", "Active share"])

    if mode == "View portfolio":
        portfolio_view_subpage()
    else:
        active_share_subpage()


def portfolio_view_subpage():
    # === View portfolio ===
    import sqlalchemy as sa

    categories = fetch_categories()
    if not categories:
        st.warning("No categories found.")
        return

    # -----------------------------
    # Frequency snapshot selector (local)
    # -----------------------------
    def _apply_frequency_snapshots(df_in: pd.DataFrame, freq: str) -> pd.DataFrame:
        if df_in.empty:
            return df_in

        df = df_in.copy()
        df["month_end"] = pd.to_datetime(df["month_end"]).dt.to_period("M").dt.to_timestamp("M")

        if freq == "Monthly":
            return df

        if freq == "Quarterly":
            df["grp"] = df["month_end"].dt.to_period("Q")
        elif freq == "Yearly":
            df["grp"] = df["month_end"].dt.to_period("Y")
        else:
            return df

        last_me = df.groupby("grp")["month_end"].transform("max")
        df = df[df["month_end"].eq(last_me)].copy()
        df.drop(columns=["grp"], inplace=True, errors="ignore")
        return df

    # -----------------------------
    # UI: Categories
    # -----------------------------
    st.subheader("1. Select categories")
    selected_categories = []
    cols = st.columns(min(4, len(categories)))
    for i, cat in enumerate(categories):
        col = cols[i % len(cols)]
        if col.checkbox(cat, value=False, key=f"port_cat_{cat}"):
            selected_categories.append(cat)

    if not selected_categories:
        st.info("Select at least one category to continue.")
        return

    funds_df = fetch_funds_for_categories(selected_categories)
    if funds_df.empty:
        st.warning("No funds found for the selected categories.")
        return

    # -----------------------------
    # UI: Fund
    # -----------------------------
    st.subheader("2. Select fund")
    cat_col = "category_name" if "category_name" in funds_df.columns else "category"

    fund_options = {
        f"{row['fund_name']} ({row[cat_col]})": row["fund_id"]
        for _, row in funds_df.iterrows()
    }

    fund_label = st.selectbox("Fund", options=list(fund_options.keys()))
    fund_id = int(fund_options[fund_label])

    # -----------------------------
    # UI: Period
    # -----------------------------
    st.subheader("3. Select period")

    current_year = dt.date.today().year
    years = list(range(current_year - 15, current_year + 1))
    month_options = list(range(1, 13))
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    col1, col2 = st.columns(2)
    with col1:
        start_year = st.selectbox("Start year", options=years, index=0, key="port_start_year")
        start_month = st.selectbox(
            "Start month",
            options=month_options,
            index=0,
            key="port_start_month",
            format_func=lambda m: month_names[m - 1],
        )
    with col2:
        end_year = st.selectbox("End year", options=years, index=len(years) - 1, key="port_end_year")
        end_month = st.selectbox(
            "End month",
            options=month_options,
            index=11,
            key="port_end_month",
            format_func=lambda m: month_names[m - 1],
        )

    start_date = month_year_to_last_day(start_year, start_month)
    end_date = month_year_to_last_day(end_year, end_month)

    if start_date > end_date:
        st.error("Start date must be earlier than end date.")
        return

    # -----------------------------
    # UI: Frequency
    # -----------------------------
    st.subheader("4. Frequency")
    freq = st.radio("Aggregation", options=["Monthly", "Quarterly", "Yearly"], horizontal=True)

    # -----------------------------
    # Formatting helper
    # -----------------------------
    def _fmt(x):
        try:
            if pd.isna(x) or abs(float(x)) < 0.0001:
                return "-"
            return f"{float(x):.1f}"
        except Exception:
            return "-"

    # -----------------------------
    # Action
    # -----------------------------
    if st.button("Show portfolio"):
        with st.spinner("Loading portfolio..."):
            engine = get_engine()

            # Use actual schema columns + aliases to keep downstream logic stable
            query = """
                SELECT
                    fund_id,
                    month_end,
                    instrument_name AS company_name,
                    isin,
                    asset_type,
                    holding_weight AS weight_pct
                FROM fundlab.fund_portfolio
                WHERE fund_id = :fund_id
                  AND month_end >= :start_date
                  AND month_end <= :end_date
            """
            port_df = pd.read_sql(
                sa.text(query),
                engine,
                params={"fund_id": fund_id, "start_date": start_date, "end_date": end_date},
            )

            if port_df.empty:
                st.warning("No portfolio data found for this fund and period.")
                return

            # normalize types
            port_df["month_end"] = pd.to_datetime(port_df["month_end"]).dt.to_period("M").dt.to_timestamp("M")
            port_df["weight_pct"] = pd.to_numeric(port_df["weight_pct"], errors="coerce").fillna(0.0)
            port_df["company_name"] = port_df["company_name"].fillna("").astype(str)

            # Apply frequency snapshots for instrument table
            port_snap = _apply_frequency_snapshots(port_df, freq)
            if port_snap.empty:
                st.warning("No portfolio snapshots found after applying the selected frequency.")
                return

            # Pull size band using correct schema column name: band_date
            sb_query = """
                SELECT
                    isin,
                    band_date AS month_end,
                    size_band
                FROM fundlab.stock_size_band
                WHERE band_date >= :start_date
                  AND band_date <= :end_date
            """
            size_band_df = pd.read_sql(
                sa.text(sb_query),
                engine,
                params={"start_date": start_date, "end_date": end_date},
            )

            if not size_band_df.empty:
                size_band_df["month_end"] = pd.to_datetime(size_band_df["month_end"]).dt.to_period("M").dt.to_timestamp("M")
                size_band_df["isin"] = size_band_df["isin"].fillna("").astype(str)

        # -----------------------------
        # Instrument-level pivot + TOTAL row
        # -----------------------------
        port_snap["period"] = port_snap["month_end"].dt.strftime("%b %Y")

        period_order_df = (
            port_snap[["period", "month_end"]]
            .drop_duplicates()
            .sort_values("month_end")
        )
        period_order = period_order_df["period"].tolist()

        pivot = port_snap.pivot_table(
            index="company_name",
            columns="period",
            values="weight_pct",
            aggfunc="sum",
            fill_value=0.0,
        ).reindex(columns=period_order)

        # Sort rows by latest period weight (desc)
        latest_period = period_order[-1]
        if latest_period in pivot.columns:
            pivot = pivot.sort_values(by=latest_period, ascending=False)

        # Add TOTAL row (numeric) BEFORE formatting
        total_series = pivot.sum(axis=0)
        pivot = pd.concat([pivot, pd.DataFrame([total_series], index=["Total"])])

        df_display = pivot.reset_index()  # company_name is column now

        # Ensure the first column is named 'company_name' after reset_index()
        first_col = df_display.columns[0]
        if first_col != "company_name":
            df_display = df_display.rename(columns={first_col: "company_name"})


        # Insert S.No. with blank for Total row
        df_display.insert(0, "S.No.", range(1, len(df_display) + 1))
        df_display.loc[df_display["company_name"] == "Total", "S.No."] = ""

        # Format weights as strings with 1 decimal, '-' for tiny/zero
        for c in df_display.columns:
            if c not in ["S.No.", "company_name"]:
                df_display[c] = df_display[c].apply(_fmt)

        st.subheader("5. Portfolio holdings:")
        st.caption(
            f"Rows: instruments · Columns: {freq.lower()} snapshots from {period_order[0]} to {period_order[-1]}"
        )

        # Freeze through 3rd column (S.No., company_name, first period)
        render_sticky_first_col_table(df_display, height_px=520, freeze_cols=3)

        # -----------------------------
        # Allocation table (uses your global helper with Jun/Dec carry-forward) + TOTAL row
        # -----------------------------
        st.subheader("6. Size / asset-type allocation")

        alloc_df = build_size_asset_allocation_pivot(
            holdings=port_df,
            size_band=size_band_df,
            freq=freq,
            period_col="month_end",
        )

        if alloc_df.empty:
            st.info("No allocation data available.")
            return

        # Add TOTAL row (numeric) BEFORE formatting
        num_cols = [c for c in alloc_df.columns if c != "Allocation"]
        alloc_num = alloc_df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        totals = alloc_num.sum(axis=0)
        alloc_df = pd.concat(
            [alloc_df, pd.DataFrame([{"Allocation": "Total", **totals.to_dict()}])],
            ignore_index=True,
        )

        # Format
        for c in alloc_df.columns:
            if c != "Allocation":
                alloc_df[c] = alloc_df[c].apply(_fmt)

        # Freeze through 2nd column (Allocation + first period)
        render_sticky_first_col_table(alloc_df, height_px=320, freeze_cols=2)


def active_share_subpage():
    st.subheader("Active share between two portfolios")

    categories = fetch_categories()
    if not categories:
        st.warning("No categories found.")
        return

    # 1. Category selector (checkboxes)
    st.subheader("1. Select categories")
    selected_categories = []
    cols = st.columns(min(4, len(categories)))
    for i, cat in enumerate(categories):
        col = cols[i % len(cols)]
        if col.checkbox(cat, value=False, key=f"as_cat_{cat}"):
            selected_categories.append(cat)

    if not selected_categories:
        st.info("Select at least one category to continue.")
        return

    funds_df = fetch_funds_for_categories(selected_categories)
    if funds_df.empty:
        st.warning("No funds found for the selected categories.")
        return

    cat_col = "category_name" if "category_name" in funds_df.columns else "category"

    fund_labels = [
        f"{row['fund_name']} ({row[cat_col]})"
        for _, row in funds_df.iterrows()
    ]
    fund_label_to_id = {
        f"{row['fund_name']} ({row[cat_col]})": row["fund_id"]
        for _, row in funds_df.iterrows()
    }

    # 2. Two portfolio selections
    st.subheader("2. Select funds for each portfolio")

    sel_colA, sel_colB = st.columns(2)

    with sel_colA:
        st.markdown("**Portfolio A – Funds**")
        selected_labels_A = st.multiselect(
            "Funds A",
            options=fund_labels,
            default=[],
            key="as_funds_A",
        )
    with sel_colB:
        st.markdown("**Portfolio B – Funds**")
        selected_labels_B = st.multiselect(
            "Funds B",
            options=fund_labels,
            default=[],
            key="as_funds_B",
        )

    if not selected_labels_A or not selected_labels_B:
        st.info("Select at least one fund for each portfolio to continue.")
        return

    fund_ids_A = [fund_label_to_id[label] for label in selected_labels_A]
    fund_ids_B = [fund_label_to_id[label] for label in selected_labels_B]

    # 3. Proportion tables side by side (wider layout, inputs beside names)
    st.subheader("3. Set fund proportions (%) in each portfolio")

    comp_colA, comp_colB = st.columns([1, 1])  # full width, two equal halves

    props_A = {}
    props_B = {}

    with comp_colA:
        st.markdown("**Portfolio A composition**")
        for label in selected_labels_A:
            fund_id = fund_label_to_id[label]
            name_col, input_col = st.columns([4, 1])
            with name_col:
                # Scrollable, fixed-width cell with full name on hover
                st.markdown(
                    f"""
                    <div style="
                        white-space: nowrap;
                        overflow-x: auto;
                        text-overflow: clip;
                        width: 100%;
                    " title="{label}">
                        {label}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with input_col:
                props_A[fund_id] = st.number_input(
                    "",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    step=1.0,
                    key=f"asA_{fund_id}",
                )

    with comp_colB:
        st.markdown("**Portfolio B composition**")
        for label in selected_labels_B:
            fund_id = fund_label_to_id[label]
            name_col, input_col = st.columns([4, 1])
            with name_col:
                st.markdown(
                    f"""
                    <div style="
                        white-space: nowrap;
                        overflow-x: auto;
                        text-overflow: clip;
                        width: 100%;
                    " title="{label}">
                        {label}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with input_col:
                props_B[fund_id] = st.number_input(
                    "",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    step=1.0,
                    key=f"asB_{fund_id}",
                )

    # Live totals shown in a new row so they align horizontally
    sum_A = sum(props_A.values())
    sum_B = sum(props_B.values())

    total_colA, total_colB = st.columns(2)
    with total_colA:
        st.markdown(f"**Total A: {sum_A:.1f}%**")
    with total_colB:
        st.markdown(f"**Total B: {sum_B:.1f}%**")

    # 4. Period selection
    st.subheader("4. Select period")

    current_year = dt.date.today().year
    years = list(range(current_year - 15, current_year + 1))
    month_options = list(range(1, 13))
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    col1, col2 = st.columns(2)
    with col1:
        start_year = st.selectbox(
            "Start year",
            options=years,
            index=0,
            key="as_start_year",
        )
        start_month = st.selectbox(
            "Start month",
            options=month_options,
            index=0,
            key="as_start_month",
            format_func=lambda m: month_names[m - 1],
        )
    with col2:
        end_year = st.selectbox(
            "End year",
            options=years,
            index=len(years) - 1,
            key="as_end_year",
        )
        end_month = st.selectbox(
            "End month",
            options=month_options,
            index=11,
            key="as_end_month",
            format_func=lambda m: month_names[m - 1],
        )

    start_date = month_year_to_last_day(start_year, start_month)
    end_date = month_year_to_last_day(end_year, end_month)

    if start_date > end_date:
        st.error("Start date must be earlier than end date.")
        return

    # 5. Frequency
    st.subheader("5. Frequency")
    freq = st.radio(
        "Aggregation",
        options=["Monthly", "Quarterly", "Yearly"],
        horizontal=True,
        key="as_freq",
    )

    # 6. Calculate active share – heavy work only ON BUTTON CLICK
    if st.button("Calculate active share"):
        # Validate totals only when button is pressed
        sum_A = sum(props_A.values())
        sum_B = sum(props_B.values())

        if abs(sum_A - 100.0) > 0.01:
            st.error(f"Portfolio A proportions must sum to 100. Currently: {sum_A:.1f}%")
            return

        if abs(sum_B - 100.0) > 0.01:
            st.error(f"Portfolio B proportions must sum to 100. Currently: {sum_B:.1f}%")
            return

        # Normalize to 1.0
        norm_props_A = {fid: val / 100.0 for fid, val in props_A.items()}
        norm_props_B = {fid: val / 100.0 for fid, val in props_B.items()}

        with st.spinner("Calculating active share..."):
            all_fund_ids = list(set(fund_ids_A + fund_ids_B))
            df_all = fetch_multi_fund_portfolios(all_fund_ids, start_date, end_date, freq)

            if df_all.empty:
                st.warning("No portfolio data found for the selected funds and period.")
                return

            df_as = compute_active_share_series(df_all, norm_props_A, norm_props_B)

        if df_as.empty:
            st.warning("Could not compute active share for any period.")
            return

        # 7. Line chart + horizontal table
        st.subheader("6. Active share over time")

        df_chart = df_as.dropna(subset=["active_share_pct"]).copy()
        if df_chart.empty:
            st.warning("Active share is NaN for all periods.")
            return

        import altair as alt

        chart = (
            alt.Chart(df_chart)
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    "period_date:T",
                    title="Period",
                    axis=alt.Axis(format="%b %Y", labelAngle=-45),
                ),
                y=alt.Y(
                    "active_share_pct:Q",
                    title="Active share (%)",
                ),
                tooltip=[
                    alt.Tooltip("period_date:T", title="Period", format="%b %Y"),
                    alt.Tooltip("active_share_pct:Q", title="Active share (%)", format=".1f"),
                ],
            )
            .properties(height=300)
        )

        st.altair_chart(chart, use_container_width=True)

        st.subheader("7. Active share table")

        df_show = df_as[["period_label", "active_share_pct"]].copy()
        df_horizontal = df_show.set_index("period_label").T
        df_horizontal.index = ["Active share (%)"]

        st.dataframe(df_horizontal.style.format("{:.1f}"))


# Validate stock prices + market cap upload
STOCK_PRICE_COLS = {
    "isin":       ["isin"],
    "price_date": ["date", "nav date", "price date", "month_end", "month-end"],
    "market_cap": ["market cap", "mcap", "market_cap", "market capitalisation", "market capitalization"],
    "price":      ["stock price", "price", "close", "close price", "last price"],
}
def validate_stock_prices_mc(df_raw: pd.DataFrame):
    """
    Validate stock prices + market cap upload.

    Expected logical roles:
      - isin
      - price_date (dd-mm-yyyy)
      - market_cap
      - price
    """
    required = {"isin", "price_date", "market_cap", "price"}
    df = map_headers(df_raw.copy(), STOCK_PRICE_COLS, required)

    # Clean types
    df["isin"] = df["isin"].astype(str).str.strip()

    # Parse dd-mm-yyyy
    try:
        df["price_date"] = pd.to_datetime(df["price_date"], dayfirst=True).dt.date
    except Exception as e:
        raise ValueError(f"Could not parse dates as dd-mm-yyyy: {e}")

    df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    if df["market_cap"].isna().any():
        raise ValueError("Some market cap values could not be parsed as numbers.")
    if df["price"].isna().any():
        raise ValueError("Some stock price values could not be parsed as numbers.")

    # ------------------------------
    # 🔥 FOREIGN KEY VALIDATION
    # ------------------------------
    valid_isins = load_stock_master_isins()
    isins_in_file = set(df["isin"])
    missing_isins = sorted(isins_in_file - valid_isins)

    if missing_isins:
        raise ValueError(
            "The following ISINs do not exist in Stock Master:\n"
            + "\n".join(missing_isins)
            + "\n\n→ Please upload/update Stock Master first."
        )

    # In-file duplicate check
    dup_keys = df.groupby(["isin", "price_date", "market_cap", "price"]).size()
    dups = dup_keys[dup_keys > 1]
    if not dups.empty:
        raise ValueError(
            f"Duplicate (isin, date, market_cap, price) rows found in file: {len(dups)} duplicates."
        )

    summary = {
        "rows": int(len(df)),
        "unique_isin_date_mcap_price": int(len(dup_keys)),
    }
    return df, summary


def update_db_page():
    home_button()
    st.header("Update underlying data")

    upload_type = st.selectbox(
        "What would you like to update?",
        [
            "Fund NAVs",
            "Benchmark NAVs",
            "Fund portfolios",
            "Fund manager tenure",
            "Stock ISIN, industry, financial/non-financial",
            "Company RoE / RoCE",
            "Stock prices and market cap",
            "Company PAT (quarterly)",
            "Company sales (quarterly)",
            "Company book value (annual)",
        ],
    )

    # Single file upload for the chosen type
    uploaded = st.file_uploader(
        "Upload Excel file",
        type=["xlsx"],
        key=f"upload_{upload_type}",
    )

    # Show expected format preview (extend this function for new types)
    show_expected_format(upload_type)

    if not uploaded:
        st.info("Please upload the appropriate Excel file to continue.")
        return

    # Read the file into a DataFrame
    try:
        df_raw = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Could not read Excel file: {e}")
        return

    # State keys to remember validated data
    state_key_df = f"validated_df_{upload_type}"
    state_key_ok = f"validated_ok_{upload_type}"

    # Dry run button
    if st.button("Validate (dry run)"):
        try:
            if upload_type == "Fund NAVs":
                df_clean, summary = validate_fund_navs(df_raw)
            elif upload_type == "Benchmark NAVs":
                df_clean, summary = validate_bench_navs(df_raw)
            elif upload_type == "Fund portfolios":
                df_clean, summary = validate_fund_portfolios(df_raw)
            elif upload_type == "Stock ISIN, industry, financial/non-financial":
                df_clean, summary = validate_stock_master(df_raw)
            elif upload_type == "Company RoE / RoCE":
                df_clean, summary = validate_roe_roce(df_raw)
            elif upload_type == "Stock prices and market cap":
                df_clean, summary = validate_stock_prices_mc(df_raw)
            elif upload_type == "Company PAT (quarterly)":
                df_clean, summary = validate_quarterly_pat(df_raw)
            elif upload_type == "Company sales (quarterly)":
                df_clean, summary = validate_quarterly_sales(df_raw)
            elif upload_type == "Company book value (annual)":
                df_clean, summary = validate_annual_book_value(df_raw)
            elif upload_type == "Fund manager tenure":
                df_clean, summary = validate_fund_manager_tenure(df_raw)
            else:
                st.error("Unsupported upload type.")
                return

            st.session_state[state_key_df] = df_clean
            st.session_state[state_key_ok] = True

            st.session_state[f"validated_summary_{upload_type}"] = summary


            st.success("Dry run successful. No critical format errors detected.")
            st.write("Summary:")
            st.json(summary)

        except ValueError as ve:
            st.error(f"Validation error: {ve}")
            st.session_state[state_key_ok] = False
            return
        except Exception as e:
            st.error(f"Unexpected error during validation: {e}")
            st.session_state[state_key_ok] = False
            return

    # Upload button (only if validation passed)
        # Upload button (only if validation passed)
    if st.session_state.get(state_key_ok):
        df_clean = st.session_state.get(state_key_df)
        summary = st.session_state.get(f"validated_summary_{upload_type}", {}) or {}

        resolutions = {}
        can_proceed = True
        block_reason = ""

        if upload_type == "Fund manager tenure":
            missing = summary.get("missing_funds", []) or []

            if missing:
                st.warning("Some funds in this upload are not present in the fund master. Resolve them below before uploading.")

                # Build a fresh set of existing fund names for validating old-name inputs
                engine2 = get_engine()
                existing_names = set(pd.read_sql("select fund_name from fundlab.fund", engine2)["fund_name"].astype(str).str.strip())

                categories = fetch_categories()
                if not categories:
                    can_proceed = False
                    block_reason = "No categories available in fundlab.category (excluding 'portfolio')."

                with st.expander("Resolve missing fund names (rename or create)", expanded=True):
                    for new_name in missing:
                        st.markdown(f"**{new_name}**")
                        c1, c2 = st.columns([2, 2])

                        old_name = c1.text_input(
                            "Old name (optional – fill only if this is a rename)",
                            key=f"fm_old_{new_name}",
                            placeholder="e.g., ICICI Value Discovery",
                        ).strip()

                        if old_name == "":
                            cat_sel = c2.selectbox(
                                "Category (required for new fund)",
                                options=["-- Select --"] + categories,
                                key=f"fm_cat_{new_name}",
                            )
                            category_name = None if cat_sel == "-- Select --" else cat_sel

                            if category_name is None:
                                can_proceed = False
                                block_reason = f"Select a category for new fund: {new_name}"
                        else:
                            c2.write("Category not required for rename.")
                            category_name = None

                            if old_name not in existing_names:
                                can_proceed = False
                                block_reason = f"Old name not found in fund master: '{old_name}' (for rename to '{new_name}')"

                        resolutions[new_name] = {"old_name": old_name, "category_name": category_name}

        if not can_proceed:
            st.error(block_reason)
            return

        if st.button("Confirm upload to database"):
            if df_clean is None:
                st.error("No validated data found in session. Please run validation again.")
                return

            try:
                if upload_type == "Fund NAVs":
                    upload_fund_navs(df_clean)
                elif upload_type == "Benchmark NAVs":
                    upload_bench_navs(df_clean)
                elif upload_type == "Fund portfolios":
                    upload_fund_portfolios(df_clean)
                elif upload_type == "Stock ISIN, industry, financial/non-financial":
                    upload_stock_master(df_clean)
                elif upload_type == "Company RoE / RoCE":
                    upload_roe_roce(df_clean)
                elif upload_type == "Stock prices and market cap":
                    upload_stock_prices_mc(df_clean)
                elif upload_type == "Company PAT (quarterly)":
                    upload_quarterly_pat(df_clean)
                elif upload_type == "Company sales (quarterly)":
                    upload_quarterly_sales(df_clean)
                elif upload_type == "Company book value (annual)":
                    upload_annual_book_value(df_clean)
                elif upload_type == "Fund manager tenure":
                    upload_fund_manager_tenure(df_clean, resolutions)

                st.success("✅ Upload completed successfully.")

            except SQLAlchemyError as e:
                msg = str(getattr(e, "orig", e))
                lower_msg = msg.lower()

                if "foreign key constraint" in lower_msg:
                    st.error(
                        "❌ Some ISINs in this file do not exist in Stock Master.\n"
                        "Please update Stock Master before uploading this data.\n\n"
                        f"Database message: {msg}"
                    )
                elif "duplicate key value" in lower_msg or "unique constraint" in lower_msg:
                    st.error(
                        "❌ Duplicate rows detected against existing database records.\n"
                        "These ISIN + period combinations already exist.\n\n"
                        f"Database message: {msg}"
                    )
                else:
                    st.error(
                        f"Database error while uploading:\n"
                        f"{e.__class__.__name__}: {msg}"
                    )
            except Exception as e:
                st.error(f"Unexpected error during upload: {e}")




# Housekeeping page
def housekeeping_page():
    home_button()
    st.header("Housekeeping – Derived Tables")

    st.write(
        "Run these steps after uploading new raw data (NAVs, portfolios, RoE/RoCE, prices) "
        "to refresh derived tables."
    )

    if st.button("1. Recompute size bands (Large/Mid/Small)"):
        recompute_size_bands()
        st.success("Size bands updated.")

    if st.button("2. Recompute 5-year median RoE/RoCE"):
        recompute_quality_medians()
        st.success("5-year medians updated.")

    if st.button("3. Recompute quality quartiles (Q1–Q4)"):
        recompute_quality_quartiles()
        st.success("Quality quartiles updated.")
    
    if st.button("4. Refresh stock valuations"):
        rebuild_stock_monthly_valuations()
        st.success("Stock valuations updated")   

    if st.button("5. Refresh fund valuations"):
        rebuild_fund_monthly_valuations()
        st.success("Fund valuations updated")

    st.markdown("---")
    st.subheader("Upload precomputed stock valuations to DB")

    uploaded_val_file = st.file_uploader(
        "Select a stock valuations Excel workbook (.xlsx)",
        type=["xlsx"],
        key="stock_val_upload",
    )

    if uploaded_val_file is not None:
        st.write(f"Selected file: **{uploaded_val_file.name}**")

    if st.button("6. Upload this workbook to Supabase"):
        upload_stock_monthly_valuations_from_excel(uploaded_val_file)

    # ---------------------------------------------------------------
    # Debug valuation point (sanity-check tool)
    # ---------------------------------------------------------------
    with st.expander("🔍 Debug valuation point (advanced)", expanded=False):
        dbg_fund = st.number_input("Fund ID", min_value=1, step=1)
        dbg_date = st.date_input("Month (any day in month)")
        dbg_segment = st.selectbox("Segment", ["Total", "Financials", "Non-financials"])
        dbg_metric = st.selectbox("Metric", ["P/E", "P/B", "P/S"])
        dbg_mode = st.selectbox(
            "Mode",
            ["Valuations of historical portfolios", "Historical valuations of current portfolio"],
        )

        if st.button("Run valuation debug"):
            debug_portfolio_valuation_point(
                fund_id=int(dbg_fund),
                target_date=dbg_date,
                segment_choice=dbg_segment,
                metric_choice=dbg_metric,
                mode=dbg_mode,
            )

  




# ======================== Fund Attribution Page ========================

CASH_ANNUAL_RETURN = 0.07

BENCH_NAME_NIFTY50  = "NIFTY 50 - TRI"
BENCH_NAME_NIFTY500 = "NIFTY 500 - TRI"
BENCH_NAME_NIFTY100 = "NIFTY 100 - TRI"  # assumed present in DB (you will upload)
BENCH_NAME_MID150   = "Nifty Midcap 150 - TRI"
BENCH_NAME_SMALL250 = "Nifty Smallcap 250 - TRI"

LIKE_FOR_LIKE_BENCH = {
    "Large": BENCH_NAME_NIFTY100,
    "Mid":   BENCH_NAME_MID150,
    "Small": BENCH_NAME_SMALL250,
}

def _to_month_end(d: dt.date) -> dt.date:
    # Convert any date to month-end date
    d = pd.to_datetime(d).date()
    return month_year_to_last_day(d.year, d.month)

def _month_ends_between(start_me: dt.date, end_me: dt.date) -> list:
    idx = pd.date_range(start=pd.Timestamp(start_me), end=pd.Timestamp(end_me), freq="M")
    return [x.date() for x in idx]

def _monthly_cash_return() -> float:
    # 7% annualized -> monthly compounded
    return (1.0 + CASH_ANNUAL_RETURN) ** (1.0/12.0) - 1.0

@st.cache_data(ttl=3600, show_spinner=False)
def _load_attrib_raw_window(fund_id: int, end_month_end: dt.date, lookback_years: int = 10, data_version: str = "v1"):
    """Load raw inputs for attribution for a single fund over [end-LOOKBACK, end]."""
    engine = get_engine()
    end_me = _to_month_end(end_month_end)
    start_me = (pd.Timestamp(end_me) - pd.DateOffset(years=lookback_years)).to_period("M").to_timestamp("M").date()

    # Holdings (all asset types)
    q_hold = text("""
        SELECT fund_id,
               month_end::date AS month_end,
               instrument_name,
               asset_type,
               isin,
               holding_weight
        FROM fundlab.fund_portfolio
        WHERE fund_id = :fund_id
          AND month_end BETWEEN :start_me AND :end_me
        ORDER BY month_end, instrument_name
    """)
    h = pd.read_sql(q_hold, engine, params={"fund_id": fund_id, "start_me": start_me, "end_me": end_me})
    if h.empty:
        return {
            "window_start": start_me,
            "window_end": end_me,
            "holdings": h,
            "prices": pd.DataFrame(),
            "size_band": pd.DataFrame(),
            "bench_nav": pd.DataFrame(),
            "stock_master": pd.DataFrame(),
            "weight_scale": None,
        }

    h["month_end"] = pd.to_datetime(h["month_end"]).dt.to_period("M").dt.to_timestamp("M").dt.date

    # Normalise holding_weight scale to 0-1.
    # Different feeds store weights as decimals (0-1), percent (0-100) or basis points (0-10000).
    hw = pd.to_numeric(h["holding_weight"], errors="coerce")
    hw_nonnull = hw.dropna()

    # Use both per-row magnitudes and per-month totals to infer scale robustly.
    if hw_nonnull.empty:
        weight_scale = 1.0
        h["holding_weight"] = hw
    else:
        per_month_sum = hw_nonnull.groupby(h.loc[hw_nonnull.index, "month_end"]).sum()
        med_sum = float(per_month_sum.median()) if not per_month_sum.empty else float(hw_nonnull.sum())
        max_w = float(hw_nonnull.max())

        # Heuristic scale detection:
        # - If totals look like ~10000 (bps) => divide by 10000
        # - Else if totals look like ~100 (percent) or max weight > 1 => divide by 100
        # - Else already 0-1
        if med_sum > 1500 or max_w > 1500:
            weight_scale = 1.0 / 10000.0
        elif med_sum > 15 or max_w > 1.5:
            weight_scale = 1.0 / 100.0
        else:
            weight_scale = 1.0

        h["holding_weight"] = hw * weight_scale

    # ISIN list (exclude null/blank)
    isins = sorted([x for x in h["isin"].dropna().astype(str).unique().tolist() if x.strip()])

    # Stock master (names)
    if isins:
        q_sm = text("""
            SELECT isin, company_name, industry, is_financial
            FROM fundlab.stock_master
            WHERE isin = ANY(:isins)
        """)
        sm = pd.read_sql(q_sm, engine, params={"isins": isins})
    else:
        sm = pd.DataFrame(columns=["isin", "company_name", "industry", "is_financial"])

    # Prices (monthly)
    if isins:
        q_px = text("""
            SELECT isin, price_date::date AS month_end, price
            FROM fundlab.stock_price
            WHERE isin = ANY(:isins)
              AND price_date BETWEEN :start_me AND :end_me
            ORDER BY isin, price_date
        """)
        px = pd.read_sql(q_px, engine, params={"isins": isins, "start_me": start_me, "end_me": end_me})
        if not px.empty:
            px["month_end"] = pd.to_datetime(px["month_end"]).dt.to_period("M").dt.to_timestamp("M").dt.date
            px["price"] = pd.to_numeric(px["price"], errors="coerce")
    else:
        px = pd.DataFrame(columns=["isin", "month_end", "price"])

    # Size band (monthly)
    if isins:
        q_sz = text("""
            SELECT isin, band_date::date AS month_end, size_band
            FROM fundlab.stock_size_band
            WHERE isin = ANY(:isins)
              AND band_date BETWEEN :start_me AND :end_me
            ORDER BY isin, band_date
        """)
        sz = pd.read_sql(q_sz, engine, params={"isins": isins, "start_me": start_me, "end_me": end_me})
        if not sz.empty:
            sz["month_end"] = pd.to_datetime(sz["month_end"]).dt.to_period("M").dt.to_timestamp("M").dt.date
            sz["size_band"] = sz["size_band"].astype(str)
    else:
        sz = pd.DataFrame(columns=["isin", "month_end", "size_band"])

    # Benchmarks (monthly)
    wanted = [BENCH_NAME_NIFTY50, BENCH_NAME_NIFTY500, BENCH_NAME_NIFTY100, BENCH_NAME_MID150, BENCH_NAME_SMALL250]
    q_b = text("""
        SELECT b.bench_id,
               b.bench_name,
               bn.nav_date::date AS month_end,
               bn.nav_value
        FROM fundlab.benchmark b
        JOIN fundlab.bench_nav bn
          ON b.bench_id = bn.bench_id
        WHERE b.bench_name = ANY(:names)
          AND bn.nav_date BETWEEN :start_me AND :end_me
        ORDER BY b.bench_name, bn.nav_date
    """)
    bn = pd.read_sql(q_b, engine, params={"names": wanted, "start_me": start_me, "end_me": end_me})
    if not bn.empty:
        bn["month_end"] = pd.to_datetime(bn["month_end"]).dt.to_period("M").dt.to_timestamp("M").dt.date
        bn["nav_value"] = pd.to_numeric(bn["nav_value"], errors="coerce")

    return {
        "window_start": start_me,
        "window_end": end_me,
        "holdings": h,
        "prices": px,
        "size_band": sz,
        "bench_nav": bn,
        "stock_master": sm,
        "weight_scale": float(weight_scale),
    }

def _compute_attribution(raw: dict, start_date: dt.date, end_date: dt.date, bench_mode: str):
    """Return (stock_df, hit_df, category_df, diag). Contributions are Rs per 100 base."""
    start_me = _to_month_end(start_date)
    end_me = _to_month_end(end_date)

    months = _month_ends_between(start_me, end_me)
    if len(months) < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"error": "Select at least two month-ends."}

    h = raw["holdings"].copy()
    px = raw["prices"].copy()
    sz = raw["size_band"].copy()
    bn = raw["bench_nav"].copy()
    sm = raw["stock_master"].copy()

    # Filter holdings to selected window (we will ffill)
    h = h[(h["month_end"] >= months[0]) & (h["month_end"] <= months[-1])].copy()

    # Ensure holding_weight is numeric (NO scaling, NO /100 here)
    h["holding_weight"] = pd.to_numeric(h["holding_weight"], errors="coerce").fillna(0.0)

    # --- SAFETY: ensure required columns exist before groupby ---
    missing = [c for c in ["month_end", "holding_weight"] if c not in h.columns]
    if missing:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {
            "error": f"_compute_attribution(): holdings dataframe missing columns: {missing}. "
                    f"Available columns: {list(h.columns)}"
        }

    # asset_type should exist per schema; if not, create a placeholder (logic will still work)
    if "asset_type" not in h.columns:
        h["asset_type"] = "Unknown"

    # instrument_key must exist for the rebasing logic; create it if missing
    if "instrument_key" not in h.columns:
        # Ensure supporting cols exist (fallbacks)
        if "instrument_name" not in h.columns:
            h["instrument_name"] = ""
        if "isin" not in h.columns:
            h["isin"] = np.nan

        isin_str = h["isin"].astype(str)
        h["instrument_key"] = np.where(
            h["isin"].notna() & (isin_str.str.strip() != "") & (isin_str.str.lower() != "nan"),
            isin_str.str.strip(),
            "NOISIN::" + h["instrument_name"].astype(str)
        )

    # Ensure month_end is month-end timestamp (consistent with the rest of the code)
    h["month_end"] = pd.to_datetime(h["month_end"]).dt.to_period("M").dt.to_timestamp("M")


    # Aggregate weights by month/instrument (avoid duplicates)
    w = (
        h.groupby(["month_end", "instrument_key", "asset_type"], as_index=False)["holding_weight"]
        .sum()
    )

    # Pivot to month x instrument matrix
    w_piv = w.pivot_table(
        index="month_end",
        columns="instrument_key",
        values="holding_weight",
        aggfunc="sum"
    )

    # Reindex to full month grid, forward-fill holdings between snapshots, fill missing with 0
    w_piv = w_piv.reindex(months).ffill().fillna(0.0)

    # NORMALIZE EACH MONTH so weights sum to 1.0 using only available rows (stocks + cash)
    row_sum = w_piv.sum(axis=1)

    # Avoid division by zero in months with no data (keep all zeros)
    w_piv = w_piv.div(row_sum.replace(0.0, np.nan), axis=0).fillna(0.0)

    # Identify explicit cash instruments ONLY from asset_type == 'Cash' (no residual cash)
    cash_keys = set(
        w.loc[w["asset_type"].astype(str).str.strip().str.lower() == "cash", "instrument_key"]
        .unique()
        .tolist()
    )

    instr_cols = w_piv.columns.tolist()

    # Start weights (t0) and period ends (t1)
    t0 = months[:-1]
    t1 = months[1:]
    w0_df = w_piv.loc[t0, :].copy()


    # Price pivot for ISINs
    isins = [c for c in instr_cols if not c.startswith("NOISIN::") and not c.startswith("CASH::")]
    px_piv = pd.DataFrame(index=months, columns=isins, dtype=float)
    if not px.empty and isins:
        px2 = px[px["isin"].astype(str).isin(isins)].copy()
        if not px2.empty:
            pxp = px2.pivot_table(index="month_end", columns="isin", values="price", aggfunc="last").reindex(months)
            for c in isins:
                if c in pxp.columns:
                    px_piv[c] = pd.to_numeric(pxp[c], errors="coerce")

    # Benchmark nav pivot
    bn_piv = pd.DataFrame(index=months)
    if not bn.empty:
        bnp = bn.pivot_table(index="month_end", columns="bench_name", values="nav_value", aggfunc="last").reindex(months)
        bn_piv = bnp

    def _bench_period_returns(bench_name: str) -> np.ndarray:
        if bench_name not in bn_piv.columns:
            return np.zeros(len(t1), dtype=float)
        v0 = bn_piv.loc[t0, bench_name].to_numpy(dtype=float)
        v1 = bn_piv.loc[t1, bench_name].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = (v1 / v0) - 1.0
        r = np.where(np.isfinite(r), r, 0.0)
        return r

    r_n50 = _bench_period_returns(BENCH_NAME_NIFTY50)
    r_n500 = _bench_period_returns(BENCH_NAME_NIFTY500)
    r_n100 = _bench_period_returns(BENCH_NAME_NIFTY100)
    r_mid150 = _bench_period_returns(BENCH_NAME_MID150)
    r_small250 = _bench_period_returns(BENCH_NAME_SMALL250)

    # Size band lookup at t0 for ISINs
    size_lookup = {}
    if not sz.empty and isins:
        sz2 = sz[sz["isin"].astype(str).isin(isins)].copy()
        sz2 = sz2[sz2["month_end"].isin(t0)]
        if not sz2.empty:
            size_lookup = sz2.set_index(["month_end", "isin"])["size_band"].to_dict()

    # Asset type lookup per instrument_key (take most recent non-null asset_type)
    asset_map = (
        w.sort_values(["instrument_key", "month_end"])
         .groupby("instrument_key")["asset_type"]
         .last()
         .to_dict()
    )

    # Instrument returns (fund) per period
    r_instr = pd.DataFrame(index=t1, columns=instr_cols, dtype=float)

    # ISIN returns from price; missing => 0
    if isins:
        p0 = px_piv.loc[t0, isins].to_numpy(dtype=float)
        p1 = px_piv.loc[t1, isins].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = (p1 / p0) - 1.0
        r = np.where(np.isfinite(r), r, 0.0)
        r_instr.loc[:, isins] = r

    # Non-ISIN instruments default 0
    for c in instr_cols:
        if c.startswith("NOISIN::"):
            r_instr[c] = 0.0

    # Cash returns (7% annualized monthly)
    cash_r = _monthly_cash_return()
    for c in cash_keys:
        if c in r_instr.columns:
            r_instr[c] = cash_r

    r_instr = r_instr.fillna(0.0)

    # Benchmark returns per instrument
    r_bm = pd.DataFrame(index=t1, columns=instr_cols, dtype=float)

    if bench_mode == "Vs NIFTY 50":
        broad = r_n50
        for c in instr_cols:
            r_bm[c] = broad
    elif bench_mode == "Vs NIFTY 500":
        broad = r_n500
        for c in instr_cols:
            r_bm[c] = broad
    else:
        # Like-for-like:
        # - Cash: benchmarked to itself (same return) => zero alpha
        # - ISINs: large/mid/small mapping (default Nifty50 if missing band or missing series)
        # - Non-ISIN non-cash (e.g., 'Other Equities' without ISIN): benchmark to Nifty 50
        for c in instr_cols:
            if c in cash_keys or c.startswith("CASH::"):
                r_bm[c] = r_instr[c].to_numpy(dtype=float)
            elif c.startswith("NOISIN::"):
                r_bm[c] = r_n50
            else:
                # ISIN
                vals = np.zeros(len(t1), dtype=float)
                for i in range(len(t1)):
                    band = str(size_lookup.get((t0[i], c), "")).strip()
                    bench_name = LIKE_FOR_LIKE_BENCH.get(band, BENCH_NAME_NIFTY50)
                    if bench_name == BENCH_NAME_NIFTY100:
                        vals[i] = r_n100[i]
                    elif bench_name == BENCH_NAME_MID150:
                        vals[i] = r_mid150[i]
                    elif bench_name == BENCH_NAME_SMALL250:
                        vals[i] = r_small250[i]
                    else:
                        vals[i] = r_n50[i]
                r_bm[c] = vals

    # Override rule: Overseas/ADR/Others equities benchmark to Nifty 50 in like-for-like
    # (and cash handled already)
    if bench_mode == "Like-for-like":
        for c in instr_cols:
            if c in cash_keys or c.startswith("CASH::") or c.startswith("NOISIN::"):
                continue
            at = str(asset_map.get(c, "")).strip()
            if at in {"Overseas Equities", "ADRs & GDRs", "Others Equities", "Other Equities"}:
                r_bm[c] = r_n50

    r_bm = r_bm.fillna(0.0)

    # Convert to numpy
    w0 = w0_df.to_numpy(dtype=float)
    r0 = r_instr.to_numpy(dtype=float)
    rb = r_bm.to_numpy(dtype=float)

    # Portfolio and benchmark-portfolio returns
    rp = np.sum(w0 * r0, axis=1)
    rbp = np.sum(w0 * rb, axis=1)

    # Sanity check: with weights correctly scaled to decimals, extreme monthly Rp should be rare.
    # Keep the check, but this should no longer trigger due to scaling.
    if np.nanmax(np.abs(rp)) > 2.0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {
            "error": f"Unrealistic monthly portfolio return detected (max |Rp|={np.nanmax(np.abs(rp)):.2f}). Check holdings/price data integrity.",
        }

    # Linking factors (base=100)
    base = 100.0
    link_fund = np.ones(len(t1), dtype=float)
    link_bm = np.ones(len(t1), dtype=float)
    for i in range(1, len(t1)):
        link_fund[i] = link_fund[i-1] * (1.0 + rp[i-1])
        link_bm[i] = link_bm[i-1] * (1.0 + rbp[i-1])

    contrib_fund = (w0 * r0) * link_fund[:, None] * base
    contrib_bm = (w0 * rb) * link_bm[:, None] * base

    total_fund = contrib_fund.sum(axis=0)
    total_bench = contrib_bm.sum(axis=0)
    total_alpha = total_fund - total_bench

    # Name map
    name_map = {}
    if not sm.empty:
        name_map = dict(zip(sm["isin"].astype(str), sm["company_name"].astype(str)))

    def _disp(instr: str) -> str:
        if instr.startswith("NOISIN::"):
            return instr.replace("NOISIN::", "")
        if instr.startswith("CASH::"):
            return "Cash (Residual)"
        if instr in cash_keys:
            return "Cash"
        return name_map.get(instr, instr)

    # Holding period months (periods where weight>0 at t0)
    held_mask = (w0_df > 0).to_numpy()
    held_months = held_mask.sum(axis=0).astype(int)
    holding_years = held_months / 12.0

    def _masked_cagr(ret_mat: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = np.full(ret_mat.shape[1], np.nan, dtype=float)
        for j in range(ret_mat.shape[1]):
            m = mask[:, j]
            n = int(m.sum())
            if n <= 0:
                continue
            gross = float(np.prod(1.0 + ret_mat[m, j]))
            yrs = n / 12.0
            if yrs > 0 and gross > 0:
                out[j] = gross ** (1.0 / yrs) - 1.0
        return out

    stock_cagr = _masked_cagr(r0, held_mask)
    bench_cagr = _masked_cagr(rb, held_mask)
    outperf = stock_cagr - bench_cagr

    stock_df = pd.DataFrame({
        "Stock name": [_disp(c) for c in instr_cols],
        "Stock contribution (Rs.)": np.round(total_fund, 2),
        "Benchmark contribution (Rs.)": np.round(total_bench, 2),
        "Alpha contribution (Rs.)": np.round(total_alpha, 2),
        "Holding period (years)": np.round(holding_years, 2),
        "Stock CAGR": np.round(stock_cagr * 100.0, 2),
        "Benchmark CAGR": np.round(bench_cagr * 100.0, 2),
        "Stock outperformance (pp)": np.round(outperf * 100.0, 2),
    }).sort_values("Alpha contribution (Rs.)", ascending=False).reset_index(drop=True)

    # Hit-rate summary (exclude cash)
    is_cash_row = stock_df["Stock name"].astype(str).str.strip().str.lower().str.startswith("cash")
    non_cash = stock_df.loc[~is_cash_row].copy()

    winners = non_cash.loc[non_cash["Stock outperformance (pp)"] > 0]
    losers = non_cash.loc[non_cash["Stock outperformance (pp)"] <= 0]

    hit_df = pd.DataFrame([{
        "Hit-rate (winners/total)": f"{len(winners)} / {len(non_cash)}" if len(non_cash) else "0 / 0",
        "Avg holding period winners (yrs)": round(float(winners["Holding period (years)"].mean()), 2) if len(winners) else np.nan,
        "Avg holding period losers (yrs)": round(float(losers["Holding period (years)"].mean()), 2) if len(losers) else np.nan,
        "Avg alpha on winners (%)": round(float(winners["Stock outperformance (pp)"].mean()), 2) if len(winners) else np.nan,
        "Avg alpha on losers (%)": round(float(losers["Stock outperformance (pp)"].mean()), 2) if len(losers) else np.nan,
    }])

    # Category table: Large/Mid/Small/Cash
    categories = ["Large", "Mid", "Small", "Cash"]
    cat_fund = {c: 0.0 for c in categories}
    cat_bench = {c: 0.0 for c in categories}

    contrib_f_df = pd.DataFrame(contrib_fund, index=t1, columns=instr_cols)
    contrib_b_df = pd.DataFrame(contrib_bm, index=t1, columns=instr_cols)

    for i in range(len(t0)):
        start_m = t0[i]
        end_m = t1[i]
        for instr in instr_cols:
            if instr in cash_keys or instr.startswith("CASH::") or instr.startswith("NOISIN::"):
                cat = "Cash"
            else:
                band = str(size_lookup.get((start_m, instr), "")).strip()
                cat = band if band in {"Large", "Mid", "Small"} else "Large"
            cat_fund[cat] += float(contrib_f_df.loc[end_m, instr])
            cat_bench[cat] += float(contrib_b_df.loc[end_m, instr])

    # Category CAGR: compute using category-level weighted returns each period
    r_instr_df = pd.DataFrame(r0, index=t1, columns=instr_cols)
    r_bm_df = pd.DataFrame(rb, index=t1, columns=instr_cols)
    w_start_df = w0_df.copy()

    cat_cagr = {}
    cat_bm_cagr = {}

    for cat in categories:
        gross_cat = 1.0
        gross_bm_cat = 1.0
        m_count = 0
        for i in range(len(t0)):
            start_m = t0[i]
            end_m = t1[i]

            members = []
            for instr in instr_cols:
                if cat == "Cash":
                    if instr in cash_keys or instr.startswith("CASH::") or instr.startswith("NOISIN::"):
                        members.append(instr)
                else:
                    if instr in cash_keys or instr.startswith("CASH::") or instr.startswith("NOISIN::"):
                        continue
                    band = str(size_lookup.get((start_m, instr), "")).strip()
                    band = band if band in {"Large", "Mid", "Small"} else "Large"
                    if band == cat:
                        members.append(instr)

            if not members:
                continue
            denom = float(w_start_df.loc[start_m, members].sum())
            if denom <= 0:
                continue

            r_cat = float((w_start_df.loc[start_m, members] * r_instr_df.loc[end_m, members]).sum() / denom)
            r_cat_bm = float((w_start_df.loc[start_m, members] * r_bm_df.loc[end_m, members]).sum() / denom)

            gross_cat *= (1.0 + r_cat)
            gross_bm_cat *= (1.0 + r_cat_bm)
            m_count += 1

        if m_count > 0:
            yrs = m_count / 12.0
            cat_cagr[cat] = gross_cat ** (1.0 / yrs) - 1.0
            cat_bm_cagr[cat] = gross_bm_cat ** (1.0 / yrs) - 1.0
        else:
            cat_cagr[cat] = np.nan
            cat_bm_cagr[cat] = np.nan

    cat_df = pd.DataFrame([{
        "Category": c,
        "Category contribution (Rs.)": round(cat_fund[c], 2),
        "Benchmark contribution (Rs.)": round(cat_bench[c], 2),
        "Category alpha contribution (Rs.)": round(cat_fund[c] - cat_bench[c], 2),
        "Category CAGR": round(cat_cagr[c] * 100.0, 2) if pd.notna(cat_cagr[c]) else np.nan,
        "Benchmark CAGR": round(cat_bm_cagr[c] * 100.0, 2) if pd.notna(cat_bm_cagr[c]) else np.nan,
        "Category outperformance (pp)": round((cat_cagr[c] - cat_bm_cagr[c]) * 100.0, 2) if pd.notna(cat_cagr[c]) else np.nan,
    } for c in categories])

    diag = {
        "months": months,
        "missing_benchmarks": [x for x in [BENCH_NAME_NIFTY50, BENCH_NAME_NIFTY500, BENCH_NAME_NIFTY100, BENCH_NAME_MID150, BENCH_NAME_SMALL250] if x not in bn_piv.columns],
    }
    return stock_df, hit_df, cat_df, diag


def fund_attribution_page():
    home_button()
    st.title("Fund attribution")

    # 1) Category
    categories = fetch_categories()
    if not categories:
        st.warning("No categories found.")
        return

    cat = st.radio("1. Select category", categories, horizontal=True, key="fa_category")

    # 2) Focus fund
    funds_df = fetch_funds_for_categories([cat])
    if funds_df.empty:
        st.warning("No funds found for this category.")
        return

    focus_label = st.selectbox(
        "2. Select focus fund",
        options=funds_df["fund_name"].tolist(),
        key="fa_focus_fund",
    )
    focus_id = int(funds_df.loc[funds_df["fund_name"] == focus_label, "fund_id"].iloc[0])

    # 3) Benchmark mode (3 options)
    bench_mode = st.radio(
        "3. Attribution benchmark mode",
        options=["Vs NIFTY 50", "Vs NIFTY 500", "Like-for-like"],
        horizontal=True,
        key="fa_bench_mode",
    )

    # 4) Start/end period below radio
    with st.form("fa_form"):
        today = dt.date.today()
        years = list(range(today.year - 15, today.year + 1))
        months = list(range(1, 13))

        def _mname(m: int) -> str:
            return dt.date(2000, m, 1).strftime("%b")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            start_year = st.selectbox("4. Start year", years, index=years.index(today.year - 3) if (today.year - 3) in years else 0, key="fa_start_year")
        with c2:
            start_month = st.selectbox("Start month", months, index=today.month - 1, format_func=_mname, key="fa_start_month")
        with c3:
            end_year = st.selectbox("5. End year", years, index=len(years) - 1, key="fa_end_year")
        with c4:
            end_month = st.selectbox("End month", months, index=today.month - 1, format_func=_mname, key="fa_end_month")

        lookback = st.selectbox("Lookback window (years)", options=[5, 10, 15], index=1, key="fa_lookback")
        submit = st.form_submit_button("Submit", type="primary")

    if not submit:
        st.info("Select inputs above and click Submit.")
        return

    start_me = _to_month_end(dt.date(int(start_year), int(start_month), 1))
    end_me = _to_month_end(dt.date(int(end_year), int(end_month), 1))
    if start_me > end_me:
        st.error("Start period must be before end period.")
        return

    # 10-year cache anchored to end period
    with st.spinner("Loading fund history (cached) ..."):
        raw = _load_attrib_raw_window(focus_id, end_me, lookback_years=int(lookback), data_version="v1")

    if raw.get("weight_scale") not in (None, 1.0):
        st.caption(f"Note: holding_weight scale detected as {raw['weight_scale']:.6f} and normalised accordingly.")


    if raw["holdings"].empty:
        st.warning("No holdings data found for this fund in the selected window.")
        return

    if start_me < raw["window_start"]:
        st.warning(f"Start period {start_me} is older than cached window start {raw['window_start']}. Increase lookback window.")
        return

    with st.spinner("Computing attribution ..."):
        stock_df, hit_df, cat_df, diag = _compute_attribution(raw, start_me, end_me, bench_mode)

    if "error" in diag:
        st.error(diag["error"])
        return

    # Benchmark warnings
    miss = diag.get("missing_benchmarks", [])
    if bench_mode == "Like-for-like":
        needed = [BENCH_NAME_NIFTY100, BENCH_NAME_MID150, BENCH_NAME_SMALL250]
        miss_needed = [x for x in needed if x in miss]
        if miss_needed:
            st.warning("Missing benchmark NAV series in DB (will default missing benchmark returns to 0): " + ", ".join(miss_needed))
    else:
        needed = BENCH_NAME_NIFTY50 if bench_mode == "Vs NIFTY 50" else BENCH_NAME_NIFTY500
        if needed in miss:
            st.warning(f"Missing benchmark NAV series in DB for {needed} (benchmark returns will default to 0).")

    st.subheader("1) Stock-level attribution")
    st.dataframe(stock_df, use_container_width=True)

    st.subheader("2) Hit-rate summary (excluding cash)")
    st.dataframe(hit_df, use_container_width=True)

    st.subheader("3) Category-level attribution")
    st.dataframe(cat_df, use_container_width=True)



def home_page():
    st.subheader("Welcome to the Fund Analytics Dashboard")

    st.markdown(
        """
        This app currently has these main sections:

        - **Performance** – NAV-based rolling returns, yearly returns, P2P, and PDF export.
        - **Portfolio quality** – RoE / RoCE and quality buckets based on portfolios.
        - **Portfolio valuations** – P/E, P/B, P/S for funds vs peers.
        - **Portfolio** – Holdings explorer, active share, look-through.
        - **Update DB** – Upload and refresh raw datasets.
        - **Housekeeping** – Rebuild precomputed tables and diagnostics.

        Use the buttons below to jump directly to a section.
        """
    )

    st.subheader("Navigation")

    if st.button("📈 Performance"):
        st.session_state["page"] = "Performance"
        st.rerun()

    if st.button("📊 Portfolio quality"):
        st.session_state["page"] = "Portfolio quality"
        st.rerun()

    if st.button("💹 Portfolio valuations"):
        st.session_state["page"] = "Portfolio valuations"
        st.rerun()

    if st.button("📉 Fund attribution"):
        st.session_state["page"] = "Fund attribution"
        st.rerun()

    if st.button("📂 Portfolio"):
        st.session_state["page"] = "Portfolio"
        st.rerun()

    if st.button("🧑‍💼 Fund manager tenure"):
        st.session_state["page"] = "Fund manager tenure"
        st.rerun()

    if st.button("🛠️ Update DB"):
        st.session_state["page"] = "Update DB"
        st.rerun()

    if st.button("🧹 Housekeeping"):
        st.session_state["page"] = "Housekeeping"
        st.rerun()






def main():
    # Initialise page once
    if "page" not in st.session_state:
        st.session_state["page"] = "Home"

    page = st.session_state["page"]

    #st.title("Fund Analytics Dashboard")
    st.markdown("---")

    if page == "Home":
        home_page()
    elif page == "Performance":
        performance_page()
    elif page == "Portfolio quality":
        portfolio_quality_page()
    elif page == "Portfolio valuations":
        portfolio_valuations_page()
    elif page == "Fund attribution":
        fund_attribution_page()
    elif page == "Portfolio":
        portfolio_page()
    elif page == "Fund manager tenure":
        fund_manager_tenure_page()
    elif page == "Update DB":
        update_db_page()
    elif page == "Housekeeping":
        housekeeping_page()
    




# ------------------------ Router ------------------------
if  __name__ == "__main__":
    main()
    
