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
    """Render a Home button that jumps back to the Home page."""
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
    values = % of domestic equity weight in that quality bucket.
    """
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(
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
              AND fp.month_end = ANY(:dates)
              AND fp.asset_type = 'Domestic Equities'
            """,
            conn,
            params={"fid": fund_id, "dates": month_ends},
        )

    if df.empty:
        return pd.DataFrame()

    # Rebase weights within each month to 100% (domestic equities only)
    df["holding_weight"] = df["holding_weight"].astype(float)
    totals = df.groupby("month_end")["holding_weight"].transform("sum")
    df["re_based_weight"] = df["holding_weight"] / totals * 100.0

    # Aggregate by quartile
    pivot = (
        df.groupby(["quality_quartile", "month_end"])["re_based_weight"]
          .sum()
          .unstack("month_end")
          .reindex(index=["Q1", "Q2", "Q3", "Q4"])
    )

    # Pretty month labels
    if pivot is not None and not pivot.empty:
        pivot.columns = [d.strftime("%b %Y") for d in pivot.columns]

    return pivot



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


@st.cache_data(show_spinner="Loading fund NAVs from database...")
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


    # ------------------------ Rolling Returns ------------------------
    st.header("Rolling Returns")

    # Month + Year pickers
    def eom(y: int, m: int) -> pd.Timestamp:
        return pd.Timestamp(year=y, month=m, day=1).to_period("M").to_timestamp("M")

    date_years = sorted(pd.to_datetime(filtered["date"]).dt.year.unique().tolist())
    if not date_years:
        st.stop()
    min_y, max_y = min(date_years), max(date_years)
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    m2n = {m:i+1 for i,m in enumerate(months)}

    cA1, cA2, cB1, cB2 = st.columns([1,1,1,1])
    with cA1:
        start_month = st.selectbox("Start month (start-domain)", months, index=0, key="rr_start_m")
    with cA2:
        start_year  = st.selectbox("Start year  (start-domain)", list(range(min_y, max_y+1)), index=0, key="rr_start_y")
    with cB1:
        end_month   = st.selectbox("End month   (end-domain)", months, index=len(months)-1, key="rr_end_m")
    with cB2:
        end_year    = st.selectbox("End year    (end-domain)", list(range(min_y, max_y+1)), index=len(range(min_y, max_y+1))-1, key="rr_end_y")

    start_domain = eom(int(start_year), m2n[start_month])
    end_domain   = eom(int(end_year),   m2n[end_month])

    def window_ok(start_dt, end_dt, months):
        return (start_dt + pd.DateOffset(months=months)) <= end_dt

    print_items = []  # (caption:str, figure:plotly Figure)

    # 3Y main chart
    st.subheader("3Y Rolling — Focus vs Peer avg vs Benchmark")

    # Only include non-empty labels in the options
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
                else pd.DataFrame({"info": ["Not enough overlapping 3Y windows"]})
            )
            if st.checkbox("To print", key="print_fig3"):
                print_items.append(("3Y Rolling — Focus/Peers/Benchmark", fig3))


    # 3Y multi-fund
    st.subheader("3Y Rolling — Multiple Selected Funds")
    mf3 = st.multiselect("Pick funds to plot (3Y multi-fund)", options=funds_selected, default=[focus_fund], key="mf3")
    if not window_ok(start_domain, end_domain, 36):
        st.info("Selected range too short for 3Y windows.")
    else:
        df3m = make_multi_fund_rolling_df(filtered, mf3, 36, start_domain, end_domain)
        fig3m = plot_multi_fund_rolling(df3m, 36, focus_name=focus_fund, chart_height=560)
        if fig3m is None:
            st.info("Not enough data to plot selected funds (3Y).")
        else:
            st.plotly_chart(fig3m, use_container_width=True)
            if st.checkbox("To print", key="print_fig3m"):
                print_items.append(("3Y Rolling — Multiple funds", fig3m))

    # 1Y main chart
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
                else pd.DataFrame({"info": ["Not enough overlapping 1Y windows"]})
            )
            if st.checkbox("To print", key="print_fig1"):
                print_items.append(("1Y Rolling — Focus/Peers/Benchmark", fig1))


    # 1Y multi-fund
    st.subheader("1Y Rolling — Multiple Selected Funds")
    mf1 = st.multiselect("Pick funds to plot (1Y multi-fund)", options=funds_selected, default=[focus_fund], key="mf1")
    if not window_ok(start_domain, end_domain, 12):
        st.info("Selected range too short for 1Y windows.")
    else:
        df1m = make_multi_fund_rolling_df(filtered, mf1, 12, start_domain, end_domain)
        fig1m = plot_multi_fund_rolling(df1m, 12, focus_name=focus_fund, chart_height=560)
        if fig1m is None:
            st.info("Not enough data to plot selected funds (1Y).")
        else:
            st.plotly_chart(fig1m, use_container_width=True)
            if st.checkbox("To print", key="print_fig1m"):
                print_items.append(("1Y Rolling — Multiple funds", fig1m))



    # ------------------------ Yearly Returns (STRICT) ------------------------
    st.header("Yearly Returns (Strict FY/CY endpoints)")

    # Explicit month & year pickers (independent of Rolling)
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    m2n = {m:i+1 for i,m in enumerate(months)}

    date_years_all = sorted({d.year for d in pd.to_datetime(filtered["date"].unique())}) if not filtered.empty else [2010, 2011]


    colA, colB, colC, colD = st.columns(4)
    with colA:
        y_start = st.selectbox("Start Year", options=date_years_all, index=0, key="strict_y_start")
    with colB:
        m_start = st.selectbox("Start Month", options=months, index=0, key="strict_m_start")
    with colC:
        y_end = st.selectbox("End Year", options=date_years_all, index=len(date_years_all)-1, key="strict_y_end")
    with colD:
        m_end = st.selectbox("End Month", options=months, index=len(months)-1, key="strict_m_end")

    start_domain = pd.Timestamp(year=int(y_start), month=m2n[m_start], day=1).to_period("M").to_timestamp("M")
    end_domain   = pd.Timestamp(year=int(y_end),   month=m2n[m_end],   day=1).to_period("M").to_timestamp("M")

    yr_type_strict = st.radio("Year type", options=["Financial (Apr–Mar)","Calendar (Jan–Dec)"], index=0, horizontal=True, key="strict_year_type")
    use_fy_strict = yr_type_strict.startswith("Financial")

    # Compute per-fund yearly returns with trimmed first/last years
    yr_rows = {}
    for f in funds_selected:
        s = filtered.loc[filtered["fund"] == f, ["date","nav"]].drop_duplicates("date").set_index("date")["nav"]
        yr_rows[f] = yearly_returns_with_custom_domain(s, start_domain, end_domain, fy=use_fy_strict)
    yr_df = pd.DataFrame(yr_rows).T

    # Benchmark
    yr_bench = None
    if bench_ser is not None and not bench_ser.empty:
        yr_bench = yearly_returns_with_custom_domain(bench_ser, start_domain, end_domain, fy=use_fy_strict).rename("Benchmark")

    if yr_df.empty:
        st.info("Not enough data to compute yearly returns for the selected funds.")
    else:
        # Align columns across funds and benchmark; keep original order
        cols_order = list(yr_df.columns)
        if yr_bench is not None and not yr_bench.empty:
            for c in yr_bench.index:
                if c not in cols_order: cols_order.append(c)
        yr_df = yr_df.reindex(columns=cols_order)

        # Actual table
        st.subheader("Actual yearly returns — Funds (rows) vs Years (columns)")

        # Build table (keep your FY/CY logic above intact)
        disp_actual = (yr_df.loc[funds_selected, cols_order] * 100.0).copy()
        disp_actual.insert(0, "Fund", disp_actual.index)
        disp_actual = disp_actual.reset_index(drop=True)  # remove unnamed index column

        # Round numeric columns to 1 decimal and style with 1-decimal display
        num_cols = [c for c in disp_actual.columns if c != "Fund"]
        disp_actual[num_cols] = disp_actual[num_cols].round(1)

        st.dataframe(
        disp_actual.style
            .format({c: "{:.1f}" for c in num_cols}, na_rep="—")
            .set_table_styles([
                {"selector": "table", "props": "table-layout:fixed"},
                {"selector": "th.col_heading",
                "props": "white-space:normal; line-height:1.1; height:56px"}
            ]),
        use_container_width=True
        )
        p_act = st.checkbox("To print", key="print_actual_tbl")


        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

        # Benchmark table
        st.subheader(f"Benchmark yearly returns — {bench_label}")
        if yr_bench is None or yr_bench.empty:
            st.info("Benchmark not available.")
            bench_df_print = pd.DataFrame()
        else:
            bench_df_print = ((yr_bench[cols_order] * 100.0).to_frame().T).round(2)
            bench_df_print.index = [bench_label]
            bench_df_print = bench_df_print.reset_index().rename(columns={"index":"Benchmark"})
            st.dataframe(
                bench_df_print.style
                .format({c: "{:.1f}" for c in bench_df_print.columns if c != "Benchmark"}, na_rep="—")
                .set_table_styles([
                    {"selector": "table", "props": "table-layout:fixed"},
                    {"selector": "th.col_heading",
                    "props": "white-space:normal; line-height:1.1; height:56px"}
                ]),
                use_container_width=True
            )

        p_bench = st.checkbox("To print", key="print_bench_tbl")

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

        # Relative (ppt) vs benchmark
        st.subheader("Relative yearly returns (ppt) — Funds (rows) vs Years (columns)")
        rel_df = pd.DataFrame()
        if yr_bench is not None and not yr_bench.empty:
            common = [c for c in cols_order if c in yr_bench.index]
            if common:
                rel_df = (yr_df.loc[funds_selected, common].subtract(yr_bench[common], axis=1) * 100.0).round(2)
        if rel_df.empty:
            st.info("Not enough overlap to compute relative returns.")
            disp_rel = pd.DataFrame()
        else:
            disp_rel = rel_df.copy()
            disp_rel.insert(0, "Fund", disp_rel.index)
            disp_rel = disp_rel.reset_index(drop=True)   # remove unnamed index column
            def rel_colors(df):
                styles = pd.DataFrame("", index=df.index, columns=df.columns)
                for c in df.columns:
                    if c == "Fund": continue
                    styles[c] = df[c].apply(lambda v: "background-color:#e6f4ea;color:#0b8043" if pd.notna(v) and v>0
                                                    else "background-color:#fdecea;color:#a50e0e" if pd.notna(v) and v<0
                                                    else "")
                return styles
            st.write(
            disp_rel.style
                .apply(rel_colors, axis=None)
                .format({c: "{:.1f}" for c in disp_rel.columns if c != "Fund"}, na_rep="—")
                .set_table_styles([
                    {"selector": "table", "props": "table-layout:fixed"},
                    {"selector": "th.col_heading",
                    "props": "white-space:normal; line-height:1.1; height:56px"}
                ])
            )

        p_rel = st.checkbox("To print", key="print_rel_tbl")
    st.header("Point-to-Point (P2P) Returns — Custom period CAGR")

    date_years_all = sorted(pd.to_datetime(filtered["date"]).dt.year.unique().tolist())
    p2p_min_y, p2p_max_y = min(date_years_all), max(date_years_all)
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    m2n = {m:i+1 for i,m in enumerate(months)}

    cP1, cP2, cP3, cP4 = st.columns([1,1,1,1])
    with cP1: p2p_start_m = st.selectbox("Start month", months, index=0, key="p2p_sm")
    with cP2: p2p_start_y = st.selectbox("Start year", list(range(p2p_min_y, p2p_max_y+1)), index=0, key="p2p_sy")
    with cP3: p2p_end_m   = st.selectbox("End month", months, index=len(months)-1, key="p2p_em")
    with cP4: p2p_end_y   = st.selectbox("End year", list(range(p2p_min_y, p2p_max_y+1)), index=len(range(p2p_min_y, p2p_max_y+1))-1, key="p2p_ey")

    p2p_start = eom(int(p2p_start_y), m2n[p2p_start_m])
    p2p_end   = eom(int(p2p_end_y),   m2n[p2p_end_m])

    def months_between(a: pd.Timestamp, b: pd.Timestamp) -> int:
        return (b.year - a.year)*12 + (b.month - a.month)

    def series_cagr_between(s: pd.Series, start_eom: pd.Timestamp, end_eom: pd.Timestamp):
        """Exact month-end CAGR (decimal) between two month-ends."""
        s = s.dropna().sort_index()
        if start_eom not in s.index or end_eom not in s.index or end_eom <= start_eom:
            return np.nan
        m = months_between(start_eom, end_eom)
        if m <= 0: return np.nan
        try:
            return (s.loc[end_eom] / s.loc[start_eom]) ** (12.0/m) - 1.0
        except Exception:
            return np.nan

    if p2p_end <= p2p_start:
        st.warning("P2P End must be after Start.")
    else:
        rows = []
        for f in funds_selected:
            s = filtered.loc[filtered["fund"] == f, ["date","nav"]].drop_duplicates("date").set_index("date")["nav"]
            val = series_cagr_between(s, p2p_start, p2p_end)
            rows.append({"Fund": f, "Start": f"{p2p_start:%b %Y}", "End": f"{p2p_end:%b %Y}",
                        "Months": months_between(p2p_start, p2p_end),
                        "CAGR %": None if np.isnan(val) else round(val*100.0, 2)})
        if bench_ser is not None and not bench_ser.empty:
            bval = series_cagr_between(bench_ser, p2p_start, p2p_end)
            rows.append({"Fund": bench_label, "Start": f"{p2p_start:%b %Y}", "End": f"{p2p_end:%b %Y}",
                        "Months": months_between(p2p_start, p2p_end),
                        "CAGR %": None if np.isnan(bval) else round(bval*100.0, 2)})

        p2p_df = pd.DataFrame(rows)
    # Coerce/round/sort CAGR columns
    for col in list(p2p_df.columns):
        if isinstance(col, str) and col.endswith('CAGR %'):
            p2p_df[col] = pd.to_numeric(p2p_df[col], errors='coerce').round(1)
    if 'CAGR %' in p2p_df.columns:
        p2p_df = p2p_df.sort_values(by='CAGR %', ascending=False)
    # Ensure all % columns are numeric, 1 decimal, and sort by main CAGR % desc
    for col in list(p2p_df.columns):
        if isinstance(col, str) and col.endswith('CAGR %'):
            p2p_df[col] = pd.to_numeric(p2p_df[col], errors='coerce').round(1)
    # Round CAGR % to 1 decimal
    if "CAGR %" in p2p_df.columns:
        p2p_df["CAGR %"] = pd.to_numeric(p2p_df["CAGR %"], errors="coerce").round(1)
        p2p_df = p2p_df.sort_values(by="CAGR %", ascending=False)

    st.dataframe(
        p2p_df.style.format(
            {c: "{:.1f}%" for c in p2p_df.columns if c.endswith("CAGR %")},
            na_rep="—"
        ),
        use_container_width=True
    )
    p_p2p = st.checkbox("To print", key="print_p2p_tbl")


    # ------------------------ Relative Multi-Horizon CAGR vs Benchmark ------------------------
    st.subheader("Relative CAGR vs Benchmark — 1Y / 3Y / 5Y / 7Y (as of P2P end month)")

    def end_aligned_cagr(series: pd.Series, end_eom: pd.Timestamp, months: int) -> float:
        s = series.dropna().sort_index()
        if end_eom not in s.index:
            return np.nan
        start_eom = (end_eom - pd.DateOffset(months=months)).to_period("M").to_timestamp("M")
        if start_eom not in s.index or end_eom <= start_eom:
            return np.nan
        try:
            return (s.loc[end_eom] / s.loc[start_eom]) ** (12.0 / months) - 1.0
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
        s = filtered.loc[filtered["fund"] == f, ["date","nav"]].drop_duplicates("date").set_index("date")["nav"]
        row = {"Fund": f}
        for m, lbl in horizons:
            fc = end_aligned_cagr(s, p2p_end, m)
            bc = bench_cagrs.get(lbl, np.nan)
            row[lbl] = None if (np.isnan(fc) or np.isnan(bc)) else round((fc - bc) * 100.0, 2)
        rel_rows.append(row)

    rel_mh_df = pd.DataFrame(rel_rows).set_index("Fund")

    def style_rel_mh(df: pd.DataFrame):
        """
        Style for Relative (multi-horizon) table:
        - color positive/negative numbers
        - format numeric cols to 2 decimals
        - safe for None/NaN via na_rep
        """
        df2 = df.copy()

        # Identify numeric columns (skip label columns like 'Fund')
        num_cols = [c for c in df2.columns
                    if c != "Fund" and pd.api.types.is_numeric_dtype(df2[c])]

        # If some numeric columns arrived as 'object' (strings), coerce them
        for c in df2.columns:
            if c != "Fund" and not pd.api.types.is_numeric_dtype(df2[c]):
                try:
                    tmp = pd.to_numeric(df2[c], errors="coerce")
                    if tmp.notna().any():
                        df2[c] = tmp
                except Exception:
                    pass

        # Recompute numeric columns after coercion
        num_cols = [c for c in df2.columns
                    if c != "Fund" and pd.api.types.is_numeric_dtype(df2[c])]

        def rel_colors(dfin: pd.DataFrame):
            styles = pd.DataFrame("", index=dfin.index, columns=dfin.columns)
            for c in num_cols:
                styles[c] = dfin[c].apply(
                    lambda v: (
                        "background-color:#e6f4ea;color:#0b8043" if pd.notna(v) and v > 0 else
                        "background-color:#fdecea;color:#a50e0e" if pd.notna(v) and v < 0 else
                        ""
                    )
                )
            return styles

        sty = df2.style.apply(rel_colors, axis=None)

        # Format only numeric columns; use na_rep so None/NaN won't crash
        fmt_map = {c: "{:.2f}" for c in num_cols}
        sty = sty.format(fmt_map, na_rep="—")

        # Optional: uniform header height / fixed layout
        sty = sty.set_table_styles([
            {"selector": "table", "props": "table-layout:fixed"},
            {"selector": "th.col_heading",
            "props": "white-space:normal; line-height:1.1; height:56px"}
        ])

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


def portfolio_fundamentals_page():
    home_button()
    st.header("Portfolio fundamentals – RoE / RoCE")

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
        if col.checkbox(cat, value=False, key=f"cat_{cat}"):
            selected_categories.append(cat)

    if not selected_categories:
        st.info("Please select at least one category.")
        return

    # 2) Fund multi-select
    st.subheader("2. Select funds")
    funds_df = fetch_funds_for_categories(selected_categories)
    if funds_df.empty:
        st.warning("No funds found for selected categories.")
        return

    fund_options = {
    f"{row['fund_name']} ({row['category_name']})": row["fund_id"]
    for _, row in funds_df.iterrows()
    }


    selected_fund_labels = st.multiselect(
        "Funds",
        options=list(fund_options.keys()),
        default=[]  # Default none selected
    )
    selected_fund_ids = [fund_options[label] for label in selected_fund_labels]

    if not selected_fund_ids:
        st.info("Please select at least one fund.")
        return

    # 3) Date range selectors (month & year separately)
    st.subheader("3. Select period (March / September portfolios only)")

    current_year = dt.date.today().year
    years = list(range(current_year - 15, current_year + 1))

    month_options = [3, 9]  # Mar, Sep

    col1, col2 = st.columns(2)
    with col1:
        start_year = st.selectbox("Start year", options=years, index=0, key="pf_start_year")
        start_month = st.selectbox(
            "Start month",
            options=month_options,
            index=0,  # default: Mar
            key="pf_start_month",
            format_func=lambda m: "Mar" if m == 3 else "Sep",
        )
    with col2:
        end_year = st.selectbox("End year", options=years, index=len(years) - 1, key="pf_end_year")
        end_month = st.selectbox(
            "End month",
            options=month_options,
            index=1,  # default: Sep
            key="pf_end_month",
            format_func=lambda m: "Mar" if m == 3 else "Sep",
        )


    start_date = month_year_to_last_day(start_year, start_month)
    end_date = month_year_to_last_day(end_year, end_month)

    if start_date > end_date:
        st.error("Start date must be earlier than end date.")
        return
    
    quality_table = compute_quality_bucket_exposure(fund_id, month_ends_list)
    if quality_table is None or quality_table.empty:
        st.info("No Q1–Q4 quality bucket data available for the selected fund and period.")
    else:
        st.subheader("Quality bucket exposures (Q1–Q4)")
        st.dataframe(
            quality_table.style.format("{:.1f}"),
            use_container_width=True
        )


    # 4) Segment radio buttons
    st.subheader("4. Segment")
    segment_choice = st.radio(
        "Show metrics for:",
        options=["Financials", "Non-financials", "Total"],
        horizontal=True
    )

    # 5) Fetch data & compute
    with st.spinner("Computing portfolio fundamentals..."):
        roe_roce_dict = load_stock_roe_roce()
        df_portfolio = fetch_portfolio_raw(selected_fund_ids, start_date, end_date)
        if df_portfolio.empty:
            st.warning("No portfolio data found for selected funds and period.")
            return

        df_result = compute_portfolio_fundamentals(df_portfolio, roe_roce_dict, segment_choice)

    if df_result.empty:
        st.warning("No fundamentals could be computed (check data availability).")
        return

    # 6) Line chart
    st.subheader("5. RoE / RoCE time series")

    df_chart = df_result.dropna(subset=["metric"]).copy()
    df_chart["month_end"] = pd.to_datetime(df_chart["month_end"])

    if df_chart.empty:
        st.info("No data to plot for selected filters.")
    else:
        # Robust y-axis domain so differences are clearly visible
        y_min = float(df_chart["metric"].min())
        y_max = float(df_chart["metric"].max())
        if y_min == y_max:
            padding = max(1.0, abs(y_min) * 0.1 if y_min != 0 else 1.0)
            domain = (y_min - padding, y_max + padding)
        else:
            padding = (y_max - y_min) * 0.1
            domain = (y_min - padding, y_max + padding)

        chart = (
            alt.Chart(df_chart)
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    "month_end:T",
                    title="Period",
                    axis=alt.Axis(format="%b %Y", labelAngle=-45),  # e.g., "Mar 2018"
                ),
                y=alt.Y(
                    "metric:Q",
                    title="RoE / RoCE (%)",
                    scale=alt.Scale(domain=domain),
                ),
                color=alt.Color("fund_name:N", title="Fund"),
                tooltip=[
                    alt.Tooltip("fund_name:N", title="Fund"),
                    alt.Tooltip("month_end:T", title="Period", format="%b %Y"),
                    alt.Tooltip("metric:Q", title="RoE / RoCE (%)", format=".2f"),
                ],
            )
            .properties(height=400)
        )

        st.altair_chart(chart, use_container_width=True)

    
    # 7) Data table
    st.subheader("6. Underlying data")

    df_table = df_result.copy()
    df_table["month_end"] = pd.to_datetime(df_table["month_end"])

    # Use the actual date for column ordering
    df_table["period_date"] = df_table["month_end"]

    df_pivot = df_table.pivot_table(
        index="fund_name",
        columns="period_date",
        values="metric"
    ).sort_index(axis=0).sort_index(axis=1)

    # After sorting, relabel columns as "Mar 2018", "Sep 2018", etc.
    df_pivot.columns = [col.strftime("%b %Y") for col in df_pivot.columns]

    st.dataframe(df_pivot.style.format("{:.2f}"))


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
    categories = fetch_categories()
    if not categories:
        st.warning("No categories found.")
        return

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

    st.subheader("2. Select fund")

    # Handle either 'category_name' or 'category' depending on your fetch_funds_for_categories
    cat_col = "category_name" if "category_name" in funds_df.columns else "category"

    fund_options = {
        f"{row['fund_name']} ({row[cat_col]})": row["fund_id"]
        for _, row in funds_df.iterrows()
    }

    fund_label = st.selectbox("Fund", options=list(fund_options.keys()))
    fund_id = fund_options[fund_label]

    # 3. Period selection
    st.subheader("3. Select period")

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
            key="port_start_year",
        )
        start_month = st.selectbox(
            "Start month",
            options=month_options,
            index=0,
            key="port_start_month",
            format_func=lambda m: month_names[m - 1],
        )
    with col2:
        end_year = st.selectbox(
            "End year",
            options=years,
            index=len(years) - 1,
            key="port_end_year",
        )
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

    # 4. Frequency selection
    st.subheader("4. Frequency")
    freq = st.radio(
        "Aggregation",
        options=["Monthly", "Quarterly", "Yearly"],
        horizontal=True,
    )

    # 5. Action button
    if st.button("Show portfolio"):
        with st.spinner("Loading portfolio..."):
            df = fetch_fund_portfolio_timeseries(fund_id, start_date, end_date, freq)

        if df.empty:
            st.warning("No portfolio data found for this fund and period.")
            return

        # Build pivot table: rows = stock names, columns = periods, values = weights
        df["period"] = pd.to_datetime(df["month_end"]).dt.strftime("%b %Y")

        # Ensure column order is chronological
        period_order_df = (
            df[["period", "month_end"]]
            .drop_duplicates()
            .sort_values("month_end")
        )
        period_order = period_order_df["period"].tolist()

        pivot = df.pivot_table(
            index="company_name",
            columns="period",
            values="weight_pct",
            aggfunc="sum",
            fill_value=0.0,
        )

        # Align columns to chronological order
        pivot = pivot.reindex(columns=period_order)

        # Sort rows by latest period weight (desc)
        latest_period = period_order[-1]
        if latest_period in pivot.columns:
            pivot = pivot.sort_values(by=latest_period, ascending=False)

        # Add serial number column
        df_display = pivot.reset_index()  # company_name becomes a normal column
        df_display.insert(0, "S.No.", range(1, len(df_display) + 1))

        st.subheader("5. Portfolio holdings")
        st.caption(
            f"Rows: stocks · Columns: {freq.lower()} snapshots from {period_order[0]} to {period_order[-1]}"
        )

        def format_weight(x):
            if pd.isna(x) or abs(x) < 0.0001:
                return "-"
            return f"{x:.1f}"

        # Format only the weight columns; leave S.No. and company_name as-is
        weight_cols = [c for c in df_display.columns if c not in ["S.No.", "company_name"]]

        styler = df_display.style
        styler = styler.format(format_weight, subset=weight_cols)

        st.dataframe(styler)


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
            "Stock ISIN, industry, financial/non-financial",
            "Company RoE / RoCE",
            "Stock prices and market cap",
            "Company sales, book value, PAT (stub)",
        ],
    )

    # Single file upload for the chosen type
    uploaded = st.file_uploader("Upload Excel file", type=["xlsx"], key=f"upload_{upload_type}")

    # Show expected format preview
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

    # Dry-run + upload pipeline
    if upload_type == "Company sales, book value, PAT (stub)":
        st.warning("This uploader is a stub. Format and ingestion are not yet implemented.")
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
            else:
                st.error("Unsupported upload type.")
                return

            st.session_state[state_key_df] = df_clean
            st.session_state[state_key_ok] = True

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
    if st.session_state.get(state_key_ok):
        if st.button("Confirm upload to database"):
            df_clean = st.session_state.get(state_key_df)
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
                st.success("✅ Upload completed successfully.")
            except SQLAlchemyError as e:
                msg = str(getattr(e, "orig", e))
                if "foreign key constraint" in msg.lower():
                    st.error(
                        "❌ Some ISINs in this file do not exist in Stock Master.\n"
                        "Please update Stock Master before uploading stock price data.\n\n"
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



def home_page():
    st.subheader("Welcome to the Fund Analytics Dashboard")

    st.markdown(
        """
        This app currently has three main sections:

        - **Performance** – NAV-based rolling returns, yearly returns, P2P, and PDF export.
        - **Fundamentals** – Portfolio-level RoE / RoCE using stock-level metrics and portfolios.
        - **Portfolio** – Holdings explorer, active share, look-through.

        Use the links below to jump directly to a section.
                """
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("📈 Performance"):
            st.session_state["page"] = "Performance"
            st.rerun()
    with col2:
        if st.button("📊 Fundamentals"):
            st.session_state["page"] = "Fundamentals"
            st.rerun()
    with col3:
        if st.button("📂 Portfolio"):
            st.session_state["page"] = "Portfolio"
            st.rerun()
    with col4:
        if st.button("🛠️ Update DB"):
            st.session_state["page"] = "Update DB"
            st.rerun()
    with col5:
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
    elif page == "Fundamentals":
        portfolio_fundamentals_page()
    elif page == "Portfolio":
        portfolio_page()
    elif page == "Update DB":
        update_db_page()
    elif page == "Housekeeping":
        housekeeping_page()
    




# ------------------------ Router ------------------------
if  __name__ == "__main__":
    main()
    
