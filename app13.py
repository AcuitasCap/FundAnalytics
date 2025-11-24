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

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

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

import plotly.express as px
import plotly.graph_objects as go
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader


# ------------------------ Page Setup ------------------------
st.set_page_config(page_title="Fund vs Benchmark & Peers Dashboard", layout="wide")
st.title("Fund Analytics v3.13")
st.caption("Rolling Start = start-date domain; End = end-date domain.")


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
        - 'Window' (i.e. asof_date)
    """

    # ----------------------------
    # 1) Load precomputed FUND rolling returns for all selected funds
    # ----------------------------
    fund_roll = load_fund_rolling(
        window_months=months,
        fund_names=selected_funds,
        start=start_domain,
        end=end_domain,
    )
    if fund_roll.empty:
        return pd.DataFrame()

    # Pivot → wide matrix: rows = dates, columns = fund_name
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
    # 2) Add BENCHMARK if provided
    # ----------------------------
    bench_label = None
    if bench_ser is not None and hasattr(bench_ser, "name"):
        bench_label = bench_ser.name

    if bench_label:
        bench_roll = load_bench_rolling(
            window_months=months,
            bench_name=bench_label,
            start=start_domain,
            end=end_domain,
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
    # 3) Compute PEER AVERAGE
    # ----------------------------
    # Fund columns are ALL columns except benchmark.
    cols_excluding_bench = [
        c for c in wide.columns
        if c != bench_label
    ]

    # Peer columns exclude FOCUS FUND
    peer_cols = [
        c for c in cols_excluding_bench
        if c != focus_fund
    ]

    # Compute Peer avg
    if peer_cols:
        wide["Peer avg"] = wide[peer_cols].mean(axis=1)
    else:
        # If no peers, do NOT include Peer avg at all
        if "Peer avg" in wide.columns:
            wide = wide.drop(columns=["Peer avg"])

    # ----------------------------
    # 4) Final formatting
    # ----------------------------
    wide.index.name = "Window"
    return wide




def make_multi_fund_rolling(funds_df, selected_funds, months, start_domain, end_domain):
    """
    Build a wide DataFrame of precomputed rolling CAGRs for multiple funds,
    to be used in the '3Y Rolling — Multiple Selected Funds' and
    '1Y Rolling — Multiple Selected Funds' tables.

    It uses the precomputed values in fundlab.fund_rolling_return instead of
    recomputing from NAVs.
    """

    # 1) Guard: no funds selected
    if not selected_funds:
        return pd.DataFrame()

    # 2) Load precomputed rolling from DB
    roll = load_fund_rolling(
        window_months=months,
        fund_names=selected_funds,
        start=start_domain,
        end=end_domain,
    )
    if roll.empty:
        return pd.DataFrame()

    # roll columns: ['fund_name', 'asof_date', 'rolling_cagr']
    # 3) Pivot to wide format: rows = date, columns = fund name
    wide = roll.pivot_table(
        index="asof_date",
        columns="fund_name",
        values="rolling_cagr",
        aggfunc="first",
    ).sort_index()

    # 4) For consistency with the rest of the code
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



def rolling_outperf_stats(df: pd.DataFrame, focus_name: str):
    if focus_name not in df.columns or "Benchmark" not in df.columns:
        return None
    f = df[focus_name] / 100.0
    b = df["Benchmark"] / 100.0
    op = (f - b).dropna()
    if op.empty:
        return None
    return pd.DataFrame({
        "windows":[int(op.notna().count())],
        "median (ppt)":[float(np.nanmedian(op)*100.0)],
        "mean   (ppt)":[float(np.nanmean(op)*100.0)],
        "min    (ppt)":[float(np.nanmin(op)*100.0)],
        "max    (ppt)":[float(np.nanmax(op)*100.0)],
        "prob. of outperformance":[float((op > 0).mean()*100.0)],
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
    st.markdown("**Benchmarks (tick multiple as needed)**")
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
        stats3 = rolling_outperf_stats(df3, focus_fund)
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
        stats1 = rolling_outperf_stats(df1, focus_fund)
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
