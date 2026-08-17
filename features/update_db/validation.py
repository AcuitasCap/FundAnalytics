from __future__ import annotations

import pandas as pd

FUND_NAV_COLS = {
    "fund_name": ["fund", "scheme", "scheme name", "fund_name"],
    "nav_date": ["date", "nav date", "month_end", "month-end"],
    "nav_value": ["nav", "nav value"],
    "fund_category": ["category", "fund category"],
    "fund_style": ["style", "style name"],
}

BENCH_NAV_COLS = {
    "benchmark_name": ["benchmark", "benchmark name", "index name"],
    "nav_date": ["date", "nav date", "month_end", "month-end"],
    "nav_value": ["nav", "nav value"],
    "bench_category": ["category", "index category"],
    "bench_style": ["style", "style name"],
}

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

ROE_ROCE_COLS = {
    "isin": ["isin"],
    "company_name": ["company name", "company_name", "name"],
    "year_end": ["year-end (yyyymm)", "year end (yyyymm)", "year_end", "yearend", "yyyymm"],
    "roe": ["roe"],
    "roce": ["roce"],
}

FM_TENURE_COLS = {
    "fund_name": ["fund name", "fund", "scheme", "scheme name", "fund_name"],
    "inception_date": ["inception date", "inception", "start date"],
    "fund_manager": ["fund manager", "manager", "fm", "portfolio manager"],
    "from_period": ["from date", "from", "from_period", "from period"],
    "to_period": ["to date", "to", "to_period", "to period"],
}

STOCK_PRICE_COLS = {
    "isin":       ["isin"],
    "price_date": ["date", "nav date", "price date", "month_end", "month-end"],
    "market_cap": ["market cap", "mcap", "market_cap", "market capitalisation", "market capitalization"],
    "price":      ["stock price", "price", "close", "close price", "last price"],
}

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
    dup_keys = df.groupby(["scheme_name", "month_end", "holding_pct", "isin"]).size()
    dups = dup_keys[dup_keys > 1]
    if not dups.empty:
        raise ValueError(
            f"Duplicate (scheme_name, month_end, holding %, isin) rows found in file: {len(dups)} duplicates."
        )

    summary = {
        "rows": int(len(df)),
        "unique_scheme_month_isin": int(len(dup_keys)),
    }
    return df, summary


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

    df["period_end"] = parse_period_to_month_end(df["yyyymm"])

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

    df["period_end"] = parse_period_to_month_end(df["yyyymm"])

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

    df["year_end"] = parse_period_to_month_end(df["yyyymm"])
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


def validate_fund_manager_tenure(
    df_raw: pd.DataFrame, existing_fund_names: set[str] | None = None
) -> tuple[pd.DataFrame, dict]:
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
    upload_funds = sorted(df["fund_name"].unique().tolist())
    if existing_fund_names is None:
        raise ValueError("Existing fund names must be supplied for validation.")
    existing = {str(name).strip() for name in existing_fund_names}
    missing_funds = sorted([f for f in upload_funds if f not in existing])

    summary = {
        "rows": int(len(df)),
        "funds_in_upload": int(len(upload_funds)),
        "missing_funds": missing_funds,
        "notes": "from/to parsed as Mmm-YYYY; stored as canonical month-date (day=1). to_date blank => current.",
    }
    return df.reset_index(drop=True), summary


def validate_stock_dividends(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Validates and cleans dividend upload.

    Accepts flexible column naming:
      - ISIN / isin
      - Ex Date / ex_date
      - DPS / dps / Dividend Per Share

    Returns:
      df_clean with columns: isin, ex_date, dps
      summary dict
    """
    if df_raw is None or df_raw.empty:
        raise ValueError("The uploaded file is empty.")

    df = df_raw.copy()

    # Normalize column names
    col_map = {}
    for c in df.columns:
        k = str(c).strip().lower()
        col_map[c] = k
    df = df.rename(columns=col_map)

    # Allow common variants
    rename = {}
    if "isin" not in df.columns:
        # nothing else reasonable to map to
        pass
    if "ex_date" not in df.columns:
        for cand in ["ex date", "ex-date", "exdate", "ex_dividend_date", "exdividenddate"]:
            if cand in df.columns:
                rename[cand] = "ex_date"
                break
    if "dps" not in df.columns:
        for cand in ["dividend per share", "div_per_share", "dividend", "dividend_per_share"]:
            if cand in df.columns:
                rename[cand] = "dps"
                break
    df = df.rename(columns=rename)

    required = ["isin", "ex_date", "dps"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Expected: {required}")

    out = df[required].copy()
    out["isin"] = out["isin"].astype(str).str.strip()
    out["ex_date"] = pd.to_datetime(out["ex_date"], errors="coerce").dt.date
    out["dps"] = pd.to_numeric(out["dps"], errors="coerce")

    # Basic validations
    bad_isin = out["isin"].isna() | (out["isin"] == "") | (out["isin"].str.lower() == "nan")
    bad_date = out["ex_date"].isna()
    bad_dps = out["dps"].isna() | (out["dps"] < 0)

    invalid = bad_isin | bad_date | bad_dps
    if invalid.any():
        examples = out.loc[invalid].head(15)
        raise ValueError(
            f"{int(invalid.sum())} invalid rows found (blank ISIN / invalid ex_date / invalid dps).\n"
            f"Examples:\n{examples.to_string(index=False)}"
        )

    # De-dupe within file
    before = len(out)
    out = out.drop_duplicates(subset=["isin", "ex_date"], keep="last").reset_index(drop=True)
    after = len(out)

    summary = {
        "rows_in_file": int(before),
        "rows_after_dedup": int(after),
        "unique_isins": int(out["isin"].nunique()),
        "min_ex_date": str(min(out["ex_date"])),
        "max_ex_date": str(max(out["ex_date"])),
    }
    return out, summary


def validate_stock_prices_mc(df_raw: pd.DataFrame, valid_isins: set[str] | None = None):
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
    if valid_isins is None:
        raise ValueError("Stock Master ISINs must be supplied for validation.")
    isins_in_file = set(df["isin"])
    missing_isins = sorted(isins_in_file - valid_isins)

    if missing_isins:
        raise ValueError(
            "The following ISINs do not exist in Stock Master:\n"
            + "\n".join(missing_isins)
            + "\n\n→ Please upload/update Stock Master first."
        )

    # In-file duplicate check: one logical stock-price row per ISIN/date.
    dup_keys = df.groupby(["isin", "price_date"]).size()
    dups = dup_keys[dup_keys > 1]
    if not dups.empty:
        raise ValueError(
            f"Duplicate (isin, date) rows found in file: {len(dups)} duplicates. "
            "Price and market cap must be unique per ISIN/date."
        )

    summary = {
        "rows": int(len(df)),
        "unique_isin_date": int(len(dup_keys)),
    }
    return df, summary
