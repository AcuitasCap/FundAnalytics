from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from core.db import get_engine


class MissingPortfolioAceCodesError(ValueError):
    """Carry the complete missing-master report back to the upload page."""

    def __init__(self, report: pd.DataFrame):
        self.report = report
        super().__init__(
            "Portfolio upload aborted. Add the missing ACE codes to "
            "fundlab.stock_master_ace before uploading the portfolio. No rows were uploaded."
        )


def find_missing_portfolio_ace_codes(df_clean: pd.DataFrame, conn=None) -> pd.DataFrame:
    """Report domestic holding ACE codes absent from the ACE stock master."""
    report_columns = ["ace_code", "rows_in_upload", "example_instrument"]
    if df_clean.empty:
        return pd.DataFrame(columns=report_columns)

    domestic = df_clean["asset_type"].astype("string").str.strip().str.casefold().eq("domestic equities")
    holdings = df_clean.loc[domestic.fillna(False), ["ace_code", "instrument"]].copy()
    if holdings.empty:
        return pd.DataFrame(columns=report_columns)

    holdings["ace_code"] = holdings["ace_code"].astype("string").str.strip().replace("", pd.NA)
    report = holdings.groupby("ace_code", as_index=False, dropna=False).agg(
        rows_in_upload=("ace_code", "size"),
        example_instrument=("instrument", "first"),
    )
    if conn is None:
        with get_engine().connect() as opened_conn:
            return find_missing_portfolio_ace_codes(df_clean, opened_conn)

    codes = report["ace_code"].dropna().astype(str).tolist()
    existing = set()
    if codes:
        existing = set(conn.execute(text("""
            SELECT ace_code FROM fundlab.stock_master_ace
            WHERE ace_code = ANY(CAST(:ace_codes AS varchar[]))
        """), {"ace_codes": codes}).scalars())
    missing = report.loc[
        report["ace_code"].isna() | ~report["ace_code"].isin(existing), report_columns
    ].copy()
    missing["ace_code"] = missing["ace_code"].fillna("(blank)")
    return missing.sort_values("ace_code").reset_index(drop=True)


class CompanyNameAceConflictError(ValueError):
    """An uploaded company name is already mapped to another ACE code."""

    def __init__(self, report: pd.DataFrame):
        self.report = report
        super().__init__(
            "Stock-master upload aborted: company names already map to different ACE codes."
        )


def load_ace_industry_flags(conn=None) -> dict[str, bool]:
    """Return case-insensitive industry classifications from stock_master_ace."""
    if conn is None:
        with get_engine().connect() as opened_conn:
            return load_ace_industry_flags(opened_conn)

    flags = {}
    rows = conn.execute(text("""
        SELECT industry, is_financial FROM fundlab.stock_master_ace
        WHERE industry IS NOT NULL
    """))
    for industry, is_financial in rows:
        key = industry.strip().casefold()
        if not key or is_financial is None:
            continue
        if key in flags and flags[key] != bool(is_financial):
            raise ValueError(
                f"Existing stock_master_ace rows disagree on financial classification for '{industry}'."
            )
        flags[key] = bool(is_financial)
    return flags


def find_company_name_ace_conflicts(df_clean: pd.DataFrame, conn=None) -> pd.DataFrame:
    """Flag existing company-name mappings that disagree with the workbook."""
    columns = ["company_name", "uploaded_ace_code", "existing_name", "existing_ace_code"]
    if df_clean.empty:
        return pd.DataFrame(columns=columns)
    if conn is None:
        with get_engine().connect() as opened_conn:
            return find_company_name_ace_conflicts(df_clean, opened_conn)

    keys = df_clean["company_name_key"].drop_duplicates().astype(str).tolist()
    existing = conn.execute(text("""
        SELECT company_name, ace_code FROM fundlab.company_name_acecode_map
        WHERE lower(trim(company_name)) = ANY(CAST(:names AS text[]))
    """), {"names": keys}).fetchall()
    existing_by_key = {}
    for company_name, ace_code in existing:
        existing_by_key.setdefault(company_name.strip().casefold(), []).append(
            (company_name, ace_code)
        )

    conflicts = []
    for row in df_clean.itertuples(index=False):
        for existing_name, existing_code in existing_by_key.get(row.company_name_key, []):
            if existing_code is not None and existing_code != row.ace_code:
                conflicts.append({
                    "company_name": row.company_name,
                    "uploaded_ace_code": row.ace_code,
                    "existing_name": existing_name,
                    "existing_ace_code": existing_code,
                })
    return pd.DataFrame(conflicts, columns=columns).drop_duplicates().reset_index(drop=True)

def find_conflicts_annual(df_clean: pd.DataFrame):
    """Check (ace_code, year_end) conflicts for annual book value."""
    if df_clean.empty:
        return []

    engine = get_engine()

    pairs = df_clean[["ace_code", "year_end"]].drop_duplicates()
    max_rows = min(len(pairs), 5000)
    pairs = pairs.iloc[:max_rows]

    values_clause = ", ".join(
        f"(:ace_code{i}, :year_end{i})" for i in range(len(pairs))
    )

    params = {}
    for i, (_, row) in enumerate(pairs.iterrows()):
        params[f"ace_code{i}"] = row["ace_code"]
        params[f"year_end{i}"] = row["year_end"]

    query = text(
        f"""
        SELECT v.ace_code, v.year_end
        FROM (VALUES
            {values_clause}
        ) AS v(ace_code, year_end)
        JOIN fundlab.stock_annual_bv_ace b
        USING (ace_code, year_end)
        LIMIT 20;
        """
    )

    with engine.begin() as conn:
        rows = conn.execute(query, params).fetchall()

    return rows


def load_stock_master_ace_codes():
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql("SELECT ace_code FROM fundlab.stock_master_ace", conn)
    return set(df["ace_code"].astype(str))


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

def load_existing_fund_names() -> set[str]:
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql("select fund_name from fundlab.fund", conn)
    return set(df["fund_name"].astype(str).str.strip())
