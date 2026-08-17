from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from core.db import get_engine

def find_conflicts_quarterly(df_clean: pd.DataFrame, table: str):
    """
    Check for (isin, period_end) conflicts in either PAT or sales upload.
    Returns a small list of conflicts if found.
    """
    if df_clean.empty:
        return []

    engine = get_engine()

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

    engine = get_engine()

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


def load_stock_master_isins():
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql("SELECT isin FROM fundlab.stock_master", conn)
    return set(df["isin"].astype(str))


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
