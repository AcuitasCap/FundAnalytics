"""Database loaders for the Portfolio Quality feature."""

from datetime import date

import pandas as pd
import streamlit as st
from sqlalchemy import text

from core.db import get_engine


@st.cache_data(ttl=60, show_spinner=False)
def load_stock_roe_roce() -> dict[str, pd.DataFrame]:
    """Load RoE/RoCE history keyed by ISIN for fast calculation lookups."""
    query = """
        SELECT isin, year_end_date, roe, roce
        FROM fundlab.stock_roe_roce
        WHERE roe IS NOT NULL OR roce IS NOT NULL
        ORDER BY isin, year_end_date
    """
    df = pd.read_sql(query, get_engine())
    if df.empty:
        return {}
    df["year_end_date"] = pd.to_datetime(df["year_end_date"]).dt.date
    return {
        isin: sub_df.sort_values("year_end_date").reset_index(drop=True)
        for isin, sub_df in df.groupby("isin")
    }


@st.cache_data(ttl=60, show_spinner=False)
def fetch_portfolio_raw(
    fund_ids: list[int], start_date: date, end_date: date
) -> pd.DataFrame:
    """Fetch March/September fund holdings required for Quality calculations."""
    if not fund_ids:
        return pd.DataFrame()
    query = text("""
        SELECT fp.fund_id, fm.fund_name, fp.month_end, fp.isin,
               fp.holding_weight AS weight_pct, fp.asset_type, sm.is_financial
        FROM fundlab.fund_portfolio fp
        JOIN fundlab.fund fm ON fp.fund_id = fm.fund_id
        JOIN fundlab.stock_master sm ON fp.isin = sm.isin
        WHERE fp.fund_id = ANY(:fund_ids)
          AND fp.month_end BETWEEN :start_date AND :end_date
          AND EXTRACT(MONTH FROM fp.month_end) IN (3, 9)
        ORDER BY fp.fund_id, fp.month_end, fp.isin
    """)
    df = pd.read_sql(query, get_engine(), params={
        "fund_ids": fund_ids, "start_date": start_date, "end_date": end_date,
    })
    if not df.empty:
        df["month_end"] = pd.to_datetime(df["month_end"]).dt.date
    return df


@st.cache_data(ttl=60, show_spinner=False)
def fetch_quality_bucket_rows(
    fund_id: int, start_date: date, end_date: date
) -> pd.DataFrame:
    """Fetch raw domestic-equity quality-quartile rows for one fund."""
    query = text("""
        SELECT fp.month_end, fp.holding_weight, sq.quality_quartile
        FROM fundlab.fund_portfolio fp
        JOIN fundlab.stock_quality_quartile sq
          ON sq.isin = fp.isin AND sq.month_end = fp.month_end
        WHERE fp.fund_id = :fund_id
          AND fp.month_end BETWEEN :start_date AND :end_date
          AND fp.asset_type = 'Domestic Equities'
    """)
    return pd.read_sql(query, get_engine(), params={
        "fund_id": fund_id, "start_date": start_date, "end_date": end_date,
    })
