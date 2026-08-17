"""Database loaders for the Portfolio explorer feature."""

from datetime import date

import pandas as pd
from sqlalchemy import text

from core.db import get_engine


def _apply_frequency_filter(df: pd.DataFrame, end_date: date, freq: str) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["month_end"] = pd.to_datetime(df["month_end"]).dt.date
    if freq == "Quarterly":
        return df[df["month_end"].apply(lambda value: value.month in (3, 6, 9, 12))]
    if freq == "Yearly":
        return df[df["month_end"].apply(lambda value: value.month == end_date.month)]
    return df


def fetch_fund_portfolio_timeseries(fund_id: int, start_date: date, end_date: date, freq: str) -> pd.DataFrame:
    """Fetch a fund's basic holdings time series, filtered to the requested frequency."""
    query = text("""
        SELECT fp.month_end, fp.isin, fp.holding_weight AS weight_pct, sm.company_name
        FROM fundlab.fund_portfolio fp
        JOIN fundlab.stock_master sm ON fp.isin = sm.isin
        WHERE fp.fund_id = :fund_id AND fp.month_end BETWEEN :start_date AND :end_date
        ORDER BY fp.month_end, sm.company_name
    """)
    df = pd.read_sql(query, get_engine(), params={"fund_id": fund_id, "start_date": start_date, "end_date": end_date})
    return _apply_frequency_filter(df, end_date, freq)


def fetch_multi_fund_portfolios(fund_ids: list[int], start_date: date, end_date: date, freq: str) -> pd.DataFrame:
    """Fetch holdings for multiple funds, filtered to the requested frequency."""
    if not fund_ids:
        return pd.DataFrame()
    query = text("""
        SELECT fp.fund_id, fp.month_end, fp.isin, fp.holding_weight AS weight_pct
        FROM fundlab.fund_portfolio fp
        WHERE fp.fund_id = ANY(:fund_ids) AND fp.month_end BETWEEN :start_date AND :end_date
        ORDER BY fp.fund_id, fp.month_end, fp.isin
    """)
    df = pd.read_sql(query, get_engine(), params={"fund_ids": fund_ids, "start_date": start_date, "end_date": end_date})
    return _apply_frequency_filter(df, end_date, freq)


def fetch_portfolio_view_data(fund_id: int, start_date: date, end_date: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch raw holdings and size-band rows needed by the holdings display."""
    holdings_query = text("""
        SELECT fund_id, month_end, instrument_name AS company_name, isin, asset_type,
               holding_weight AS weight_pct
        FROM fundlab.fund_portfolio
        WHERE fund_id = :fund_id AND month_end >= :start_date AND month_end <= :end_date
    """)
    holdings = pd.read_sql(holdings_query, get_engine(), params={"fund_id": fund_id, "start_date": start_date, "end_date": end_date})
    size_band_query = text("""
        SELECT isin, band_date AS month_end, size_band
        FROM fundlab.stock_size_band
        WHERE band_date >= :start_date AND band_date <= :end_date
    """)
    size_bands = pd.read_sql(size_band_query, get_engine(), params={"start_date": start_date, "end_date": end_date})
    return holdings, size_bands
