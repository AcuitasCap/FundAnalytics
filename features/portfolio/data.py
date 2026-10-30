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
        SELECT fp.month_end, fp.ace_code, fp.holding_weight AS weight_pct,
               fp.instrument_name AS company_name
        FROM fundlab.fund_portfolio_ace fp
        JOIN fundlab.stock_master_ace sm ON fp.ace_code = sm.ace_code
        WHERE fp.fund_id = :fund_id AND fp.month_end BETWEEN :start_date AND :end_date
        ORDER BY fp.month_end, fp.instrument_name
    """)
    df = pd.read_sql(query, get_engine(), params={"fund_id": fund_id, "start_date": start_date, "end_date": end_date})
    return _apply_frequency_filter(df, end_date, freq)


def fetch_multi_fund_portfolios(fund_ids: list[int], start_date: date, end_date: date, freq: str) -> pd.DataFrame:
    """Fetch holdings for multiple funds, filtered to the requested frequency."""
    if not fund_ids:
        return pd.DataFrame()
    query = text("""
        SELECT fp.fund_id, fp.month_end, fp.ace_code, fp.holding_weight AS weight_pct
        FROM fundlab.fund_portfolio_ace fp
        WHERE fp.fund_id = ANY(:fund_ids) AND fp.month_end BETWEEN :start_date AND :end_date
        ORDER BY fp.fund_id, fp.month_end, fp.ace_code
    """)
    df = pd.read_sql(query, get_engine(), params={"fund_ids": fund_ids, "start_date": start_date, "end_date": end_date})
    return _apply_frequency_filter(df, end_date, freq)


def fetch_portfolio_view_data(fund_id: int, start_date: date, end_date: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch raw holdings and size-band rows needed by the holdings display."""
    holdings_query = text("""
        SELECT fund_id, month_end, instrument_name AS company_name, ace_code, asset_type,
               holding_weight AS weight_pct
        FROM fundlab.fund_portfolio_ace
        WHERE fund_id = :fund_id AND month_end >= :start_date AND month_end <= :end_date
    """)
    holdings = pd.read_sql(holdings_query, get_engine(), params={"fund_id": fund_id, "start_date": start_date, "end_date": end_date})
    size_band_query = text("""
        WITH relevant_codes AS (
            SELECT DISTINCT ace_code
            FROM fundlab.fund_portfolio_ace
            WHERE fund_id = :fund_id
              AND month_end BETWEEN :start_date AND :end_date
              AND ace_code IS NOT NULL
        )
        SELECT sb.ace_code, sb.band_date AS month_end, sb.size_band
        FROM fundlab.stock_size_band_ace sb
        JOIN relevant_codes c ON c.ace_code = sb.ace_code
        WHERE sb.band_date BETWEEN :start_date AND :end_date
        UNION ALL
        SELECT c.ace_code, prior.band_date AS month_end, prior.size_band
        FROM relevant_codes c
        CROSS JOIN LATERAL (
            SELECT band_date, size_band
            FROM fundlab.stock_size_band_ace
            WHERE ace_code = c.ace_code AND band_date < :start_date
            ORDER BY band_date DESC
            LIMIT 1
        ) prior
        ORDER BY ace_code, month_end
    """)
    size_bands = pd.read_sql(size_band_query, get_engine(), params={"fund_id": fund_id, "start_date": start_date, "end_date": end_date})
    return holdings, size_bands
