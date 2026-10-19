"""Read-only Supabase inputs for market valuation analysis."""

import datetime as dt

import pandas as pd
import streamlit as st
from sqlalchemy import text

from core.dataframes import ensure_unique_monthly_rows
from core.db import get_engine


METRIC_COLUMNS = {"P/S": "ps", "P/E": "pe", "P/B": "pb"}
COHORT_BANDS = {"Large caps": "Large", "Mid caps": "Mid", "Small caps": "Small"}
SEGMENTS = {"Total", "Financials", "Non-financials"}
NIFTY_50_ETF_NAME = "Nippon India ETF Nifty 50 BeES"


@st.cache_data(ttl=60 * 30, show_spinner=False)
def market_valuation_bounds() -> tuple[dt.date | None, dt.date | None]:
    """Return the available monthly multiple range without loading stock rows."""
    query = text(
        "SELECT MIN(month_end)::date AS first_month, MAX(month_end)::date AS last_month "
        "FROM fundlab.stock_monthly_valuations"
    )
    with get_engine().connect() as conn:
        first_month, last_month = conn.execute(query).one()
    return first_month, last_month


@st.cache_data(ttl=60 * 30, show_spinner=False)
def nifty_50_holdings_bounds() -> tuple[dt.date | None, dt.date | None]:
    """Return the available monthly domestic-equity holdings range for the ETF."""
    query = text(
        """
        SELECT MIN(fp.month_end)::date, MAX(fp.month_end)::date
        FROM fundlab.fund_portfolio fp
        JOIN fundlab.fund f ON f.fund_id = fp.fund_id
        WHERE f.fund_name = :fund_name
          AND fp.asset_type = 'Domestic Equities'
          AND fp.holding_weight > 0
          AND fp.isin IS NOT NULL
        """
    )
    with get_engine().connect() as conn:
        return tuple(conn.execute(query, {"fund_name": NIFTY_50_ETF_NAME}).one())


def load_market_stock_multiples(
    cohort: str,
    segment: str,
    metric: str,
    start_date: dt.date,
    end_date: dt.date,
) -> pd.DataFrame:
    """Load stock multiples using same-month ETF holdings or historical size bands."""
    if cohort not in {*COHORT_BANDS, "Nifty 50"}:
        raise ValueError(f"Unsupported market cohort: {cohort}")
    if segment not in SEGMENTS:
        raise ValueError(f"Unsupported segment: {segment}")
    if metric not in METRIC_COLUMNS:
        raise ValueError(f"Unsupported valuation metric: {metric}")
    if start_date > end_date:
        raise ValueError("Start date must be on or before end date.")

    metric_column = METRIC_COLUMNS[metric]
    params = {
        "start_date": start_date.replace(day=1),
        "end_date": end_date,
        "segment": segment,
    }
    if cohort == "Nifty 50":
        query = text(
            f"""
            WITH constituents AS (
                SELECT DISTINCT fp.isin,
                       date_trunc('month', fp.month_end)::date AS month_start
                FROM fundlab.fund_portfolio fp
                JOIN fundlab.fund f ON f.fund_id = fp.fund_id
                WHERE f.fund_name = :fund_name
                  AND fp.asset_type = 'Domestic Equities'
                  AND fp.holding_weight > 0
                  AND fp.isin IS NOT NULL
                  AND fp.month_end BETWEEN :start_date AND :end_date
            )
            SELECT v.isin, v.month_end::date AS month_end,
                   v.{metric_column} AS multiple
            FROM constituents c
            JOIN fundlab.stock_monthly_valuations v
              ON v.isin = c.isin
             AND v.month_end >= c.month_start
             AND v.month_end < c.month_start + INTERVAL '1 month'
            JOIN fundlab.stock_master sm ON sm.isin = v.isin
            WHERE (:segment = 'Total'
                   OR (:segment = 'Financials' AND sm.is_financial IS TRUE)
                   OR (:segment = 'Non-financials' AND sm.is_financial IS FALSE))
            ORDER BY v.month_end, v.isin
            """
        )
        params["fund_name"] = NIFTY_50_ETF_NAME
    else:
        query = text(
            f"""
            SELECT v.isin, v.month_end::date AS month_end,
                   v.{metric_column} AS multiple
            FROM fundlab.stock_monthly_valuations v
            JOIN fundlab.stock_master sm ON sm.isin = v.isin
            JOIN LATERAL (
                SELECT band.size_band
                FROM fundlab.stock_size_band band
                WHERE band.isin = v.isin AND band.band_date <= v.month_end
                ORDER BY band.band_date DESC
                LIMIT 1
            ) b ON TRUE
            WHERE v.month_end BETWEEN :start_date AND :end_date
              AND b.size_band = :size_band
              AND (:segment = 'Total'
                   OR (:segment = 'Financials' AND sm.is_financial IS TRUE)
                   OR (:segment = 'Non-financials' AND sm.is_financial IS FALSE))
            ORDER BY v.month_end, v.isin
            """
        )
        params["size_band"] = COHORT_BANDS[cohort]
    with get_engine().connect() as conn:
        frame = pd.read_sql(query, conn, params=params)
    if frame.empty:
        return pd.DataFrame(columns=["isin", "month_end", "multiple"])
    frame["isin"] = frame["isin"].astype(str).str.strip()
    frame["month_end"] = (
        pd.to_datetime(frame["month_end"], errors="coerce")
        .dt.to_period("M")
        .dt.to_timestamp("M")
    )
    frame["multiple"] = pd.to_numeric(frame["multiple"], errors="coerce")
    frame = frame.dropna(subset=["isin", "month_end"])
    return ensure_unique_monthly_rows(
        frame,
        key_cols=["isin", "month_end"],
        value_cols=["multiple"],
        context="fundlab.stock_monthly_valuations",
    ).reset_index(drop=True)


@st.cache_data(ttl=60 * 30, show_spinner=False)
def cached_market_stock_multiples(
    cohort: str,
    segment: str,
    metric: str,
    start_date: dt.date,
    end_date: dt.date,
) -> pd.DataFrame:
    """Cache read-only stock inputs; percentile calculations remain local."""
    return load_market_stock_multiples(cohort, segment, metric, start_date, end_date)
