"""Read-only batched inputs for live Portfolio Valuations analysis."""

import datetime as dt

import pandas as pd
import streamlit as st
from sqlalchemy import text

from core.dataframes import ensure_unique_monthly_rows
from core.db import get_engine


HOLDING_COLUMNS = [
    "fund_id",
    "holding_month_end",
    "anchor_month_end",
    "isin",
    "weight_pct",
    "is_financial",
]
MULTIPLE_COLUMNS = ["isin", "month_end", "ps", "pe", "pb"]


def _empty_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.DataFrame(columns=HOLDING_COLUMNS), pd.DataFrame(columns=MULTIPLE_COLUMNS)


def load_portfolio_valuation_inputs(
    fund_ids: list[int],
    start_date: dt.date,
    end_date: dt.date,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load holdings and stock multiples for all selected funds in two batched queries."""
    if not fund_ids:
        return _empty_inputs()

    if mode not in {
        "Valuations of historical portfolios",
        "Historical valuations of current portfolio",
    }:
        raise ValueError(f"Unsupported valuation mode: {mode}")

    engine = get_engine()
    if mode == "Valuations of historical portfolios":
        holdings_sql = text(
            """
            SELECT
                fp.fund_id,
                fp.month_end::date AS holding_month_end,
                fp.month_end::date AS anchor_month_end,
                fp.isin,
                fp.holding_weight AS weight_pct,
                sm.is_financial
            FROM fundlab.fund_portfolio fp
            JOIN fundlab.stock_master sm ON sm.isin = fp.isin
            WHERE fp.fund_id = ANY(:fund_ids)
              AND fp.asset_type = 'Domestic Equities'
              AND fp.month_end BETWEEN :start_date AND :end_date
            ORDER BY fp.fund_id, fp.month_end, fp.holding_weight DESC, fp.isin
            """
        )
        holdings_params = {
            "fund_ids": fund_ids,
            "start_date": start_date,
            "end_date": end_date,
        }
    else:
        holdings_sql = text(
            """
            WITH anchors AS (
                SELECT fund_id, MAX(month_end) AS anchor_month_end
                FROM fundlab.fund_portfolio
                WHERE fund_id = ANY(:fund_ids)
                  AND asset_type = 'Domestic Equities'
                  AND month_end <= :end_date
                GROUP BY fund_id
            )
            SELECT
                fp.fund_id,
                fp.month_end::date AS holding_month_end,
                a.anchor_month_end::date AS anchor_month_end,
                fp.isin,
                fp.holding_weight AS weight_pct,
                sm.is_financial
            FROM anchors a
            JOIN fundlab.fund_portfolio fp
              ON fp.fund_id = a.fund_id
             AND fp.month_end = a.anchor_month_end
            JOIN fundlab.stock_master sm ON sm.isin = fp.isin
            WHERE fp.asset_type = 'Domestic Equities'
            ORDER BY fp.fund_id, fp.holding_weight DESC, fp.isin
            """
        )
        holdings_params = {"fund_ids": fund_ids, "end_date": end_date}

    with engine.begin() as conn:
        holdings = pd.read_sql(holdings_sql, conn, params=holdings_params)

    if holdings.empty:
        return _empty_inputs()

    holdings["fund_id"] = pd.to_numeric(holdings["fund_id"], errors="coerce")
    holdings["isin"] = holdings["isin"].astype(str).str.strip()
    holdings["weight_pct"] = pd.to_numeric(holdings["weight_pct"], errors="coerce")
    holdings["is_financial"] = holdings["is_financial"].fillna(False).astype(bool)
    for column in ("holding_month_end", "anchor_month_end"):
        holdings[column] = (
            pd.to_datetime(holdings[column], errors="coerce")
            .dt.to_period("M")
            .dt.to_timestamp("M")
        )
    holdings = holdings.dropna(
        subset=["fund_id", "holding_month_end", "anchor_month_end", "isin", "weight_pct"]
    )
    holdings["fund_id"] = holdings["fund_id"].astype(int)
    holdings = holdings[holdings["weight_pct"] > 0].copy()
    if holdings.empty:
        return _empty_inputs()

    isins = sorted(holdings["isin"].unique().tolist())
    multiples_sql = text(
        """
        SELECT isin, month_end::date AS month_end, ps, pe, pb
        FROM fundlab.stock_monthly_valuations
        WHERE isin = ANY(:isins)
          AND month_end BETWEEN :start_date AND :end_date
        ORDER BY isin, month_end
        """
    )
    with engine.begin() as conn:
        multiples = pd.read_sql(
            multiples_sql,
            conn,
            params={"isins": isins, "start_date": start_date, "end_date": end_date},
        )

    if multiples.empty:
        return holdings[HOLDING_COLUMNS].reset_index(drop=True), pd.DataFrame(columns=MULTIPLE_COLUMNS)

    multiples["isin"] = multiples["isin"].astype(str).str.strip()
    multiples["month_end"] = (
        pd.to_datetime(multiples["month_end"], errors="coerce")
        .dt.to_period("M")
        .dt.to_timestamp("M")
    )
    for column in ("ps", "pe", "pb"):
        multiples[column] = pd.to_numeric(multiples[column], errors="coerce")
    multiples = multiples.dropna(subset=["isin", "month_end"])
    multiples["month_key"] = multiples["month_end"].dt.to_period("M")
    multiples = ensure_unique_monthly_rows(
        multiples,
        key_cols=["isin", "month_key"],
        value_cols=["ps", "pe", "pb"],
        context="fundlab.stock_monthly_valuations",
    ).drop(columns=["month_key"])

    return (
        holdings[HOLDING_COLUMNS].reset_index(drop=True),
        multiples[MULTIPLE_COLUMNS].reset_index(drop=True),
    )


@st.cache_data(ttl=60 * 30, show_spinner=False)
def cached_portfolio_valuation_inputs(
    fund_ids: tuple[int, ...],
    start_date: dt.date,
    end_date: dt.date,
    mode: str,
    data_version: str = "stock-multiples-v2",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cache the read-only input frames; calculations remain live and local."""
    del data_version
    return load_portfolio_valuation_inputs(list(fund_ids), start_date, end_date, mode)
