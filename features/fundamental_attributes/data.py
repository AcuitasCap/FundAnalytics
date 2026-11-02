"""Batched, read-only ACE inputs for fundamental attribute comparisons."""

import datetime as dt

import pandas as pd
import streamlit as st
from sqlalchemy import text

from core.dataframes import ensure_unique_monthly_rows
from core.db import get_engine
from . import BANDS, COMPARATORS, CURRENT, HISTORICAL


HOLDING_COLUMNS = ["fund_id", "holding_month_end", "anchor_month_end", "holding_key",
                   "ace_code", "weight_pct", "is_financial"]
MEMBER_COLUMNS = ["ace_code", "month_end", "is_financial"]


def load_attribute_inputs(fund_ids, start_date, end_date, mode, compare, kind):
    """Read holdings, historical cohort membership and stock facts in one snapshot.

    Holdings with missing identifiers remain in the weight ranking. No migration
    map or company-name join is needed, and no derived data is written.
    """
    if mode not in {HISTORICAL, CURRENT} or compare not in COMPARATORS or kind not in {"valuation", "growth"}:
        raise ValueError("Unsupported fundamental attribute settings.")
    if start_date > end_date:
        raise ValueError("Start date must be on or before end date.")
    fact_columns = (["ace_code", "month_end", "ps", "pe", "pb"] if kind == "valuation"
                    else ["ace_code", "period_end", "sales", "pat"])
    empty = (pd.DataFrame(columns=HOLDING_COLUMNS), pd.DataFrame(columns=MEMBER_COLUMNS),
             pd.DataFrame(columns=fact_columns))
    if not fund_ids:
        return empty
    params = {"fund_ids": list(fund_ids), "start_date": start_date, "end_date": end_date}
    if mode == CURRENT:
        anchor_sql = """
            WITH anchors AS (
                SELECT fund_id, MAX(month_end) AS anchor_month_end
                FROM fundlab.fund_portfolio_ace
                WHERE fund_id = ANY(:fund_ids) AND month_end <= :end_date
                  AND asset_type = 'Domestic Equities' AND holding_weight > 0
                GROUP BY fund_id
            )
        """
        source_sql = """FROM anchors a JOIN fundlab.fund_portfolio_ace fp
                        ON fp.fund_id = a.fund_id AND fp.month_end = a.anchor_month_end"""
        period_sql = ""
    else:
        anchor_sql = ""
        source_sql = "FROM fundlab.fund_portfolio_ace fp"
        period_sql = "AND fp.month_end BETWEEN :start_date AND :end_date"
    holdings_sql = text(f"""
        {anchor_sql}
        SELECT fp.fund_id, fp.month_end::date AS holding_month_end,
               fp.month_end::date AS anchor_month_end, fp.holding_id::text AS holding_key,
               fp.ace_code, fp.holding_weight AS weight_pct, sm.is_financial
        {source_sql}
        LEFT JOIN fundlab.stock_master_ace sm ON sm.ace_code = fp.ace_code
        WHERE fp.fund_id = ANY(:fund_ids) AND fp.asset_type = 'Domestic Equities'
          AND fp.holding_weight > 0 {period_sql}
        ORDER BY fp.fund_id, fp.month_end, fp.holding_weight DESC, fp.holding_id
    """)
    with get_engine().begin() as conn:
        conn.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        holdings = pd.read_sql(holdings_sql, conn, params=params)
        members = empty[1]
        if holdings.empty:
            return empty
        # Use one complete dated snapshot per month. Per-stock as-of joins can
        # keep delisted/removed constituents from older snapshots in the cohort.
        if compare != "Peers":
            cohort_params = dict(params, size_band=BANDS.get(compare), compare=compare)
            members = pd.read_sql(text("""
                WITH months AS (
                    SELECT (d + INTERVAL '1 month - 1 day')::date AS month_end
                    FROM generate_series(date_trunc('month', CAST(:start_date AS date)),
                         date_trunc('month', CAST(:end_date AS date)), INTERVAL '1 month') d
                )
                SELECT b.ace_code, m.month_end, sm.is_financial
                FROM months m
                JOIN LATERAL (
                    SELECT MAX(band.band_date) AS band_date
                    FROM fundlab.stock_size_band_ace band
                    WHERE band.band_date <= m.month_end
                ) snapshot ON TRUE
                JOIN fundlab.stock_size_band_ace b ON b.band_date = snapshot.band_date
                JOIN fundlab.stock_master_ace sm ON sm.ace_code = b.ace_code
                WHERE (:compare = 'Top 500' AND b.size_band IN ('Large', 'Mid', 'Small'))
                   OR (:compare <> 'Top 500' AND b.size_band = :size_band)
                ORDER BY m.month_end, b.ace_code
            """), conn, params=cohort_params)
        codes = sorted(set(holdings["ace_code"].dropna().astype(str)) |
                       set(members["ace_code"].dropna().astype(str)))
        facts = empty[2]
        if codes:
            if kind == "valuation":
                fact_sql = """
                    SELECT ace_code, month_end::date AS month_end, ps, pe, pb
                    FROM fundlab.stock_monthly_valuations_ace
                    WHERE ace_code = ANY(:ace_codes) AND month_end BETWEEN :fact_start AND :end_date
                    ORDER BY ace_code, month_end
                """
                fact_start = start_date
            else:
                fact_sql = """
                    SELECT ace_code, period_end::date AS period_end, sales, pat
                    FROM fundlab.stock_quarterly_financials_ace
                    WHERE ace_code = ANY(:ace_codes) AND period_end BETWEEN :fact_start AND :end_date
                    ORDER BY ace_code, period_end
                """
                # Three-year base TTM needs four earlier consecutive quarters.
                fact_start = (pd.Timestamp(start_date) - pd.DateOffset(years=4)).date().replace(day=1)
            facts = pd.read_sql(text(fact_sql), conn,
                                params={"ace_codes": codes, "fact_start": fact_start, "end_date": end_date})
    for frame in (holdings, members, facts):
        frame["ace_code"] = frame["ace_code"].astype("string").str.strip().replace("", pd.NA)
    for column in ("holding_month_end", "anchor_month_end"):
        holdings[column] = pd.to_datetime(holdings[column]).dt.to_period("M").dt.to_timestamp("M")
    holdings["weight_pct"] = pd.to_numeric(holdings["weight_pct"], errors="coerce")
    holdings["fund_id"] = holdings["fund_id"].astype(int)
    members["month_end"] = pd.to_datetime(members["month_end"]).dt.to_period("M").dt.to_timestamp("M")
    date_column = "month_end" if kind == "valuation" else "period_end"
    facts[date_column] = pd.to_datetime(facts[date_column])
    value_columns = ["ps", "pe", "pb"] if kind == "valuation" else ["sales", "pat"]
    for column in value_columns:
        facts[column] = pd.to_numeric(facts[column], errors="coerce")
    facts = ensure_unique_monthly_rows(facts, ["ace_code", date_column], value_columns,
                                       "Fundamental attribute stock inputs")
    return holdings, members, facts


@st.cache_data(ttl=60 * 30, show_spinner=False)
def cached_attribute_inputs(fund_ids: tuple, start_date: dt.date, end_date: dt.date,
                            mode: str, compare: str, kind: str):
    return load_attribute_inputs(fund_ids, start_date, end_date, mode, compare, kind)
