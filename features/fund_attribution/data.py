import datetime as dt

import pandas as pd
import streamlit as st
from sqlalchemy import text

from core.db import get_engine
from core.dates import to_month_end
from features.fund_attribution.fundamentals import FUNDAMENTAL_FIELDS

# ======================== Fund Attribution Page ========================

CASH_ANNUAL_RETURN = 0.07

BENCH_NAME_NIFTY50  = "NIFTY 50 - TRI"
BENCH_NAME_NIFTY500 = "NIFTY 500 - TRI"
BENCH_NAME_NIFTY100 = "NIFTY 100 - TRI"  # assumed present in DB (you will upload)
BENCH_NAME_MID150   = "Nifty Midcap 150 - TRI"
BENCH_NAME_SMALL250 = "Nifty Smallcap 250 - TRI"

LIKE_FOR_LIKE_BENCH = {
    "Large": BENCH_NAME_NIFTY100,
    "Mid":   BENCH_NAME_MID150,
    "Small": BENCH_NAME_SMALL250,
}

def _monthly_cash_return() -> float:
    # 7% annualized -> monthly compounded
    return (1.0 + CASH_ANNUAL_RETURN) ** (1.0/12.0) - 1.0

@st.cache_data(ttl=3600, show_spinner=False)
def _load_attrib_raw_window(
    fund_id: int,
    end_month_end: dt.date,
    lookback_years: int = 10,
    data_version: str = "ace-v3",
):
    """Load ACE-keyed raw inputs for attribution for a single fund over [end-LOOKBACK, end]."""
    engine = get_engine()
    end_me = to_month_end(end_month_end)
    start_me = (pd.Timestamp(end_me) - pd.DateOffset(years=lookback_years)).to_period("M").to_timestamp("M").date()
    # Records can use the last trading day while the selector uses calendar month-end.
    # Include the entire first month in each query before normalizing to month-end.
    bounds = {"start_me": start_me.replace(day=1), "end_me": end_me}

    # Holdings (all asset types)
    q_hold = text("""
        SELECT fund_id,
               month_end::date AS month_end,
               instrument_name,
               asset_type,
               ace_code,
               holding_weight
        FROM fundlab.fund_portfolio_ace
        WHERE fund_id = :fund_id
          AND month_end BETWEEN :start_me AND :end_me
        ORDER BY month_end, instrument_name
    """)
    h = pd.read_sql(q_hold, engine, params={"fund_id": fund_id, **bounds})
    if h.empty:
        return {
            "window_start": start_me,
            "window_end": end_me,
            "holdings": h,
            "prices": pd.DataFrame(),
            "size_band": pd.DataFrame(),
            "bench_nav": pd.DataFrame(),
            "stock_master": pd.DataFrame(),
            "weight_scale": None,
        }

    h["month_end"] = pd.to_datetime(h["month_end"]).dt.to_period("M").dt.to_timestamp("M").dt.date

    # Normalise holding_weight scale to 0-1.
    hw = pd.to_numeric(h["holding_weight"], errors="coerce")
    hw_nonnull = hw.dropna()

    if hw_nonnull.empty:
        weight_scale = 1.0
        h["holding_weight"] = hw
    else:
        per_month_sum = hw_nonnull.groupby(h.loc[hw_nonnull.index, "month_end"]).sum()
        med_sum = float(per_month_sum.median()) if not per_month_sum.empty else float(hw_nonnull.sum())
        max_w = float(hw_nonnull.max())

        if med_sum > 1500 or max_w > 1500:
            weight_scale = 1.0 / 10000.0
        elif med_sum > 15 or max_w > 1.5:
            weight_scale = 1.0 / 100.0
        else:
            weight_scale = 1.0

        h["holding_weight"] = hw * weight_scale

    # ACE code list (exclude null/blank)
    ace_codes = sorted([x for x in h["ace_code"].dropna().astype(str).unique().tolist() if x.strip()])

    # ACE master has no name column. Use one deterministic mapped name per
    # company, so multiple aliases cannot multiply holdings or return rows.
    if ace_codes:
        q_sm = text("""
            SELECT sm.ace_code, names.company_name, sm.industry, sm.is_financial
            FROM fundlab.stock_master_ace sm
            LEFT JOIN (
                SELECT ace_code, MIN(company_name) AS company_name
                FROM fundlab.company_name_acecode_map
                WHERE ace_code = ANY(:ace_codes)
                GROUP BY ace_code
            ) names ON names.ace_code = sm.ace_code
            WHERE sm.ace_code = ANY(:ace_codes)
        """)
        sm = pd.read_sql(q_sm, engine, params={"ace_codes": ace_codes})
        holding_names = (
            h.dropna(subset=["ace_code", "instrument_name"])
            .sort_values(["month_end", "instrument_name"])
            .drop_duplicates("ace_code", keep="last")
            .set_index("ace_code")["instrument_name"]
        )
        sm["company_name"] = sm["company_name"].fillna(sm["ace_code"].map(holding_names)).fillna(sm["ace_code"])
    else:
        sm = pd.DataFrame(columns=["ace_code", "company_name", "industry", "is_financial"])

    # Prices (monthly) with adjusted price (returns use adj_price basis)
    if ace_codes:
        q_px = text("""
            SELECT ace_code,
                   price_date::date AS month_end,
                   adj_price,
                   price,
                   dividend_yield
            FROM fundlab.stock_price_ace
            WHERE ace_code = ANY(:ace_codes)
              AND price_date BETWEEN :start_me AND :end_me
            ORDER BY ace_code, price_date
        """)
        px = pd.read_sql(q_px, engine, params={"ace_codes": ace_codes, **bounds})
        if not px.empty:
            px["month_end"] = pd.to_datetime(px["month_end"]).dt.to_period("M").dt.to_timestamp("M").dt.date
            px["adj_price"] = pd.to_numeric(px["adj_price"], errors="coerce")
            px["price"] = pd.to_numeric(px["price"], errors="coerce")
            px["dividend_yield"] = pd.to_numeric(px["dividend_yield"], errors="coerce")
    else:
        px = pd.DataFrame(columns=["ace_code", "month_end", "adj_price", "price", "dividend_yield"])

    # Include the last band before the window so the first snapshots can use it.
    if ace_codes:
        q_sz = text("""
            WITH prior_bands AS (
                SELECT DISTINCT ON (ace_code) ace_code, band_date, size_band
                FROM fundlab.stock_size_band_ace
                WHERE ace_code = ANY(:ace_codes) AND band_date < :start_me
                ORDER BY ace_code, band_date DESC
            )
            SELECT ace_code, band_date::date AS month_end, size_band
            FROM fundlab.stock_size_band_ace
            WHERE ace_code = ANY(:ace_codes)
              AND band_date BETWEEN :start_me AND :end_me
            UNION ALL
            SELECT ace_code, band_date::date, size_band FROM prior_bands
            ORDER BY ace_code, month_end
        """)
        sz = pd.read_sql(q_sz, engine, params={"ace_codes": ace_codes, **bounds})
        if not sz.empty:
            sz["month_end"] = pd.to_datetime(sz["month_end"]).dt.to_period("M").dt.to_timestamp("M").dt.date
            sz["size_band"] = sz["size_band"].astype(str)
    else:
        sz = pd.DataFrame(columns=["ace_code", "month_end", "size_band"])

    # Benchmarks (monthly)
    wanted = [BENCH_NAME_NIFTY50, BENCH_NAME_NIFTY500, BENCH_NAME_NIFTY100, BENCH_NAME_MID150, BENCH_NAME_SMALL250]
    q_b = text("""
        SELECT b.bench_id,
               b.bench_name,
               bn.nav_date::date AS month_end,
               bn.nav_value
        FROM fundlab.benchmark b
        JOIN fundlab.bench_nav bn
          ON b.bench_id = bn.bench_id
        WHERE b.bench_name = ANY(:names)
          AND bn.nav_date BETWEEN :start_me AND :end_me
        ORDER BY b.bench_name, bn.nav_date
    """)
    bn = pd.read_sql(q_b, engine, params={"names": wanted, **bounds})
    if not bn.empty:
        bn["month_end"] = pd.to_datetime(bn["month_end"]).dt.to_period("M").dt.to_timestamp("M").dt.date
        bn["nav_value"] = pd.to_numeric(bn["nav_value"], errors="coerce")

    return {
        "window_start": start_me,
        "window_end": end_me,
        "holdings": h,
        "prices": px,
        "size_band": sz,
        "bench_nav": bn,
        "stock_master": sm,
        "weight_scale": float(weight_scale),
    }


@st.cache_data(ttl=3600, show_spinner=False)
def load_attrib_fundamental_endpoints(endpoints: tuple[tuple[str, dt.date], ...]) -> pd.DataFrame:
    """Load just the requested ACE/month-year pairs, including last trading day records."""
    columns = ["ace_code", "month_end", "ttm_pat", "ttm_sales", "book_value"]
    if not endpoints:
        return pd.DataFrame(columns=columns)

    ace_codes, dates = zip(*endpoints)
    query = text("""
        SELECT v.ace_code, v.month_end, v.ttm_pat, v.ttm_sales, v.book_value
        FROM unnest(CAST(:ace_codes AS text[]), CAST(:month_starts AS date[])) AS e(ace_code, month_start)
        JOIN fundlab.stock_monthly_valuations_ace v
          ON v.ace_code = e.ace_code
         AND v.month_end >= e.month_start
         AND v.month_end < e.month_start + INTERVAL '1 month'
    """)
    return pd.read_sql(query, get_engine(), params={
        "ace_codes": list(ace_codes), "month_starts": [date.replace(day=1) for date in dates],
    })


def load_attrib_fundamental_history(ace_codes, start_month, end_month, metric):
    """Read the selected metric for a stock universe and inclusive month range."""
    field = FUNDAMENTAL_FIELDS[metric]
    if not ace_codes:
        return pd.DataFrame(columns=["ace_code", "month_end", "value"])

    query = text(f"""
        SELECT ace_code, month_end, {field} AS value
        FROM fundlab.stock_monthly_valuations_ace
        WHERE ace_code = ANY(:ace_codes) AND month_end BETWEEN :start_date AND :end_date
    """)
    return pd.read_sql(query, get_engine(), params={
        "ace_codes": list(ace_codes),
        "start_date": to_month_end(start_month).replace(day=1),
        "end_date": to_month_end(end_month),
    })
