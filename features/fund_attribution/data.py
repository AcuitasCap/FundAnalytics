import datetime as dt

import pandas as pd
import streamlit as st
from sqlalchemy import text

from core.db import get_engine
from core.dates import to_month_end

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
    data_version: str = "v1",
):
    """Load raw inputs for attribution for a single fund over [end-LOOKBACK, end]."""
    engine = get_engine()
    end_me = to_month_end(end_month_end)
    start_me = (pd.Timestamp(end_me) - pd.DateOffset(years=lookback_years)).to_period("M").to_timestamp("M").date()

    # Holdings (all asset types)
    q_hold = text("""
        SELECT fund_id,
               month_end::date AS month_end,
               instrument_name,
               asset_type,
               isin,
               holding_weight
        FROM fundlab.fund_portfolio
        WHERE fund_id = :fund_id
          AND month_end BETWEEN :start_me AND :end_me
        ORDER BY month_end, instrument_name
    """)
    h = pd.read_sql(q_hold, engine, params={"fund_id": fund_id, "start_me": start_me, "end_me": end_me})
    if h.empty:
        return {
            "window_start": start_me,
            "window_end": end_me,
            "holdings": h,
            "prices": pd.DataFrame(),
            "multiples": pd.DataFrame(),
            "yields": pd.DataFrame(),
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

    # ISIN list (exclude null/blank)
    isins = sorted([x for x in h["isin"].dropna().astype(str).unique().tolist() if x.strip()])

    # Stock master (names)
    if isins:
        q_sm = text("""
            SELECT isin, company_name, industry, is_financial
            FROM fundlab.stock_master
            WHERE isin = ANY(:isins)
        """)
        sm = pd.read_sql(q_sm, engine, params={"isins": isins})
    else:
        sm = pd.DataFrame(columns=["isin", "company_name", "industry", "is_financial"])

    # Prices (monthly) with adjusted price (returns use adj_price basis)
    if isins:
        q_px = text("""
            SELECT isin,
                   price_date::date AS month_end,
                   adj_price,
                   price,
                   dividend_yield
            FROM fundlab.stock_price
            WHERE isin = ANY(:isins)
              AND price_date BETWEEN :start_me AND :end_me
            ORDER BY isin, price_date
        """)
        px = pd.read_sql(q_px, engine, params={"isins": isins, "start_me": start_me, "end_me": end_me})
        if not px.empty:
            px["month_end"] = pd.to_datetime(px["month_end"]).dt.to_period("M").dt.to_timestamp("M").dt.date
            px["adj_price"] = pd.to_numeric(px["adj_price"], errors="coerce")
            px["price"] = pd.to_numeric(px["price"], errors="coerce")
            px["dividend_yield"] = pd.to_numeric(px["dividend_yield"], errors="coerce")
    else:
        px = pd.DataFrame(columns=["isin", "month_end", "adj_price", "price", "dividend_yield"])

    # Stock valuation multiples (monthly): P/S, P/E and P/B computed on Supabase.
    if isins:
        q_mul = text("""
            SELECT
                isin,
                month_end::date AS month_end,
                ps,
                pe,
                pb
            FROM fundlab.stock_monthly_valuations
            WHERE isin = ANY(:isins)
              AND month_end BETWEEN :start_me AND :end_me
            ORDER BY isin, month_end
        """)
        yld = pd.read_sql(q_mul, engine, params={"isins": isins, "start_me": start_me, "end_me": end_me})
        if not yld.empty:
            yld["month_end"] = pd.to_datetime(yld["month_end"]).dt.to_period("M").dt.to_timestamp("M").dt.date
            for c in ["ps", "pe", "pb"]:
                if c in yld.columns:
                    yld[c] = pd.to_numeric(yld[c], errors="coerce")
    else:
        yld = pd.DataFrame(columns=["isin", "month_end", "ps", "pe", "pb"])

    # Size band (monthly)
    if isins:
        q_sz = text("""
            SELECT isin, band_date::date AS month_end, size_band
            FROM fundlab.stock_size_band
            WHERE isin = ANY(:isins)
              AND band_date BETWEEN :start_me AND :end_me
            ORDER BY isin, band_date
        """)
        sz = pd.read_sql(q_sz, engine, params={"isins": isins, "start_me": start_me, "end_me": end_me})
        if not sz.empty:
            sz["month_end"] = pd.to_datetime(sz["month_end"]).dt.to_period("M").dt.to_timestamp("M").dt.date
            sz["size_band"] = sz["size_band"].astype(str)
    else:
        sz = pd.DataFrame(columns=["isin", "month_end", "size_band"])

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
    bn = pd.read_sql(q_b, engine, params={"names": wanted, "start_me": start_me, "end_me": end_me})
    if not bn.empty:
        bn["month_end"] = pd.to_datetime(bn["month_end"]).dt.to_period("M").dt.to_timestamp("M").dt.date
        bn["nav_value"] = pd.to_numeric(bn["nav_value"], errors="coerce")

    return {
        "window_start": start_me,
        "window_end": end_me,
        "holdings": h,
        "prices": px,
        "multiples": yld,
        "yields": yld,
        "size_band": sz,
        "bench_nav": bn,
        "stock_master": sm,
        "weight_scale": float(weight_scale),
    }
