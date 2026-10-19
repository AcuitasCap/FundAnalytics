"""Market valuation controls and read-only analysis orchestration."""

import datetime as dt

import pandas as pd
import streamlit as st

from core.dates import month_year_to_last_day
from core.navigation import home_button
from .compute import compute_market_valuation_ranges
from .data import (
    cached_market_stock_multiples,
    market_valuation_bounds,
    nifty_50_holdings_bounds,
)
from .display import display_market_valuation_ranges


def market_valuations_page() -> None:
    home_button()
    if st.button("← Valuations"):
        st.session_state["page"] = "Valuations"
        st.rerun()
    st.header("Market valuations")

    try:
        first_month, last_month = market_valuation_bounds()
    except Exception as error:
        st.error(f"Could not determine available valuation dates: {error}")
        return
    if first_month is None or last_month is None:
        st.info("No monthly stock valuations are available.")
        return

    years = list(range(first_month.year, last_month.year + 1))
    months = list(range(1, 13))
    default_start_year = max(first_month.year, last_month.year - 5)
    with st.form("market_valuation_controls"):
        cohort = st.radio("Market cohort", ["Nifty 50", "Large caps", "Mid caps", "Small caps"])
        segment = st.radio("Segment", ["Total", "Financials", "Non-financials"], horizontal=True)
        metric = st.radio("Valuation metric", ["P/S", "P/E", "P/B"], horizontal=True)
        start_col, end_col = st.columns(2)
        with start_col:
            start_month = st.selectbox(
                "Start month", months, index=last_month.month - 1,
                format_func=lambda month: dt.date(2000, month, 1).strftime("%b"),
            )
            start_year = st.selectbox(
                "Start year", years, index=years.index(default_start_year)
            )
        with end_col:
            end_month = st.selectbox(
                "End month", months, index=last_month.month - 1,
                format_func=lambda month: dt.date(2000, month, 1).strftime("%b"),
            )
            end_year = st.selectbox("End year", years, index=len(years) - 1)
        submitted = st.form_submit_button("Submit", type="primary")

    if not submitted:
        return
    start_date = month_year_to_last_day(start_year, start_month)
    end_date = month_year_to_last_day(end_year, end_month)
    if start_date > end_date:
        st.error("Start date must be on or before end date.")
        return
    if ((start_year, start_month) < (first_month.year, first_month.month)
            or (end_year, end_month) > (last_month.year, last_month.month)):
        st.error(
            f"Valuations are available from {first_month:%b %Y} through {last_month:%b %Y}."
        )
        return

    try:
        with st.spinner("Loading market valuations…"):
            if cohort == "Nifty 50":
                holdings_first, holdings_last = nifty_50_holdings_bounds()
                if holdings_first is None or holdings_last is None:
                    st.error("No domestic-equity holdings were found for Nippon India ETF Nifty 50 BeES.")
                    return
                if ((start_year, start_month) < (holdings_first.year, holdings_first.month)
                        or (end_year, end_month) > (holdings_last.year, holdings_last.month)):
                    st.warning(
                        f"ETF holdings are available from {holdings_first:%b %Y} through "
                        f"{holdings_last:%b %Y}. Months outside this range will be omitted."
                    )
            stock_multiples = cached_market_stock_multiples(
                cohort, segment, metric, start_date, end_date
            )
            ranges = compute_market_valuation_ranges(stock_multiples)
    except ValueError as error:
        st.error(str(error))
        return
    except Exception as error:
        st.error(f"Error while loading market valuations: {error}")
        return
    omitted_months = len(pd.period_range(start_date, end_date, freq="M")) - len(ranges)
    if omitted_months > 0:
        st.warning(
            f"{omitted_months} selected month(s) have no displayed percentile because "
            "the cohort or enough valid stock multiples were unavailable."
        )
    display_market_valuation_ranges(ranges, cohort=cohort, segment=segment, metric=metric)
