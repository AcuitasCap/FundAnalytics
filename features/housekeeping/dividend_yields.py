from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd


EXCEPTION_COLUMNS = [
    "isin",
    "ex_month_start",
    "month_dps",
    "min_ex_date",
    "max_ex_date",
    "n_events",
    "prev_month_end_date",
    "base_price",
    "computed_yield",
    "reason",
]


def subtract_years_safe(d: dt.date, years: int) -> dt.date:
    try:
        return d.replace(year=d.year - years)
    except ValueError:
        # Handles Feb 29 -> Feb 28 on non-leap target year.
        return d.replace(year=d.year - years, month=2, day=28)


def compute_dividend_yield_updates(mapped_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Input columns expected in mapped_df:
      isin, ex_month_start, month_dps, min_ex_date, max_ex_date, n_events,
      prev_month_end_date, base_price

    Returns:
      updates_df columns: isin, price_date, dividend_yield
      exceptions_df columns: EXCEPTION_COLUMNS
    """
    if mapped_df is None or mapped_df.empty:
        return (
            pd.DataFrame(columns=["isin", "price_date", "dividend_yield"]),
            pd.DataFrame(columns=EXCEPTION_COLUMNS),
        )

    d = mapped_df.copy()
    d["isin"] = d["isin"].astype(str).str.strip()
    d["month_dps"] = pd.to_numeric(d["month_dps"], errors="coerce")
    d["base_price"] = pd.to_numeric(d["base_price"], errors="coerce")
    d["computed_yield"] = np.where(
        d["base_price"] > 0,
        d["month_dps"] / d["base_price"],
        np.nan,
    )

    missing_prev = d["prev_month_end_date"].isna()
    bad_base = d["base_price"].isna() | (d["base_price"] <= 0)
    bad_month_dps = d["month_dps"].isna()

    invalid = missing_prev | bad_base | bad_month_dps
    d["reason"] = np.where(missing_prev, "missing_prev_month_price_row", None)
    d.loc[d["reason"].isna() & (bad_base | bad_month_dps), "reason"] = "missing_or_bad_base_price"

    updates_df = d.loc[~invalid, ["isin", "prev_month_end_date", "computed_yield"]].copy()
    updates_df = updates_df.rename(
        columns={"prev_month_end_date": "price_date", "computed_yield": "dividend_yield"}
    )
    updates_df = updates_df.reset_index(drop=True)

    exceptions_df = d.loc[invalid, EXCEPTION_COLUMNS].copy().reset_index(drop=True)
    return updates_df, exceptions_df
