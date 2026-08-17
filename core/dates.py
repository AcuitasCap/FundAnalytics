"""Generic date normalization helpers shared across dashboard features."""

import datetime as dt

import numpy as np
import pandas as pd


def month_year_to_last_day(year: int, month: int) -> dt.date:
    """Return the final calendar day of ``year``/``month``."""
    if month == 12:
        return dt.date(year, 12, 31)
    first_next = dt.date(year + (month // 12), ((month % 12) + 1), 1)
    return first_next - dt.timedelta(days=1)


def to_month_end(value) -> dt.date:
    """Normalize a date-like value to its calendar month-end date."""
    value_date = pd.to_datetime(value).date()
    return month_year_to_last_day(value_date.year, value_date.month)


def month_ends_between(start_month_end: dt.date, end_month_end: dt.date) -> list[dt.date]:
    """Return inclusive month-end dates between two month-end boundaries."""
    idx = pd.date_range(
        start=pd.Timestamp(start_month_end),
        end=pd.Timestamp(end_month_end),
        freq="ME",
    )
    return [value.date() for value in idx]


def parse_period_to_month_end(series: pd.Series) -> pd.Series:
    """Convert Excel-style quarter/year-end values into month-end timestamps.

    Handles datetime values plus numeric or string ``YYYYMM`` and ``YYYYMMDD``
    representations.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        dates = pd.to_datetime(series)
        return dates + pd.offsets.MonthEnd(0)

    values = series.astype(str).str.strip()
    digits = values.str.extract(r"(\d+)", expand=False)
    is_yyyymm = digits.str.len() == 6
    is_yyyymmdd = digits.str.len() == 8

    if not (is_yyyymm | is_yyyymmdd).all():
        bad = values[~(is_yyyymm | is_yyyymmdd)].unique()[:10]
        raise ValueError(
            "Invalid period values found (expected YYYYMM or YYYYMMDD). "
            f"Sample invalid values: {bad}"
        )

    parsed = pd.to_datetime(
        np.where(is_yyyymm, digits + "01", digits),
        format="%Y%m%d",
        errors="raise",
    )
    return parsed + pd.offsets.MonthEnd(0)
