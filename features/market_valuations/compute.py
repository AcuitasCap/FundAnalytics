"""Unweighted monthly market valuation percentiles."""

import numpy as np
import pandas as pd


RESULT_COLUMNS = ["month_end", "p25", "median", "p75", "stock_count", "historical_median"]


def compute_market_valuation_ranges(
    stock_multiples: pd.DataFrame,
    *,
    minimum_stocks: int = 5,
) -> pd.DataFrame:
    """Calculate each month's stock percentiles and the median of monthly medians."""
    if minimum_stocks < 1:
        raise ValueError("minimum_stocks must be positive")
    if stock_multiples.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    frame = stock_multiples[["ace_code", "month_end", "multiple"]].copy()
    frame["month_end"] = pd.to_datetime(frame["month_end"], errors="coerce")
    frame["multiple"] = pd.to_numeric(frame["multiple"], errors="coerce")
    frame = frame.dropna(subset=["ace_code", "month_end", "multiple"])
    frame = frame[np.isfinite(frame["multiple"]) & (frame["multiple"] > 0)]
    frame = frame.drop_duplicates(subset=["ace_code", "month_end"])
    if frame.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    monthly = frame.groupby("month_end")["multiple"].agg(
        stock_count="size",
        p25=lambda values: values.quantile(0.25),
        median="median",
        p75=lambda values: values.quantile(0.75),
    ).reset_index()
    monthly = monthly[monthly["stock_count"] >= minimum_stocks].copy()
    if monthly.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    monthly["historical_median"] = monthly["median"].median()
    return monthly[RESULT_COLUMNS].sort_values("month_end").reset_index(drop=True)
