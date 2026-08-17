import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from features.housekeeping.dividend_yields import compute_dividend_yield_updates


def _mapped_df(rows):
    return pd.DataFrame(rows)[
        [
            "isin",
            "ex_month_start",
            "month_dps",
            "min_ex_date",
            "max_ex_date",
            "n_events",
            "prev_month_end_date",
            "base_price",
        ]
    ]


def test_standard_mapping_updates_prior_month_end():
    mapped = _mapped_df(
        [
            {
                "isin": "INE000A01001",
                "ex_month_start": dt.date(2025, 2, 1),
                "month_dps": 5.0,
                "min_ex_date": dt.date(2025, 2, 10),
                "max_ex_date": dt.date(2025, 2, 10),
                "n_events": 1,
                "prev_month_end_date": dt.date(2025, 1, 31),
                "base_price": 200.0,
            }
        ]
    )

    updates_df, exceptions_df = compute_dividend_yield_updates(mapped)

    assert len(exceptions_df) == 0
    assert len(updates_df) == 1
    assert updates_df.iloc[0]["isin"] == "INE000A01001"
    assert updates_df.iloc[0]["price_date"] == dt.date(2025, 1, 31)
    assert updates_df.iloc[0]["dividend_yield"] == 0.025


def test_missing_prior_month_price_row_goes_to_exceptions():
    mapped = _mapped_df(
        [
            {
                "isin": "INE000A01002",
                "ex_month_start": dt.date(2025, 2, 1),
                "month_dps": 5.0,
                "min_ex_date": dt.date(2025, 2, 12),
                "max_ex_date": dt.date(2025, 2, 12),
                "n_events": 1,
                "prev_month_end_date": pd.NaT,
                "base_price": pd.NA,
            }
        ]
    )

    updates_df, exceptions_df = compute_dividend_yield_updates(mapped)

    assert len(updates_df) == 0
    assert len(exceptions_df) == 1
    assert exceptions_df.iloc[0]["reason"] == "missing_prev_month_price_row"


def test_bad_base_price_goes_to_exceptions():
    mapped = _mapped_df(
        [
            {
                "isin": "INE000A01003",
                "ex_month_start": dt.date(2025, 2, 1),
                "month_dps": 5.0,
                "min_ex_date": dt.date(2025, 2, 5),
                "max_ex_date": dt.date(2025, 2, 5),
                "n_events": 1,
                "prev_month_end_date": dt.date(2025, 1, 31),
                "base_price": 0.0,
            },
            {
                "isin": "INE000A01004",
                "ex_month_start": dt.date(2025, 2, 1),
                "month_dps": 5.0,
                "min_ex_date": dt.date(2025, 2, 6),
                "max_ex_date": dt.date(2025, 2, 6),
                "n_events": 1,
                "prev_month_end_date": dt.date(2025, 1, 31),
                "base_price": pd.NA,
            },
        ]
    )

    updates_df, exceptions_df = compute_dividend_yield_updates(mapped)

    assert len(updates_df) == 0
    assert len(exceptions_df) == 2
    assert set(exceptions_df["reason"].tolist()) == {"missing_or_bad_base_price"}


def test_multiple_dividends_same_month_uses_aggregated_month_dps():
    mapped = _mapped_df(
        [
            {
                "isin": "INE000A01005",
                "ex_month_start": dt.date(2025, 2, 1),
                "month_dps": 5.0,  # 2 + 3 already aggregated by SQL step
                "min_ex_date": dt.date(2025, 2, 4),
                "max_ex_date": dt.date(2025, 2, 20),
                "n_events": 2,
                "prev_month_end_date": dt.date(2025, 1, 31),
                "base_price": 250.0,
            }
        ]
    )

    updates_df, exceptions_df = compute_dividend_yield_updates(mapped)

    assert len(exceptions_df) == 0
    assert len(updates_df) == 1
    assert updates_df.iloc[0]["dividend_yield"] == 0.02
