import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from features.housekeeping.adjusted_prices import (
    build_month_end_map,
    compute_action_multiplier,
    compute_cumulative_multipliers,
    map_actions_to_month_end,
)


def test_compute_action_multiplier_bonus_and_split():
    bonus = compute_action_multiplier(
        pd.Series({"action_type": "BONUS", "denominator": 1, "numerator": 1})
    )
    split = compute_action_multiplier(
        pd.Series({"action_type": "SPLIT", "denominator": 2, "numerator": 10})
    )
    assert bonus == 2.0
    assert split == 5.0


def test_map_actions_to_month_end_maps_to_month_max_price_date_and_flags_missing():
    price_df = pd.DataFrame(
        [
            {"isin": "A", "price_date": dt.date(2020, 1, 30), "price": 100},
            {"isin": "A", "price_date": dt.date(2020, 1, 31), "price": 101},
            {"isin": "A", "price_date": dt.date(2020, 2, 28), "price": 110},
        ]
    )
    actions_df = pd.DataFrame(
        [
            {
                "isin": "A",
                "action_type": "BONUS",
                "ex_date": dt.date(2020, 1, 15),
                "denominator": 1,
                "numerator": 1,
            },
            {
                "isin": "B",
                "action_type": "SPLIT",
                "ex_date": dt.date(2020, 1, 20),
                "denominator": 2,
                "numerator": 10,
            },
        ]
    )

    month_end_map = build_month_end_map(price_df)
    month_multiplier_df, exceptions_df = map_actions_to_month_end(actions_df, month_end_map)

    assert len(month_multiplier_df) == 1
    assert month_multiplier_df.iloc[0]["isin"] == "A"
    assert month_multiplier_df.iloc[0]["mapped_month_end_price_date"] == dt.date(2020, 1, 31)
    assert month_multiplier_df.iloc[0]["month_multiplier"] == 2.0

    assert len(exceptions_df) == 1
    assert exceptions_df.iloc[0]["isin"] == "B"
    assert exceptions_df.iloc[0]["reason"] == "missing_stock_price_month_row"


def test_map_actions_same_month_multipliers_are_multiplied():
    price_df = pd.DataFrame(
        [
            {"isin": "A", "price_date": dt.date(2020, 1, 31), "price": 100},
            {"isin": "A", "price_date": dt.date(2020, 2, 29), "price": 110},
        ]
    )
    actions_df = pd.DataFrame(
        [
            {
                "isin": "A",
                "action_type": "BONUS",
                "ex_date": dt.date(2020, 1, 5),
                "denominator": 1,
                "numerator": 1,
            },
            {
                "isin": "A",
                "action_type": "SPLIT",
                "ex_date": dt.date(2020, 1, 20),
                "denominator": 2,
                "numerator": 10,
            },
        ]
    )

    month_end_map = build_month_end_map(price_df)
    month_multiplier_df, exceptions_df = map_actions_to_month_end(actions_df, month_end_map)
    assert exceptions_df.empty
    assert len(month_multiplier_df) == 1
    assert month_multiplier_df.iloc[0]["month_multiplier"] == 10.0  # 2.0 * 5.0


def test_compute_cumulative_multipliers_applies_from_effective_month_end_inclusive():
    price_df = pd.DataFrame(
        [
            {"isin": "A", "price_date": dt.date(2019, 12, 31), "price": 100},
            {"isin": "A", "price_date": dt.date(2020, 1, 31), "price": 120},
            {"isin": "A", "price_date": dt.date(2020, 2, 29), "price": 130},
            {"isin": "A", "price_date": dt.date(2021, 2, 26), "price": 200},
            {"isin": "A", "price_date": dt.date(2021, 2, 28), "price": 210},
            {"isin": "A", "price_date": dt.date(2021, 3, 31), "price": 220},
        ]
    )
    month_multiplier_df = pd.DataFrame(
        [
            {"isin": "A", "mapped_month_end_price_date": dt.date(2020, 1, 31), "month_multiplier": 2.0},
            {"isin": "A", "mapped_month_end_price_date": dt.date(2021, 2, 28), "month_multiplier": 5.0},
        ]
    )

    out = compute_cumulative_multipliers(price_df, month_multiplier_df)
    out = out.sort_values("price_date").reset_index(drop=True)

    expected_multipliers = [1.0, 2.0, 2.0, 2.0, 10.0, 10.0]
    assert out["adj_multiplier"].tolist() == expected_multipliers

    expected_adj_prices = [
        100.0,
        240.0,
        260.0,
        400.0,
        2100.0,
        2200.0,
    ]
    assert out["adj_price"].tolist() == expected_adj_prices


def test_invalid_ratio_goes_to_exceptions():
    price_df = pd.DataFrame(
        [{"isin": "A", "price_date": dt.date(2020, 1, 31), "price": 100}]
    )
    actions_df = pd.DataFrame(
        [
            {
                "isin": "A",
                "action_type": "BONUS",
                "ex_date": dt.date(2020, 1, 15),
                "denominator": 0,
                "numerator": 1,
            }
        ]
    )

    month_end_map = build_month_end_map(price_df)
    month_multiplier_df, exceptions_df = map_actions_to_month_end(actions_df, month_end_map)
    assert month_multiplier_df.empty
    assert len(exceptions_df) == 1
    assert exceptions_df.iloc[0]["reason"] == "invalid_ratio"
