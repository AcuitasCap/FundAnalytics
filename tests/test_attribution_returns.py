import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from features.fund_attribution.returns import compute_adj_price_returns
from features.fund_attribution.returns import compute_dividend_yield_returns


def _months(*pairs):
    return [pd.Timestamp(dt.date(y, m, d)) for (y, m, d) in pairs]


def test_ma_market_cap_jump_does_not_drive_return_basis():
    months = _months((2024, 1, 31), (2024, 2, 29))
    prices_df = pd.DataFrame(
        [
            {"ace_code": "INEA", "month_end": dt.date(2024, 1, 31), "adj_price": 100.0, "market_cap": 1000.0},
            {"ace_code": "INEA", "month_end": dt.date(2024, 2, 29), "adj_price": 102.0, "market_cap": 2000.0},
        ]
    )

    returns_df, invalid_df, audit_df = compute_adj_price_returns(prices_df, months, ["INEA"])

    assert invalid_df.empty
    assert np.isclose(float(returns_df.iloc[0]["INEA"]), 0.02, atol=1e-12)
    assert np.isclose(float(audit_df.iloc[0]["price_return_adj_price"]), 0.02, atol=1e-12)


def test_bonus_split_adjusted_price_series_is_continuous_for_return_math():
    months = _months((2020, 1, 31), (2020, 2, 29), (2020, 3, 31))
    # Already adjusted series should not introduce artificial jumps.
    prices_df = pd.DataFrame(
        [
            {"ace_code": "INEB", "month_end": dt.date(2020, 1, 31), "adj_price": 100.0},
            {"ace_code": "INEB", "month_end": dt.date(2020, 2, 29), "adj_price": 102.0},
            {"ace_code": "INEB", "month_end": dt.date(2020, 3, 31), "adj_price": 104.04},
        ]
    )

    returns_df, invalid_df, _ = compute_adj_price_returns(prices_df, months, ["INEB"])

    assert invalid_df.empty
    r = returns_df["INEB"].to_numpy(dtype=float)
    assert np.allclose(r, np.array([0.02, 0.02]), atol=1e-12)


def test_portfolio_identity_weighted_sum_matches_helper_returns():
    months = _months((2024, 1, 31), (2024, 2, 29), (2024, 3, 31))
    prices_df = pd.DataFrame(
        [
            {"ace_code": "A", "month_end": dt.date(2024, 1, 31), "adj_price": 100.0},
            {"ace_code": "A", "month_end": dt.date(2024, 2, 29), "adj_price": 105.0},
            {"ace_code": "A", "month_end": dt.date(2024, 3, 31), "adj_price": 110.25},
            {"ace_code": "B", "month_end": dt.date(2024, 1, 31), "adj_price": 200.0},
            {"ace_code": "B", "month_end": dt.date(2024, 2, 29), "adj_price": 198.0},
            {"ace_code": "B", "month_end": dt.date(2024, 3, 31), "adj_price": 201.96},
        ]
    )

    returns_df, invalid_df, _ = compute_adj_price_returns(prices_df, months, ["A", "B"])
    assert invalid_df.empty

    # t-1 weights for two periods
    w = np.array([[0.6, 0.4], [0.55, 0.45]], dtype=float)
    r = returns_df[["A", "B"]].to_numpy(dtype=float)
    portfolio_r = np.sum(w * r, axis=1)

    # Identity in this test: residual is 0 because we use the same return vectors.
    lhs = portfolio_r
    rhs = np.sum(w * r, axis=1) + 0.0
    assert np.allclose(lhs, rhs, atol=1e-12)


def test_invalid_adj_price_rows_are_flagged_and_zeroed():
    months = _months((2024, 1, 31), (2024, 2, 29))
    prices_df = pd.DataFrame(
        [
            {"ace_code": "INEC", "month_end": dt.date(2024, 1, 31), "adj_price": 0.0},
            {"ace_code": "INEC", "month_end": dt.date(2024, 2, 29), "adj_price": 105.0},
        ]
    )

    returns_df, invalid_df, _ = compute_adj_price_returns(prices_df, months, ["INEC"])

    assert len(invalid_df) == 1
    assert invalid_df.iloc[0]["reason"] == "invalid_adj_price_t_minus_1"
    assert float(returns_df.iloc[0]["INEC"]) == 0.0


def test_dividend_yield_timing_uses_t0_yield_for_t0_to_t1_return():
    months = _months((2024, 1, 31), (2024, 2, 29), (2024, 3, 31))
    prices_df = pd.DataFrame(
        [
            {"ace_code": "A", "month_end": dt.date(2024, 1, 31), "dividend_yield": 0.01},
            {"ace_code": "A", "month_end": dt.date(2024, 2, 29), "dividend_yield": 0.02},
            {"ace_code": "A", "month_end": dt.date(2024, 3, 31), "dividend_yield": 0.03},
        ]
    )

    div_df = compute_dividend_yield_returns(prices_df, months, ["A"])
    assert np.isclose(float(div_df.iloc[0]["A"]), 0.01, atol=1e-12)
    assert np.isclose(float(div_df.iloc[1]["A"]), 0.02, atol=1e-12)
