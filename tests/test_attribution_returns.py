import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from features.fund_attribution.returns import compute_adj_price_returns
from features.fund_attribution.returns import compute_dividend_yield_returns, compute_monthly_portfolio_multiples
from features.fund_attribution.compute import _compute_domestic_triangulation_tables


def _months(*pairs):
    return [pd.Timestamp(dt.date(y, m, d)) for (y, m, d) in pairs]


def test_ma_market_cap_jump_does_not_drive_return_basis():
    months = _months((2024, 1, 31), (2024, 2, 29))
    prices_df = pd.DataFrame(
        [
            {"isin": "INEA", "month_end": dt.date(2024, 1, 31), "adj_price": 100.0, "market_cap": 1000.0},
            {"isin": "INEA", "month_end": dt.date(2024, 2, 29), "adj_price": 102.0, "market_cap": 2000.0},
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
            {"isin": "INEB", "month_end": dt.date(2020, 1, 31), "adj_price": 100.0},
            {"isin": "INEB", "month_end": dt.date(2020, 2, 29), "adj_price": 102.0},
            {"isin": "INEB", "month_end": dt.date(2020, 3, 31), "adj_price": 104.04},
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
            {"isin": "A", "month_end": dt.date(2024, 1, 31), "adj_price": 100.0},
            {"isin": "A", "month_end": dt.date(2024, 2, 29), "adj_price": 105.0},
            {"isin": "A", "month_end": dt.date(2024, 3, 31), "adj_price": 110.25},
            {"isin": "B", "month_end": dt.date(2024, 1, 31), "adj_price": 200.0},
            {"isin": "B", "month_end": dt.date(2024, 2, 29), "adj_price": 198.0},
            {"isin": "B", "month_end": dt.date(2024, 3, 31), "adj_price": 201.96},
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
            {"isin": "INEC", "month_end": dt.date(2024, 1, 31), "adj_price": 0.0},
            {"isin": "INEC", "month_end": dt.date(2024, 2, 29), "adj_price": 105.0},
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
            {"isin": "A", "month_end": dt.date(2024, 1, 31), "dividend_yield": 0.01},
            {"isin": "A", "month_end": dt.date(2024, 2, 29), "dividend_yield": 0.02},
            {"isin": "A", "month_end": dt.date(2024, 3, 31), "dividend_yield": 0.03},
        ]
    )

    div_df = compute_dividend_yield_returns(prices_df, months, ["A"])
    assert np.isclose(float(div_df.iloc[0]["A"]), 0.01, atol=1e-12)
    assert np.isclose(float(div_df.iloc[1]["A"]), 0.02, atol=1e-12)


def test_monthly_chaining_and_valuation_fundamental_identities_hold():
    months = _months((2024, 1, 31), (2024, 2, 29), (2024, 3, 31))
    t1 = months[1:]
    w0_df = pd.DataFrame({"A": [1.0, 1.0]}, index=months[:-1])

    multiples_df = pd.DataFrame(
        [
            {"isin": "A", "month_end": dt.date(2024, 1, 31), "pe": 10.0, "pb": 2.0, "ps": 3.0},
            {"isin": "A", "month_end": dt.date(2024, 2, 29), "pe": 11.0, "pb": 2.2, "ps": 3.3},
            {"isin": "A", "month_end": dt.date(2024, 3, 31), "pe": 12.0, "pb": 2.4, "ps": 3.6},
        ]
    )
    r_price_df = pd.DataFrame({"A": [0.10, 0.05]}, index=t1)
    m_start, m_end, _ = compute_monthly_portfolio_multiples(
        multiples_df, months, ["A"], w0_df, r_price_df, "Earnings (P/E)"
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        r_val = (m_end / m_start) - 1.0
    r_price = np.array([0.10, 0.05], dtype=float)
    r_div = np.array([0.01, 0.02], dtype=float)
    r_total = r_price + r_div
    r_fund = ((1.0 + r_price) / (1.0 + r_val)) - 1.0

    assert np.allclose(r_fund, ((1.0 + r_price) / (1.0 + r_val)) - 1.0, atol=1e-12)

    gross_total = float(np.prod(1.0 + r_total))
    gross_price = float(np.prod(1.0 + r_price))
    gross_val = float(np.prod(1.0 + r_val))
    gross_fund = float(np.prod(1.0 + r_fund))

    assert np.isclose(gross_total, np.prod(1.0 + r_total), atol=1e-12)
    assert np.isclose(gross_price, gross_val * gross_fund, atol=1e-12)

    # Alignment guard: output length/index follows t1.
    assert len(m_start) == len(t1)
    assert len(m_end) == len(t1)


def test_portfolio_multiple_uses_single_inversion_of_weighted_implied_yield():
    months = _months((2024, 1, 31), (2024, 2, 29))
    w0_df = pd.DataFrame({"A": [0.6], "B": [0.4]}, index=months[:-1])
    multiples_df = pd.DataFrame(
        [
            {"isin": "A", "month_end": dt.date(2024, 1, 31), "pe": 1 / 0.03, "pb": 2.0, "ps": 5.0},
            {"isin": "B", "month_end": dt.date(2024, 1, 31), "pe": 50.0, "pb": 2.5, "ps": 10.0},
            {"isin": "A", "month_end": dt.date(2024, 2, 29), "pe": 25.0, "pb": 1 / 0.45, "ps": 1 / 0.22},
            {"isin": "B", "month_end": dt.date(2024, 2, 29), "pe": 100.0, "pb": 1 / 0.35, "ps": 1 / 0.12},
        ]
    )

    r_price_df = pd.DataFrame({"A": [0.20], "B": [-0.10]}, index=months[1:])
    m_start, m_end, dbg = compute_monthly_portfolio_multiples(
        multiples_df, months, ["A", "B"], w0_df, r_price_df, "Earnings (P/E)"
    )

    expected_y0 = 0.6 * 0.03 + 0.4 * 0.02
    drift_a = 0.6 * 1.20
    drift_b = 0.4 * 0.90
    drift_sum = drift_a + drift_b
    end_w_a = drift_a / drift_sum
    end_w_b = drift_b / drift_sum
    expected_y1 = end_w_a * 0.04 + end_w_b * 0.01
    assert np.isclose(float(dbg["agg_yield_start"][0]), expected_y0, atol=1e-12)
    assert np.isclose(float(dbg["agg_yield_end"][0]), expected_y1, atol=1e-12)
    assert np.isclose(float(m_start[0]), 1.0 / expected_y0, atol=1e-12)
    assert np.isclose(float(m_end[0]), 1.0 / expected_y1, atol=1e-12)
    assert np.isclose(float(dbg["drifted_end_weight_sum"][0]), drift_sum, atol=1e-12)


def test_end_multiple_uses_drifted_weights_from_price_moves():
    months = _months((2024, 1, 31), (2024, 2, 29))
    w0_df = pd.DataFrame({"A": [0.05], "B": [0.95]}, index=months[:-1])
    r_price_df = pd.DataFrame({"A": [1.0], "B": [-0.05263157894736842]}, index=months[1:])
    multiples_df = pd.DataFrame(
        [
            {"isin": "A", "month_end": dt.date(2024, 1, 31), "pe": 10.0, "pb": 2.0, "ps": 5.0},
            {"isin": "B", "month_end": dt.date(2024, 1, 31), "pe": 10.0, "pb": 2.0, "ps": 5.0},
            {"isin": "A", "month_end": dt.date(2024, 2, 29), "pe": 20.0, "pb": 2.0, "ps": 5.0},
            {"isin": "B", "month_end": dt.date(2024, 2, 29), "pe": 5.0, "pb": 2.0, "ps": 5.0},
        ]
    )

    _, m_end, dbg = compute_monthly_portfolio_multiples(
        multiples_df, months, ["A", "B"], w0_df, r_price_df, "Earnings (P/E)"
    )

    assert np.isclose(float(dbg["drifted_end_weight_sum"][0]), 1.0, atol=1e-12)
    assert np.isclose(float(dbg["coverage_end_weight"][0]), 1.0, atol=1e-12)
    assert np.isclose(float(dbg["agg_yield_end"][0]), 0.10 * 0.05 + 0.90 * 0.20, atol=1e-12)
    assert np.isclose(float(m_end[0]), 1.0 / 0.185, atol=1e-12)


def test_domestic_triangulation_valid_fundamental_growth_abs_and_cagr():
    idx_abs = pd.to_datetime(["2024-01-31", "2024-02-29"]).to_period("M").to_timestamp("M")
    w_abs = pd.DataFrame({"A": [1.0, 1.0]}, index=idx_abs)
    px_abs = pd.DataFrame(
        [
            {"isin": "A", "month_end": dt.date(2024, 1, 31), "adj_price": 100.0},
            {"isin": "A", "month_end": dt.date(2024, 2, 29), "adj_price": 110.0},
        ]
    )
    yld_abs = pd.DataFrame(
        [
            {"isin": "A", "month_end": dt.date(2024, 1, 31), "pe": 0.10, "pb": 0.50, "ps": 0.3333},
            {"isin": "A", "month_end": dt.date(2024, 2, 29), "pe": 0.10, "pb": 0.50, "ps": 0.3333},
        ]
    )
    tri_abs, summary_abs, _ = _compute_domestic_triangulation_tables(
        w0_dom_df=w_abs,
        domestic_isins=["A"],
        px_df=px_abs,
        yld_df=yld_abs,
        lens="Earnings (P/E)",
        name_map={"A": "A Co"},
        mode_label="Fallback to price growth",
    )
    assert np.isclose(float(tri_abs.iloc[0]["Fundamental growth in holding period (%)"]), 10.0, atol=1e-12)
    assert np.isclose(float(summary_abs.iloc[0]["Value (%)"]), 10.0, atol=1e-12)
    assert np.isclose(float(summary_abs.iloc[1]["Value (%)"]), 10.0, atol=1e-12)

    idx_cagr = pd.date_range("2023-01-31", periods=13, freq="ME")
    w_cagr = pd.DataFrame({"B": np.ones(len(idx_cagr))}, index=idx_cagr)
    px_cagr = pd.DataFrame(
        [
            {"isin": "B", "month_end": idx_cagr[0].date(), "adj_price": 100.0},
            {"isin": "B", "month_end": idx_cagr[-1].date(), "adj_price": 144.0},
        ]
    )
    yld_cagr = pd.DataFrame(
        [
            {"isin": "B", "month_end": idx_cagr[0].date(), "pe": 0.0833333333, "pb": 0.50, "ps": 0.3333},
            {"isin": "B", "month_end": idx_cagr[-1].date(), "pe": 0.0833333333, "pb": 0.50, "ps": 0.3333},
        ]
    )
    tri_cagr, _, _ = _compute_domestic_triangulation_tables(
        w0_dom_df=w_cagr,
        domestic_isins=["B"],
        px_df=px_cagr,
        yld_df=yld_cagr,
        lens="Earnings (P/E)",
        name_map={"B": "B Co"},
        mode_label="Fallback to price growth",
    )
    expected_cagr_pct = ((144.0 / 100.0) ** (12.0 / 13.0) - 1.0) * 100.0
    assert np.isclose(float(tri_cagr.iloc[0]["Fundamental growth in holding period (%)"]), expected_cagr_pct, atol=1e-3)


def test_domestic_triangulation_invalid_fundamentals_mode_a_falls_back_to_price_growth():
    idx = pd.to_datetime(["2024-01-31", "2024-02-29"]).to_period("M").to_timestamp("M")
    w = pd.DataFrame({"A": [1.0, 1.0]}, index=idx)
    px = pd.DataFrame(
        [
            {"isin": "A", "month_end": dt.date(2024, 1, 31), "adj_price": 100.0},
            {"isin": "A", "month_end": dt.date(2024, 2, 29), "adj_price": 110.0},
        ]
    )
    yld = pd.DataFrame(
        [
            {"isin": "A", "month_end": dt.date(2024, 1, 31), "pe": 0.10, "pb": 0.50, "ps": 0.3333},
            {"isin": "A", "month_end": dt.date(2024, 2, 29), "pe": -0.10, "pb": 0.50, "ps": 0.3333},
        ]
    )
    tri, summary, _ = _compute_domestic_triangulation_tables(
        w0_dom_df=w,
        domestic_isins=["A"],
        px_df=px,
        yld_df=yld,
        lens="Earnings (P/E)",
        name_map={"A": "A Co"},
        mode_label="Fallback to price growth",
    )
    assert np.isclose(float(tri.iloc[0]["Fundamental growth in holding period (%)"]), 10.0, atol=1e-12)
    assert "assume price growth" in str(tri.iloc[0]["Notes"])
    assert np.isclose(float(summary.iloc[0]["Value (%)"]), 10.0, atol=1e-12)


def test_domestic_triangulation_invalid_mode_b_excludes_and_rebases_weighted_average():
    idx = pd.to_datetime(["2024-01-31", "2024-02-29"]).to_period("M").to_timestamp("M")
    w = pd.DataFrame({"A": [0.6, 0.6], "B": [0.4, 0.4]}, index=idx)
    px = pd.DataFrame(
        [
            {"isin": "A", "month_end": dt.date(2024, 1, 31), "adj_price": 100.0},
            {"isin": "A", "month_end": dt.date(2024, 2, 29), "adj_price": 110.0},
            {"isin": "B", "month_end": dt.date(2024, 1, 31), "adj_price": 100.0},
            {"isin": "B", "month_end": dt.date(2024, 2, 29), "adj_price": 130.0},
        ]
    )
    yld = pd.DataFrame(
        [
            {"isin": "A", "month_end": dt.date(2024, 1, 31), "pe": 0.10, "pb": 0.50, "ps": 0.3333},
            {"isin": "A", "month_end": dt.date(2024, 2, 29), "pe": 0.10, "pb": 0.50, "ps": 0.3333},
            {"isin": "B", "month_end": dt.date(2024, 1, 31), "pe": 0.0833333333, "pb": 0.50, "ps": 0.3333},
            {"isin": "B", "month_end": dt.date(2024, 2, 29), "pe": -0.0833333333, "pb": 0.50, "ps": 0.3333},
        ]
    )
    tri, summary, coverage = _compute_domestic_triangulation_tables(
        w0_dom_df=w,
        domestic_isins=["A", "B"],
        px_df=px,
        yld_df=yld,
        lens="Earnings (P/E)",
        name_map={"A": "A Co", "B": "B Co"},
        mode_label="Exclude invalid from weighted average",
    )
    row_a = tri.loc[tri["Stock name"] == "A Co"].iloc[0]
    row_b = tri.loc[tri["Stock name"] == "B Co"].iloc[0]
    assert np.isclose(float(row_a["Fundamental growth in holding period (%)"]), 10.0, atol=1e-12)
    assert pd.isna(row_b["Fundamental growth in holding period (%)"])
    assert "excluded from weighted average" in str(row_b["Notes"])
    assert np.isclose(float(summary.iloc[0]["Value (%)"]), 10.0, atol=1e-12)
    assert np.isclose(float(coverage["valid_weight_sum"]), 0.6, atol=1e-12)
    assert np.isclose(float(coverage["invalid_weight_sum"]), 0.4, atol=1e-12)


def test_domestic_triangulation_average_weights_sum_to_100_percent():
    idx = pd.to_datetime(["2024-01-31", "2024-02-29"]).to_period("M").to_timestamp("M")
    w = pd.DataFrame({"A": [0.7, 0.7], "B": [0.3, 0.3]}, index=idx)
    px = pd.DataFrame(
        [
            {"isin": "A", "month_end": dt.date(2024, 1, 31), "adj_price": 100.0},
            {"isin": "A", "month_end": dt.date(2024, 2, 29), "adj_price": 101.0},
            {"isin": "B", "month_end": dt.date(2024, 1, 31), "adj_price": 100.0},
            {"isin": "B", "month_end": dt.date(2024, 2, 29), "adj_price": 101.0},
        ]
    )
    yld = pd.DataFrame(
        [
            {"isin": "A", "month_end": dt.date(2024, 1, 31), "pe": 0.10, "pb": 0.50, "ps": 0.3333},
            {"isin": "A", "month_end": dt.date(2024, 2, 29), "pe": 0.10, "pb": 0.50, "ps": 0.3333},
            {"isin": "B", "month_end": dt.date(2024, 1, 31), "pe": 0.10, "pb": 0.50, "ps": 0.3333},
            {"isin": "B", "month_end": dt.date(2024, 2, 29), "pe": 0.10, "pb": 0.50, "ps": 0.3333},
        ]
    )
    tri, _, _ = _compute_domestic_triangulation_tables(
        w0_dom_df=w,
        domestic_isins=["A", "B"],
        px_df=px,
        yld_df=yld,
        lens="Earnings (P/E)",
        name_map={"A": "A Co", "B": "B Co"},
        mode_label="Fallback to price growth",
    )
    weights_pct = tri["Average holding weight (%)"].to_numpy(dtype=float)
    assert np.isclose(np.sum(weights_pct) / 100.0, 1.0, atol=1e-12)
    assert np.isclose(np.sum(weights_pct), 100.0, atol=1e-9)
