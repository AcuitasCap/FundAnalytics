from __future__ import annotations

import pandas as pd
import numpy as np


def compute_adj_price_returns(
    prices_df: pd.DataFrame,
    months: list[pd.Timestamp],
    isins: list[str],
    audit_rows: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute monthly stock returns from adj_price:
      r_t = (adj_price_t / adj_price_{t-1}) - 1

    Returns:
      returns_df: index=t1 months, columns=isins
      invalid_rows_df: rows where adj_price_t or adj_price_{t-1} is missing/non-positive
      audit_df: sample rows with adj_price_{t-1}, adj_price_t, return
    """
    t0 = months[:-1]
    t1 = months[1:]

    returns_df = pd.DataFrame(index=t1, columns=isins, dtype=float)
    if prices_df is None or prices_df.empty or not isins:
        return returns_df.fillna(0.0), pd.DataFrame(), pd.DataFrame()

    px = prices_df.copy()
    px["isin"] = px["isin"].astype(str)
    px = px[px["isin"].isin(isins)].copy()
    if px.empty:
        return returns_df.fillna(0.0), pd.DataFrame(), pd.DataFrame()

    px["month_end"] = pd.to_datetime(px["month_end"], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
    px["adj_price"] = pd.to_numeric(px["adj_price"], errors="coerce")

    adj_piv = px.pivot_table(index="month_end", columns="isin", values="adj_price", aggfunc="last").reindex(months)
    adj_piv = adj_piv.reindex(columns=isins)

    p0 = adj_piv.loc[t0, isins].to_numpy(dtype=float)
    p1 = adj_piv.loc[t1, isins].to_numpy(dtype=float)
    valid = np.isfinite(p0) & np.isfinite(p1) & (p0 > 0) & (p1 > 0)

    with np.errstate(divide="ignore", invalid="ignore"):
        r = (p1 / p0) - 1.0
    r = np.where(valid, r, 0.0)
    returns_df.loc[:, isins] = r
    returns_df = returns_df.fillna(0.0)

    # Invalid rows diagnostics
    invalid_mask = ~valid
    invalid_records = []
    for i, end_m in enumerate(t1):
        for j, isin in enumerate(isins):
            if not invalid_mask[i, j]:
                continue
            p0v = p0[i, j]
            p1v = p1[i, j]
            if not np.isfinite(p0v) or p0v <= 0:
                reason = "invalid_adj_price_t_minus_1"
            elif not np.isfinite(p1v) or p1v <= 0:
                reason = "invalid_adj_price_t"
            else:
                reason = "invalid_adj_price_pair"
            invalid_records.append(
                {
                    "isin": isin,
                    "month_start": t0[i].date(),
                    "month_end": end_m.date(),
                    "adj_price_t_minus_1": None if not np.isfinite(p0v) else float(p0v),
                    "adj_price_t": None if not np.isfinite(p1v) else float(p1v),
                    "reason": reason,
                }
            )
    invalid_rows_df = pd.DataFrame(invalid_records)

    # Return audit sample
    audit_records = []
    for i, end_m in enumerate(t1):
        for j, isin in enumerate(isins):
            if len(audit_records) >= audit_rows:
                break
            p0v = p0[i, j]
            p1v = p1[i, j]
            rv = r[i, j]
            audit_records.append(
                {
                    "isin": isin,
                    "month_start": t0[i].date(),
                    "month_end": end_m.date(),
                    "adj_price_t_minus_1": None if not np.isfinite(p0v) else float(p0v),
                    "adj_price_t": None if not np.isfinite(p1v) else float(p1v),
                    "price_return_adj_price": float(rv),
                }
            )
        if len(audit_records) >= audit_rows:
            break
    audit_df = pd.DataFrame(audit_records)

    return returns_df, invalid_rows_df, audit_df


def compute_dividend_yield_returns(
    prices_df: pd.DataFrame,
    months: list[pd.Timestamp],
    isins: list[str],
) -> pd.DataFrame:
    """
    Compute monthly dividend return series aligned to return months (t1):
      div_return[t1, isin] = dividend_yield observed at t0.
    Missing values are treated as 0.
    """
    t1 = months[1:]
    out = pd.DataFrame(index=t1, columns=isins, dtype=float)
    if prices_df is None or prices_df.empty or not isins:
        return out.fillna(0.0)

    px = prices_df.copy()
    px["isin"] = px["isin"].astype(str)
    px = px[px["isin"].isin(isins)].copy()
    if px.empty:
        return out.fillna(0.0)

    if "month_end" not in px.columns and "price_date" in px.columns:
        px["month_end"] = px["price_date"]
    px["month_end"] = pd.to_datetime(px["month_end"], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
    px["dividend_yield"] = pd.to_numeric(px.get("dividend_yield"), errors="coerce")

    div_piv = px.pivot_table(index="month_end", columns="isin", values="dividend_yield", aggfunc="last").reindex(months)
    div_piv = div_piv.reindex(columns=isins)
    out.loc[:, isins] = div_piv.shift(1).loc[t1, isins].to_numpy(dtype=float)
    return out.fillna(0.0)


def compute_monthly_portfolio_multiples(
    multiples_df: pd.DataFrame,
    months: list[pd.Timestamp],
    isins: list[str],
    w0_df: pd.DataFrame,
    lens: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Compute portfolio multiple at start/end of each month using t0 weights.

    Returns:
      M_start: len(t1), portfolio multiple at t0
      M_end: len(t1), portfolio multiple at t1 with t0 weights
      debug: coverage/aggregation diagnostics
    """
    t0 = months[:-1]
    t1 = months[1:]
    n = len(t1)

    m_start = np.full(n, np.nan, dtype=float)
    m_end = np.full(n, np.nan, dtype=float)
    agg_y0_arr = np.full(n, np.nan, dtype=float)
    agg_y1_arr = np.full(n, np.nan, dtype=float)
    cov0_arr = np.zeros(n, dtype=float)
    cov1_arr = np.zeros(n, dtype=float)
    valid0_n = np.zeros(n, dtype=int)
    valid1_n = np.zeros(n, dtype=int)

    mult_col = "ps" if str(lens).startswith("Sales") else ("pe" if str(lens).startswith("Earnings") else "pb")
    valid_rule = "nonmissing_yield"

    if multiples_df is None or multiples_df.empty or not isins:
        debug = {
            "mult_col": mult_col,
            "validity_rule": valid_rule,
            "agg_yield_start": agg_y0_arr,
            "agg_yield_end": agg_y1_arr,
            "coverage_start_weight": cov0_arr,
            "coverage_end_weight": cov1_arr,
            "valid_count_start": valid0_n,
            "valid_count_end": valid1_n,
        }
        return m_start, m_end, debug

    mul = multiples_df.copy()
    mul["isin"] = mul["isin"].astype(str)
    mul = mul[mul["isin"].isin(isins)].copy()
    if mul.empty or mult_col not in mul.columns:
        debug = {
            "mult_col": mult_col,
            "validity_rule": valid_rule,
            "agg_yield_start": agg_y0_arr,
            "agg_yield_end": agg_y1_arr,
            "coverage_start_weight": cov0_arr,
            "coverage_end_weight": cov1_arr,
            "valid_count_start": valid0_n,
            "valid_count_end": valid1_n,
        }
        return m_start, m_end, debug

    if "month_end" not in mul.columns and "price_date" in mul.columns:
        mul["month_end"] = mul["price_date"]
    mul["month_end"] = pd.to_datetime(mul["month_end"], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
    mul[mult_col] = pd.to_numeric(mul[mult_col], errors="coerce")

    m_piv = mul.pivot_table(index="month_end", columns="isin", values=mult_col, aggfunc="last").reindex(months)
    m_piv = m_piv.reindex(columns=isins)
    y_vals = m_piv.to_numpy(dtype=float)

    w_aligned = w0_df.reindex(index=t0, columns=isins).fillna(0.0)
    w_mat = w_aligned.to_numpy(dtype=float)

    for i in range(n):
        w = w_mat[i, :]
        y0 = y_vals[i, :]
        y1 = y_vals[i + 1, :]

        m0 = np.isfinite(y0)
        m1 = np.isfinite(y1)

        valid0_n[i] = int(m0.sum())
        valid1_n[i] = int(m1.sum())
        cov0_arr[i] = float(np.sum(w[m0])) if np.any(m0) else 0.0
        cov1_arr[i] = float(np.sum(w[m1])) if np.any(m1) else 0.0

        y0_filled = np.where(m0, y0, 0.0)
        y1_filled = np.where(m1, y1, 0.0)
        agg_y0 = float(np.sum(w * y0_filled))
        agg_y1 = float(np.sum(w * y1_filled))
        agg_y0_arr[i] = agg_y0
        agg_y1_arr[i] = agg_y1

        m_start[i] = (1.0 / agg_y0) if np.isfinite(agg_y0) and agg_y0 != 0.0 else np.nan
        m_end[i] = (1.0 / agg_y1) if np.isfinite(agg_y1) and agg_y1 != 0.0 else np.nan

    debug = {
        "mult_col": mult_col,
        "validity_rule": valid_rule,
        "agg_yield_start": agg_y0_arr,
        "agg_yield_end": agg_y1_arr,
        "coverage_start_weight": cov0_arr,
        "coverage_end_weight": cov1_arr,
        "valid_count_start": valid0_n,
        "valid_count_end": valid1_n,
    }
    return m_start, m_end, debug
