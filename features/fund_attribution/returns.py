from __future__ import annotations

import pandas as pd
import numpy as np


def compute_adj_price_returns(
    prices_df: pd.DataFrame,
    months: list[pd.Timestamp],
    ace_codes: list[str],
    audit_rows: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute adjusted price returns between consecutive supplied endpoints.
    Endpoints can be monthly or farther apart when holdings are reported less often.

    Returns:
      returns_df: index=t1 months, columns=ace_codes
      invalid_rows_df: rows where adj_price_t or adj_price_{t-1} is missing/non-positive
      audit_df: sample rows with adj_price_{t-1}, adj_price_t, return
    """
    t0 = months[:-1]
    t1 = months[1:]

    returns_df = pd.DataFrame(index=t1, columns=ace_codes, dtype=float)
    if prices_df is None or prices_df.empty or not ace_codes:
        return returns_df.fillna(0.0), pd.DataFrame(), pd.DataFrame()

    px = prices_df.copy()
    px["ace_code"] = px["ace_code"].astype(str)
    px = px[px["ace_code"].isin(ace_codes)].copy()
    if px.empty:
        return returns_df.fillna(0.0), pd.DataFrame(), pd.DataFrame()

    px["month_end"] = pd.to_datetime(px["month_end"], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
    px["adj_price"] = pd.to_numeric(px["adj_price"], errors="coerce")

    adj_piv = px.pivot_table(index="month_end", columns="ace_code", values="adj_price", aggfunc="last").reindex(months)
    adj_piv = adj_piv.reindex(columns=ace_codes)

    p0 = adj_piv.loc[t0, ace_codes].to_numpy(dtype=float)
    p1 = adj_piv.loc[t1, ace_codes].to_numpy(dtype=float)
    valid = np.isfinite(p0) & np.isfinite(p1) & (p0 > 0) & (p1 > 0)

    with np.errstate(divide="ignore", invalid="ignore"):
        r = (p1 / p0) - 1.0
    r = np.where(valid, r, 0.0)
    returns_df.loc[:, ace_codes] = r
    returns_df = returns_df.fillna(0.0)

    # Invalid rows diagnostics
    invalid_mask = ~valid
    invalid_records = []
    for i, end_m in enumerate(t1):
        for j, ace_code in enumerate(ace_codes):
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
                    "ace_code": ace_code,
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
        for j, ace_code in enumerate(ace_codes):
            if len(audit_records) >= audit_rows:
                break
            p0v = p0[i, j]
            p1v = p1[i, j]
            rv = r[i, j]
            audit_records.append(
                {
                    "ace_code": ace_code,
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
    ace_codes: list[str],
) -> pd.DataFrame:
    """
    Compute dividend returns in each interval, using the yield at the start
    of each month. For longer intervals, each monthly dividend is measured
    against the initial stock allocation using the intervening price change.
    Missing yields contribute zero.
    """
    t1 = months[1:]
    out = pd.DataFrame(index=t1, columns=ace_codes, dtype=float)
    if prices_df is None or prices_df.empty or not ace_codes or not t1:
        return out.fillna(0.0)

    px = prices_df.copy()
    px["ace_code"] = px["ace_code"].astype(str)
    px = px[px["ace_code"].isin(ace_codes)].copy()
    if px.empty:
        return out.fillna(0.0)

    if "month_end" not in px.columns and "price_date" in px.columns:
        px["month_end"] = px["price_date"]
    px["month_end"] = pd.to_datetime(px["month_end"], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
    px["dividend_yield"] = pd.to_numeric(px.get("dividend_yield"), errors="coerce")

    div_piv = px.pivot_table(index="month_end", columns="ace_code", values="dividend_yield", aggfunc="last")
    if len(pd.date_range(months[0], months[-1], freq="ME")) == len(months):
        out.loc[:, ace_codes] = div_piv.reindex(index=months[:-1], columns=ace_codes).to_numpy(dtype=float)
        return out.fillna(0.0)

    adj_piv = pd.DataFrame()
    if "adj_price" in px:
        px["adj_price"] = pd.to_numeric(px["adj_price"], errors="coerce")
        adj_piv = px.pivot_table(index="month_end", columns="ace_code", values="adj_price", aggfunc="last")
    for start, end in zip(months[:-1], t1):
        accrual_months = pd.date_range(start, end, freq="ME", inclusive="left")
        monthly_yields = div_piv.reindex(index=accrual_months, columns=ace_codes).fillna(0.0)
        if len(accrual_months) == 1:
            out.loc[end] = monthly_yields.iloc[0]
            continue
        prices = adj_piv.reindex(index=accrual_months, columns=ace_codes)
        prices = prices.where(prices > 0)
        drift = prices.div(prices.iloc[0]).replace([np.inf, -np.inf], np.nan)
        # Starting yields are already expressed relative to the opening price.
        drift.iloc[0] = 1.0
        out.loc[end] = (monthly_yields * drift).sum(axis=0)
    return out.fillna(0.0)
