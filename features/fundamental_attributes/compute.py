"""Pure stock medians, peer comparisons and rolling three-year growth."""

import numpy as np
import pandas as pd

from core.dataframes import ensure_unique_monthly_rows
from . import CURRENT, HISTORICAL


FUND_COLUMNS = ["fund_id", "month_end", "anchor_month_end", "median",
                "selected_stock_count", "valid_stock_count", "coverage_pct"]
SERIES_COLUMNS = ["month_end", "series", "median", "valid_stock_count", "peer_fund_count"]


def select_segment(frame, segment):
    if segment == "Total":
        return frame.copy()
    if segment not in {"Financials", "Non-financials"}:
        raise ValueError(f"Unsupported segment: {segment}")
    return frame.loc[frame["is_financial"].eq(segment == "Financials").fillna(False)].copy()


def portfolio_rows(holdings, start_date, end_date, mode, focus_id, top_half=False):
    """Consolidate ACE aliases, select top holdings, then apply segment elsewhere.

    Top half means reported weights through the stock taking the cumulative
    weight above 50 percentage points of the entire fund, before segment or
    valuation availability filtering. Unmapped holdings still consume weight.
    """
    if mode not in {HISTORICAL, CURRENT}:
        raise ValueError(f"Unsupported portfolio mode: {mode}")
    h = holdings.copy()
    if h.empty:
        h["month_end"] = pd.Series(dtype="datetime64[ns]")
        return h, pd.DataFrame(columns=["fund_id", "month_end", "anchor_month_end"])
    h["weight_pct"] = pd.to_numeric(h["weight_pct"], errors="coerce")
    h = h[h["weight_pct"] > 0].copy()
    h["ace_code"] = h["ace_code"].astype("string").str.strip().replace("", pd.NA)
    for column in ("holding_month_end", "anchor_month_end"):
        h[column] = pd.to_datetime(h[column]).dt.to_period("M").dt.to_timestamp("M")
    h["_stock_key"] = "ACE:" + h["ace_code"]
    h["_stock_key"] = h["_stock_key"].fillna("HOLDING:" + h["holding_key"].astype(str))
    h = h.groupby(["fund_id", "holding_month_end", "anchor_month_end", "_stock_key"],
                  as_index=False, dropna=False).agg(
                      ace_code=("ace_code", "first"), weight_pct=("weight_pct", "sum"),
                      is_financial=("is_financial", "first"))
    months = pd.period_range(start_date, end_date, freq="M").to_timestamp("M")
    if mode == CURRENT:
        h = h.merge(pd.DataFrame({"month_end": months}), how="cross")
    else:
        h["month_end"] = h["holding_month_end"]
        h = h[h["month_end"].isin(months)].copy()
    grid = h[["fund_id", "month_end", "anchor_month_end"]].drop_duplicates()
    if top_half and not h.empty:
        h = h.sort_values(["fund_id", "month_end", "weight_pct", "_stock_key"],
                          ascending=[True, True, False, True])
        previous = h.groupby(["fund_id", "month_end"])["weight_pct"].cumsum() - h["weight_pct"]
        h = h[(h["fund_id"] != focus_id) | (previous <= 50.0)].copy()
    return h, grid


def stock_growth(financials, metric):
    """TTM CAGR from four consecutive quarters versus exactly 12 quarters ago.

    Missing quarters break the TTM window. Zero/negative current or base TTM
    values have no CAGR; negative individual quarters remain valid in the sum.
    """
    if metric not in {"Sales", "PAT"}:
        raise ValueError(f"Unsupported growth metric: {metric}")
    columns = ["ace_code", "quarter_end", "value", "current_ttm", "base_ttm"]
    if financials.empty:
        return pd.DataFrame(columns=columns)
    frame = financials.copy()
    frame["quarter_end"] = pd.to_datetime(frame["period_end"]).dt.to_period("Q")
    column = "sales" if metric == "Sales" else "pat"
    frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = ensure_unique_monthly_rows(frame, ["ace_code", "quarter_end"], [column],
                                       "Quarterly growth inputs")
    records = []
    for code, group in frame.groupby("ace_code"):
        series = group.set_index("quarter_end")[column].sort_index()
        series = series.reindex(pd.period_range(series.index.min(), series.index.max(), freq="Q"))
        ttm = series.rolling(4, min_periods=4).sum()
        base = ttm.shift(12)
        valid = np.isfinite(ttm) & np.isfinite(base) & (ttm > 0) & (base > 0)
        cagr = ((ttm.where(valid) / base.where(valid)) ** (1 / 3) - 1) * 100
        records.append(pd.DataFrame({"ace_code": code, "quarter_end": series.index.to_timestamp("Q"),
                                     "value": cagr.to_numpy(), "current_ttm": ttm.to_numpy(),
                                     "base_ttm": base.to_numpy()}))
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame(columns=columns)


def attach_stock_values(rows, facts, kind, metric, growth_values=None):
    """Match valuations by ACE/month-year, or growth by completed quarter."""
    rows = rows.copy()
    if kind == "valuation":
        column = {"P/S": "ps", "P/E": "pe", "P/B": "pb"}.get(metric)
        if column is None:
            raise ValueError(f"Unsupported valuation metric: {metric}")
        values = facts[["ace_code", "month_end", column]].rename(columns={column: "value"}).copy()
        rows["month_key"] = pd.to_datetime(rows["month_end"]).dt.to_period("M")
        values["month_key"] = pd.to_datetime(values.pop("month_end")).dt.to_period("M")
        keys = ["ace_code", "month_key"]
    elif kind == "growth":
        # At a quarter-end the same quarter's results are not yet available.
        rows["quarter_end"] = rows["month_end"].dt.to_period("Q").dt.start_time - pd.Timedelta(days=1)
        values = (stock_growth(facts, metric) if growth_values is None else growth_values)[
            ["ace_code", "quarter_end", "value"]]
        keys = ["ace_code", "quarter_end"]
    else:
        raise ValueError(f"Unsupported attribute kind: {kind}")
    values = ensure_unique_monthly_rows(values, keys, ["value"], "Fundamental attribute values")
    # Never merge NULL keys: pandas otherwise matches all identifier-less rows.
    values = values[values["ace_code"].notna()].copy()
    values["ace_code"] = values["ace_code"].astype("string")
    joined = rows.merge(values, on=keys, how="left", validate="many_to_one")
    joined["value"] = pd.to_numeric(joined["value"], errors="coerce")
    joined.loc[~np.isfinite(joined["value"]), "value"] = np.nan
    if kind == "valuation":
        joined.loc[joined["value"] <= 0, "value"] = np.nan
    return joined


def compute_attribute_comparison(holdings, members, facts, *, start_date, end_date,
                                 mode, focus_id, focus_label, compare, segment, kind,
                                 metric, top_half=False, minimum_stocks=5):
    """Return focus/comparator median lines and per-fund coverage diagnostics."""
    if minimum_stocks < 1 or pd.Period(start_date, freq="M") > pd.Period(end_date, freq="M"):
        raise ValueError("Check the selected period and minimum stock count.")
    h, grid = portfolio_rows(holdings, start_date, end_date, mode, focus_id, top_half)
    if h.empty:
        return pd.DataFrame(columns=SERIES_COLUMNS), pd.DataFrame(columns=FUND_COLUMNS)
    chosen = select_segment(h, segment)
    growth_values = stock_growth(facts, metric) if kind == "growth" else None
    valued = attach_stock_values(chosen, facts, kind, metric, growth_values)
    records = []
    for (fund_id, month_end), group in valued.groupby(["fund_id", "month_end"]):
        valid = group["value"].notna()
        count = int(valid.sum())
        weight = group["weight_pct"].sum()
        records.append({"fund_id": fund_id, "month_end": month_end,
                        "median": group["value"].median() if count >= minimum_stocks else np.nan,
                        "selected_stock_count": len(group), "valid_stock_count": count,
                        "coverage_pct": 100 * group.loc[valid, "weight_pct"].sum() / weight if weight else 0})
    stats = pd.DataFrame(records, columns=[c for c in FUND_COLUMNS if c != "anchor_month_end"])
    stats = grid.merge(stats, on=["fund_id", "month_end"], how="left")
    for column in ("selected_stock_count", "valid_stock_count", "coverage_pct"):
        stats[column] = pd.to_numeric(stats[column], errors="coerce").fillna(0)
    focus = stats[stats["fund_id"] == focus_id].copy()
    if focus.empty:
        return pd.DataFrame(columns=SERIES_COLUMNS), stats[FUND_COLUMNS]
    focus["series"] = focus_label
    focus["peer_fund_count"] = np.nan
    comparison = focus[["month_end"]].copy()
    if compare == "Peers":
        peers = stats[(stats["fund_id"] != focus_id) & stats["median"].notna()]
        summary = peers.groupby("month_end").agg(median=("median", "median"),
                                                  peer_fund_count=("fund_id", "nunique")).reset_index()
        comparison = comparison.merge(summary, on="month_end", how="left")
        comparison["valid_stock_count"] = np.nan
        comparison["peer_fund_count"] = comparison["peer_fund_count"].fillna(0)
    else:
        cohort = members.copy()
        cohort["month_end"] = pd.to_datetime(cohort["month_end"]).dt.to_period("M").dt.to_timestamp("M")
        cohort = select_segment(cohort[cohort["month_end"].isin(focus["month_end"])], segment)
        cohort = attach_stock_values(cohort, facts, kind, metric, growth_values)
        summary = cohort.groupby("month_end").agg(median=("value", "median"),
                                                   valid_stock_count=("value", "count")).reset_index()
        summary.loc[summary["valid_stock_count"] < minimum_stocks, "median"] = np.nan
        comparison = comparison.merge(summary, on="month_end", how="left")
        comparison["valid_stock_count"] = comparison["valid_stock_count"].fillna(0)
        comparison["peer_fund_count"] = np.nan
    comparison["series"] = compare
    series = pd.concat([focus[SERIES_COLUMNS], comparison[SERIES_COLUMNS]], ignore_index=True)
    return series.sort_values(["month_end", "series"]).reset_index(drop=True), stats[FUND_COLUMNS]
