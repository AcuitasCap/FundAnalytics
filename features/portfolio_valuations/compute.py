"""Pure live percentile calculations for Portfolio Valuations."""

import datetime as dt

import numpy as np
import pandas as pd


RANGE_COLUMNS = [
    "fund_id",
    "month_end",
    "anchor_month_end",
    "p25",
    "median",
    "p75",
    "selected_stock_count",
    "valid_stock_count",
    "coverage_pct",
]
SERIES_COLUMNS = [
    "month_end",
    "series",
    "p25",
    "median",
    "p75",
    "valid_stock_count",
    "peer_fund_count",
]


def _top_half_by_weight(group: pd.DataFrame) -> pd.DataFrame:
    """Include descending holdings through the stock that crosses 50% weight."""
    ranked = group.sort_values(["weight_pct", "isin"], ascending=[False, True]).copy()
    weights = pd.to_numeric(ranked["weight_pct"], errors="coerce").fillna(0.0).clip(lower=0.0)
    total = float(weights.sum())
    if total <= 0:
        return ranked.iloc[0:0].copy()
    previous_cumulative = weights.cumsum() - weights
    return ranked.loc[previous_cumulative < (0.5 * total)].copy()


def compute_fund_valuation_ranges(
    holdings: pd.DataFrame,
    multiples: pd.DataFrame,
    *,
    start_date: dt.date,
    end_date: dt.date,
    mode: str,
    segment: str,
    metric: str,
    valuation_range: str,
    minimum_stocks: int = 5,
) -> pd.DataFrame:
    """Return unweighted P25/median/P75 for every eligible fund and month."""
    if holdings.empty or multiples.empty:
        return pd.DataFrame(columns=RANGE_COLUMNS)
    if segment not in {"Total", "Financials", "Non-financials"}:
        raise ValueError(f"Unsupported segment: {segment}")
    metric_column = {"P/S": "ps", "P/E": "pe", "P/B": "pb"}.get(metric)
    if metric_column is None:
        raise ValueError(f"Unsupported valuation metric: {metric}")
    if valuation_range not in {"Full valuation range", "Valuation range of top 50% stocks"}:
        raise ValueError(f"Unsupported valuation range: {valuation_range}")
    if minimum_stocks < 1:
        raise ValueError("minimum_stocks must be positive")

    start_period = pd.Period(start_date, freq="M")
    end_period = pd.Period(end_date, freq="M")
    month_periods = pd.period_range(start_period, end_period, freq="M")
    if len(month_periods) == 0:
        return pd.DataFrame(columns=RANGE_COLUMNS)

    h = holdings.copy()
    h["holding_month_end"] = pd.to_datetime(h["holding_month_end"], errors="coerce")
    h["anchor_month_end"] = pd.to_datetime(h["anchor_month_end"], errors="coerce")
    h["weight_pct"] = pd.to_numeric(h["weight_pct"], errors="coerce")
    h = h.dropna(subset=["fund_id", "isin", "holding_month_end", "anchor_month_end", "weight_pct"])
    h = h[h["weight_pct"] > 0].copy()
    h["is_financial"] = h["is_financial"].fillna(False).astype(bool)

    if segment == "Financials":
        h = h[h["is_financial"]].copy()
    elif segment == "Non-financials":
        h = h[~h["is_financial"]].copy()
    if h.empty:
        return pd.DataFrame(columns=RANGE_COLUMNS)

    if mode == "Valuations of historical portfolios":
        h["month_key"] = h["holding_month_end"].dt.to_period("M")
        h = h[h["month_key"].isin(month_periods)].copy()
    elif mode == "Historical valuations of current portfolio":
        anchors = h.drop(columns=["month_key"], errors="ignore")
        month_grid = pd.DataFrame({"month_key": month_periods})
        anchors["_join_key"] = 1
        month_grid["_join_key"] = 1
        h = anchors.merge(month_grid, on="_join_key").drop(columns=["_join_key"])
    else:
        raise ValueError(f"Unsupported valuation mode: {mode}")

    if valuation_range == "Valuation range of top 50% stocks":
        selected = [
            _top_half_by_weight(group)
            for _, group in h.groupby(["fund_id", "month_key"], sort=False)
        ]
        h = pd.concat(selected, ignore_index=True) if selected else h.iloc[0:0].copy()
    if h.empty:
        return pd.DataFrame(columns=RANGE_COLUMNS)

    v = multiples.copy()
    v["month_end"] = pd.to_datetime(v["month_end"], errors="coerce")
    v["month_key"] = v["month_end"].dt.to_period("M")
    v[metric_column] = pd.to_numeric(v[metric_column], errors="coerce")
    joined = h.merge(
        v[["isin", "month_key", metric_column]],
        on=["isin", "month_key"],
        how="left",
    )

    records: list[dict] = []
    for (fund_id, month_key), group in joined.groupby(["fund_id", "month_key"], sort=True):
        values = pd.to_numeric(group[metric_column], errors="coerce")
        valid = values[np.isfinite(values) & (values > 0)]
        selected_weight = float(group["weight_pct"].sum())
        valid_weight = float(group.loc[valid.index, "weight_pct"].sum()) if len(valid) else 0.0
        enough = len(valid) >= minimum_stocks
        percentiles = np.percentile(valid.to_numpy(dtype=float), [25, 50, 75]) if enough else [np.nan] * 3
        anchor_values = pd.to_datetime(group["anchor_month_end"], errors="coerce").dropna()
        records.append(
            {
                "fund_id": int(fund_id),
                "month_end": month_key.to_timestamp("M"),
                "anchor_month_end": anchor_values.max() if not anchor_values.empty else pd.NaT,
                "p25": float(percentiles[0]) if enough else np.nan,
                "median": float(percentiles[1]) if enough else np.nan,
                "p75": float(percentiles[2]) if enough else np.nan,
                "selected_stock_count": int(len(group)),
                "valid_stock_count": int(len(valid)),
                "coverage_pct": (100.0 * valid_weight / selected_weight) if selected_weight > 0 else np.nan,
            }
        )

    return pd.DataFrame.from_records(records, columns=RANGE_COLUMNS)


def build_focus_peer_series(ranges: pd.DataFrame, focus_fund_id: int) -> pd.DataFrame:
    """Compare focus ranges with the median corresponding statistic across peer funds."""
    if ranges.empty:
        return pd.DataFrame(columns=SERIES_COLUMNS)

    focus = ranges[ranges["fund_id"] == focus_fund_id].copy()
    focus["series"] = "Focus fund"
    focus["peer_fund_count"] = np.nan
    focus_out = focus[
        ["month_end", "series", "p25", "median", "p75", "valid_stock_count", "peer_fund_count"]
    ]

    peers = ranges[ranges["fund_id"] != focus_fund_id].dropna(subset=["p25", "median", "p75"])
    peer_rows: list[dict] = []
    for month_end, group in peers.groupby("month_end", sort=True):
        peer_rows.append(
            {
                "month_end": month_end,
                "series": "Peer-set",
                "p25": float(group["p25"].median()),
                "median": float(group["median"].median()),
                "p75": float(group["p75"].median()),
                "valid_stock_count": np.nan,
                "peer_fund_count": int(group["fund_id"].nunique()),
            }
        )
    peers_out = pd.DataFrame.from_records(peer_rows, columns=SERIES_COLUMNS)
    return (
        pd.concat([focus_out, peers_out], ignore_index=True)
        .sort_values(["month_end", "series"])
        .reset_index(drop=True)
    )
