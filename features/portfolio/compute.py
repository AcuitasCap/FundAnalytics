"""Pure Active Share computations for the Portfolio explorer."""

import numpy as np
import pandas as pd


def _build_composite_portfolio(df_all: pd.DataFrame, fund_proportions: dict[int, float], period) -> pd.Series | None:
    subset = df_all[(df_all["fund_id"].isin(fund_proportions)) & (df_all["month_end"] == period)].copy()
    if subset.empty:
        return None
    subset["fund_prop"] = subset["fund_id"].map(fund_proportions).astype(float)
    subset["allocation"] = subset["weight_pct"].astype(float) * subset["fund_prop"]
    weights = subset.groupby("isin")["allocation"].sum()
    total = weights.sum()
    return None if total == 0 else weights / total


def compute_active_share(df_all: pd.DataFrame, fund_proportions_a: dict[int, float], fund_proportions_b: dict[int, float]) -> pd.DataFrame:
    """Return Active Share (%) for each available holdings period."""
    columns = ["period_date", "period_label", "active_share_pct"]
    if df_all.empty:
        return pd.DataFrame(columns=columns)
    records = []
    for period in sorted(df_all["month_end"].unique()):
        weights_a = _build_composite_portfolio(df_all, fund_proportions_a, period)
        weights_b = _build_composite_portfolio(df_all, fund_proportions_b, period)
        if weights_a is None or weights_b is None:
            active_share_pct = np.nan
        else:
            shared_isins = weights_a.index.union(weights_b.index)
            overlap = sum(min(float(weights_a.get(isin, 0)), float(weights_b.get(isin, 0))) for isin in shared_isins)
            active_share_pct = (1.0 - overlap) * 100.0
        timestamp = pd.to_datetime(period)
        records.append({"period_date": timestamp, "period_label": timestamp.strftime("%b %Y"), "active_share_pct": active_share_pct})
    return pd.DataFrame.from_records(records, columns=columns).sort_values("period_date")
