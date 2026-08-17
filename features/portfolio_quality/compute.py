"""Pure calculation layer for the Portfolio Quality feature."""

import numpy as np
import pandas as pd


ROC_MODE = "Return on capital vs peers / universe"
QUARTILE_MODE = "Quality quartile exposures (Q1–Q4)"
_FUNDAMENTAL_COLUMNS = ["fund_id", "fund_name", "month_end", "metric"]


def _empty_fundamentals() -> pd.DataFrame:
    return pd.DataFrame(columns=_FUNDAMENTAL_COLUMNS)


def _median_metric(roe_roce_data: dict[str, pd.DataFrame], isin, eval_date, is_financial) -> float:
    sub = roe_roce_data.get(isin)
    if sub is None or sub.empty:
        return 0.0
    sub = sub.loc[sub["year_end_date"] < eval_date].tail(5)
    metric = sub["roe" if is_financial else "roce"].dropna()
    return float(metric.median()) if not metric.empty else 0.0


def _portfolio_fundamentals(
    portfolio_data: pd.DataFrame, roe_roce_data: dict[str, pd.DataFrame], segment: str
) -> pd.DataFrame:
    if portfolio_data.empty:
        return _empty_fundamentals()
    df = portfolio_data.loc[portfolio_data["asset_type"] == "Domestic Equities"].copy()
    if df.empty:
        return _empty_fundamentals()
    keys = ["fund_id", "fund_name", "month_end"]
    df["weight_pct"] = df["weight_pct"].astype(float)
    totals = df.groupby(keys)["weight_pct"].transform("sum")
    df["dom_weight"] = np.where(totals > 0, df["weight_pct"] / totals, 0.0)
    df["stock_metric"] = [
        _median_metric(roe_roce_data, row.isin, row.month_end, bool(row.is_financial))
        for row in df.itertuples(index=False)
    ]
    records = []
    for (fund_id, fund_name, month_end), group in df.groupby(keys):
        if segment == "Financials":
            chosen = group.loc[group["is_financial"] == True].copy()
        elif segment == "Non-financials":
            chosen = group.loc[group["is_financial"] == False].copy()
        else:
            chosen = group.copy()
        if chosen.empty:
            metric = np.nan
        elif segment == "Total":
            metric = (chosen["dom_weight"] * chosen["stock_metric"].fillna(0.0)).sum()
        else:
            total = chosen["dom_weight"].sum()
            weights = np.where(total > 0, chosen["dom_weight"] / total, 0.0)
            metric = (weights * chosen["stock_metric"].fillna(0.0)).sum()
        records.append({"fund_id": fund_id, "fund_name": fund_name,
                        "month_end": month_end, "metric": metric})
    return pd.DataFrame.from_records(records, columns=_FUNDAMENTAL_COLUMNS).sort_values(
        ["fund_name", "month_end"]
    )


def _quartile_exposure(rows: pd.DataFrame, valid_months: list) -> pd.DataFrame:
    if rows.empty or not valid_months:
        return pd.DataFrame()
    df = rows.copy()
    df["month_end"] = pd.to_datetime(df["month_end"]).dt.date
    df = df.loc[df["month_end"].isin(valid_months)]
    if df.empty:
        return pd.DataFrame()
    df["holding_weight"] = df["holding_weight"].astype(float)
    totals = df.groupby("month_end")["holding_weight"].transform("sum")
    df["re_based_weight"] = np.where(totals > 0, df["holding_weight"] / totals * 100.0, 0.0)
    pivot = (df.groupby(["quality_quartile", "month_end"])["re_based_weight"].sum()
               .unstack("month_end").reindex(index=["Q1", "Q2", "Q3", "Q4"]))
    if pivot.empty:
        return pd.DataFrame()
    pivot.columns = [pd.to_datetime(column).strftime("%b %Y") for column in pivot.columns]
    total = pivot.sum(axis=0).to_frame().T
    total.index = ["Total"]
    return pd.concat([pivot, total])


def compute_quality_analysis(
    mode: str,
    portfolio_data: pd.DataFrame,
    roe_roce_data: dict[str, pd.DataFrame],
    quality_bucket_data: pd.DataFrame | None,
    context: dict,
) -> pd.DataFrame:
    """Return the display-ready table for the requested Quality analysis mode."""
    fundamentals = _portfolio_fundamentals(
        portfolio_data, roe_roce_data, context.get("segment_choice", "Total")
    )
    if mode == ROC_MODE:
        if fundamentals.empty:
            return fundamentals
        focus_id = context["focus_fund_id"]
        focus = fundamentals.loc[fundamentals["fund_id"] == focus_id].copy()
        if focus.empty:
            return focus
        other_ids = [fid for fid in context["selected_fund_ids"] if fid != focus_id]
        if context.get("comparison_mode") == "Universe median" and other_ids:
            others = fundamentals.loc[fundamentals["fund_id"].isin(other_ids)]
            if not others.empty:
                peers = others.groupby("month_end")["metric"].median().reset_index()
                peers["fund_name"] = "Universe median (others)"
                peers["fund_id"] = -1
                return pd.concat([focus, peers], ignore_index=True)
            return focus
        return fundamentals.loc[fundamentals["fund_id"].isin(context["selected_fund_ids"])].copy()
    if mode == QUARTILE_MODE:
        focus_months = sorted(
            fundamentals.loc[fundamentals["fund_id"] == context["focus_fund_id"], "month_end"].unique()
        )
        bucket_rows = quality_bucket_data if quality_bucket_data is not None else pd.DataFrame()
        return _quartile_exposure(bucket_rows, focus_months)
    raise ValueError(f"Unsupported Quality analysis mode: {mode}")
