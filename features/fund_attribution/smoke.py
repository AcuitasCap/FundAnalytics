import datetime as dt

import numpy as np
import pandas as pd

from core.dates import month_ends_between, to_month_end
from core.fund_catalog import fetch_categories, fetch_funds_for_categories
from features.fund_attribution.data import BENCH_NAME_NIFTY50, _load_attrib_raw_window
from features.fund_attribution.compute import _compute_attribution

def _run_attrib_smoke_test():
    """
    Exercise a representative live attribution run.

    This is intentionally a structural smoke test, not a data-regression test:
    market data and portfolio holdings are expected to change over time. It
    verifies that the selected fund can be loaded, attribution completes, and
    the key result tables contain usable, internally consistent values.
    """
    category = "Flexicap Fund"
    focus = "UTI Flexicap Fund"
    fallback_category = "Flexi Cap Fund"
    fallback_focus = "UTI Flexi Cap Fund"
    lookback_years = 15
    start_me = to_month_end(dt.date(2020, 12, 1))
    end_me = to_month_end(dt.date(2025, 9, 1))
    bench_mode = "Vs NIFTY 50"
    lens = "Earnings (P/E)"

    categories = fetch_categories()
    cat_list = [category] if category in categories else [fallback_category] if fallback_category in categories else categories
    funds_df = fetch_funds_for_categories(cat_list)
    if funds_df.empty or (focus not in funds_df["fund_name"].tolist() and fallback_focus not in funds_df["fund_name"].tolist()):
        # fallback: scan all categories
        if categories and cat_list != categories:
            funds_df = fetch_funds_for_categories(categories)
    if funds_df.empty or (focus not in funds_df["fund_name"].tolist() and fallback_focus not in funds_df["fund_name"].tolist()):
        raise AssertionError(f"Smoke test fund not found: {focus} in category {category}")
    focus_name = focus if focus in funds_df["fund_name"].tolist() else fallback_focus
    focus_id = int(funds_df.loc[funds_df["fund_name"] == focus_name, "fund_id"].iloc[0])

    raw = _load_attrib_raw_window(focus_id, end_me, lookback_years=lookback_years, data_version="v1")
    stock_df, hit_df, cat_df, diag = _compute_attribution(
        raw, start_me, end_me, bench_mode, lens=lens, universe_mode="Full holdings"
    )
    if "error" in diag:
        raise AssertionError(diag["error"])

    dom_sum = diag.get("domestic_decomp_summary_df", pd.DataFrame())
    dom_total = np.nan
    if isinstance(dom_sum, pd.DataFrame) and not dom_sum.empty and "Metric" in dom_sum.columns:
        row = dom_sum.loc[dom_sum["Metric"] == "Domestic sleeve total return"]
        if not row.empty:
            dom_total = float(row["Total return (%)"].iloc[0])

    # Benchmark total return from NAV series
    bench_name = BENCH_NAME_NIFTY50
    months = month_ends_between(start_me, end_me)
    bench_total = np.nan
    bn = raw.get("bench_nav", pd.DataFrame())
    if not bn.empty and "bench_name" in bn.columns:
        b = bn[bn["bench_name"] == bench_name].copy()
        if not b.empty:
            b["month_end"] = pd.to_datetime(b["month_end"]).dt.to_period("M").dt.to_timestamp("M")
            month_idx = [pd.Timestamp(m).to_period("M").to_timestamp("M") for m in months]
            bnp = b.pivot_table(index="month_end", values="nav_value", aggfunc="last").reindex(month_idx)
            v0 = bnp.iloc[:-1, 0].to_numpy(dtype=float)
            v1 = bnp.iloc[1:, 0].to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = (v1 / v0) - 1.0
            r = np.where(np.isfinite(r), r, 0.0)
            if len(r):
                bench_total = float(np.prod(1.0 + r) - 1.0)

    # Hit-rate winners count (exclude cash)
    winners_count = 0
    if isinstance(stock_df, pd.DataFrame) and not stock_df.empty:
        is_cash_row = stock_df["Stock name"].astype(str).str.strip().str.lower().str.startswith("cash")
        non_cash = stock_df.loc[~is_cash_row].copy()
        winners_count = int((non_cash["Stock outperformance (pp)"] > 0).sum())

    non_cash_count = len(non_cash) if isinstance(stock_df, pd.DataFrame) and not stock_df.empty else 0
    required_domestic_metrics = {
        "Domestic sleeve total return",
        "Price return",
        "Dividend yield return",
    }
    available_domestic_metrics = set(dom_sum.get("Metric", pd.Series(dtype=str)).dropna().astype(str))

    results_ok = (
        isinstance(stock_df, pd.DataFrame)
        and not stock_df.empty
        and isinstance(hit_df, pd.DataFrame)
        and not hit_df.empty
        and isinstance(cat_df, pd.DataFrame)
        and not cat_df.empty
        and required_domestic_metrics.issubset(available_domestic_metrics)
        and np.isfinite(dom_total)
        and np.isfinite(bench_total)
        and 0 <= winners_count <= non_cash_count
    )

    print("Attribution smoke test:")
    print(f"- Period: {start_me} to {end_me}")
    print(f"- Domestic sleeve total return (%): {dom_total:.2f}")
    print(f"- Benchmark return {bench_name} (%): {bench_total * 100.0:.2f}")
    print(f"- Winners count: {winners_count} / {non_cash_count} non-cash stocks")

    if not results_ok:
        dom_debug = diag.get("domestic_decomp_debug", {})
        print("Diagnostics:")
        print(f"- Domestic ISIN count: {diag.get('domestic_decomp_stock_count')}")
        print(f"- Start multiple: {dom_debug.get('start_multiple')}")
        print(f"- End multiple: {dom_debug.get('end_multiple')}")
        print(f"- Start coverage weight: {dom_debug.get('start_coverage_weight')}")
        print(f"- End coverage weight: {dom_debug.get('end_coverage_weight')}")
        print(f"- Benchmark series used: {bench_name}")
        raise AssertionError("Attribution smoke test returned incomplete or invalid results.")

