import datetime as dt

import numpy as np
import pandas as pd

from .returns import (
    compute_adj_price_returns,
    compute_dividend_yield_returns,
)
from core.dates import to_month_end
from features.fund_attribution.data import (
    BENCH_NAME_MID150, BENCH_NAME_NIFTY100, BENCH_NAME_NIFTY50,
    BENCH_NAME_NIFTY500, BENCH_NAME_SMALL250, LIKE_FOR_LIKE_BENCH,
    _monthly_cash_return,
)

def _compute_attribution(
    raw: dict,
    start_date: dt.date,
    end_date: dt.date,
    bench_mode: str,
    universe_mode: str = "Full holdings",
    include_dividends: bool = True,
    benchmark_stock_keys: tuple[str, ...] = (),
):
    """Return (stock_df, hit_df, category_df, diag). Contributions are Rs per 100 base."""
    start_me = to_month_end(start_date)
    end_me = to_month_end(end_date)

    if start_me >= end_me:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"error": "Select at least two month-ends."}

    h = raw["holdings"].copy()
    px = raw["prices"].copy()
    sz = raw["size_band"].copy()
    bn = raw["bench_nav"].copy()
    sm = raw["stock_master"].copy()

    if px is None or px.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"error": "_compute_attribution(): prices dataframe is empty."}

    if "month_end" not in px.columns:
        if "price_date" in px.columns:
            px["month_end"] = px["price_date"]
        else:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {
                "error": "_compute_attribution(): prices dataframe missing month_end/price_date."
            }
    px_required = ["ace_code", "adj_price", "month_end"]
    if include_dividends:
        px_required.append("dividend_yield")
    px_missing = [c for c in px_required if c not in px.columns]
    if px_missing:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {
            "error": f"_compute_attribution(): prices dataframe missing columns: {px_missing}. "
                     f"Available columns: {list(px.columns)}"
        }

    missing = [c for c in ["month_end", "holding_weight"] if c not in h.columns]
    if missing:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {
            "error": f"_compute_attribution(): holdings dataframe missing columns: {missing}. "
                     f"Available columns: {list(h.columns)}"
        }

    # Every reported snapshot opens a holding interval. The end month closes
    # the final interval even when it has no snapshot of its own.
    h["month_end"] = pd.to_datetime(h["month_end"], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
    start_month, end_month = pd.Timestamp(start_me), pd.Timestamp(end_me)
    h = h[h["month_end"].between(start_month, end_month)].copy()
    reported_months = pd.DatetimeIndex(h["month_end"].unique()).sort_values()
    t0 = reported_months[reported_months < end_month].tolist()
    if not t0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {
            "error": "No reported holdings at or after the selected start month and before the end month."
        }
    months = [*t0, end_month]
    t1 = months[1:]
    period_months = np.array([
        (end.year - start.year) * 12 + end.month - start.month
        for start, end in zip(t0, t1)
    ])
    years = float(period_months.sum()) / 12.0
    h["holding_weight"] = pd.to_numeric(h["holding_weight"], errors="coerce").fillna(0.0)

    if "asset_type" not in h.columns:
        h["asset_type"] = "Unknown"

    if "instrument_key" not in h.columns:
        if "instrument_name" not in h.columns:
            h["instrument_name"] = ""
        if "ace_code" not in h.columns:
            h["ace_code"] = np.nan

        ace_str = h["ace_code"].astype(str)
        h["instrument_key"] = np.where(
            h["ace_code"].notna() & (ace_str.str.strip() != "") & (ace_str.str.lower() != "nan"),
            ace_str.str.strip(),
            "NOACE::" + h["instrument_name"].astype(str)
        )

    # The same report count applies to every position, so an absence adds zero.
    # Averaging before filtering to equities preserves the weights in the full fund.
    average_weights_pct = h.groupby("instrument_key")["holding_weight"].sum() * (100 / len(reported_months))
    # Balance reported weights with cash, including any negative residual.
    average_weights_pct["CASH::RESIDUAL"] = 100.0 - average_weights_pct.sum()

    domestic_only = str(universe_mode).strip() == "Domestic equities only"
    if domestic_only:
        h = h[h["asset_type"].astype(str).str.strip() == "Domestic Equities"].copy()

    # Aggregate weights by month/instrument
    reported_weights = (
        h.groupby(["month_end", "instrument_key", "asset_type"], as_index=False)["holding_weight"]
         .sum()
    )
    w = reported_weights.loc[reported_weights["month_end"] < end_month]

    # Pivot weights: month x instrument
    snapshot_weights = reported_weights.pivot_table(
        index="month_end",
        columns="instrument_key",
        values="holding_weight",
        aggfunc="sum"
    )

    # Explicit snapshots store zero weight for stocks absent from that snapshot.
    w_piv = snapshot_weights.reindex(t0).fillna(0.0)
    w_piv = w_piv.loc[:, w_piv.ne(0).any(axis=0)]
    if w_piv.empty and domestic_only:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {
            "error": "No domestic-equity positions were held during the selected period."
        }

    cash_keys = set(
        w.loc[w["asset_type"].astype(str).str.strip().str.lower() == "cash", "instrument_key"]
         .unique()
         .tolist()
    )

    if domestic_only:
        # Rebase the domestic sleeve to 100% at each rebalance.
        dom_tot = w_piv.sum(axis=1).replace(0.0, np.nan)
        w_piv = w_piv.div(dom_tot, axis=0).fillna(0.0)
    else:
        # Add residual cash (if holdings do not sum to 1)
        tot = w_piv.sum(axis=1)
        residual = (1.0 - tot).clip(lower=0.0)
        if residual.max() > 1e-8:
            w_piv["CASH::RESIDUAL"] = residual
            cash_keys.add("CASH::RESIDUAL")

    instr_cols = w_piv.columns.tolist()
    w0_df = w_piv

    # Adjusted-price return diagnostics
    ace_codes = [c for c in instr_cols if not c.startswith("NOACE::") and not c.startswith("CASH::")]
    invalid_price_rows_df = pd.DataFrame()
    price_return_audit_df = pd.DataFrame()

    # Benchmark nav pivot
    bn_piv = pd.DataFrame(index=months)
    if not bn.empty:
        bn["month_end"] = pd.to_datetime(bn["month_end"]).dt.to_period("M").dt.to_timestamp("M")
        bnp = bn.pivot_table(index="month_end", columns="bench_name", values="nav_value", aggfunc="last")
        bn_piv = bnp.reindex(months)

    def _bench_period_returns(bench_name: str) -> np.ndarray:
        if bench_name not in bn_piv.columns:
            return np.full(len(t1), np.nan)
        v0 = bn_piv.loc[t0, bench_name].to_numpy(dtype=float)
        v1 = bn_piv.loc[t1, bench_name].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = (v1 / v0) - 1.0
        return np.where(np.isfinite(r) & (v0 > 0) & (v1 > 0), r, np.nan)

    r_n50 = _bench_period_returns(BENCH_NAME_NIFTY50)
    r_n500 = _bench_period_returns(BENCH_NAME_NIFTY500)
    r_n100 = _bench_period_returns(BENCH_NAME_NIFTY100)
    r_mid150 = _bench_period_returns(BENCH_NAME_MID150)
    r_small250 = _bench_period_returns(BENCH_NAME_SMALL250)

    # Use the most recent size band available at each holding interval's start.
    size_lookup = {}
    if not sz.empty and ace_codes:
        sz2 = sz[sz["ace_code"].astype(str).isin(ace_codes)].copy()
        sz2["month_end"] = pd.to_datetime(sz2["month_end"]).dt.to_period("M").dt.to_timestamp("M")
        if not sz2.empty:
            bands = sz2.pivot_table(index="month_end", columns="ace_code", values="size_band", aggfunc="last")
            bands = bands.sort_index().ffill().reindex(t0, method="ffill")
            size_lookup = bands.stack().to_dict()

    asset_map = (
        w.sort_values(["instrument_key", "month_end"])
         .groupby("instrument_key")["asset_type"]
         .last()
         .to_dict()
    )
    equity_types = {"Domestic Equities", "Overseas Equities", "ADRs & GDRs", "Others Equities", "Other Equities"}
    equity_keys = tuple(
        c for c in instr_cols if c not in cash_keys and (
            str(asset_map.get(c, "")).strip() in equity_types
            or (asset_map.get(c) == "Unknown" and c in ace_codes)
        )
    )
    selected = set(benchmark_stock_keys)
    if selected.difference(equity_keys):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"error": "Select stocks from the current attribution table."}

    # ----------------------------
    # Instrument returns
    # - price return from adj_price
    # - dividend-inclusive attribution uses price plus interval dividends
    # ----------------------------
    r_price_instr = pd.DataFrame(index=t1, columns=instr_cols, dtype=float)
    r_div_instr = pd.DataFrame(index=t1, columns=instr_cols, dtype=float)

    if ace_codes:
        r_ace_df, invalid_price_rows_df, price_return_audit_df = compute_adj_price_returns(
            prices_df=px,
            months=months,
            ace_codes=ace_codes,
        )
        if include_dividends:
            div_ace_df = compute_dividend_yield_returns(px, months, ace_codes)
            r_div_instr.loc[:, ace_codes] = div_ace_df
        for c in ace_codes:
            if c in r_ace_df.columns:
                r_price_instr[c] = pd.to_numeric(r_ace_df[c], errors="coerce")
        if not invalid_price_rows_df.empty:
            held_pairs = {
                (month.date(), code) for month in t0 for code in ace_codes
                if w0_df.at[month, code] != 0 and code not in cash_keys
            }
            invalid_price_rows_df = invalid_price_rows_df.loc[
                invalid_price_rows_df.set_index(["month_start", "ace_code"]).index.isin(held_pairs)
            ].copy()

    for c in instr_cols:
        if c.startswith("NOACE::"):
            r_price_instr[c] = 0.0
            r_div_instr[c] = 0.0

    cash_r = (1.0 + _monthly_cash_return()) ** period_months - 1.0
    for c in cash_keys:
        if c in r_price_instr.columns:
            r_price_instr[c] = cash_r
            r_div_instr[c] = 0.0

    r_price_instr = r_price_instr.fillna(0.0)
    r_div_instr = r_div_instr.fillna(0.0)

    # ----------------------------
    # Benchmark returns per instrument. Keep missing endpoints until after validating substitutions.
    # ----------------------------
    r_bm = pd.DataFrame(index=t1, columns=instr_cols, dtype=float)

    if bench_mode == "Vs NIFTY 50":
        broad = r_n50
        for c in instr_cols:
            r_bm[c] = broad
    elif bench_mode == "Vs NIFTY 500":
        broad = r_n500
        for c in instr_cols:
            r_bm[c] = broad
    else:
        for c in instr_cols:
            if c in cash_keys or c.startswith("CASH::"):
                r_bm[c] = r_price_instr[c]
            elif c.startswith("NOACE::"):
                r_bm[c] = r_n50
            else:
                vals = np.zeros(len(t1), dtype=float)
                for i in range(len(t1)):
                    band = str(size_lookup.get((t0[i], c), "")).strip()
                    bench_name = LIKE_FOR_LIKE_BENCH.get(band, BENCH_NAME_NIFTY50)
                    if bench_name == BENCH_NAME_NIFTY100:
                        vals[i] = r_n100[i]
                    elif bench_name == BENCH_NAME_MID150:
                        vals[i] = r_mid150[i]
                    elif bench_name == BENCH_NAME_SMALL250:
                        vals[i] = r_small250[i]
                    else:
                        vals[i] = r_n50[i]
                r_bm[c] = vals

        for c in instr_cols:
            if c in cash_keys or c.startswith("CASH::") or c.startswith("NOACE::"):
                continue
            at = str(asset_map.get(c, "")).strip()
            if at in {"Overseas Equities", "ADRs & GDRs", "Others Equities", "Other Equities"}:
                r_bm[c] = r_n50

    # Use stable instrument keys rather than names or row positions after sorting.
    substitute = [key for key in equity_keys if key in selected]
    for key in substitute:
        missing_held = r_bm[key].isna().to_numpy() & w0_df[key].ne(0).to_numpy()
        if missing_held.any():
            name_rows = sm.loc[sm["ace_code"].astype(str).eq(key), "company_name"] if not sm.empty else pd.Series(dtype=str)
            name = name_rows.iloc[0] if not name_rows.empty else key.removeprefix("NOACE::")
            period = t1[np.flatnonzero(missing_held)[0]]
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {
                "error": f"Benchmark NAV is missing for {name} in the interval ending {period:%b %Y}."
            }
    r_bm = r_bm.fillna(0.0)
    if substitute:
        r_price_instr[substitute] = r_bm[substitute]
        r_div_instr[substitute] = 0.0
        if not invalid_price_rows_df.empty:
            invalid_price_rows_df = invalid_price_rows_df.loc[~invalid_price_rows_df["ace_code"].isin(substitute)]

    r_instr = r_price_instr + r_div_instr

    # ----------------------------
    # Brinson-style attribution (unchanged)
    # ----------------------------
    w0 = w0_df.to_numpy(dtype=float)
    r0 = r_instr.to_numpy(dtype=float)
    rb = r_bm.to_numpy(dtype=float)

    rp = np.sum(w0 * r0, axis=1)
    rbp = np.sum(w0 * rb, axis=1)

    base = 100.0
    link_fund = np.ones(len(t1), dtype=float)
    link_bm = np.ones(len(t1), dtype=float)
    for i in range(1, len(t1)):
        link_fund[i] = link_fund[i-1] * (1.0 + rp[i-1])
        link_bm[i] = link_bm[i-1] * (1.0 + rbp[i-1])

    contrib_fund = (w0 * r0) * link_fund[:, None] * base
    contrib_bm = (w0 * rb) * link_bm[:, None] * base

    total_fund = contrib_fund.sum(axis=0)
    total_bench = contrib_bm.sum(axis=0)
    total_alpha = total_fund - total_bench

    gross_fund = float(np.prod(1.0 + rp))
    gross_price = float(np.prod(1.0 + np.sum(w0 * r_price_instr.to_numpy(dtype=float), axis=1)))
    gross_dividend = gross_fund / gross_price if gross_price > 0 else np.nan
    portfolio_label = "Domestic equity sleeve" if domestic_only else "Fund"
    return_label = "return (price and dividends)" if include_dividends else "price return"
    portfolio_return_summary_df = pd.DataFrame([
        {
            "Series": label,
            "Total return (%)": round((gross - 1.0) * 100.0, 2),
            "CAGR (%)": round((gross ** (1.0 / years) - 1.0) * 100.0, 2) if gross > 0 else np.nan,
        }
        for label, gross in [
            (f"{portfolio_label} {return_label}", gross_fund),
            ("Benchmark", float(np.prod(1.0 + rbp))),
            ("Price return", gross_price),
            ("Dividend yield return", gross_dividend),
        ]
    ])

    # Name map
    name_map = {}
    if not sm.empty:
        name_map = dict(zip(sm["ace_code"].astype(str), sm["company_name"].astype(str)))

    def _disp(instr: str) -> str:
        if instr.startswith("NOACE::"):
            return instr.replace("NOACE::", "")
        if instr.startswith("CASH::"):
            return "Cash (Residual)"
        if instr in cash_keys:
            return "Cash"
        return name_map.get(instr, instr)

    held_mask = w0_df.ne(0).to_numpy()
    held_months = held_mask.T @ period_months
    holding_years = held_months / 12.0

    def _holding_returns(ret_mat: np.ndarray) -> np.ndarray:
        out = np.full(ret_mat.shape[1], np.nan, dtype=float)
        for j, years_held in enumerate(holding_years):
            if years_held <= 0:
                continue
            gross = float(np.prod(1.0 + ret_mat[held_mask[:, j], j]))
            if not np.isfinite(gross):
                continue
            if held_months[j] < 12:
                out[j] = gross - 1.0
            elif gross >= 0:
                out[j] = gross ** (1.0 / years_held) - 1.0
        return out

    stock_return = _holding_returns(r0)
    bench_return = _holding_returns(rb)
    outperf = stock_return - bench_return

    stock_df = pd.DataFrame({
        "Stock name": [_disp(c) for c in instr_cols],
        "Stock contribution (Rs.)": np.round(total_fund, 2),
        "Benchmark contribution (Rs.)": np.round(total_bench, 2),
        "Alpha contribution (Rs.)": np.round(total_alpha, 2),
        "Holding period (years)": np.round(holding_years, 2),
        "Stock return (%)": np.round(stock_return * 100.0, 2),
        "Benchmark return (%)": np.round(bench_return * 100.0, 2),
        "Return basis": np.where(held_months < 12, "Cumulative", "Annualized"),
        "Stock outperformance (pp)": np.round(outperf * 100.0, 2),
    }, index=pd.Index(instr_cols, name="instrument_key"))

    equity_rows = stock_df.index.isin(equity_keys)
    winners = equity_rows & (total_alpha > 0)
    losers = equity_rows & (total_alpha <= 0)

    def _mean_or_nan(values: np.ndarray, selected: np.ndarray) -> float:
        return round(float(values[selected].mean()), 2) if selected.any() else np.nan

    hit_df = pd.DataFrame([{
        "Hit-rate (winners/total)": f"{int(winners.sum())} / {int(equity_rows.sum())}",
        "Avg holding period winners (yrs)": _mean_or_nan(holding_years, winners),
        "Avg holding period losers (yrs)": _mean_or_nan(holding_years, losers),
        "Avg alpha contribution on winners (Rs.)": _mean_or_nan(total_alpha, winners),
        "Avg alpha contribution on losers (Rs.)": _mean_or_nan(total_alpha, losers),
    }])
    stock_df = stock_df.sort_values("Alpha contribution (Rs.)", ascending=False)

    # Category-level attribution (unchanged)
    categories = ["Large", "Mid", "Small", "Cash"]
    cat_fund = {c: 0.0 for c in categories}
    cat_bench = {c: 0.0 for c in categories}

    contrib_f_df = pd.DataFrame(contrib_fund, index=t1, columns=instr_cols)
    contrib_b_df = pd.DataFrame(contrib_bm, index=t1, columns=instr_cols)

    for i in range(len(t0)):
        start_m = t0[i]
        end_m = t1[i]
        for instr in instr_cols:
            if instr in cash_keys or instr.startswith("CASH::") or instr.startswith("NOACE::"):
                cat = "Cash"
            else:
                band = str(size_lookup.get((start_m, instr), "")).strip()
                cat = band if band in {"Large", "Mid", "Small"} else "Large"
            cat_fund[cat] += float(contrib_f_df.loc[end_m, instr])
            cat_bench[cat] += float(contrib_b_df.loc[end_m, instr])

    r_instr_df = pd.DataFrame(r0, index=t1, columns=instr_cols)
    r_bm_df = pd.DataFrame(rb, index=t1, columns=instr_cols)
    w_start_df = w0_df.copy()

    cat_cagr = {}
    cat_bm_cagr = {}

    for cat in categories:
        gross_cat = 1.0
        gross_bm_cat = 1.0
        m_count = 0
        for i in range(len(t0)):
            start_m = t0[i]
            end_m = t1[i]

            members = []
            for instr in instr_cols:
                if cat == "Cash":
                    if instr in cash_keys or instr.startswith("CASH::") or instr.startswith("NOACE::"):
                        members.append(instr)
                else:
                    if instr in cash_keys or instr.startswith("CASH::") or instr.startswith("NOACE::"):
                        continue
                    band = str(size_lookup.get((start_m, instr), "")).strip()
                    band = band if band in {"Large", "Mid", "Small"} else "Large"
                    if band == cat:
                        members.append(instr)

            if not members:
                continue
            denom = float(w_start_df.loc[start_m, members].sum())
            if denom <= 0:
                continue

            r_cat = float((w_start_df.loc[start_m, members] * r_instr_df.loc[end_m, members]).sum() / denom)
            r_cat_bm = float((w_start_df.loc[start_m, members] * r_bm_df.loc[end_m, members]).sum() / denom)

            gross_cat *= (1.0 + r_cat)
            gross_bm_cat *= (1.0 + r_cat_bm)
            m_count += period_months[i]

        if m_count > 0:
            yrs = m_count / 12.0
            cat_cagr[cat] = gross_cat ** (1.0 / yrs) - 1.0
            cat_bm_cagr[cat] = gross_bm_cat ** (1.0 / yrs) - 1.0
        else:
            cat_cagr[cat] = np.nan
            cat_bm_cagr[cat] = np.nan

    cat_df = pd.DataFrame([{
        "Category": c,
        "Category contribution (Rs.)": round(cat_fund[c], 2),
        "Benchmark contribution (Rs.)": round(cat_bench[c], 2),
        "Category alpha contribution (Rs.)": round(cat_fund[c] - cat_bench[c], 2),
        "Category CAGR": round(cat_cagr[c] * 100.0, 2) if pd.notna(cat_cagr[c]) else np.nan,
        "Benchmark CAGR": round(cat_bm_cagr[c] * 100.0, 2) if pd.notna(cat_bm_cagr[c]) else np.nan,
        "Category outperformance (pp)": round((cat_cagr[c] - cat_bm_cagr[c]) * 100.0, 2) if pd.notna(cat_cagr[c]) else np.nan,
    } for c in categories])

    # Fundamental endpoints come from reports that actually list the stock.
    # Price attribution still runs each opening snapshot through the next rebalance.
    reported_positions = snapshot_weights.reindex(index=reported_months, columns=equity_keys).fillna(0.0).ne(0)
    report_gaps = np.diff(reported_months.to_period("M").asi8)
    fundamental_windows = []
    for instrument in equity_keys:
        listed = reported_positions[instrument].to_numpy()
        months_held = int(report_gaps[listed[:-1] & listed[1:]].sum())
        if months_held <= 12:
            continue
        held_reports = reported_months[listed]
        fundamental_windows.append({
            "instrument_key": instrument,
            "stock_name": _disp(instrument),
            "ace_code": instrument if instrument in ace_codes else None,
            "holding_months": months_held,
            "average_weight_pct": float(average_weights_pct[instrument]),
            "first_month": held_reports[0].date(),
            "last_month": held_reports[-1].date(),
        })

    # Include stocks first reported at the end of the review, even though they
    # did not have a price attribution interval inside the window.
    review_assets = reported_weights["asset_type"].astype(str).str.strip()
    review_keys = reported_weights["instrument_key"]
    identified = ~review_keys.str.startswith(("NOACE::", "CASH::"))
    review_equities = review_assets.isin(equity_types) | (review_assets.eq("Unknown") & identified)
    review_stocks = tuple(
        (_disp(key), key if not key.startswith("NOACE::") else None)
        for key in review_keys[review_equities & reported_weights["holding_weight"].ne(0)].unique()
        if key not in cash_keys and not key.startswith("CASH::")
    )

    diag = {
        "months": months,
        "period_months": period_months,
        "requested_start_month": start_month,
        "effective_start_month": t0[0],
        "portfolio_return_summary_df": portfolio_return_summary_df,
        "equity_keys": equity_keys,
        "average_portfolio_weights_pct": average_weights_pct,
        "fundamental_windows": fundamental_windows,
        "fundamental_review_stocks": review_stocks,
        "missing_benchmarks": [x for x in [BENCH_NAME_NIFTY50, BENCH_NAME_NIFTY500, BENCH_NAME_NIFTY100, BENCH_NAME_MID150, BENCH_NAME_SMALL250] if x not in bn_piv.columns],
        "adj_price_invalid_rows_df": invalid_price_rows_df,
        "adj_price_return_audit_df": price_return_audit_df,
    }
    return stock_df, hit_df, cat_df, diag
