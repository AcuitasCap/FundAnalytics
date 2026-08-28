import datetime as dt
import inspect

import numpy as np
import pandas as pd

from .returns import (
    compute_adj_price_returns,
    compute_dividend_yield_returns,
    compute_monthly_portfolio_multiples,
)
from core.dates import month_ends_between, to_month_end
from features.fund_attribution.data import (
    BENCH_NAME_MID150, BENCH_NAME_NIFTY100, BENCH_NAME_NIFTY50,
    BENCH_NAME_NIFTY500, BENCH_NAME_SMALL250, LIKE_FOR_LIKE_BENCH,
    _monthly_cash_return,
)

def _compute_domestic_triangulation_tables(
    w0_dom_df: pd.DataFrame,
    domestic_isins: list[str],
    px_df: pd.DataFrame,
    yld_df: pd.DataFrame,
    lens: str,
    name_map: dict[str, str],
    mode_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    eps = 1e-12
    mode_a = str(mode_label).strip() == "Fallback to price growth"
    mult_col = "ps" if str(lens).startswith("Sales") else ("pe" if str(lens).startswith("Earnings") else "pb")

    out_cols = ["Stock name", "Average holding weight (%)", "Fundamental growth in holding period (%)", "Notes"]
    if w0_dom_df is None or w0_dom_df.empty or not domestic_isins:
        summary = pd.DataFrame(
            [
                {"Metric": "Weighted average fundamental growth (%)", "Value (%)": np.nan},
                {"Metric": "Median fundamental growth (%)", "Value (%)": np.nan},
            ]
        )
        return pd.DataFrame(columns=out_cols), summary, {
            "valid_weight_sum": 0.0,
            "invalid_weight_sum": 0.0,
        }

    w_dom = w0_dom_df.reindex(columns=domestic_isins).fillna(0.0).copy()
    active_isins = [c for c in domestic_isins if bool((w_dom[c] > eps).any())]
    if not active_isins:
        summary = pd.DataFrame(
            [
                {"Metric": "Weighted average fundamental growth (%)", "Value (%)": np.nan},
                {"Metric": "Median fundamental growth (%)", "Value (%)": np.nan},
            ]
        )
        return pd.DataFrame(columns=out_cols), summary, {
            "valid_weight_sum": 0.0,
            "invalid_weight_sum": 0.0,
        }

    w_dom = w_dom.loc[:, active_isins]
    avg_w = w_dom.mean(axis=0).astype(float)

    px = px_df.copy()
    if "month_end" not in px.columns and "price_date" in px.columns:
        px["month_end"] = px["price_date"]
    px["isin"] = px["isin"].astype(str)
    px["month_end"] = pd.to_datetime(px["month_end"], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
    px["adj_price"] = pd.to_numeric(px["adj_price"], errors="coerce")
    price_piv = (
        px[px["isin"].isin(active_isins)]
        .pivot_table(index="month_end", columns="isin", values="adj_price", aggfunc="last")
        .reindex(index=w_dom.index, columns=active_isins)
    )

    mul = yld_df.copy()
    if "month_end" not in mul.columns and "price_date" in mul.columns:
        mul["month_end"] = mul["price_date"]
    mul["isin"] = mul["isin"].astype(str)
    mul["month_end"] = pd.to_datetime(mul["month_end"], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
    mul[mult_col] = pd.to_numeric(mul.get(mult_col), errors="coerce")
    mult_piv = (
        mul[mul["isin"].isin(active_isins)]
        .pivot_table(index="month_end", columns="isin", values=mult_col, aggfunc="last")
        .reindex(index=w_dom.index, columns=active_isins)
    )

    def _period_growth(v0: float, v1: float, n_hold_months: int) -> float:
        if not (np.isfinite(v0) and np.isfinite(v1) and v0 > 0 and v1 > 0):
            return np.nan
        if n_hold_months < 12:
            return (v1 / v0) - 1.0
        years = n_hold_months / 12.0
        return (v1 / v0) ** (1.0 / years) - 1.0 if years > 0 else np.nan

    rows = []
    for isin in active_isins:
        w_ser = w_dom[isin].fillna(0.0)
        held = w_ser > eps
        hold_months = list(w_ser.index[held])
        n_hold_months = int(held.sum())
        avg_weight = float(avg_w.get(isin, 0.0))

        if n_hold_months <= 0:
            continue

        hold_start = hold_months[0]
        hold_end = hold_months[-1]

        p_start = float(price_piv.loc[hold_start, isin]) if (hold_start in price_piv.index and isin in price_piv.columns) else np.nan
        p_end = float(price_piv.loc[hold_end, isin]) if (hold_end in price_piv.index and isin in price_piv.columns) else np.nan
        m_start = float(mult_piv.loc[hold_start, isin]) if (hold_start in mult_piv.index and isin in mult_piv.columns) else np.nan
        m_end = float(mult_piv.loc[hold_end, isin]) if (hold_end in mult_piv.index and isin in mult_piv.columns) else np.nan

        # P/M = fundamental, since stock_monthly_valuations now stores M
        # directly (P/S, P/E, or P/B), rather than its inverse yield.
        f_start = p_start / m_start if np.isfinite(p_start) and np.isfinite(m_start) and m_start > 0 else np.nan
        f_end = p_end / m_end if np.isfinite(p_end) and np.isfinite(m_end) and m_end > 0 else np.nan

        if n_hold_months == 1:
            growth = 0.0
            note = "Held only 1 month"
            fundamental_valid = True
        else:
            missing_ep = not (
                np.isfinite(p_start)
                and np.isfinite(p_end)
                and np.isfinite(m_start)
                and np.isfinite(m_end)
                and m_start > 0
                and m_end > 0
                and np.isfinite(f_start)
                and np.isfinite(f_end)
            )
            sign_or_zero = (not missing_ep) and ((f_start == 0.0) or (f_end == 0.0) or ((f_start * f_end) <= 0.0))
            fundamental_valid = (not missing_ep) and (not sign_or_zero)
            if fundamental_valid:
                growth = _period_growth(float(f_start), float(f_end), n_hold_months)
                note = ""
            else:
                reasons = []
                if missing_ep:
                    reasons.append("Missing endpoint data")
                if sign_or_zero:
                    reasons.append("Fundamental sign/zero issue")
                reason_txt = ", ".join(reasons) if reasons else "Invalid fundamentals"
                price_growth = _period_growth(float(p_start), float(p_end), n_hold_months)
                if mode_a and np.isfinite(price_growth):
                    growth = float(price_growth)
                    note = f"{reason_txt}, assume price growth"
                elif mode_a:
                    growth = np.nan
                    note = f"{reason_txt}, price fallback unavailable"
                else:
                    growth = np.nan
                    note = f"{reason_txt}, excluded from weighted average"

        rows.append(
            {
                "Stock name": name_map.get(isin, isin),
                "Average holding weight (%)": avg_weight * 100.0,
                "Fundamental growth in holding period (%)": growth * 100.0 if np.isfinite(growth) else np.nan,
                "Notes": note,
                "__avg_w": avg_weight,
                "__fund_valid": bool(fundamental_valid),
            }
        )

    dom_triangulation_df = pd.DataFrame(rows)
    if dom_triangulation_df.empty:
        summary = pd.DataFrame(
            [
                {"Metric": "Weighted average fundamental growth (%)", "Value (%)": np.nan},
                {"Metric": "Median fundamental growth (%)", "Value (%)": np.nan},
            ]
        )
        return pd.DataFrame(columns=out_cols), summary, {
            "valid_weight_sum": 0.0,
            "invalid_weight_sum": 0.0,
        }

    dom_triangulation_df = dom_triangulation_df.sort_values("Average holding weight (%)", ascending=False).reset_index(drop=True)
    w_vec = dom_triangulation_df["__avg_w"].to_numpy(dtype=float)
    g_vec = dom_triangulation_df["Fundamental growth in holding period (%)"].to_numpy(dtype=float) / 100.0
    fund_valid_vec = dom_triangulation_df["__fund_valid"].astype(bool).to_numpy(dtype=bool)

    valid_weight_sum = float(np.sum(w_vec[fund_valid_vec])) if len(w_vec) else 0.0
    invalid_weight_sum = float(np.sum(w_vec[~fund_valid_vec])) if len(w_vec) else 0.0

    m_wavg = np.isfinite(g_vec)
    w_sum_wavg = float(np.sum(w_vec[m_wavg])) if np.any(m_wavg) else 0.0
    if w_sum_wavg > 0:
        w_norm = w_vec[m_wavg] / w_sum_wavg
        weighted_avg = float(np.sum(w_norm * g_vec[m_wavg]))
    else:
        weighted_avg = np.nan

    m_med = fund_valid_vec & np.isfinite(g_vec)
    median_val = float(np.median(g_vec[m_med])) if np.any(m_med) else np.nan

    dom_triangulation_summary_df = pd.DataFrame(
        [
            {"Metric": "Weighted average fundamental growth (%)", "Value (%)": round(weighted_avg * 100.0, 4) if np.isfinite(weighted_avg) else np.nan},
            {"Metric": "Median fundamental growth (%)", "Value (%)": round(median_val * 100.0, 4) if np.isfinite(median_val) else np.nan},
        ]
    )

    dom_triangulation_df = dom_triangulation_df[out_cols].copy()
    dom_triangulation_df["Average holding weight (%)"] = dom_triangulation_df["Average holding weight (%)"].round(4)
    dom_triangulation_df["Fundamental growth in holding period (%)"] = dom_triangulation_df[
        "Fundamental growth in holding period (%)"
    ].round(4)

    return dom_triangulation_df, dom_triangulation_summary_df, {
        "valid_weight_sum": valid_weight_sum,
        "invalid_weight_sum": invalid_weight_sum,
    }


def _compute_attribution(
    raw: dict,
    start_date: dt.date,
    end_date: dt.date,
    bench_mode: str,
    lens: str = "Earnings (P/E)",
    universe_mode: str = "Full holdings",
):
    """Return (stock_df, hit_df, category_df, diag). Contributions are Rs per 100 base."""
    start_me = to_month_end(start_date)
    end_me = to_month_end(end_date)

    months = month_ends_between(start_me, end_me)
    if len(months) < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"error": "Select at least two month-ends."}

    h = raw["holdings"].copy()
    px = raw["prices"].copy()
    yld = raw.get("yields", raw.get("multiples", pd.DataFrame())).copy()
    sz = raw["size_band"].copy()
    bn = raw["bench_nav"].copy()
    sm = raw["stock_master"].copy()

    # Required input checks for domestic sleeve monthly decomposition.
    if px is None or px.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"error": "_compute_attribution(): prices dataframe is empty."}
    if yld is None or yld.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"error": "_compute_attribution(): multiples dataframe is empty."}

    if "month_end" not in px.columns:
        if "price_date" in px.columns:
            px["month_end"] = px["price_date"]
        else:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {
                "error": "_compute_attribution(): prices dataframe missing month_end/price_date."
            }
    px_missing = [c for c in ["isin", "adj_price", "dividend_yield", "month_end"] if c not in px.columns]
    if px_missing:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {
            "error": f"_compute_attribution(): prices dataframe missing columns: {px_missing}. "
                     f"Available columns: {list(px.columns)}"
        }

    if "month_end" not in yld.columns:
        if "price_date" in yld.columns:
            yld["month_end"] = yld["price_date"]
        else:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {
                "error": "_compute_attribution(): multiples dataframe missing month_end."
            }
    yld_missing = [c for c in ["isin", "month_end", "pe", "pb", "ps"] if c not in yld.columns]
    if yld_missing:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {
            "error": f"_compute_attribution(): multiples dataframe missing columns: {yld_missing}. "
                     f"Available columns: {list(yld.columns)}"
        }

    # Filter holdings to selected window (we will ffill)
    h = h[(h["month_end"] >= months[0]) & (h["month_end"] <= months[-1])].copy()
    h["holding_weight"] = pd.to_numeric(h["holding_weight"], errors="coerce").fillna(0.0)

    missing = [c for c in ["month_end", "holding_weight"] if c not in h.columns]
    if missing:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {
            "error": f"_compute_attribution(): holdings dataframe missing columns: {missing}. "
                     f"Available columns: {list(h.columns)}"
        }

    if "asset_type" not in h.columns:
        h["asset_type"] = "Unknown"

    if "instrument_key" not in h.columns:
        if "instrument_name" not in h.columns:
            h["instrument_name"] = ""
        if "isin" not in h.columns:
            h["isin"] = np.nan

        isin_str = h["isin"].astype(str)
        h["instrument_key"] = np.where(
            h["isin"].notna() & (isin_str.str.strip() != "") & (isin_str.str.lower() != "nan"),
            isin_str.str.strip(),
            "NOISIN::" + h["instrument_name"].astype(str)
        )

    # Canonicalise to calendar month-end label (your current convention)
    h["month_end"] = pd.to_datetime(h["month_end"]).dt.to_period("M").dt.to_timestamp("M")
    months = [pd.Timestamp(m).to_period("M").to_timestamp("M") for m in months]
    months = [pd.Timestamp(m) for m in months]

    # Attribution universe selection:
    # - Full holdings: current behavior (all asset types).
    # - Domestic equities only: use only Domestic Equities and rebase sleeve to 100% each month.
    if str(universe_mode).strip() == "Domestic equities only":
        h = h[h["asset_type"].astype(str).str.strip() == "Domestic Equities"].copy()

    # Aggregate weights by month/instrument
    w = (
        h.groupby(["month_end", "instrument_key", "asset_type"], as_index=False)["holding_weight"]
         .sum()
    )

    # Pivot weights: month x instrument
    w_piv = w.pivot_table(
        index="month_end",
        columns="instrument_key",
        values="holding_weight",
        aggfunc="sum"
    )

    # Use raw weights only (no forward-fill)
    w_piv = w_piv.reindex(months).fillna(0.0)

    cash_keys = set(
        w.loc[w["asset_type"].astype(str).str.strip().str.lower() == "cash", "instrument_key"]
         .unique()
         .tolist()
    )

    if str(universe_mode).strip() == "Domestic equities only":
        # Rebase domestic sleeve to 100% each month.
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
    t0 = months[:-1]
    t1 = months[1:]
    w0_df = w_piv.loc[t0, :].copy()

    # Adjusted-price return diagnostics
    isins = [c for c in instr_cols if not c.startswith("NOISIN::") and not c.startswith("CASH::")]
    invalid_price_rows_df = pd.DataFrame()
    price_return_audit_df = pd.DataFrame()

    # Benchmark nav pivot
    bn_piv = pd.DataFrame(index=months)
    if not bn.empty:
        bnp = bn.pivot_table(index="month_end", columns="bench_name", values="nav_value", aggfunc="last")
        bnp.index = pd.to_datetime(bnp.index).to_period("M").to_timestamp("M")
        bn_piv = bnp.reindex(months)

    def _bench_period_returns(bench_name: str) -> np.ndarray:
        if bench_name not in bn_piv.columns:
            return np.zeros(len(t1), dtype=float)
        v0 = bn_piv.loc[t0, bench_name].to_numpy(dtype=float)
        v1 = bn_piv.loc[t1, bench_name].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = (v1 / v0) - 1.0
        r = np.where(np.isfinite(r), r, 0.0)
        return r

    r_n50 = _bench_period_returns(BENCH_NAME_NIFTY50)
    r_n500 = _bench_period_returns(BENCH_NAME_NIFTY500)
    r_n100 = _bench_period_returns(BENCH_NAME_NIFTY100)
    r_mid150 = _bench_period_returns(BENCH_NAME_MID150)
    r_small250 = _bench_period_returns(BENCH_NAME_SMALL250)

    # Size band lookup at t0 for ISINs
    size_lookup = {}
    if not sz.empty and isins:
        sz2 = sz[sz["isin"].astype(str).isin(isins)].copy()
        sz2["month_end"] = pd.to_datetime(sz2["month_end"]).dt.to_period("M").dt.to_timestamp("M")
        sz2 = sz2[sz2["month_end"].isin(t0)]
        if not sz2.empty:
            size_lookup = sz2.set_index(["month_end", "isin"])["size_band"].to_dict()

    asset_map = (
        w.sort_values(["instrument_key", "month_end"])
         .groupby("instrument_key")["asset_type"]
         .last()
         .to_dict()
    )

    # ----------------------------
    # Instrument returns
    # - price return from adj_price
    # - dividend return uses month t yield for month t+1 return
    # - stock attribution uses total return = price + dividend
    # ----------------------------
    r_price_instr = pd.DataFrame(index=t1, columns=instr_cols, dtype=float)
    r_div_instr = pd.DataFrame(index=t1, columns=instr_cols, dtype=float)
    r_instr = pd.DataFrame(index=t1, columns=instr_cols, dtype=float)

    if isins:
        r_isin_df, invalid_price_rows_df, price_return_audit_df = compute_adj_price_returns(
            prices_df=px,
            months=months,
            isins=isins,
        )
        div_isin_df = compute_dividend_yield_returns(px, months, isins)
        for c in isins:
            if c in r_isin_df.columns:
                r_price_instr[c] = pd.to_numeric(r_isin_df[c], errors="coerce")
            if c in div_isin_df.columns:
                r_div_instr[c] = pd.to_numeric(div_isin_df[c], errors="coerce")

    for c in instr_cols:
        if c.startswith("NOISIN::"):
            r_price_instr[c] = 0.0
            r_div_instr[c] = 0.0

    cash_r = _monthly_cash_return()
    for c in cash_keys:
        if c in r_price_instr.columns:
            r_price_instr[c] = cash_r
            r_div_instr[c] = 0.0

    r_price_instr = r_price_instr.fillna(0.0)
    r_div_instr = r_div_instr.fillna(0.0)
    r_instr = (r_price_instr + r_div_instr).fillna(0.0)

    # ----------------------------
    # Benchmark returns per instrument (unchanged)
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
                r_bm[c] = r_instr[c].to_numpy(dtype=float)
            elif c.startswith("NOISIN::"):
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
            if c in cash_keys or c.startswith("CASH::") or c.startswith("NOISIN::"):
                continue
            at = str(asset_map.get(c, "")).strip()
            if at in {"Overseas Equities", "ADRs & GDRs", "Others Equities", "Other Equities"}:
                r_bm[c] = r_n50

    r_bm = r_bm.fillna(0.0)

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

    # Name map
    name_map = {}
    if not sm.empty:
        name_map = dict(zip(sm["isin"].astype(str), sm["company_name"].astype(str)))

    def _disp(instr: str) -> str:
        if instr.startswith("NOISIN::"):
            return instr.replace("NOISIN::", "")
        if instr.startswith("CASH::"):
            return "Cash (Residual)"
        if instr in cash_keys:
            return "Cash"
        return name_map.get(instr, instr)

    held_mask = (w0_df > 0).to_numpy()
    held_months = held_mask.sum(axis=0).astype(int)
    holding_years = held_months / 12.0

    def _masked_cagr(ret_mat: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = np.full(ret_mat.shape[1], np.nan, dtype=float)
        for j in range(ret_mat.shape[1]):
            m = mask[:, j]
            n = int(m.sum())
            if n <= 0:
                continue
            gross = float(np.prod(1.0 + ret_mat[m, j]))
            yrs = n / 12.0
            if yrs > 0 and gross > 0:
                out[j] = gross ** (1.0 / yrs) - 1.0
        return out

    stock_cagr = _masked_cagr(r0, held_mask)
    bench_cagr = _masked_cagr(rb, held_mask)
    outperf = stock_cagr - bench_cagr

    stock_df = pd.DataFrame({
        "Stock name": [_disp(c) for c in instr_cols],
        "Stock contribution (Rs.)": np.round(total_fund, 2),
        "Benchmark contribution (Rs.)": np.round(total_bench, 2),
        "Alpha contribution (Rs.)": np.round(total_alpha, 2),
        "Holding period (years)": np.round(holding_years, 2),
        "Stock CAGR": np.round(stock_cagr * 100.0, 2),
        "Benchmark CAGR": np.round(bench_cagr * 100.0, 2),
        "Stock outperformance (pp)": np.round(outperf * 100.0, 2),
    }).sort_values("Alpha contribution (Rs.)", ascending=False).reset_index(drop=True)

    winners = stock_df.loc[stock_df["Alpha contribution (Rs.)"] > 0]
    losers = stock_df.loc[stock_df["Alpha contribution (Rs.)"] <= 0]

    hit_df = pd.DataFrame([{
        "Hit-rate (winners/total)": f"{len(winners)} / {len(stock_df)}" if len(stock_df) else "0 / 0",
        "Avg holding period winners (yrs)": round(float(winners["Holding period (years)"].mean()), 2) if len(winners) else np.nan,
        "Avg holding period losers (yrs)": round(float(losers["Holding period (years)"].mean()), 2) if len(losers) else np.nan,
        "Avg alpha contribution on winners (Rs.)": round(float(winners["Alpha contribution (Rs.)"].mean()), 2) if len(winners) else np.nan,
        "Avg alpha contribution on losers (Rs.)": round(float(losers["Alpha contribution (Rs.)"].mean()), 2) if len(losers) else np.nan,
    }])

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
            if instr in cash_keys or instr.startswith("CASH::") or instr.startswith("NOISIN::"):
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
                    if instr in cash_keys or instr.startswith("CASH::") or instr.startswith("NOISIN::"):
                        members.append(instr)
                else:
                    if instr in cash_keys or instr.startswith("CASH::") or instr.startswith("NOISIN::"):
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
            m_count += 1

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

    # ============================================================
    # Domestic-equity sleeve decomposition (one-shot multiple backout)
    # ============================================================
    domestic_isins = [c for c in isins if str(asset_map.get(c, "")).strip() == "Domestic Equities"]
    domestic_isins = [c for c in domestic_isins if c in w0_df.columns]

    dom_decomp_summary_df = pd.DataFrame()
    dom_decomp_period_label = None
    dom_decomp_debug = {}
    dom_decomp_contrib_df = pd.DataFrame()
    dom_decomp_monthly_audit_df = pd.DataFrame()
    dom_triangulation_df = pd.DataFrame()
    dom_triangulation_summary_df = pd.DataFrame()
    dom_triangulation_mode = str(raw.get("dom_triangulation_mode", "Fallback to price growth"))
    dom_triangulation_coverage = {"valid_weight_sum": np.nan, "invalid_weight_sum": np.nan}
    dom_decomp_runtime_mode = "not_run"

    if domestic_isins:
        # Domestic sleeve weights rebased to 1.0 each month (t0)
        # Use raw (no forward-fill) monthly weights for domestic sleeve
        w0_dom_df = w0_df.loc[:, domestic_isins].copy()
        dom_sum = w0_dom_df.sum(axis=1).replace(0.0, np.nan)
        w0_dom_df = w0_dom_df.div(dom_sum, axis=0).fillna(0.0)

        # Monthly domestic returns: price from adj_price plus dividend yield shifted from t0 to t1.
        r_price_dom_m = np.sum(w0_dom_df.to_numpy(dtype=float) * r_price_instr.loc[:, domestic_isins].to_numpy(dtype=float), axis=1)
        div_isin_m = compute_dividend_yield_returns(px, months, domestic_isins)
        div_mat = div_isin_m.reindex(index=t1, columns=domestic_isins).fillna(0.0).to_numpy(dtype=float)
        w_dom_mat = w0_dom_df.reindex(index=t0, columns=domestic_isins).fillna(0.0).to_numpy(dtype=float)
        r_div_dom_m = np.sum(w_dom_mat * div_mat, axis=1)
        r_total_dom_m = r_price_dom_m + r_div_dom_m

        # Monthly valuation and residual fundamental decomposition from stock-level multiples.
        helper_sig = inspect.signature(compute_monthly_portfolio_multiples)
        if "r_price_df" in helper_sig.parameters:
            dom_decomp_runtime_mode = "monthly_drifted_end_weights"
            m_start_arr, m_end_arr, mult_debug = compute_monthly_portfolio_multiples(
                multiples_df=yld,
                months=months,
                isins=domestic_isins,
                w0_df=w0_dom_df,
                r_price_df=r_price_instr.loc[:, domestic_isins],
                lens=lens,
            )
        else:
            dom_decomp_runtime_mode = "legacy_start_weight_end_multiple"
            print(
                "WARN: compute_monthly_portfolio_multiples imported without r_price_df support; "
                "falling back to legacy valuation decomposition."
            )
            m_start_arr, m_end_arr, mult_debug = compute_monthly_portfolio_multiples(
                multiples_df=yld,
                months=months,
                isins=domestic_isins,
                w0_df=w0_dom_df,
                lens=lens,
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            r_val_raw = (m_end_arr / m_start_arr) - 1.0
        r_val_m = np.where(np.isfinite(r_val_raw), r_val_raw, 0.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            r_fund_raw = ((1.0 + r_price_dom_m) / (1.0 + r_val_m)) - 1.0
        r_fund_m = np.where(np.isfinite(r_fund_raw), r_fund_raw, 0.0)

        n_months = int(len(t1))
        years = n_months / 12.0

        gross_total = float(np.prod(1.0 + r_total_dom_m)) if n_months > 0 else np.nan
        gross_price = float(np.prod(1.0 + r_price_dom_m)) if n_months > 0 else np.nan
        gross_div = (gross_total / gross_price) if np.isfinite(gross_total) and np.isfinite(gross_price) and gross_price > 0 else np.nan
        gross_fund = float(np.prod(1.0 + r_fund_m)) if n_months > 0 else np.nan
        gross_val = (gross_price / gross_fund) if np.isfinite(gross_price) and np.isfinite(gross_fund) and gross_fund > 0 else np.nan

        total_return = gross_total - 1.0 if np.isfinite(gross_total) else np.nan
        price_return = gross_price - 1.0 if np.isfinite(gross_price) else np.nan
        div_return = gross_div - 1.0 if np.isfinite(gross_div) else np.nan
        fund_return = gross_fund - 1.0 if np.isfinite(gross_fund) else np.nan
        val_return = gross_val - 1.0 if np.isfinite(gross_val) else np.nan

        def _cagr_if_gt_1y(gross: float) -> float:
            if years > 1.0 and np.isfinite(gross) and gross > 0:
                return gross ** (1.0 / years) - 1.0
            return np.nan

        total_cagr = _cagr_if_gt_1y(gross_total)
        price_cagr = _cagr_if_gt_1y(gross_price)
        div_cagr = _cagr_if_gt_1y(gross_div)
        val_cagr = _cagr_if_gt_1y(gross_val)
        fund_cagr = _cagr_if_gt_1y(gross_fund)

        dom_decomp_summary_df = pd.DataFrame([
            {"Metric": "Domestic sleeve total return", "Total return (%)": round(total_return * 100.0, 2) if pd.notna(total_return) else np.nan, "CAGR (%)": round(total_cagr * 100.0, 2) if pd.notna(total_cagr) else np.nan},
            {"Metric": "Price return", "Total return (%)": round(price_return * 100.0, 2) if pd.notna(price_return) else np.nan, "CAGR (%)": round(price_cagr * 100.0, 2) if pd.notna(price_cagr) else np.nan},
            {"Metric": "Dividend yield return", "Total return (%)": round(div_return * 100.0, 2) if pd.notna(div_return) else np.nan, "CAGR (%)": round(div_cagr * 100.0, 2) if pd.notna(div_cagr) else np.nan},
            {"Metric": "Valuation change", "Total return (%)": round(val_return * 100.0, 2) if pd.notna(val_return) else np.nan, "CAGR (%)": round(val_cagr * 100.0, 2) if pd.notna(val_cagr) else np.nan},
            {"Metric": "Fundamental growth", "Total return (%)": round(fund_return * 100.0, 2) if pd.notna(fund_return) else np.nan, "CAGR (%)": round(fund_cagr * 100.0, 2) if pd.notna(fund_cagr) else np.nan},
        ])

        start_label = t0[0].strftime("%b %Y")
        end_label = t1[-1].strftime("%b %Y")
        dom_decomp_period_label = f"{start_label} to {end_label}"

        dom_decomp_debug = {
            "start_month": t0[0],
            "end_month": t1[-1],
            "start_agg_yield": float(mult_debug["agg_yield_start"][0]) if len(mult_debug.get("agg_yield_start", [])) else np.nan,
            "end_agg_yield": float(mult_debug["agg_yield_end"][-1]) if len(mult_debug.get("agg_yield_end", [])) else np.nan,
            "start_coverage_weight": float(mult_debug["coverage_start_weight"][0]) if len(mult_debug.get("coverage_start_weight", [])) else np.nan,
            "end_coverage_weight": float(mult_debug["coverage_end_weight"][-1]) if len(mult_debug.get("coverage_end_weight", [])) else np.nan,
            "start_multiple": float(m_start_arr[0]) if len(m_start_arr) else np.nan,
            "end_multiple": float(m_end_arr[-1]) if len(m_end_arr) else np.nan,
            "multiple_gross": gross_val,
            "domestic_gross": gross_total,
            "multiple_source": "fundlab.stock_monthly_valuations",
            "multiple_column": mult_debug.get("mult_col"),
            "multiple_validity_rule": mult_debug.get("validity_rule"),
            "gross_price": gross_price,
            "gross_dividend": gross_div,
            "gross_fundamental": gross_fund,
            "runtime_mode": dom_decomp_runtime_mode,
            "helper_signature": str(helper_sig),
        }

        dom_decomp_monthly_audit_df = pd.DataFrame({
            "month_end": [x.date() for x in t1],
            "price_return": r_price_dom_m,
            "dividend_return": r_div_dom_m,
            "total_return": r_total_dom_m,
            "valuation_change": r_val_m,
            "fundamental_return": r_fund_m,
            "M_start": m_start_arr,
            "M_end": m_end_arr,
            "coverage_start_weight": mult_debug.get("coverage_start_weight"),
            "coverage_end_weight": mult_debug.get("coverage_end_weight"),
            "valid_count_start": mult_debug.get("valid_count_start"),
            "valid_count_end": mult_debug.get("valid_count_end"),
            "drifted_end_weight_sum": mult_debug.get("drifted_end_weight_sum"),
        })

        # Stock-level diagnostics: average weight + exact chained contributions (no scaling)
        r_stock_m = r_instr.loc[:, domestic_isins].to_numpy(dtype=float)
        w0_dom = w0_dom_df.to_numpy(dtype=float)
        r_dom_m = r_price_dom_m

        link_dom = np.ones(len(r_dom_m), dtype=float)
        for i in range(1, len(r_dom_m)):
            link_dom[i] = link_dom[i-1] * (1.0 + r_dom_m[i-1])

        contrib = np.sum((w0_dom * r_stock_m) * link_dom[:, None], axis=0)
        avg_w = w0_dom_df.mean(axis=0).to_numpy(dtype=float)
        domestic_weight_avg = float(w0_dom_df.sum(axis=1).mean())

        dom_decomp_contrib_df = pd.DataFrame({
            "Stock name": [name_map.get(x, x) for x in domestic_isins],
            "Average holding weight (%)": np.round(avg_w * 100.0, 4),
            "Contribution (%)": np.round(contrib * 100.0, 4),
        }).sort_values("Contribution (%)", ascending=False).reset_index(drop=True)

        total_row = pd.DataFrame([{
            "Stock name": "TOTAL (sum of contributions)",
            "Average holding weight (%)": np.round(domestic_weight_avg * 100.0, 4),
            "Contribution (%)": np.round(contrib.sum() * 100.0, 4),
        }])
        dom_decomp_contrib_df = pd.concat([dom_decomp_contrib_df, total_row], ignore_index=True)

        dom_triangulation_df, dom_triangulation_summary_df, dom_triangulation_coverage = _compute_domestic_triangulation_tables(
            w0_dom_df=w0_dom_df,
            domestic_isins=domestic_isins,
            px_df=px,
            yld_df=yld,
            lens=lens,
            name_map=name_map,
            mode_label=dom_triangulation_mode,
        )

    diag = {
        "months": months,
        "missing_benchmarks": [x for x in [BENCH_NAME_NIFTY50, BENCH_NAME_NIFTY500, BENCH_NAME_NIFTY100, BENCH_NAME_MID150, BENCH_NAME_SMALL250] if x not in bn_piv.columns],
        "adj_price_invalid_rows_df": invalid_price_rows_df,
        "adj_price_return_audit_df": price_return_audit_df,
        "domestic_decomp_period_label": dom_decomp_period_label,
        "domestic_decomp_summary_df": dom_decomp_summary_df,
        "domestic_decomp_stock_count": len(domestic_isins),
        "domestic_decomp_lens": lens,
        "domestic_decomp_debug": dom_decomp_debug,
        "domestic_decomp_contrib_df": dom_decomp_contrib_df,
        "domestic_decomp_monthly_audit_df": dom_decomp_monthly_audit_df,
        "dom_triangulation_df": dom_triangulation_df,
        "dom_triangulation_summary_df": dom_triangulation_summary_df,
        "dom_triangulation_mode": dom_triangulation_mode,
        "dom_triangulation_valid_weight_sum": dom_triangulation_coverage.get("valid_weight_sum"),
        "dom_triangulation_invalid_weight_sum": dom_triangulation_coverage.get("invalid_weight_sum"),
    }
    return stock_df, hit_df, cat_df, diag
