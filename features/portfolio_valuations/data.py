"""Read-only data retrieval and cached valuation panels for Portfolio Valuations."""

import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy.sql import text

from core.dataframes import ensure_unique_monthly_rows
from core.db import get_engine

def compute_portfolio_valuations_cube(
    fund_ids: list[int],
    start_date: dt.date,
    end_date: dt.date,
    mode: str,
    agg_choice: str = "Weighted average multiple",  # NEW
) -> pd.DataFrame:
    """
    Output cube columns:
      fund_id (int)
      month_end (datetime64[ns]) canonical month-end timestamp
      segment (str) in {"Total","Financials","Non-financials"}
      metric  (str) in {"P/S","P/B","P/E"}
      value   (float) portfolio multiple

    mode:
      - "Valuations of historical portfolios"
      - "Historical valuations of current portfolio"

    agg_choice:
      - "Weighted average multiple" => existing behavior
      - "Median multiple"           => median stock multiple within the portfolio each month

    For median:
      multiple_stock = 1 / yield_stock  (yield != 0)
      value = median(multiple_stock) over stocks with non-null multiple
      Segment uses segment membership; NO weight rebasing required for median.
    """
    import numpy as np
    import pandas as pd
    from sqlalchemy.sql import text

    if not fund_ids:
        return pd.DataFrame(columns=["fund_id", "month_end", "segment", "metric", "value"])

    if agg_choice not in ("Weighted average multiple", "Median multiple"):
        raise ValueError(f"Unsupported agg_choice: {agg_choice}")

    engine = get_engine()

    # Canonical month periods
    month_periods = pd.period_range(start=start_date, end=end_date, freq="M")
    if len(month_periods) == 0:
        return pd.DataFrame(columns=["fund_id", "month_end", "segment", "metric", "value"])

    # ------------------------------------------------------------
    # Helper: compute median multiples from a holdings panel + yields
    # holdings required cols: fund_id, month_key, isin, is_financial, weight_pct (weight_pct unused for median)
    # yields required cols:    isin, month_key, ps_yield, pb_yield, pe_yield
    # ------------------------------------------------------------
    def _median_cube_from_holdings_and_yields(holdings: pd.DataFrame, vals: pd.DataFrame) -> pd.DataFrame:
        if holdings.empty or vals.empty:
            return pd.DataFrame(columns=["fund_id", "month_end", "segment", "metric", "value"])

        h = holdings.copy()
        v = vals.copy()

        h["isin"] = h["isin"].astype(str).str.strip()
        v["isin"] = v["isin"].astype(str).str.strip()

        # join
        df = h.merge(
            v[["isin", "month_key", "ps_yield", "pb_yield", "pe_yield"]],
            on=["isin", "month_key"],
            how="left",
        )

        for c in ("ps_yield", "pb_yield", "pe_yield"):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

        # implied multiples
        df["mul_ps"] = np.where(df["ps_yield"] != 0, 1.0 / df["ps_yield"], np.nan)
        df["mul_pb"] = np.where(df["pb_yield"] != 0, 1.0 / df["pb_yield"], np.nan)
        df["mul_pe"] = np.where(df["pe_yield"] != 0, 1.0 / df["pe_yield"], np.nan)

        seg_defs = [
            ("Total", None),
            ("Financials", True),
            ("Non-financials", False),
        ]

        out_rows = []
        grouped = df.groupby(["fund_id", "month_key"], sort=True)

        for (fund_id, month_key), grp in grouped:
            month_end_ts = month_key.to_timestamp("M").normalize()

            for seg_name, fin_flag in seg_defs:
                if fin_flag is None:
                    seg_grp = grp
                else:
                    seg_grp = grp[grp["is_financial"] == fin_flag]

                if seg_grp.empty:
                    continue

                ps_med = seg_grp["mul_ps"].median(skipna=True)
                pb_med = seg_grp["mul_pb"].median(skipna=True)
                pe_med = seg_grp["mul_pe"].median(skipna=True)

                if not np.isnan(ps_med):
                    out_rows.append({"fund_id": int(fund_id), "month_end": month_end_ts, "segment": seg_name, "metric": "P/S", "value": float(ps_med)})
                if not np.isnan(pb_med):
                    out_rows.append({"fund_id": int(fund_id), "month_end": month_end_ts, "segment": seg_name, "metric": "P/B", "value": float(pb_med)})
                if not np.isnan(pe_med):
                    out_rows.append({"fund_id": int(fund_id), "month_end": month_end_ts, "segment": seg_name, "metric": "P/E", "value": float(pe_med)})

        if not out_rows:
            return pd.DataFrame(columns=["fund_id", "month_end", "segment", "metric", "value"])

        out = pd.DataFrame.from_records(out_rows)
        out["month_end"] = pd.to_datetime(out["month_end"], errors="coerce")
        out = out.dropna(subset=["month_end"])
        return out.sort_values(["fund_id", "segment", "metric", "month_end"]).reset_index(drop=True)

    # ============================================================
    # BRANCH A: MEDIAN MULTIPLE (new)
    # ============================================================
    if agg_choice == "Median multiple":
        # We must compute from stock-level yields for BOTH modes.
        # Mode 1 uses historical holdings per month; Mode 2 uses anchor holdings replicated across months.

        # ---- Get holdings panel depending on mode ----
        if mode == "Valuations of historical portfolios":
            with engine.begin() as conn:
                h_sql = text(
                    """
                    SELECT
                        fp.fund_id,
                        fp.month_end,
                        fp.isin,
                        fp.holding_weight AS weight_pct,
                        sm.is_financial
                    FROM fundlab.fund_portfolio fp
                    JOIN fundlab.stock_master sm
                      ON sm.isin = fp.isin
                    WHERE fp.fund_id = ANY(:fund_ids)
                      AND fp.asset_type = 'Domestic Equities'
                      AND fp.month_end BETWEEN :start_date AND :end_date;
                    """
                )
                holdings = pd.read_sql(
                    h_sql,
                    conn,
                    params={"fund_ids": fund_ids, "start_date": start_date, "end_date": end_date},
                )

            if holdings.empty:
                return pd.DataFrame(columns=["fund_id", "month_end", "segment", "metric", "value"])

            holdings["month_end"] = pd.to_datetime(holdings["month_end"], errors="coerce")
            holdings = holdings.dropna(subset=["month_end"])
            holdings["month_key"] = holdings["month_end"].dt.to_period("M")
            holdings = holdings[holdings["month_key"].isin(month_periods)].copy()
            if holdings.empty:
                return pd.DataFrame(columns=["fund_id", "month_end", "segment", "metric", "value"])

            holdings["is_financial"] = holdings["is_financial"].astype(bool)

        elif mode == "Historical valuations of current portfolio":
            with engine.begin() as conn:
                holdings_sql = text(
                    """
                    SELECT
                        fp.fund_id,
                        fp.month_end,
                        fp.isin,
                        fp.holding_weight AS weight_pct,
                        sm.is_financial
                    FROM fundlab.fund_portfolio fp
                    JOIN fundlab.stock_master sm
                      ON sm.isin = fp.isin
                    JOIN (
                        SELECT fund_id, MAX(month_end) AS anchor_month_end
                        FROM fundlab.fund_portfolio
                        WHERE fund_id = ANY(:fund_ids)
                          AND month_end <= :end_date
                        GROUP BY fund_id
                    ) a
                      ON a.fund_id = fp.fund_id
                     AND a.anchor_month_end = fp.month_end
                    WHERE fp.asset_type = 'Domestic Equities';
                    """
                )
                anchor = pd.read_sql(
                    holdings_sql,
                    conn,
                    params={"fund_ids": fund_ids, "end_date": end_date},
                )

            if anchor.empty:
                return pd.DataFrame(columns=["fund_id", "month_end", "segment", "metric", "value"])

            anchor["isin"] = anchor["isin"].astype(str).str.strip()
            anchor["weight_pct"] = pd.to_numeric(anchor["weight_pct"], errors="coerce").fillna(0.0)
            anchor["is_financial"] = anchor["is_financial"].astype(bool)

            months_df = pd.DataFrame({"month_key": month_periods})
            base = anchor[["fund_id", "isin", "weight_pct", "is_financial"]].copy()
            base["_key"] = 1
            months_df["_key"] = 1
            holdings = base.merge(months_df, on="_key").drop(columns=["_key"])

        else:
            raise ValueError(f"Unsupported mode: {mode}")

        # ---- Pull yields for all ISINs over the window once ----
        all_isins = sorted(holdings["isin"].dropna().astype(str).str.strip().unique().tolist())
        if not all_isins:
            return pd.DataFrame(columns=["fund_id", "month_end", "segment", "metric", "value"])

        with engine.begin() as conn:
            vals_sql = text(
                """
                SELECT
                    isin,
                    month_end,
                    ps AS ps_yield,
                    pe AS pe_yield,
                    pb AS pb_yield
                FROM fundlab.stock_monthly_valuations
                WHERE isin = ANY(:isins)
                  AND month_end BETWEEN :start_date AND :end_date;
                """
            )
            vals = pd.read_sql(
                vals_sql,
                conn,
                params={"isins": all_isins, "start_date": start_date, "end_date": end_date},
            )

        if vals.empty:
            return pd.DataFrame(columns=["fund_id", "month_end", "segment", "metric", "value"])

        vals["month_end"] = pd.to_datetime(vals["month_end"], errors="coerce")
        vals = vals.dropna(subset=["month_end"])
        vals["month_key"] = vals["month_end"].dt.to_period("M")
        vals = vals[vals["month_key"].isin(month_periods)].copy()
        if vals.empty:
            return pd.DataFrame(columns=["fund_id", "month_end", "segment", "metric", "value"])

        return _median_cube_from_holdings_and_yields(holdings, vals)

    # ============================================================
    # BRANCH B: WEIGHTED AVERAGE MULTIPLE (existing behavior)
    # ============================================================

    # ============================================================
    # MODE 1: Historical portfolios => fetch stored fund valuations
    # ============================================================
    if mode == "Valuations of historical portfolios":
        with engine.begin() as conn:
            sql = text(
                """
                SELECT
                    fund_id,
                    month_end::date AS month_end,
                    segment,
                    ps,
                    pb,
                    pe
                FROM fundlab.fund_monthly_valuations
                WHERE fund_id = ANY(:fund_ids)
                  AND month_end BETWEEN :start_date AND :end_date
                ORDER BY fund_id, month_end, segment;
                """
            )
            df = pd.read_sql(
                sql,
                conn,
                params={"fund_ids": fund_ids, "start_date": start_date, "end_date": end_date},
            )

        if df.empty:
            return pd.DataFrame(columns=["fund_id", "month_end", "segment", "metric", "value"])

        df["month_end"] = pd.to_datetime(df["month_end"], errors="coerce")
        df = df.dropna(subset=["month_end"])
        if df.empty:
            return pd.DataFrame(columns=["fund_id", "month_end", "segment", "metric", "value"])

        df["segment"] = df["segment"].astype(str)

        for c in ("ps", "pb", "pe"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

        out = df.melt(
            id_vars=["fund_id", "month_end", "segment"],
            value_vars=["ps", "pb", "pe"],
            var_name="metric",
            value_name="value",
        )
        out["metric"] = out["metric"].map({"ps": "P/S", "pb": "P/B", "pe": "P/E"}).fillna(out["metric"])
        out["month_end"] = out["month_end"].dt.to_period("M").dt.to_timestamp("M").dt.normalize()

        return out[["fund_id", "month_end", "segment", "metric", "value"]].sort_values(
            ["fund_id", "segment", "metric", "month_end"]
        ).reset_index(drop=True)

    # ============================================================
    # MODE 2: Current portfolio held static; compute from stock yields
    # ============================================================
    if mode != "Historical valuations of current portfolio":
        raise ValueError(f"Unsupported mode: {mode}")

    # 1) Anchor holdings per fund at latest month_end <= end_date
    with engine.begin() as conn:
        holdings_sql = text(
            """
            SELECT
                fp.fund_id,
                fp.month_end,
                fp.isin,
                fp.holding_weight AS weight_pct,
                sm.is_financial
            FROM fundlab.fund_portfolio fp
            JOIN fundlab.stock_master sm
              ON sm.isin = fp.isin
            JOIN (
                SELECT fund_id, MAX(month_end) AS anchor_month_end
                FROM fundlab.fund_portfolio
                WHERE fund_id = ANY(:fund_ids)
                  AND month_end <= :end_date
                GROUP BY fund_id
            ) a
              ON a.fund_id = fp.fund_id
             AND a.anchor_month_end = fp.month_end
            WHERE fp.asset_type = 'Domestic Equities';
            """
        )
        anchor = pd.read_sql(
            holdings_sql,
            conn,
            params={"fund_ids": fund_ids, "end_date": end_date},
        )

    if anchor.empty:
        return pd.DataFrame(columns=["fund_id", "month_end", "segment", "metric", "value"])

    anchor["isin"] = anchor["isin"].astype(str).str.strip()
    anchor["weight_pct"] = pd.to_numeric(anchor["weight_pct"], errors="coerce").fillna(0.0)
    anchor["is_financial"] = anchor["is_financial"].astype(bool)

    # 2) Prepare month grid
    months_df = pd.DataFrame({"month_key": month_periods})

    # 3) Pull yields for all ISINs in anchor holdings across the selected window
    all_isins = sorted(anchor["isin"].dropna().unique().tolist())
    if not all_isins:
        return pd.DataFrame(columns=["fund_id", "month_end", "segment", "metric", "value"])

    with engine.begin() as conn:
        vals_sql = text(
            """
            SELECT
                isin,
                month_end,
                ps AS ps_yield,
                pe AS pe_yield,
                pb AS pb_yield
            FROM fundlab.stock_monthly_valuations
            WHERE isin = ANY(:isins)
              AND month_end BETWEEN :start_date AND :end_date;
            """
        )
        vals = pd.read_sql(
            vals_sql,
            conn,
            params={"isins": all_isins, "start_date": start_date, "end_date": end_date},
        )

    if vals.empty:
        return pd.DataFrame(columns=["fund_id", "month_end", "segment", "metric", "value"])

    vals["isin"] = vals["isin"].astype(str).str.strip()
    vals["month_end"] = pd.to_datetime(vals["month_end"], errors="coerce")
    vals = vals.dropna(subset=["month_end"])
    vals["month_key"] = vals["month_end"].dt.to_period("M")
    vals = ensure_unique_monthly_rows(
        vals,
        key_cols=["isin", "month_key"],
        value_cols=["ps_yield", "pe_yield", "pb_yield"],
        context="fundlab.stock_monthly_valuations",
    )
    vals = vals[vals["month_key"].isin(month_periods)].copy()
    if vals.empty:
        return pd.DataFrame(columns=["fund_id", "month_end", "segment", "metric", "value"])

    # 4) Expand anchor holdings across months and join yields
    base = anchor[["fund_id", "isin", "weight_pct", "is_financial"]].copy()
    base["_key"] = 1
    months_df["_key"] = 1
    combo = base.merge(months_df, on="_key").drop(columns=["_key"])

    df = combo.merge(
        vals[["isin", "month_key", "ps_yield", "pe_yield", "pb_yield"]],
        on=["isin", "month_key"],
        how="left",
    )

    # Prepare base weights once per (fund_id, month_key)
    df["weight_pct"] = pd.to_numeric(df["weight_pct"], errors="coerce").fillna(0.0)
    abs_sum_w = df.groupby(["fund_id", "month_key"])["weight_pct"].transform(lambda s: s.abs().sum())
    df["weight_scale"] = np.where(abs_sum_w > 2.0, 0.01, 1.0)
    df["w_base"] = df["weight_pct"] * df["weight_scale"]

    # 5) Compute multiples per segment and per metric with segment rebasing to 100%
    def _seg_multiple(seg_df: pd.DataFrame, yield_col: str) -> float:
        g = seg_df.copy()
        g[yield_col] = pd.to_numeric(g[yield_col], errors="coerce")

        seg_sum = float(pd.to_numeric(g["w_base"], errors="coerce").fillna(0.0).sum())
        if seg_sum <= 0:
            return np.nan
        g["w_seg"] = pd.to_numeric(g["w_base"], errors="coerce").fillna(0.0) / seg_sum

        y_filled = g[yield_col].fillna(0.0)
        agg_yield = float(np.nansum(g["w_seg"] * y_filled))

        if np.isnan(agg_yield) or agg_yield == 0:
            return np.nan

        return float(1.0 / agg_yield)

    seg_defs = [("Total", None), ("Financials", True), ("Non-financials", False)]

    records = []
    grouped = df.groupby(["fund_id", "month_key"], sort=True)

    for (fund_id, month_key), grp in grouped:
        for seg_name, fin_flag in seg_defs:
            seg_grp = grp if fin_flag is None else grp[grp["is_financial"] == fin_flag]
            if seg_grp.empty:
                continue

            ps_val = _seg_multiple(seg_grp, "ps_yield")
            pb_val = _seg_multiple(seg_grp, "pb_yield")
            pe_val = _seg_multiple(seg_grp, "pe_yield")

            month_end_ts = month_key.to_timestamp("M").normalize()

            if not np.isnan(ps_val):
                records.append({"fund_id": int(fund_id), "month_end": month_end_ts, "segment": seg_name, "metric": "P/S", "value": ps_val})
            if not np.isnan(pb_val):
                records.append({"fund_id": int(fund_id), "month_end": month_end_ts, "segment": seg_name, "metric": "P/B", "value": pb_val})
            if not np.isnan(pe_val):
                records.append({"fund_id": int(fund_id), "month_end": month_end_ts, "segment": seg_name, "metric": "P/E", "value": pe_val})

    if not records:
        return pd.DataFrame(columns=["fund_id", "month_end", "segment", "metric", "value"])

    out = pd.DataFrame.from_records(records)
    out["month_end"] = pd.to_datetime(out["month_end"], errors="coerce")
    out = out.dropna(subset=["month_end"])
    return out.sort_values(["fund_id", "segment", "metric", "month_end"]).reset_index(drop=True)


def compute_portfolio_exposures_base_panel(
    fund_ids: list[int],
    start_date: dt.date,
    end_date: dt.date,
    mode: str,
) -> pd.DataFrame:
    import pandas as pd
    from sqlalchemy.sql import text

    if not fund_ids:
        return pd.DataFrame(columns=["fund_id", "month_key", "isin", "weight_pct", "is_financial", "ps_yield", "pe_yield", "pb_yield"])

    engine = get_engine()
    month_periods = pd.period_range(start=start_date, end=end_date, freq="M")
    if len(month_periods) == 0:
        return pd.DataFrame(columns=["fund_id", "month_key", "isin", "weight_pct", "is_financial", "ps_yield", "pe_yield", "pb_yield"])

    months_df = pd.DataFrame({"month_key": month_periods})

    # -----------------------------
    # MODE 1: historical portfolios
    # -----------------------------
    if mode == "Valuations of historical portfolios":
        with engine.begin() as conn:
            h_sql = text(
                """
                SELECT
                    fp.fund_id,
                    fp.month_end,
                    fp.isin,
                    fp.holding_weight AS weight_pct,
                    sm.is_financial
                FROM fundlab.fund_portfolio fp
                JOIN fundlab.stock_master sm
                  ON sm.isin = fp.isin
                WHERE fp.fund_id = ANY(:fund_ids)
                  AND fp.asset_type = 'Domestic Equities'
                  AND fp.month_end BETWEEN :start_date AND :end_date;
                """
            )
            h = pd.read_sql(
                h_sql,
                conn,
                params={"fund_ids": fund_ids, "start_date": start_date, "end_date": end_date},
            )

        if h.empty:
            return pd.DataFrame(columns=["fund_id", "month_key", "isin", "weight_pct", "is_financial", "ps_yield", "pe_yield", "pb_yield"])

        h["month_end"] = pd.to_datetime(h["month_end"], errors="coerce")
        h = h.dropna(subset=["month_end"])
        h["month_key"] = h["month_end"].dt.to_period("M")
        h = h[h["month_key"].isin(month_periods)].copy()
        if h.empty:
            return pd.DataFrame(columns=["fund_id", "month_key", "isin", "weight_pct", "is_financial", "ps_yield", "pe_yield", "pb_yield"])

        h["isin"] = h["isin"].astype(str).str.strip()
        h["is_financial"] = h["is_financial"].astype(bool)
        h["weight_pct"] = pd.to_numeric(h["weight_pct"], errors="coerce").fillna(0.0)

        rows = h[["fund_id", "month_key", "isin", "weight_pct", "is_financial"]].copy()

    # -----------------------------
    # MODE 2: current portfolio (anchor) expanded across months
    # -----------------------------
    elif mode == "Historical valuations of current portfolio":
        with engine.begin() as conn:
            holdings_sql = text(
                """
                SELECT
                    fp.fund_id,
                    fp.month_end,
                    fp.isin,
                    fp.holding_weight AS weight_pct,
                    sm.is_financial
                FROM fundlab.fund_portfolio fp
                JOIN fundlab.stock_master sm
                  ON sm.isin = fp.isin
                JOIN (
                    SELECT fund_id, MAX(month_end) AS anchor_month_end
                    FROM fundlab.fund_portfolio
                    WHERE fund_id = ANY(:fund_ids)
                      AND month_end <= :end_date
                    GROUP BY fund_id
                ) a
                  ON a.fund_id = fp.fund_id
                 AND a.anchor_month_end = fp.month_end
                WHERE fp.asset_type = 'Domestic Equities';
                """
            )
            h = pd.read_sql(
                holdings_sql,
                conn,
                params={"fund_ids": fund_ids, "end_date": end_date},
            )

        if h.empty:
            return pd.DataFrame(columns=["fund_id", "month_key", "isin", "weight_pct", "is_financial", "ps_yield", "pe_yield", "pb_yield"])

        h["isin"] = h["isin"].astype(str).str.strip()
        h["is_financial"] = h["is_financial"].astype(bool)
        h["weight_pct"] = pd.to_numeric(h["weight_pct"], errors="coerce").fillna(0.0)

        base = h[["fund_id", "isin", "weight_pct", "is_financial"]].copy()
        base["_key"] = 1
        months_df["_key"] = 1
        rows = base.merge(months_df, on="_key").drop(columns=["_key"])

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # Pull yields once for all required isins over the window
    all_isins = sorted(rows["isin"].dropna().unique().tolist())
    if not all_isins:
        return pd.DataFrame(columns=["fund_id", "month_key", "isin", "weight_pct", "is_financial", "ps_yield", "pe_yield", "pb_yield"])

    with engine.begin() as conn:
        v_sql = text(
            """
            SELECT
                isin,
                month_end,
                ps AS ps_yield,
                pe AS pe_yield,
                pb AS pb_yield
            FROM fundlab.stock_monthly_valuations
            WHERE isin = ANY(:isins)
              AND month_end BETWEEN :start_date AND :end_date;
            """
        )
        vals = pd.read_sql(
            v_sql,
            conn,
            params={"isins": all_isins, "start_date": start_date, "end_date": end_date},
        )

    if vals.empty:
        return pd.DataFrame(columns=["fund_id", "month_key", "isin", "weight_pct", "is_financial", "ps_yield", "pe_yield", "pb_yield"])

    vals["isin"] = vals["isin"].astype(str).str.strip()
    vals["month_end"] = pd.to_datetime(vals["month_end"], errors="coerce")
    vals = vals.dropna(subset=["month_end"])
    vals["month_key"] = vals["month_end"].dt.to_period("M")
    vals = vals[vals["month_key"].isin(month_periods)].copy()
    if vals.empty:
        return pd.DataFrame(columns=["fund_id", "month_key", "isin", "weight_pct", "is_financial", "ps_yield", "pe_yield", "pb_yield"])

    out = rows.merge(
        vals[["isin", "month_key", "ps_yield", "pe_yield", "pb_yield"]],
        on=["isin", "month_key"],
        how="left",
    )

    for c in ("ps_yield", "pe_yield", "pb_yield", "weight_pct"):
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["is_financial"] = out["is_financial"].astype(bool)

    return out[["fund_id", "month_key", "isin", "weight_pct", "is_financial", "ps_yield", "pe_yield", "pb_yield"]]


def compute_portfolio_exposures_timeseries(
    fund_ids: list[int],
    focus_fund_id: int,
    start_date: dt.date,
    end_date: dt.date,
    segment_choice: str,
    metric_choice: str,
    mode: str,
) -> pd.DataFrame:
    """
    Exposure diagnostics (% bucket weights), computed from stock-level data.

    DB hit is only via cached_portfolio_exposures_base_panel() and only on:
      (fund_ids, start/end, mode)

    Segment  handling:
      - filter to segment (Financials / Non-financials / Total)
      - REBASE weights to 100% within segment:
          Mode 1 (historical portfolios): within (fund_id, month_key)
          Mode 2 (current portfolio held static): within (fund_id) on anchor weights (replicated across months),
             implemented here as within (fund_id, month_key) because month_key exists in the panel.
    """
    import numpy as np
    import pandas as pd

    if not fund_ids or focus_fund_id not in fund_ids:
        raise ValueError("Focus fund must be among selected funds.")
    other_fund_ids = [fid for fid in fund_ids if fid != focus_fund_id]
    if not other_fund_ids:
        raise ValueError("Need at least one other fund to compute universe median.")

    base = cached_portfolio_exposures_base_panel(
        fund_ids=fund_ids,
        start_date=start_date,
        end_date=end_date,
        mode=mode,
    )
    if base.empty:
        return pd.DataFrame(columns=["month_end", "series", "metric", "value"])

    # segment filter
    if segment_choice == "Financials":
        base = base[base["is_financial"]].copy()
    elif segment_choice == "Non-financials":
        base = base[~base["is_financial"]].copy()

    if base.empty:
        return pd.DataFrame(columns=["month_end", "series", "metric", "value"])

    # Rebase weights to 100% within segment
    base["weight_pct"] = pd.to_numeric(base["weight_pct"], errors="coerce").fillna(0.0)
    sum_w = base.groupby(["fund_id", "month_key"])["weight_pct"].transform("sum")
    base = base[sum_w > 0].copy()
    if base.empty:
        return pd.DataFrame(columns=["month_end", "series", "metric", "value"])
    base["w_seg"] = base["weight_pct"] / sum_w

    # Choose yield column
    ymap = {"P/S": "ps_yield", "P/E": "pe_yield", "P/B": "pb_yield"}
    ycol = ymap.get(metric_choice)
    if ycol is None:
        raise ValueError(f"Unsupported metric_choice: {metric_choice}")

    base["yield_value"] = pd.to_numeric(base[ycol], errors="coerce")

    # implied multiple safely
    base["multiple"] = np.where(
        base["yield_value"].notna() & (base["yield_value"] != 0),
        1.0 / base["yield_value"],
        np.nan,
    )

    # Bucket definitions identical to your earlier exposures code
    if metric_choice == "P/E":
        bucket_defs = [
            ("Loss-making (%)", lambda x: x.notna() & (x < 0)),
            ("P/E > 40x (%)", lambda x: x.notna() & (x > 40)),
            ("P/E < 15x (%)", lambda x: x.notna() & (x > 0) & (x < 15)),
        ]
        coverage_mask = lambda x: x.notna() & (x != 0)
    elif metric_choice == "P/S":
        bucket_defs = [
            ("P/S > 4x (%)", lambda x: x.notna() & (x > 4)),
            ("P/S < 1.5x (%)", lambda x: x.notna() & (x > 0) & (x < 1.5)),
        ]
        coverage_mask = lambda x: x.notna() & (x > 0)
    elif metric_choice == "P/B":
        bucket_defs = [
            ("P/B > 6x (%)", lambda x: x.notna() & (x > 6)),
            ("P/B < 2x (%)", lambda x: x.notna() & (x > 0) & (x < 2)),
        ]
        coverage_mask = lambda x: x.notna() & (x > 0)
    else:
        return pd.DataFrame(columns=["month_end", "series", "metric", "value"])

    base["is_covered"] = coverage_mask(base["multiple"])

    # coverage weight within segment (already rebased to 1.0, but coverage may be <1 due to missing yields)
    cov = (
        base.assign(w_cov=np.where(base["is_covered"], base["w_seg"], 0.0))
            .groupby(["fund_id", "month_key"], as_index=False)["w_cov"].sum()
            .rename(columns={"w_cov": "coverage_weight"})
    )

    out_rows = []
    for metric_label, mask_fn in bucket_defs:
        m = mask_fn(base["multiple"])
        tmp = base.copy()
        tmp["w_hit"] = np.where(m, tmp["w_seg"], 0.0)

        g = tmp.groupby(["fund_id", "month_key"], as_index=False)["w_hit"].sum()
        g = g.merge(cov, on=["fund_id", "month_key"], how="left")

        g["value"] = np.where(
            (g["coverage_weight"].notna()) & (g["coverage_weight"] > 0),
            100.0 * (g["w_hit"] / g["coverage_weight"]),
            np.nan,
        )
        g["metric"] = metric_label
        out_rows.append(g[["fund_id", "month_key", "metric", "value"]])

    fund_bucket = pd.concat(out_rows, ignore_index=True)
    fund_bucket["month_end"] = fund_bucket["month_key"].dt.to_timestamp("M").dt.normalize()

    focus = fund_bucket[fund_bucket["fund_id"] == focus_fund_id].copy()
    focus["series"] = "Focus fund"
    focus = focus[["month_end", "metric", "series", "value"]]

    others = fund_bucket[fund_bucket["fund_id"].isin(other_fund_ids)].copy()
    med = others.groupby(["month_end", "metric"], as_index=False)["value"].median()
    med["series"] = "Universe median (others)"
    med = med[["month_end", "metric", "series", "value"]]

    out = pd.concat([focus, med], ignore_index=True)
    out = out.sort_values(["metric", "month_end", "series"]).reset_index(drop=True)
    return out


@st.cache_data(ttl=60 * 30, show_spinner=False)
def cached_portfolio_valuations_cube(
    fund_ids: list[int],
    start_date: dt.date,
    end_date: dt.date,
    mode: str,
    agg_choice: str,  # NEW
) -> pd.DataFrame:
    fund_ids_norm = tuple(sorted(int(x) for x in fund_ids))
    return compute_portfolio_valuations_cube(
        fund_ids=list(fund_ids_norm),
        start_date=start_date,
        end_date=end_date,
        mode=mode,
        agg_choice=agg_choice,
    )


@st.cache_data(ttl=60 * 30, show_spinner=False)
def cached_portfolio_valuations_timeseries(
    fund_ids,
    focus_fund_id,
    start_date,
    end_date,
    segment_choice,
    metric_choice,
    mode,
    agg_choice="Weighted average multiple",  # NEW
):
    return compute_portfolio_valuations_timeseries(
        fund_ids=fund_ids,
        focus_fund_id=focus_fund_id,
        start_date=start_date,
        end_date=end_date,
        segment_choice=segment_choice,
        metric_choice=metric_choice,
        mode=mode,
        agg_choice=agg_choice,  # NEW
    )


def compute_portfolio_valuations_timeseries(
    fund_ids: list[int],
    focus_fund_id: int,
    start_date: dt.date,
    end_date: dt.date,
    segment_choice: str,
    metric_choice: str,
    mode: str,
    agg_choice: str = "Weighted average multiple",  # NEW
) -> pd.DataFrame:
    import pandas as pd

    if not fund_ids or focus_fund_id not in fund_ids:
        raise ValueError("Focus fund must be among selected funds.")
    other_fund_ids = [fid for fid in fund_ids if fid != focus_fund_id]
    if not other_fund_ids:
        raise ValueError("Need at least one other fund to compute universe median.")

    df_cube = cached_portfolio_valuations_cube(
        fund_ids=fund_ids,
        start_date=start_date,
        end_date=end_date,
        mode=mode,
        agg_choice=agg_choice,  # NEW
    )
    if df_cube is None or df_cube.empty:
        return pd.DataFrame(columns=["month_end", "series", "value"])

    df_slice = df_cube[
        (df_cube["segment"] == segment_choice) &
        (df_cube["metric"] == metric_choice)
    ].copy()

    if df_slice.empty:
        return pd.DataFrame(columns=["month_end", "series", "value"])

    focus = df_slice[df_slice["fund_id"] == focus_fund_id][["month_end", "value"]].copy()
    if focus.empty:
        return pd.DataFrame(columns=["month_end", "series", "value"])
    focus["series"] = "Focus fund"

    others = df_slice[df_slice["fund_id"].isin(other_fund_ids)][["month_end", "value"]].copy()
    if others.empty:
        return pd.DataFrame(columns=["month_end", "series", "value"])

    median_others = others.groupby("month_end", as_index=False)["value"].median()
    median_others["series"] = "Universe median (others)"

    out = pd.concat([focus, median_others], ignore_index=True)
    out["month_end"] = pd.to_datetime(out["month_end"], errors="coerce")
    out = out.dropna(subset=["month_end"])
    return out.sort_values(["month_end", "series"]).reset_index(drop=True)



@st.cache_data(ttl=60 * 30, show_spinner=False)
def cached_portfolio_median_cube(
    fund_ids: list[int],
    start_date: dt.date,
    end_date: dt.date,
    mode: str,
) -> pd.DataFrame:
    fund_ids_norm = tuple(sorted(int(x) for x in fund_ids))
    return compute_portfolio_median_cube(
        fund_ids=list(fund_ids_norm),
        start_date=start_date,
        end_date=end_date,
        mode=mode,
    )

def compute_portfolio_median_cube(
    fund_ids: list[int],
    start_date: dt.date,
    end_date: dt.date,
    mode: str,
) -> pd.DataFrame:
    """
    Median multiple cube computed from stock-level data.

    Definition used (simple, per your brief):
      For each (fund_id, month_key, segment, metric):
        - filter holdings to segment (Total/Financials/Non-financials)
        - compute stock multiple = 1 / yield (yield != 0)
        - take median of stock multiples across holdings with non-null multiple

    Notes:
      - This is an *unweighted* median of stock multiples.
      - This aligns with “median portfolio stock valuation”.
      - If later you want *weighted median*, we can add it without changing the UI.
    """
    import numpy as np
    import pandas as pd

    if not fund_ids:
        return pd.DataFrame(columns=["fund_id", "month_end", "segment", "metric", "value"])

    # Reuse your existing cached base panel (1 DB hit per boundary)
    base = cached_portfolio_exposures_base_panel(
        fund_ids=fund_ids,
        start_date=start_date,
        end_date=end_date,
        mode=mode,
    )
    if base.empty:
        return pd.DataFrame(columns=["fund_id", "month_end", "segment", "metric", "value"])

    base["is_financial"] = base["is_financial"].astype(bool)

    def _median_for_segment(seg_name: str, seg_df: pd.DataFrame) -> pd.DataFrame:
        # compute implied multiples safely
        tmp = seg_df.copy()
        for ycol, met in [("ps_yield", "P/S"), ("pb_yield", "P/B"), ("pe_yield", "P/E")]:
            tmp[ycol] = pd.to_numeric(tmp[ycol], errors="coerce").fillna(0.0)
            tmp[f"mul_{met}"] = np.where(tmp[ycol] != 0, 1.0 / tmp[ycol], np.nan)

        rows = []
        g = tmp.groupby(["fund_id", "month_key"], sort=True)

        for (fund_id, month_key), grp in g:
            month_end_ts = month_key.to_timestamp("M").normalize()

            for met in ("P/S", "P/B", "P/E"):
                col = f"mul_{met}"
                med = grp[col].median(skipna=True)
                if np.isnan(med):
                    continue
                rows.append(
                    {"fund_id": int(fund_id), "month_end": month_end_ts, "segment": seg_name, "metric": met, "value": float(med)}
                )

        if not rows:
            return pd.DataFrame(columns=["fund_id", "month_end", "segment", "metric", "value"])
        return pd.DataFrame.from_records(rows)

    parts = []

    # Total
    parts.append(_median_for_segment("Total", base))

    # Financials
    fin = base[base["is_financial"]].copy()
    parts.append(_median_for_segment("Financials", fin))

    # Non-financials
    nfin = base[~base["is_financial"]].copy()
    parts.append(_median_for_segment("Non-financials", nfin))

    out = pd.concat([p for p in parts if p is not None and not p.empty], ignore_index=True)
    if out.empty:
        return pd.DataFrame(columns=["fund_id", "month_end", "segment", "metric", "value"])

    out["month_end"] = pd.to_datetime(out["month_end"], errors="coerce")
    out = out.dropna(subset=["month_end"])
    return out.sort_values(["fund_id", "segment", "metric", "month_end"]).reset_index(drop=True)



@st.cache_data(ttl=60 * 30, show_spinner=False)
def cached_portfolio_exposures_base_panel(
    fund_ids: list[int],
    start_date: dt.date,
    end_date: dt.date,
    mode: str,
) -> pd.DataFrame:
    """
    Base panel cache boundary = (sorted fund_ids, start_date, end_date, mode)

    Output columns:
      fund_id, month_key, isin, weight_pct, is_financial, ps_yield, pe_yield, pb_yield
    """
    fund_ids_norm = tuple(sorted(int(x) for x in fund_ids))
    return compute_portfolio_exposures_base_panel(
        fund_ids=list(fund_ids_norm),
        start_date=start_date,
        end_date=end_date,
        mode=mode,
    )


@st.cache_data(ttl=60 * 30, show_spinner=False)
def cached_portfolio_exposures_timeseries(
    fund_ids: list[int],
    focus_fund_id: int,
    start_date: dt.date,
    end_date: dt.date,
    segment_choice: str,
    metric_choice: str,
    mode: str,
) -> pd.DataFrame:
    return compute_portfolio_exposures_timeseries(
        fund_ids=fund_ids,
        focus_fund_id=focus_fund_id,
        start_date=start_date,
        end_date=end_date,
        segment_choice=segment_choice,
        metric_choice=metric_choice,
        mode=mode,
    )
