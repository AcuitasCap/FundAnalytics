"""Explicit production-changing Housekeeping maintenance jobs."""

from io import BytesIO
import datetime as dt
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import streamlit as st
import sqlalchemy as sa
from sqlalchemy import text

from .adjusted_prices import refresh_adjusted_prices as _refresh_adjusted_prices_full
from .dividend_yields import EXCEPTION_COLUMNS, compute_dividend_yield_updates, subtract_years_safe
from services.performance_db import load_bench_rolling, load_fund_rolling
from core.db import get_engine
from core.dataframes import ensure_unique_monthly_rows
from .compute import ROLLING_WINDOWS_MONTHS, _compute_rolling_cagr_from_monthly_nav, _prepare_monthly_nav_series
from .data import _latest_rolling_asof_map

def _insert_incremental_rolling_rows(
    conn,
    records_df: pd.DataFrame,
    table_name: str,
    id_col: str,
    temp_table_name: str,
) -> int:
    if records_df.empty:
        return 0

    conn.execute(text(f"DROP TABLE IF EXISTS {temp_table_name};"))
    conn.execute(text(f"CREATE TEMP TABLE {temp_table_name} AS SELECT * FROM {table_name} WITH NO DATA;"))
    records_df.to_sql(temp_table_name, conn, if_exists="append", index=False)
    result = conn.execute(text(f"""
        INSERT INTO {table_name} ({id_col}, window_months, asof_date, rolling_cagr)
        SELECT {id_col}, window_months, asof_date, rolling_cagr
        FROM {temp_table_name}
        ON CONFLICT ({id_col}, window_months, asof_date) DO NOTHING;
    """))
    return int(result.rowcount or 0)


def refresh_precomputed_rolling_returns() -> dict:
    """
    Incrementally compute and store 1Y / 3Y rolling returns for funds and benchmarks.

    Existing rows in Supabase are left untouched. Only rows with asof_date greater than
    the latest stored date for each entity/window are inserted.
    """
    engine = get_engine()

    with engine.begin() as conn:
        fund_navs = pd.read_sql(text("""
            SELECT fund_id, nav_date::date AS nav_date, nav_value
            FROM fundlab.fund_nav
            ORDER BY fund_id, nav_date
        """), conn, parse_dates=["nav_date"])

        bench_navs = pd.read_sql(text("""
            SELECT bench_id, nav_date::date AS nav_date, nav_value
            FROM fundlab.bench_nav
            ORDER BY bench_id, nav_date
        """), conn, parse_dates=["nav_date"])

        fund_latest = _latest_rolling_asof_map(conn, "fundlab.fund_rolling_return", "fund_id")
        bench_latest = _latest_rolling_asof_map(conn, "fundlab.bench_rolling_return", "bench_id")

        fund_series_map = _prepare_monthly_nav_series(fund_navs, "fund_id", "nav_date", "nav_value")
        bench_series_map = _prepare_monthly_nav_series(bench_navs, "bench_id", "nav_date", "nav_value")

        fund_records = []
        bench_records = []
        summary = {
            "windows_months": list(ROLLING_WINDOWS_MONTHS),
            "fund_entities_seen": len(fund_series_map),
            "benchmark_entities_seen": len(bench_series_map),
            "fund_rows_candidate": 0,
            "benchmark_rows_candidate": 0,
            "fund_rows_inserted": 0,
            "benchmark_rows_inserted": 0,
            "fund_entities_updated": 0,
            "benchmark_entities_updated": 0,
            "fund_entities_already_current": 0,
            "benchmark_entities_already_current": 0,
        }

        for fund_id, nav_series in fund_series_map.items():
            inserted_for_entity = False
            for months in ROLLING_WINDOWS_MONTHS:
                rolling = _compute_rolling_cagr_from_monthly_nav(nav_series, months).dropna()
                if rolling.empty:
                    continue

                latest_asof = fund_latest.get((fund_id, months))
                if latest_asof is not None:
                    rolling = rolling[rolling.index > latest_asof]

                if rolling.empty:
                    summary["fund_entities_already_current"] += 1
                    continue

                inserted_for_entity = True
                for asof_date, rolling_cagr in rolling.items():
                    fund_records.append(
                        {
                            "fund_id": fund_id,
                            "window_months": months,
                            "asof_date": pd.Timestamp(asof_date).date(),
                            "rolling_cagr": float(rolling_cagr),
                        }
                    )

            if inserted_for_entity:
                summary["fund_entities_updated"] += 1

        for bench_id, nav_series in bench_series_map.items():
            inserted_for_entity = False
            for months in ROLLING_WINDOWS_MONTHS:
                rolling = _compute_rolling_cagr_from_monthly_nav(nav_series, months).dropna()
                if rolling.empty:
                    continue

                latest_asof = bench_latest.get((bench_id, months))
                if latest_asof is not None:
                    rolling = rolling[rolling.index > latest_asof]

                if rolling.empty:
                    summary["benchmark_entities_already_current"] += 1
                    continue

                inserted_for_entity = True
                for asof_date, rolling_cagr in rolling.items():
                    bench_records.append(
                        {
                            "bench_id": bench_id,
                            "window_months": months,
                            "asof_date": pd.Timestamp(asof_date).date(),
                            "rolling_cagr": float(rolling_cagr),
                        }
                    )

            if inserted_for_entity:
                summary["benchmark_entities_updated"] += 1

        fund_df = pd.DataFrame.from_records(fund_records)
        bench_df = pd.DataFrame.from_records(bench_records)
        summary["fund_rows_candidate"] = int(len(fund_df))
        summary["benchmark_rows_candidate"] = int(len(bench_df))

        summary["fund_rows_inserted"] = _insert_incremental_rolling_rows(
            conn,
            fund_df,
            "fundlab.fund_rolling_return",
            "fund_id",
            "tmp_fund_roll_incremental",
        )
        summary["benchmark_rows_inserted"] = _insert_incremental_rolling_rows(
            conn,
            bench_df,
            "fundlab.bench_rolling_return",
            "bench_id",
            "tmp_bench_roll_incremental",
        )

    load_fund_rolling.clear()
    load_bench_rolling.clear()
    return summary


def refresh_adjusted_prices() -> tuple[dict, pd.DataFrame]:
    """
    Full-history refresh of adjusted multipliers/prices in fundlab.stock_price.
    Updates only adj_multiplier and adj_price.
    """
    return _refresh_adjusted_prices_full(get_engine())


def refresh_stock_dividend_yields(scope: str = "All") -> tuple[dict, pd.DataFrame]:
    """
    Returns:
      summary: dict
      exceptions_df: DataFrame (may be empty)
    """
    scope_clean = str(scope).strip()
    if scope_clean not in {"All", "Last 3 years"}:
        raise ValueError("scope must be one of: 'All', 'Last 3 years'")

    date_from = None
    if scope_clean == "Last 3 years":
        today_ist = datetime.now(ZoneInfo("Asia/Kolkata")).date()
        date_from = subtract_years_safe(today_ist, 3)

    engine = get_engine()

    mapped_sql = sa.text(
        """
        select
          d.isin,
          d.ex_month_start,
          d.month_dps,
          d.min_ex_date,
          d.max_ex_date,
          d.n_events,
          pm.prev_month_end_date,
          sp.price as base_price
        from (
          select
            isin,
            date_trunc('month', ex_date)::date as ex_month_start,
            sum(dps)::numeric as month_dps,
            min(ex_date)::date as min_ex_date,
            max(ex_date)::date as max_ex_date,
            count(*)::int as n_events
          from fundlab.stock_dividend
          where (:date_from is null or ex_date >= :date_from)
          group by 1,2
        ) d
        left join lateral (
          select max(price_date)::date as prev_month_end_date
          from fundlab.stock_price
          where isin = d.isin
            and price_date >= (d.ex_month_start - interval '1 month')
            and price_date < d.ex_month_start
        ) pm on true
        left join fundlab.stock_price sp
          on sp.isin = d.isin
         and sp.price_date = pm.prev_month_end_date
        """
    )

    with engine.begin() as conn:
        mapped_df = pd.read_sql(mapped_sql, conn, params={"date_from": date_from})

    if mapped_df.empty:
        summary = {
            "scope": scope_clean,
            "date_from": date_from.isoformat() if date_from else None,
            "rows_aggregated": 0,
            "updates_attempted": 0,
            "updates_applied": 0,
            "exceptions": 0,
        }
        return summary, pd.DataFrame(columns=EXCEPTION_COLUMNS)

    updates_df, exceptions_df = compute_dividend_yield_updates(mapped_df)

    updates_applied = 0
    chunk_size = 1000

    if not updates_df.empty:
        with engine.begin() as conn:
            for start in range(0, len(updates_df), chunk_size):
                chunk = updates_df.iloc[start : start + chunk_size]
                values_sql_parts = []
                params = {}
                for i, row in enumerate(chunk.itertuples(index=False), start=1):
                    values_sql_parts.append(f"(:isin{i}, :price_date{i}, :yield{i})")
                    params[f"isin{i}"] = row.isin
                    params[f"price_date{i}"] = row.price_date
                    params[f"yield{i}"] = float(row.dividend_yield)

                update_sql = sa.text(
                    f"""
                    update fundlab.stock_price sp
                    set dividend_yield = v.dividend_yield
                    from (values {", ".join(values_sql_parts)}) as v(isin, price_date, dividend_yield)
                    where sp.isin = v.isin
                      and sp.price_date = v.price_date
                    """
                )
                result = conn.execute(update_sql, params)
                if result.rowcount is not None and result.rowcount >= 0:
                    updates_applied += int(result.rowcount)
                else:
                    updates_applied += int(len(chunk))

    summary = {
        "scope": scope_clean,
        "date_from": date_from.isoformat() if date_from else None,
        "rows_aggregated": int(len(mapped_df)),
        "updates_attempted": int(len(updates_df)),
        "updates_applied": int(updates_applied),
        "exceptions": int(len(exceptions_df)),
    }
    return summary, exceptions_df


def upload_stock_monthly_valuations_from_excel(uploaded_file, batch_size: int = 2000):
    """
    Upload a single precomputed stock valuations Excel workbook (.xlsx)
    into fundlab.stock_monthly_valuations using batched executemany.

    Assumptions:
    - fundlab.stock_monthly_valuations exists with columns:
        isin text,
        month_end date,
        ttm_sales numeric,
        ttm_pat numeric,
        book_value numeric,
        ps numeric,
        pe numeric,
        pb numeric
      and is currently empty (or we don't care about duplicates yet).
    - The Excel was generated by rebuild_stock_monthly_valuations and
      has columns: isin, month_end, ttm_sales, ttm_pat,
                   book_value, ps, pe, pb
    """

    if uploaded_file is None:
        st.warning("Please select an Excel file first.")
        return

    # ------------------------------------------------------------------
    # 1) Read Excel
    # ------------------------------------------------------------------
    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Could not read Excel file: {e}")
        return

    if df.empty:
        st.warning("The uploaded workbook has no rows.")
        return

    # Normalise column names
    df.columns = [str(c).strip().lower() for c in df.columns]

    required_cols = [
        "isin",
        "month_end",
        "ttm_sales",
        "ttm_pat",
        "book_value",
        "ps",
        "pe",
        "pb",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(
            "Uploaded workbook is missing required columns: "
            + ", ".join(missing)
        )
        st.write("Columns found:", list(df.columns))
        return

    # Keep only required columns in correct order
    df = df[required_cols].copy()

    # ------------------------------------------------------------------
    # 2) Type cleanup
    # ------------------------------------------------------------------
    df["isin"] = df["isin"].astype(str).str.strip()

    df["month_end"] = pd.to_datetime(
        df["month_end"], errors="coerce"
    ).dt.date

    for col in ["ttm_sales", "ttm_pat", "book_value", "ps", "pe", "pb"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows without isin or month_end
    df = df.dropna(subset=["isin", "month_end"]).reset_index(drop=True)

    n_rows = len(df)
    if n_rows == 0:
        st.warning("No valid rows (isin + month_end) to upload after cleaning.")
        return

    st.write(f"Preparing to upload **{n_rows:,}** valuation rows…")

    # Replace NaNs with None so psycopg2 sends NULLs
    df = df.where(pd.notnull(df), None)

    # ------------------------------------------------------------------
    # 3) Prepare insert statement
    # ------------------------------------------------------------------
    insert_sql = text(
        """
        INSERT INTO fundlab.stock_monthly_valuations (
            isin,
            month_end,
            ttm_sales,
            ttm_pat,
            book_value,
            ps,
            pe,
            pb
        )
        VALUES (
            :isin,
            :month_end,
            :ttm_sales,
            :ttm_pat,
            :book_value,
            :ps,
            :pe,
            :pb
        );
        """
    )

    engine = get_engine()
    progress = st.progress(0.0, text="Uploading stock valuations…")
    inserted = 0

    try:
        with engine.begin() as conn:
            for start_idx in range(0, n_rows, batch_size):
                end_idx = min(start_idx + batch_size, n_rows)
                chunk = df.iloc[start_idx:end_idx]

                params = []
                for _, r in chunk.iterrows():
                    params.append(
                        {
                            "isin": r["isin"],
                            "month_end": r["month_end"],
                            "ttm_sales": r["ttm_sales"],
                            "ttm_pat": r["ttm_pat"],
                            "book_value": r["book_value"],
                            "ps": r["ps"],
                            "pe": r["pe"],
                            "pb": r["pb"],
                        }
                    )

                if params:
                    conn.execute(insert_sql, params)
                    inserted += len(params)
                    progress.progress(
                        inserted / n_rows,
                        text=(
                            f"Uploading stock valuations… "
                            f"{inserted:,} / {n_rows:,} rows"
                        ),
                    )

    except Exception as e:
        st.error("❌ Error while inserting into stock_monthly_valuations.")
        st.write("Python exception:", repr(e))
        orig = getattr(e, "orig", None)
        if orig is not None:
            st.write("DBAPI .orig:", repr(orig))
        return

    progress.progress(1.0, text="Upload complete.")
    st.success(f"✅ Uploaded {inserted:,} rows into fundlab.stock_monthly_valuations.")


def recompute_size_bands(batch_size: int = 10000):
    """
    For each end-June and end-December in stock_price, rank by market_cap
    and assign size bands: top 100 Large, next 150 Mid, rest Small.
    Upserts into fundlab.stock_size_band.
    """
    engine = get_engine()
    with engine.begin() as conn:
        prices = pd.read_sql(
            """
            SELECT isin, price_date, market_cap
            FROM fundlab.stock_price
            WHERE EXTRACT(MONTH FROM price_date) IN (6, 12)
            ORDER BY price_date, market_cap DESC
            """,
            conn,
        )

    if prices.empty:
        st.warning("No stock_price data for June/December month-ends.")
        return

    # Rank by market cap within each date
    prices["rank_by_mcap"] = (
        prices.groupby("price_date")["market_cap"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    def band_for_rank(r):
        if r <= 100:
            return "Large"
        elif r <= 250:
            return "Mid"
        else:
            return "Small"

    prices["size_band"] = prices["rank_by_mcap"].apply(band_for_rank)
    prices = prices.rename(columns={"price_date": "band_date"})

    # Batched upsert
    insert_sql = text("""
        INSERT INTO fundlab.stock_size_band (
            isin,
            band_date,
            size_band,
            rank_by_mcap,
            market_cap
        )
        SELECT
            unnest(:isins),
            unnest(:band_dates),
            unnest(:bands),
            unnest(:ranks),
            unnest(:mcaps)
        ON CONFLICT (isin, band_date)
        DO UPDATE
        SET size_band    = EXCLUDED.size_band,
            rank_by_mcap = EXCLUDED.rank_by_mcap,
            market_cap   = EXCLUDED.market_cap
    """)

    with engine.begin() as conn:
        n = len(prices)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            chunk = prices.iloc[start:end]
            params = {
                "isins":      list(chunk["isin"].astype(str)),
                "band_dates": list(chunk["band_date"]),
                "bands":      list(chunk["size_band"].astype(str)),
                "ranks":      list(chunk["rank_by_mcap"].astype(int)),
                "mcaps":      [float(x) for x in chunk["market_cap"]],
            }
            conn.execute(insert_sql, params)


def recompute_quality_medians(batch_size: int = 10000):
    """
    For each stock and each month_end, compute median of last 5 RoE/RoCE
    observations (by year_end_date <= month_end) and store in
    fundlab.stock_quality_median.
    """
    engine = get_engine()
    with engine.begin() as conn:
        roe = pd.read_sql(
            """
            SELECT isin, year_end_date, roe, roce
            FROM fundlab.stock_roe_roce
            ORDER BY isin, year_end_date
            """,
            conn,
        )
        months = pd.read_sql(
            """
            SELECT DISTINCT month_end
            FROM fundlab.fund_portfolio
            ORDER BY month_end
            """,
            conn,
        )

    if roe.empty or months.empty:
        st.warning("RoE/RoCE or portfolio month-end data is empty.")
        return

    months_list = list(months["month_end"])
    rows = []

    grouped = roe.groupby("isin", sort=False)
    for isin, g in grouped:
        g = g.sort_values("year_end_date")
        for m in months_list:
            hist = g[g["year_end_date"] <= m].tail(5)
            if hist.empty:
                continue
            rows.append(
                {
                    "isin": isin,
                    "month_end": m,
                    "median_roe_5y": hist["roe"].median(skipna=True),
                    "median_roce_5y": hist["roce"].median(skipna=True),
                }
            )

    if not rows:
        st.warning("No 5-year medians could be computed.")
        return

    med_df = pd.DataFrame(rows)

    insert_sql = text("""
        INSERT INTO fundlab.stock_quality_median (
            isin,
            month_end,
            median_roe_5y,
            median_roce_5y
        )
        SELECT
            unnest(:isins),
            unnest(:month_ends),
            unnest(:med_roe),
            unnest(:med_roce)
        ON CONFLICT (isin, month_end)
        DO UPDATE
        SET median_roe_5y  = EXCLUDED.median_roe_5y,
            median_roce_5y = EXCLUDED.median_roce_5y
    """)

    engine = get_engine()
    with engine.begin() as conn:
        n = len(med_df)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            chunk = med_df.iloc[start:end]
            params = {
                "isins":      list(chunk["isin"].astype(str)),
                "month_ends": list(chunk["month_end"]),
                "med_roe":    [None if pd.isna(x) else float(x) for x in chunk["median_roe_5y"]],
                "med_roce":   [None if pd.isna(x) else float(x) for x in chunk["median_roce_5y"]],
            }
            conn.execute(insert_sql, params)


def recompute_quality_quartiles(batch_size: int = 10000):
    """
    For each month_end, label each stock as Q1–Q4 within its
    (is_financial, size_band) bucket based on 5y median RoE/RoCE.
    """
    engine = get_engine()
    with engine.begin() as conn:
        med = pd.read_sql(
            """
            SELECT isin, month_end, median_roe_5y, median_roce_5y
            FROM fundlab.stock_quality_median
            ORDER BY isin, month_end
            """,
            conn,
        )
        size = pd.read_sql(
            """
            SELECT isin, band_date, size_band
            FROM fundlab.stock_size_band
            ORDER BY isin, band_date
            """,
            conn,
        )
        master = pd.read_sql(
            """
            SELECT isin, is_financial
            FROM fundlab.stock_master
            """,
            conn,
        )

    if med.empty or size.empty:
        st.warning("Quality medians or size bands are empty.")
        return

    # Map size_band to each (isin, month_end) using last band_date <= month_end
    months = med["month_end"].sort_values().unique()
    size_records = []

    for isin, g in size.groupby("isin", sort=False):
        g = g.sort_values("band_date")
        for m in months:
            sub = g[g["band_date"] <= m]
            if sub.empty:
                continue
            band = sub.iloc[-1]["size_band"]
            size_records.append(
                {"isin": isin, "month_end": m, "size_band": band}
            )

    size_for_month = pd.DataFrame(size_records)

    full = (
        med.merge(size_for_month, on=["isin", "month_end"], how="inner")
           .merge(master, on="isin", how="left")
    )
    full["is_financial"] = full["is_financial"].fillna(False)

    rows = []

    for m, g_m in full.groupby("month_end", sort=False):
        for fin_flag in (True, False):
            g_f = g_m[g_m["is_financial"] == fin_flag]
            for band in ("Large", "Mid", "Small"):
                g_b = g_f[g_f["size_band"] == band]
                if g_b.empty:
                    continue

                if fin_flag:
                    g_b = g_b.assign(quality_metric=g_b["median_roe_5y"])
                else:
                    g_b = g_b.assign(quality_metric=g_b["median_roce_5y"])

                g_b = g_b[~g_b["quality_metric"].isna()]
                if g_b.empty:
                    continue

                # Assign quartiles: Q1 highest quality
                try:
                    # Use rank so ties handled deterministically
                    rank_series = g_b["quality_metric"].rank(
                        method="first", ascending=False
                    )
                    q_labels = pd.qcut(
                        rank_series,
                        4,
                        labels=["Q1", "Q2", "Q3", "Q4"],
                    )
                except ValueError:
                    # Too few stocks for qcut; manual thresholds
                    ranks = g_b["quality_metric"].rank(
                        method="first", ascending=False
                    )
                    n = len(ranks)

                    def q_of_rank(r):
                        if r <= 0.25 * n:
                            return "Q1"
                        elif r <= 0.5 * n:
                            return "Q2"
                        elif r <= 0.75 * n:
                            return "Q3"
                        else:
                            return "Q4"

                    q_labels = ranks.apply(q_of_rank)

                for isin_val, qm, qlab in zip(
                    g_b["isin"], g_b["quality_metric"], q_labels
                ):
                    rows.append(
                        {
                            "isin": isin_val,
                            "month_end": m,
                            "size_band": band,
                            "is_financial": fin_flag,
                            "quality_metric": float(qm),
                            "quality_quartile": str(qlab),
                        }
                    )

    if not rows:
        st.warning("No quartile labels could be computed.")
        return

    quart_df = pd.DataFrame(rows)

    insert_sql = text("""
        INSERT INTO fundlab.stock_quality_quartile (
            isin,
            month_end,
            size_band,
            is_financial,
            quality_metric,
            quality_quartile
        )
        SELECT
            unnest(:isins),
            unnest(:month_ends),
            unnest(:bands),
            unnest(:fin_flags),
            unnest(:metrics),
            unnest(:quartiles)
        ON CONFLICT (isin, month_end)
        DO UPDATE
        SET size_band        = EXCLUDED.size_band,
            is_financial     = EXCLUDED.is_financial,
            quality_metric   = EXCLUDED.quality_metric,
            quality_quartile = EXCLUDED.quality_quartile
    """)

    with engine.begin() as conn:
        n = len(quart_df)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            chunk = quart_df.iloc[start:end]
            params = {
                "isins":      list(chunk["isin"].astype(str)),
                "month_ends": list(chunk["month_end"]),
                "bands":      list(chunk["size_band"].astype(str)),
                "fin_flags":  list(chunk["is_financial"].astype(bool)),
                "metrics":    [float(x) for x in chunk["quality_metric"]],
                "quartiles":  list(chunk["quality_quartile"].astype(str)),
            }
            conn.execute(insert_sql, params)


def _calculate_stock_valuation_multiples(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Apply freshness, validity and clipping rules to aligned stock inputs."""
    out = df.copy()
    mc = pd.to_numeric(out["market_cap"], errors="coerce")
    ttm_sales = pd.to_numeric(out["ttm_sales"], errors="coerce")
    ttm_pat = pd.to_numeric(out["ttm_pat"], errors="coerce")
    bv = pd.to_numeric(out["book_value"], errors="coerce")
    q_anchor = pd.to_datetime(out["quarter_anchor"], errors="coerce")
    b_anchor = pd.to_datetime(out["book_anchor"], errors="coerce")
    month_end = pd.to_datetime(out["month_end"], errors="coerce")
    quarterly_fresh = q_anchor.notna() & (q_anchor >= month_end - pd.DateOffset(months=3))
    book_fresh = b_anchor.notna() & (b_anchor >= month_end - pd.DateOffset(months=12))

    diagnostics: dict[str, dict[str, int]] = {}
    for label, denominator, fresh, column, lower, upper in (
        ("P/S", ttm_sales, quarterly_fresh, "ps", 0.5, 30.0),
        ("P/E", ttm_pat, quarterly_fresh, "pe", 5.0, 200.0),
        ("P/B", bv, book_fresh, "pb", 0.5, 40.0),
    ):
        valid = mc.notna() & (mc > 0) & denominator.notna() & (denominator > 0) & fresh
        raw = mc / denominator
        out[column] = np.where(valid, raw.clip(lower=lower, upper=upper), np.nan)
        diagnostics[label] = {
            "missing": int(denominator.isna().sum()),
            "non_positive": int((denominator.notna() & (denominator <= 0)).sum()),
            "stale": int((denominator.notna() & ~fresh).sum()),
            "invalid_market_cap": int((mc.isna() | (mc <= 0)).sum()),
            "capped": int((out[column].notna() & ((raw < lower) | (raw > upper))).sum()),
        }
    return out, diagnostics


def rebuild_stock_monthly_valuations(
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
    write_to_db: bool = True,
    chunk_size: int = 50_000,
    debug_anchor_isin: str | None = None,
):
    """
    Rebuild stock_monthly_valuations as stock-level valuation multiples.

    1) Reads monthly market cap from fundlab.stock_price
    2) Attaches TTM sales / PAT (from fundlab.stock_quarterly_financials, consolidated)
    3) Attaches latest annual book value (from fundlab.stock_annual_book_value, consolidated)
    4) Uses only fundamentals strictly before each valuation month (no look-ahead),
       rejects missing, non-positive or stale inputs, then clips valid multiples.
    5) Writes directly to fundlab.stock_monthly_valuations:
          - ps stores P/S = market_cap / TTM sales, clipped to [0.5, 30]
          - pe stores P/E = market_cap / TTM PAT, clipped to [5, 200]
          - pb stores P/B = market_cap / book value, clipped to [0.5, 40]

    Output columns (aligned to fundlab.stock_monthly_valuations):
      isin, month_end, ttm_sales, ttm_pat, book_value, ps, pe, pb
    """
    engine = get_engine()

    # --------------------------------------------------------------
    # 0) Auto-derive start/end if not given, from stock_price
    # --------------------------------------------------------------
    if start_date is None or end_date is None:
        with engine.begin() as conn:
            rng = conn.execute(
                text(
                    """
                    SELECT
                        MIN(price_date)::date AS min_d,
                        MAX(price_date)::date AS max_d
                    FROM fundlab.stock_price;
                    """
                )
            ).fetchone()

        if not rng or rng.min_d is None or rng.max_d is None:
            st.warning("No data in fundlab.stock_price to infer date range.")
            return

        if start_date is None:
            start_date = rng.min_d
        if end_date is None:
            end_date = rng.max_d

    st.write(f"Using price date range: {start_date} → {end_date}")

    # --------------------------------------------------------------
    # 1) Fetch monthly market cap
    # --------------------------------------------------------------
    with engine.begin() as conn:
        price_sql = text(
            """
            SELECT
                isin,
                price_date::date AS month_end,
                market_cap
            FROM fundlab.stock_price
            WHERE price_date BETWEEN :start_date AND :end_date
              AND market_cap IS NOT NULL;
            """
        )
        prices = pd.read_sql(
            price_sql,
            conn,
            params={"start_date": start_date, "end_date": end_date},
        )

    if prices.empty:
        st.info("No stock_price data found in the given period for stock valuations.")
        return

    prices["isin"] = prices["isin"].astype(str).str.strip()
    prices["month_end"] = pd.to_datetime(prices["month_end"], errors="coerce").dt.normalize()
    prices["market_cap"] = pd.to_numeric(prices["market_cap"], errors="coerce")

    prices = prices.dropna(subset=["isin", "month_end", "market_cap"])
    prices = ensure_unique_monthly_rows(
        prices,
        key_cols=["isin", "month_end"],
        value_cols=["market_cap"],
        context="fundlab.stock_price",
    )
    if prices.empty:
        st.info("After cleaning, no usable price/market_cap rows remain.")
        return

    base = prices.copy()

    all_isins = sorted(base["isin"].unique().tolist())
    min_month = base["month_end"].min()
    max_month = base["month_end"].max()

    st.write(f"Distinct ISINs in price data: {len(all_isins)}")
    st.write(f"Month_end range in price data: {min_month.date()} → {max_month.date()}")

    # --------------------------------------------------------------
    # 2) Fetch quarterly & annual fundamentals once
    # --------------------------------------------------------------
    with engine.begin() as conn:
        q_sql = text(
            """
            SELECT isin, period_end, sales, pat
            FROM fundlab.stock_quarterly_financials
            WHERE isin = ANY(:isins)
              AND is_consolidated = TRUE
              AND period_end <= :end_date
            ORDER BY isin, period_end;
            """
        )
        qdf = pd.read_sql(
            q_sql,
            conn,
            params={"isins": all_isins, "end_date": max_month},
        )

        bv_sql = text(
            """
            SELECT isin, year_end, book_value
            FROM fundlab.stock_annual_book_value
            WHERE isin = ANY(:isins)
              AND is_consolidated = TRUE
              AND year_end <= :end_date
            ORDER BY isin, year_end;
            """
        )
        bvdf = pd.read_sql(
            bv_sql,
            conn,
            params={"isins": all_isins, "end_date": max_month},
        )

    # ---- Quarterly data / TTM ----
    if not qdf.empty:
        qdf["isin"] = qdf["isin"].astype(str).str.strip()
        qdf["period_end"] = pd.to_datetime(qdf["period_end"], errors="coerce").dt.normalize()
        qdf["sales"] = pd.to_numeric(qdf["sales"], errors="coerce")
        qdf["pat"] = pd.to_numeric(qdf["pat"], errors="coerce")

        qdf = qdf.dropna(subset=["isin", "period_end"]).sort_values(["isin", "period_end"])

        qdf["ttm_sales"] = (
            qdf.groupby("isin")["sales"]
            .rolling(window=4, min_periods=4)
            .sum()
            .reset_index(level=0, drop=True)
        )
        qdf["ttm_pat"] = (
            qdf.groupby("isin")["pat"]
            .rolling(window=4, min_periods=4)
            .sum()
            .reset_index(level=0, drop=True)
        )

        qdf_ttm = qdf[["isin", "period_end", "ttm_sales", "ttm_pat"]].dropna(
            how="all", subset=["ttm_sales", "ttm_pat"]
        )
        qdf_ttm = qdf_ttm.sort_values(["isin", "period_end"]).reset_index(drop=True)
    else:
        qdf_ttm = pd.DataFrame(columns=["isin", "period_end", "ttm_sales", "ttm_pat"])

    # ---- Annual BV ----
    if not bvdf.empty:
        bvdf["isin"] = bvdf["isin"].astype(str).str.strip()
        bvdf["year_end"] = pd.to_datetime(bvdf["year_end"], errors="coerce").dt.normalize()
        bvdf["book_value"] = pd.to_numeric(bvdf["book_value"], errors="coerce")
        bvdf = bvdf.dropna(subset=["isin", "year_end"]).sort_values(["isin", "year_end"]).reset_index(drop=True)
    else:
        bvdf = pd.DataFrame(columns=["isin", "year_end", "book_value"])

    # --------------------------------------------------------------
    # 3) Attach TTM sales / PAT and BV to (isin, month_end)
    # --------------------------------------------------------------
    base = base.reset_index(drop=True)
    base["ttm_sales"] = np.nan
    base["ttm_pat"] = np.nan
    base["book_value"] = np.nan
    base["quarter_anchor"] = pd.NaT
    base["book_anchor"] = pd.NaT

    # ---- TTM sales/PAT via "as-of" alignment (vectorized per ISIN) ----
    if not qdf_ttm.empty:
        for isin, sub_ttm in qdf_ttm.groupby("isin", sort=False):
            mask = base["isin"] == isin
            if not mask.any():
                continue

            idx = base.index[mask]
            h_dates = base.loc[idx, "month_end"].values.astype("datetime64[ns]")
            q_dates = sub_ttm["period_end"].values.astype("datetime64[ns]")

            # IMPORTANT: use side="left" to enforce fundamentals date < month_end (no lookahead at quarter ends)
            pos = np.searchsorted(q_dates, h_dates, side="left") - 1
            valid = pos >= 0
            if not np.any(valid):
                continue

            aligned_sales = np.full(h_dates.shape, np.nan)
            aligned_pat = np.full(h_dates.shape, np.nan)
            aligned_anchors = np.full(h_dates.shape, np.datetime64("NaT"), dtype="datetime64[ns]")

            aligned_sales[valid] = sub_ttm["ttm_sales"].values[pos[valid]]
            aligned_pat[valid] = sub_ttm["ttm_pat"].values[pos[valid]]
            aligned_anchors[valid] = q_dates[pos[valid]]

            base.loc[idx, "ttm_sales"] = aligned_sales
            base.loc[idx, "ttm_pat"] = aligned_pat
            base.loc[idx, "quarter_anchor"] = aligned_anchors

    # ---- Book value via "as-of" alignment ----
    if not bvdf.empty:
        for isin, sub_bv in bvdf.groupby("isin", sort=False):
            mask = base["isin"] == isin
            if not mask.any():
                continue

            idx = base.index[mask]
            h_dates = base.loc[idx, "month_end"].values.astype("datetime64[ns]")
            b_dates = sub_bv["year_end"].values.astype("datetime64[ns]")

            # IMPORTANT: use side="left" to enforce fundamentals date < month_end (no lookahead at quarter ends)
            pos = np.searchsorted(b_dates, h_dates, side="left") - 1
            valid = pos >= 0
            if not np.any(valid):
                continue

            aligned_bv = np.full(h_dates.shape, np.nan)
            aligned_anchors = np.full(h_dates.shape, np.datetime64("NaT"), dtype="datetime64[ns]")
            aligned_bv[valid] = sub_bv["book_value"].values[pos[valid]]
            aligned_anchors[valid] = b_dates[pos[valid]]
            base.loc[idx, "book_value"] = aligned_bv
            base.loc[idx, "book_anchor"] = aligned_anchors

    if debug_anchor_isin:
        _isin = str(debug_anchor_isin).strip()
        q_anchor_dates = [pd.Timestamp("2024-09-30"), pd.Timestamp("2024-10-31")]
        b_anchor_dates = [pd.Timestamp("2023-03-31"), pd.Timestamp("2023-04-30")]

        sub_ttm_dbg = qdf_ttm[qdf_ttm["isin"] == _isin] if not qdf_ttm.empty else pd.DataFrame()
        if sub_ttm_dbg.empty:
            st.warning(f"[Anchor debug] No quarterly TTM rows found for ISIN={_isin}")
        else:
            q_dates_dbg = sub_ttm_dbg["period_end"].values.astype("datetime64[ns]")
            for month_end_dbg in q_anchor_dates:
                q_pos_dbg = np.searchsorted(
                    q_dates_dbg, month_end_dbg.to_datetime64(), side="left"
                ) - 1
                q_anchor_dbg = (
                    pd.Timestamp(q_dates_dbg[q_pos_dbg]).date() if q_pos_dbg >= 0 else None
                )
                st.write(
                    f"[Anchor debug] ISIN={_isin} month_end={month_end_dbg.date()} "
                    f"quarter_end_anchor={q_anchor_dbg}"
                )

        sub_bv_dbg = bvdf[bvdf["isin"] == _isin] if not bvdf.empty else pd.DataFrame()
        if sub_bv_dbg.empty:
            st.warning(f"[Anchor debug] No annual BV rows found for ISIN={_isin}")
        else:
            b_dates_dbg = sub_bv_dbg["year_end"].values.astype("datetime64[ns]")
            for month_end_dbg in b_anchor_dates:
                b_pos_dbg = np.searchsorted(
                    b_dates_dbg, month_end_dbg.to_datetime64(), side="left"
                ) - 1
                b_anchor_dbg = (
                    pd.Timestamp(b_dates_dbg[b_pos_dbg]).date() if b_pos_dbg >= 0 else None
                )
                st.write(
                    f"[Anchor debug] ISIN={_isin} month_end={month_end_dbg.date()} "
                    f"annual_end_anchor={b_anchor_dbg}"
                )

    # --------------------------------------------------------------
    # 4) Compute capped multiples.  A quarterly TTM may be at most three
    # calendar months old; annual book value may be at most twelve.
    # --------------------------------------------------------------
    df = base.copy()

    mc = pd.to_numeric(df["market_cap"], errors="coerce")
    ttm_sales = pd.to_numeric(df["ttm_sales"], errors="coerce")
    ttm_pat = pd.to_numeric(df["ttm_pat"], errors="coerce")
    bv = pd.to_numeric(df["book_value"], errors="coerce")

    df, diagnostics = _calculate_stock_valuation_multiples(df)
    for label, detail in diagnostics.items():
        missing = detail["missing"]
        non_positive = detail["non_positive"]
        stale = detail["stale"]
        bad_market_cap = detail["invalid_market_cap"]
        capped = detail["capped"]
        if missing or non_positive or stale or bad_market_cap or capped:
            st.warning(f"{label}: nulls from missing fundamentals={missing:,}, non-positive fundamentals={non_positive:,}, stale fundamentals={stale:,}, invalid market cap={bad_market_cap:,}; capped={capped:,}.")

    valuations_df = df[["isin", "month_end", "ttm_sales", "ttm_pat", "book_value", "ps", "pe", "pb"]].copy()

    valuations_df["isin"] = valuations_df["isin"].astype(str).str.strip()
    valuations_df["month_end"] = pd.to_datetime(valuations_df["month_end"], errors="coerce").dt.date

    valuations_df = valuations_df.dropna(subset=["isin", "month_end"])
    valuations_df = valuations_df.replace({np.inf: np.nan, -np.inf: np.nan})
    valuations_df = valuations_df.sort_values(["month_end", "isin"]).reset_index(drop=True)

    n_total = len(valuations_df)
    st.write(f"Total valuation rows prepared: {n_total:,}")
    if n_total == 0:
        st.info("No rows to write after processing.")
        return

    # --------------------------------------------------------------
    # 5) Write to Supabase (delete-range + insert chunks)
    #     Because stock_monthly_valuations has NO UNIQUE/PK constraint.
    # --------------------------------------------------------------
    if not write_to_db:
        st.info("write_to_db=False; skipping DB write.")
        return valuations_df

    with engine.begin() as conn:
        st.write("Deleting existing rows in target range from fundlab.stock_monthly_valuations...")
        conn.execute(
            text(
                """
                DELETE FROM fundlab.stock_monthly_valuations
                WHERE month_end BETWEEN :start_date AND :end_date;
                """
            ),
            {"start_date": start_date, "end_date": end_date},
        )

    st.write("Inserting new rows into fundlab.stock_monthly_valuations...")
    # Use pandas to_sql for bulk insert
    # Note: method="multi" batches INSERT VALUES lists; adjust chunk_size as needed.
    valuations_df.to_sql(
        "stock_monthly_valuations",
        con=engine,
        schema="fundlab",
        if_exists="append",
        index=False,
        chunksize=chunk_size,
        method="multi",
    )

    st.success("stock_monthly_valuations rebuilt (multiples) and written to Supabase successfully.")
    return None

