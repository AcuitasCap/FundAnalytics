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


def rebuild_stock_monthly_valuations(
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
    write_to_db: bool = True,
    chunk_size: int = 50_000,
    debug_anchor_isin: str | None = None,
):
    """
    Housekeeping job (rebuild stock_monthly_valuations as YIELDS):

    1) Reads monthly market cap from fundlab.stock_price
    2) Attaches TTM sales / PAT (from fundlab.stock_quarterly_financials, consolidated)
    3) Attaches latest annual book value (from fundlab.stock_annual_book_value, consolidated)
    4) Computes stock-level YIELDS with ONLY guardrail:
          - market_cap > 0 (else yield = NULL)
       No other adjustments, no caps, no dropping, negative/zero numerators allowed.
    5) Writes directly to fundlab.stock_monthly_valuations:
          - ps stores Sales Yield  = ttm_sales / market_cap
          - pe stores Earnings Yield = ttm_pat / market_cap
          - pb stores Book Yield   = book_value / market_cap

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

            aligned_sales[valid] = sub_ttm["ttm_sales"].values[pos[valid]]
            aligned_pat[valid] = sub_ttm["ttm_pat"].values[pos[valid]]

            base.loc[idx, "ttm_sales"] = aligned_sales
            base.loc[idx, "ttm_pat"] = aligned_pat

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
            aligned_bv[valid] = sub_bv["book_value"].values[pos[valid]]
            base.loc[idx, "book_value"] = aligned_bv

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
    # 4) Compute yields (ONLY guardrail: market_cap > 0)
    # --------------------------------------------------------------
    df = base.copy()

    mc = pd.to_numeric(df["market_cap"], errors="coerce")
    ttm_sales = pd.to_numeric(df["ttm_sales"], errors="coerce")
    ttm_pat = pd.to_numeric(df["ttm_pat"], errors="coerce")
    bv = pd.to_numeric(df["book_value"], errors="coerce")

    # Yields stored in ps/pe/pb (schema unchanged)
    df["ps"] = np.where(mc > 0, ttm_sales / mc, np.nan)   # sales yield
    df["pe"] = np.where(mc > 0, ttm_pat / mc, np.nan)     # earnings yield
    df["pb"] = np.where(mc > 0, bv / mc, np.nan)          # book yield

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

    st.success("stock_monthly_valuations rebuilt (yields) and written to Supabase successfully.")
    return None


def rebuild_fund_monthly_valuations(
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
    fund_ids: list[int] | None = None,
):
    """
    Housekeeping job (CSV version) — UPDATED FOR YIELDS IN stock_monthly_valuations:

    - Pulls Domestic Equity holdings (fundlab.fund_portfolio) joined to stock_master for is_financial
    - Join to stock_monthly_valuations by (isin, month_key) (year-month)
      NOTE: stock_monthly_valuations.ps/pe/pb now store YIELDS:
        ps = sales_yield    (ttm_sales / market_cap)
        pe = earnings_yield (ttm_pat / market_cap)
        pb = book_yield     (book_value / market_cap)

    - Compute fund-level multiples for Total / Financials / Non-financials by:
        1) Rebase weights within each segment to 100%
        2) Force NaN yields to 0.0
        3) Aggregate yield = SUM( w_seg * yield_i )
        4) Multiple = 1 / aggregate_yield, only undefined when aggregate_yield == 0
      No sign-based filters and no yield-based dropping/re-normalization.

    - Reads existing fund_monthly_valuations rows in range and skips those keys.
    - Outputs a CSV for manual Supabase import.
    """

    engine = get_engine()

    # --------------------------------------------------------------
    # 0) Derive date range from fund_portfolio if needed
    # --------------------------------------------------------------
    with engine.begin() as conn:
        if start_date is None or end_date is None:
            if fund_ids:
                rng = conn.execute(
                    text(
                        """
                        SELECT
                            MIN(month_end)::date AS min_d,
                            MAX(month_end)::date AS max_d
                        FROM fundlab.fund_portfolio
                        WHERE fund_id = ANY(:fund_ids)
                        """
                    ),
                    {"fund_ids": fund_ids},
                ).fetchone()
            else:
                rng = conn.execute(
                    text(
                        """
                        SELECT
                            MIN(month_end)::date AS min_d,
                            MAX(month_end)::date AS max_d
                        FROM fundlab.fund_portfolio
                        """
                    )
                ).fetchone()

            if not rng or rng.min_d is None or rng.max_d is None:
                st.warning("No data in fundlab.fund_portfolio to infer date range.")
                return

            if start_date is None:
                start_date = rng.min_d
            if end_date is None:
                end_date = rng.max_d

    if start_date > end_date:
        st.error(f"Invalid date range: start_date {start_date} > end_date {end_date}.")
        return

    # --------------------------------------------------------------
    # 1) Fetch holdings (Domestic Equities only) for the period
    # --------------------------------------------------------------
    with engine.begin() as conn:
        if fund_ids:
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
                  ON fp.isin = sm.isin
                WHERE fp.month_end BETWEEN :start_date AND :end_date
                  AND fp.asset_type = 'Domestic Equities'
                  AND fp.fund_id = ANY(:fund_ids)
                ORDER BY fp.fund_id, fp.month_end, fp.isin
                """
            )
            holdings = pd.read_sql(
                holdings_sql,
                conn,
                params={"start_date": start_date, "end_date": end_date, "fund_ids": fund_ids},
            )
        else:
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
                  ON fp.isin = sm.isin
                WHERE fp.month_end BETWEEN :start_date AND :end_date
                  AND fp.asset_type = 'Domestic Equities'
                ORDER BY fp.fund_id, fp.month_end, fp.isin
                """
            )
            holdings = pd.read_sql(
                holdings_sql,
                conn,
                params={"start_date": start_date, "end_date": end_date},
            )

    if holdings.empty:
        st.info("No fund_portfolio holdings found in the given period.")
        return

    holdings["isin"] = holdings["isin"].astype(str).str.strip()
    holdings["month_end"] = pd.to_datetime(holdings["month_end"], errors="coerce")
    holdings["weight_pct"] = pd.to_numeric(holdings["weight_pct"], errors="coerce")
    holdings = holdings.dropna(subset=["fund_id", "isin", "month_end", "weight_pct"])

    if holdings.empty:
        st.info("No usable holdings after cleaning.")
        return

    # --------------------------------------------------------------
    # 2) Prepare base weights WITHOUT re-normalization
    # --------------------------------------------------------------
    holdings["month_key"] = holdings["month_end"].dt.to_period("M")
    abs_sum_w = holdings.groupby(["fund_id", "month_key"])["weight_pct"].transform(lambda s: s.abs().sum())
    holdings["weight_scale"] = np.where(abs_sum_w > 2.0, 0.01, 1.0)
    holdings["w_base"] = holdings["weight_pct"] * holdings["weight_scale"]

    # canonical month_end (date) for output keys
    holdings["month_end"] = holdings["month_key"].dt.to_timestamp("M").dt.date

    # --------------------------------------------------------------
    # 3) Pull stock yields for relevant ISINs and date window
    #    and merge by (isin, month_key)
    # --------------------------------------------------------------
    all_isins = sorted(holdings["isin"].unique().tolist())
    min_month = pd.Timestamp(min(holdings["month_end"]))
    max_month = pd.Timestamp(max(holdings["month_end"]))

    with engine.begin() as conn:
        val_sql = text(
            """
            SELECT
                isin,
                month_end,
                ps,
                pe,
                pb
            FROM fundlab.stock_monthly_valuations
            WHERE isin = ANY(:isins)
              AND month_end BETWEEN :start_date AND :end_date
            """
        )
        vals = pd.read_sql(
            val_sql,
            conn,
            params={
                "isins": all_isins,
                "start_date": min_month.date(),
                "end_date": max_month.date(),
            },
        )

    if vals.empty:
        st.info("No stock_monthly_valuations rows found for the given holdings.")
        return

    vals["isin"] = vals["isin"].astype(str).str.strip()
    vals["month_end"] = pd.to_datetime(vals["month_end"], errors="coerce")
    vals = vals.dropna(subset=["month_end"])
    vals["month_key"] = vals["month_end"].dt.to_period("M")
    vals = ensure_unique_monthly_rows(
        vals,
        key_cols=["isin", "month_key"],
        value_cols=["ps", "pe", "pb"],
        context="fundlab.stock_monthly_valuations",
    )

    df = holdings.merge(
        vals[["isin", "month_key", "ps", "pe", "pb"]],
        on=["isin", "month_key"],
        how="left",
    )

    # Ensure numeric
    for c in ("ps", "pe", "pb", "w_base"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if df[["ps", "pe", "pb"]].isna().all(axis=None):
        st.info("All merged yields are NaN after month_key merge; nothing to compute.")
        return

    # --------------------------------------------------------------
    # 4) Load existing fund_monthly_valuations keys (incremental mode)
    # --------------------------------------------------------------
    with engine.begin() as conn:
        if fund_ids:
            existing_sql = text(
                """
                SELECT fund_id, month_end, segment
                FROM fundlab.fund_monthly_valuations
                WHERE month_end BETWEEN :start_date AND :end_date
                  AND fund_id = ANY(:fund_ids)
                """
            )
            existing_rows = conn.execute(
                existing_sql,
                {"start_date": start_date, "end_date": end_date, "fund_ids": fund_ids},
            ).fetchall()
        else:
            existing_sql = text(
                """
                SELECT fund_id, month_end, segment
                FROM fundlab.fund_monthly_valuations
                WHERE month_end BETWEEN :start_date AND :end_date
                """
            )
            existing_rows = conn.execute(
                existing_sql,
                {"start_date": start_date, "end_date": end_date},
            ).fetchall()

    existing_keys: set[tuple[int, dt.date, str]] = set()
    for r in existing_rows:
        existing_keys.add((int(r.fund_id), r.month_end, str(r.segment)))

    # --------------------------------------------------------------
    # 5) Compute fund-level multiples from yields.
    #    Logic aligned to the retired valuation diagnostic:
    #      - Rebase weights within segment to 100%
    #      - NaN yield is forced to 0.0
    #      - Multiple = 1 / SUM(w_seg * yield_filled)
    #      - Undefined only when aggregate yield == 0
    # --------------------------------------------------------------
    df["is_financial"] = df["is_financial"].astype(bool)

    def _seg_multiple_from_yield(
        seg_grp: pd.DataFrame, yield_col: str
    ) -> tuple[float, int, float]:
        """
        Returns (portfolio_multiple, nonmissing_stock_count, coverage_weight_nonmissing)

        Steps:
          - Force NaN yields to 0.0
          - coverage_weight_nonmissing = SUM(w_seg where original yield was non-missing)
          - Aggregate yield = SUM(w_seg * yield_filled)
          - Multiple = 1 / aggregate_yield (undefined when aggregate_yield == 0)
        """
        g = seg_grp.copy()
        g[yield_col] = pd.to_numeric(g[yield_col], errors="coerce")
        g["w_seg"] = pd.to_numeric(g["w_seg"], errors="coerce").fillna(0.0)

        nonmissing = g[yield_col].notna()
        coverage_w = float(g.loc[nonmissing, "w_seg"].sum())
        y_filled = g[yield_col].fillna(0.0)
        agg_yield = float(np.nansum(g["w_seg"] * y_filled))

        if np.isnan(agg_yield) or agg_yield == 0:
            return (np.nan, int(nonmissing.sum()), coverage_w)

        return (float(1.0 / agg_yield), int(nonmissing.sum()), coverage_w)

    segments = ["Total", "Financials", "Non-financials"]
    records: list[dict] = []

    grouped = df.groupby(["fund_id", "month_end"], sort=True)
    n_groups = len(grouped)

    progress = st.progress(0)
    status_placeholder = st.empty()

    if n_groups == 0:
        st.info("No (fund, month_end) groups after merge; nothing to compute.")
        return

    for i, ((fund_id, month_end_date), grp) in enumerate(grouped, start=1):
        grp = grp.copy()

        for seg in segments:
            key = (int(fund_id), month_end_date, seg)
            if key in existing_keys:
                continue

            if seg == "Financials":
                seg_grp = grp[grp["is_financial"]].copy()
            elif seg == "Non-financials":
                seg_grp = grp[~grp["is_financial"]].copy()
            else:
                seg_grp = grp.copy()

            if seg_grp.empty:
                continue

            # Rebase within segment so each segment is treated as a standalone portfolio
            seg_sum = float(pd.to_numeric(seg_grp["w_base"], errors="coerce").fillna(0.0).sum())
            if seg_sum <= 0:
                continue
            seg_grp = seg_grp.copy()
            seg_grp["w_seg"] = pd.to_numeric(seg_grp["w_base"], errors="coerce").fillna(0.0) / seg_sum

            ps_val, ps_cnt, ps_cov = _seg_multiple_from_yield(seg_grp, "ps")  # sales yield
            pe_val, pe_cnt, pe_cov = _seg_multiple_from_yield(seg_grp, "pe")  # earnings yield
            pb_val, pb_cnt, pb_cov = _seg_multiple_from_yield(seg_grp, "pb")  # book yield

            # If nothing computable, skip
            if np.isnan(ps_val) and np.isnan(pe_val) and np.isnan(pb_val):
                continue

            stock_count = int(max(ps_cnt, pe_cnt, pb_cnt))
            # Segment weights are rebased to 100%
            total_weight_seg = float(pd.to_numeric(seg_grp["w_seg"], errors="coerce").fillna(0.0).sum())

            records.append(
                {
                    "fund_id": int(fund_id),
                    "month_end": month_end_date,
                    "segment": seg,
                    "ps": None if np.isnan(ps_val) else float(ps_val),
                    "pe": None if np.isnan(pe_val) else float(pe_val),
                    "pb": None if np.isnan(pb_val) else float(pb_val),
                    "stock_count": stock_count,
                    "total_weight": total_weight_seg,
                    "notes": (
                        f"coverage_w(ps/pe/pb)={ps_cov:.3f}/{pe_cov:.3f}/{pb_cov:.3f}"
                    ),
                }
            )

        # progress UI
        if n_groups > 0:
            pct = int(i * 100 / n_groups)
            progress.progress(min(pct, 100))
            if i % 50 == 0 or i == n_groups:
                status_placeholder.text(f"Processed {i} / {n_groups} fund-month groups")

    progress.empty()
    status_placeholder.empty()

    if not records:
        st.success(
            f"No *new* fund_monthly_valuations rows needed between {start_date} and {end_date}. "
            f"Table already up to date for this range."
        )
        return

    df_out = pd.DataFrame.from_records(records)
    df_out.sort_values(["fund_id", "month_end", "segment"], inplace=True)

    st.write(
        f"Prepared **{len(df_out)}** new (fund_id, month_end, segment) valuation rows "
        f"between {start_date} and {end_date} that do *not* yet exist in "
        f"`fundlab.fund_monthly_valuations`."
    )
    st.dataframe(df_out.head(50))

    csv_bytes = df_out.to_csv(index=False).encode("utf-8")

    # Persist for housekeeping_page() so the download button survives reruns
    st.session_state["fund_valuations_csv_bytes"] = csv_bytes
    st.session_state["fund_valuations_csv_name"] = "fund_monthly_valuations_delta.csv"
    st.session_state["fund_valuations_rows"] = int(len(df_out))

    st.download_button(
        label="⬇️ Download fund_monthly_valuations CSV (new rows only)",
        data=csv_bytes,
        file_name="fund_monthly_valuations_delta.csv",
        mime="text/csv",
    )

    st.info(
        "Upload this CSV into Supabase (fundlab.fund_monthly_valuations). "
        "Because we only included rows that don't already exist for this date range, "
        "it will act as an incremental update."
    )


