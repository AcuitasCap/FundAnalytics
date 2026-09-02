from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from core.db import get_engine
from .corporate_actions import upload_stock_corporate_actions

# Comment to refresh Streamlit 

class _NoopProgress:
    """Keeps legacy batch progress calls out of the database writer layer."""

    def progress(self, _value: float) -> None:
        return None

    def text(self, _value: str) -> None:
        return None


def _batch_progress() -> tuple[_NoopProgress, _NoopProgress]:
    return _NoopProgress(), _NoopProgress()

def upload_fund_navs(df: pd.DataFrame):
    engine = get_engine()
    with engine.begin() as conn:
        # Ensure fund names exist
        funds = sorted(set(df["fund_name"]))
        if funds:
            conn.execute(
                text("""
                    INSERT INTO fundlab.fund (fund_name)
                    SELECT unnest(:names)
                    ON CONFLICT (fund_name) DO NOTHING
                """),
                {"names": funds},
            )

        # Upsert NAVs
        ins = text("""
            INSERT INTO fundlab.fund_nav (fund_id, nav_date, nav_value)
            SELECT f.fund_id, :d, :v FROM fundlab.fund f WHERE f.fund_name = :n
            ON CONFLICT (fund_id, nav_date) DO UPDATE
            SET nav_value = EXCLUDED.nav_value
        """)
        for _, r in df.iterrows():
            conn.execute(
                ins,
                {"n": r["fund_name"], "d": r["nav_date"], "v": float(r["nav_value"])},
            )


def upload_bench_navs(df: pd.DataFrame):
    engine = get_engine()
    with engine.begin() as conn:
        benches = sorted(set(df["benchmark_name"]))
        if benches:
            conn.execute(
                text("""
                    INSERT INTO fundlab.benchmark (bench_name)
                    SELECT unnest(:names)
                    ON CONFLICT (bench_name) DO NOTHING
                """),
                {"names": benches},
            )

        ins = text("""
            INSERT INTO fundlab.bench_nav (bench_id, nav_date, nav_value)
            SELECT b.bench_id, :d, :v FROM fundlab.benchmark b WHERE b.bench_name = :n
            ON CONFLICT (bench_id, nav_date) DO UPDATE
            SET nav_value = EXCLUDED.nav_value
        """)
        for _, r in df.iterrows():
            conn.execute(
                ins,
                {"n": r["benchmark_name"], "d": r["nav_date"], "v": float(r["nav_value"])},
            )


def upload_fund_portfolios(df: pd.DataFrame, batch_size: int = 10000):
    """
    Upload cleaned fund portfolio data into fundlab.fund_portfolio in batches.

    Expected canonical columns in df:
      - scheme_name
      - month_end      (python date)
      - holding_pct    (0–100)
      - asset_type
      - isin
      - instrument     (used for instrument_name in DB)
    """
    engine = get_engine()
    with engine.begin() as conn:
        # 1) Ensure funds exist
        schemes = sorted(set(df["scheme_name"]))
        if schemes:
            conn.execute(
                text("""
                    INSERT INTO fundlab.fund (fund_name)
                    SELECT unnest(:names)
                    ON CONFLICT (fund_name) DO NOTHING
                """),
                {"names": schemes},
            )

        # 2) Batched insert into fund_portfolio
        n = len(df)
        if n == 0:
            return

        insert_sql = text("""
            INSERT INTO fundlab.fund_portfolio (
                fund_id,
                month_end,
                instrument_name,
                holding_weight,
                asset_type,
                isin
            )
            SELECT
                f.fund_id,
                t.month_end,
                t.instrument_name,
                t.holding_weight,
                t.asset_type,
                t.isin
            FROM (
                SELECT
                    -- Explicit array casts prevent PostgreSQL from inferring a
                    -- numeric type for the ISIN array when this statement is
                    -- prepared through the DB driver.
                    unnest(CAST(:scheme_names AS text[]))            AS scheme_name,
                    unnest(CAST(:month_ends AS date[]))              AS month_end,
                    unnest(CAST(:instrument_names AS text[]))        AS instrument_name,
                    unnest(CAST(:weights AS double precision[]))     AS holding_weight,
                    unnest(CAST(:asset_types AS text[]))             AS asset_type,
                    unnest(CAST(:isins AS text[]))                   AS isin
            ) t
            JOIN fundlab.fund f
              ON f.fund_name = t.scheme_name
            ON CONFLICT (fund_id, month_end, instrument_name, asset_type, holding_weight)
            DO NOTHING
        """)

        # Process in chunks
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            chunk = df.iloc[start:end]

            params = {
                "scheme_names":   list(chunk["scheme_name"].astype(str)),
                "month_ends":     list(chunk["month_end"]),        # python date → date[]
                "instrument_names": list(chunk["instrument"].astype(str)),
                "weights":        [float(x) for x in chunk["holding_pct"]],
                "asset_types":    list(chunk["asset_type"].astype(str)),
                "isins":          list(chunk["isin"].astype(str)),
            }

            conn.execute(insert_sql, params)


def upload_stock_master(df: pd.DataFrame):
    engine = get_engine()
    with engine.begin() as conn:
        ins = text("""
            INSERT INTO fundlab.stock_master (isin, company_name, industry, is_financial)
            VALUES (:isin, :name, :industry, :fin)
            ON CONFLICT (isin) DO UPDATE
            SET company_name = EXCLUDED.company_name,
                industry = EXCLUDED.industry,
                is_financial = EXCLUDED.is_financial
        """)
        for _, r in df.iterrows():
            conn.execute(
                ins,
                {
                    "isin": r["isin"],
                    "name": r["company_name"],
                    "industry": r.get("industry"),
                    "fin": bool(r["is_financial"]),
                },
            )


def upload_roe_roce(df: pd.DataFrame):
    engine = get_engine()
    with engine.begin() as conn:
        ins = text("""
            INSERT INTO fundlab.stock_roe_roce (isin, year_end_date, roe, roce, company_name)
            VALUES (:isin, :d, :roe, :roce, :name)
            ON CONFLICT (isin, year_end_date) DO UPDATE
            SET roe = COALESCE(EXCLUDED.roe, stock_roe_roce.roe),
                roce = COALESCE(EXCLUDED.roce, stock_roe_roce.roce),
                company_name = COALESCE(EXCLUDED.company_name, stock_roe_roce.company_name)
        """)
        for _, r in df.iterrows():
            conn.execute(
                ins,
                {
                    "isin": r["isin"],
                    "d": r["year_end_date"],
                    "roe": None if pd.isna(r.get("roe")) else float(r["roe"]),
                    "roce": None if pd.isna(r.get("roce")) else float(r["roce"]),
                    "name": r.get("company_name"),
                },
            )


def upload_stock_prices_mc(df: pd.DataFrame, batch_size: int = 10000):
    """
    Upload stock prices + market cap into fundlab.stock_price in batches.

    Expected canonical columns in df:
      - isin
      - price_date  (python date)
      - market_cap
      - price

    Logical key should be (isin, price_date). If the table currently allows multiple
    rows for the same ISIN/date, valuation rebuilds can be distorted downstream.
    """
    engine = get_engine()
    with engine.begin() as conn:
        n = len(df)
        if n == 0:
            return

        # IMPORTANT: the database should enforce a single row per (isin, price_date).
        insert_sql = text("""
            INSERT INTO fundlab.stock_price (
                isin,
                price_date,
                market_cap,
                price
            )
            SELECT
                unnest(:isins)       AS isin,
                unnest(:dates)       AS price_date,
                unnest(:mcaps)       AS market_cap,
                unnest(:prices)      AS price
            ON CONFLICT (isin, price_date, market_cap, price)
            DO NOTHING
        """)

        # Process in chunks
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            chunk = df.iloc[start:end]

            params = {
                "isins":  list(chunk["isin"].astype(str)),
                "dates":  list(chunk["price_date"]),                     # python date → date[]
                "mcaps":  [float(x) for x in chunk["market_cap"]],
                "prices": [float(x) for x in chunk["price"]],
            }

            conn.execute(insert_sql, params)


def upload_quarterly_pat(df_clean: pd.DataFrame) -> None:
    """
    Bulk upload quarterly PAT into fundlab.stock_quarterly_financials.

    Behaviour:
    - If (isin, period_end) does NOT exist → INSERT a new row with PAT.
    - If (isin, period_end) already exists → UPDATE PAT (and fiscal_year/quarter)
      for that row, leaving sales untouched.
    - Uses set-based INSERT/UPDATE via unnest for speed.
    """

    if df_clean.empty:
        return

    # Remove duplicates within the file
    df_clean = df_clean.drop_duplicates(subset=["isin", "period_end"]).copy()

    n = len(df_clean)
    if n == 0:
        return

    BATCH_SIZE = 10_000

    engine = get_engine()
    progress_bar, progress_text = _batch_progress()

    with engine.begin() as conn:
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            chunk = df_clean.iloc[start:end].copy()

            # 1) Find which (isin, period_end) already exist in DB
            keys = chunk[["isin", "period_end"]].drop_duplicates()
            existing_sql = text(
                """
                SELECT v.isin, v.period_end
                FROM (
                    SELECT
                        unnest(:isins)       AS isin,
                        unnest(:period_ends) AS period_end
                ) v
                JOIN fundlab.stock_quarterly_financials q
                  ON q.isin = v.isin
                 AND q.period_end = v.period_end
                 AND q.is_consolidated = TRUE
                """
            )
            key_params = {
                "isins":       list(keys["isin"].astype(str)),
                "period_ends": list(keys["period_end"]),
            }
            existing_rows = conn.execute(existing_sql, key_params).fetchall()
            existing_set = {(r.isin, r.period_end) for r in existing_rows}

            # 2) Split into new vs existing rows
            chunk["key"] = list(zip(chunk["isin"], chunk["period_end"]))
            mask_existing = chunk["key"].isin(existing_set)

            new_rows = chunk[~mask_existing].copy()
            upd_rows = chunk[mask_existing].copy()

            # 3) Insert new rows
            if not new_rows.empty:
                insert_sql = text(
                    """
                    INSERT INTO fundlab.stock_quarterly_financials (
                        isin,
                        period_end,
                        fiscal_year,
                        fiscal_quarter,
                        pat
                    )
                    SELECT
                        t.isin,
                        t.period_end,
                        t.fiscal_year,
                        t.fiscal_quarter,
                        t.pat
                    FROM (
                        SELECT
                            unnest(:isins)           AS isin,
                            unnest(:period_ends)     AS period_end,
                            unnest(:fiscal_years)    AS fiscal_year,
                            unnest(:fiscal_quarters) AS fiscal_quarter,
                            unnest(:pats)            AS pat
                    ) t
                    """
                )
                params_ins = {
                    "isins":           list(new_rows["isin"].astype(str)),
                    "period_ends":     list(new_rows["period_end"]),
                    "fiscal_years":    list(new_rows["fiscal_year"].astype(int)),
                    "fiscal_quarters": list(new_rows["fiscal_quarter"].astype(int)),
                    "pats":            [float(x) for x in new_rows["pat"]],
                }
                conn.execute(insert_sql, params_ins)

            # 4) Update existing rows (PAT only)
            if not upd_rows.empty:
                update_sql = text(
                    """
                    UPDATE fundlab.stock_quarterly_financials q
                    SET
                        pat          = v.pat,
                        fiscal_year  = v.fiscal_year,
                        fiscal_quarter = v.fiscal_quarter,
                        updated_at   = NOW()
                    FROM (
                        SELECT
                            unnest(:isins)           AS isin,
                            unnest(:period_ends)     AS period_end,
                            unnest(:fiscal_years)    AS fiscal_year,
                            unnest(:fiscal_quarters) AS fiscal_quarter,
                            unnest(:pats)            AS pat
                    ) v
                    WHERE q.isin = v.isin
                      AND q.period_end = v.period_end
                      AND q.is_consolidated = TRUE
                    """
                )
                params_upd = {
                    "isins":           list(upd_rows["isin"].astype(str)),
                    "period_ends":     list(upd_rows["period_end"]),
                    "fiscal_years":    list(upd_rows["fiscal_year"].astype(int)),
                    "fiscal_quarters": list(upd_rows["fiscal_quarter"].astype(int)),
                    "pats":            [float(x) for x in upd_rows["pat"]],
                }
                conn.execute(update_sql, params_upd)

            # Progress
            frac = end / n
            progress_bar.progress(frac)
            progress_text.text(f"Uploading PAT… {end} / {n} rows processed")

    progress_text.text(f"PAT upload complete: {n} rows processed (inserts + updates).")


def upload_quarterly_sales(df_clean: pd.DataFrame) -> None:
    """
    Bulk upload quarterly sales into fundlab.stock_quarterly_financials.

    Behaviour:
    - If (isin, period_end) does NOT exist → INSERT a new row with sales.
    - If (isin, period_end) already exists → UPDATE sales for that row,
      leaving PAT untouched.
    """

    if df_clean.empty:
        return

    df_clean = df_clean.drop_duplicates(subset=["isin", "period_end"]).copy()

    n = len(df_clean)
    if n == 0:
        return

    BATCH_SIZE = 10_000

    engine = get_engine()
    progress_bar, progress_text = _batch_progress()

    with engine.begin() as conn:
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            chunk = df_clean.iloc[start:end].copy()

            # 1) Which keys already exist?
            keys = chunk[["isin", "period_end"]].drop_duplicates()
            existing_sql = text(
                """
                SELECT v.isin, v.period_end
                FROM (
                    SELECT
                        unnest(:isins)       AS isin,
                        unnest(:period_ends) AS period_end
                ) v
                JOIN fundlab.stock_quarterly_financials q
                  ON q.isin = v.isin
                 AND q.period_end = v.period_end
                 AND q.is_consolidated = TRUE
                """
            )
            key_params = {
                "isins":       list(keys["isin"].astype(str)),
                "period_ends": list(keys["period_end"]),
            }
            existing_rows = conn.execute(existing_sql, key_params).fetchall()
            existing_set = {(r.isin, r.period_end) for r in existing_rows}

            # 2) Split into new vs existing
            chunk["key"] = list(zip(chunk["isin"], chunk["period_end"]))
            mask_existing = chunk["key"].isin(existing_set)

            new_rows = chunk[~mask_existing].copy()
            upd_rows = chunk[mask_existing].copy()

            # 3) Insert new rows
            if not new_rows.empty:
                insert_sql = text(
                    """
                    INSERT INTO fundlab.stock_quarterly_financials (
                        isin,
                        period_end,
                        fiscal_year,
                        fiscal_quarter,
                        sales
                    )
                    SELECT
                        t.isin,
                        t.period_end,
                        t.fiscal_year,
                        t.fiscal_quarter,
                        t.sales
                    FROM (
                        SELECT
                            unnest(:isins)           AS isin,
                            unnest(:period_ends)     AS period_end,
                            unnest(:fiscal_years)    AS fiscal_year,
                            unnest(:fiscal_quarters) AS fiscal_quarter,
                            unnest(:sales_vals)      AS sales
                    ) t
                    """
                )
                params_ins = {
                    "isins":           list(new_rows["isin"].astype(str)),
                    "period_ends":     list(new_rows["period_end"]),
                    "fiscal_years":    list(new_rows["fiscal_year"].astype(int)),
                    "fiscal_quarters": list(new_rows["fiscal_quarter"].astype(int)),
                    "sales_vals":      [float(x) for x in new_rows["sales"]],
                }
                conn.execute(insert_sql, params_ins)

            # 4) Update existing rows (sales only)
            if not upd_rows.empty:
                update_sql = text(
                    """
                    UPDATE fundlab.stock_quarterly_financials q
                    SET
                        sales        = v.sales,
                        fiscal_year  = v.fiscal_year,
                        fiscal_quarter = v.fiscal_quarter,
                        updated_at   = NOW()
                    FROM (
                        SELECT
                            unnest(:isins)           AS isin,
                            unnest(:period_ends)     AS period_end,
                            unnest(:fiscal_years)    AS fiscal_year,
                            unnest(:fiscal_quarters) AS fiscal_quarter,
                            unnest(:sales_vals)      AS sales
                    ) v
                    WHERE q.isin = v.isin
                      AND q.period_end = v.period_end
                      AND q.is_consolidated = TRUE
                    """
                )
                params_upd = {
                    "isins":           list(upd_rows["isin"].astype(str)),
                    "period_ends":     list(upd_rows["period_end"]),
                    "fiscal_years":    list(upd_rows["fiscal_year"].astype(int)),
                    "fiscal_quarters": list(upd_rows["fiscal_quarter"].astype(int)),
                    "sales_vals":      [float(x) for x in upd_rows["sales"]],
                }
                conn.execute(update_sql, params_upd)

            frac = end / n
            progress_bar.progress(frac)
            progress_text.text(f"Uploading sales… {end} / {n} rows processed")

    progress_text.text(f"Sales upload complete: {n} rows processed (inserts + updates).")


def upload_annual_book_value(df_clean: pd.DataFrame) -> None:
    if df_clean.empty:
        return

    engine = get_engine()
    df_clean = df_clean.drop_duplicates(subset=["isin", "year_end"])

    conflicts = find_conflicts_annual(df_clean)
    if conflicts:
        sample = "; ".join([f"{r.isin} @ {r.year_end.date()}" for r in conflicts])
        raise ValueError(
            "Duplicate book value data detected: These ISIN + year-end rows already exist.\n"
            f"Examples: {sample}\n\n"
            "Upload aborted. No rows inserted."
        )

    n = len(df_clean)
    BATCH_SIZE = 10000
    if n == 0:
        return

    insert_sql = text(
        """
        INSERT INTO fundlab.stock_annual_book_value (
            isin,
            year_end,
            fiscal_year,
            book_value
        )
        SELECT
            t.isin,
            t.year_end,
            t.fiscal_year,
            t.book_value
        FROM (
            SELECT
                unnest(:isins)        AS isin,
                unnest(:year_ends)    AS year_end,
                unnest(:fiscal_years) AS fiscal_year,
                unnest(:bvals)        AS book_value
        ) t
        """
    )

    progress_bar, progress_text = _batch_progress()

    with engine.begin() as conn:
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            chunk = df_clean.iloc[start:end]

            params = {
                "isins":        list(chunk["isin"].astype(str)),
                "year_ends":    list(chunk["year_end"]),
                "fiscal_years": list(chunk["fiscal_year"].astype(int)),
                "bvals":        [float(x) for x in chunk["book_value"]],
            }

            frac = (end / n)
            progress_bar.progress(frac)
            progress_text.text(f"Inserting book value… {end} / {n}")

            conn.execute(insert_sql, params)

    progress_text.text(f"Annual book value insert complete: {n} rows.")


def _category_name_to_id(engine) -> dict:
    df = pd.read_sql(
        """
        SELECT category_id, category_name
        FROM fundlab.category
        WHERE LOWER(category_name) <> 'portfolio'
        """,
        engine,
    )
    return dict(zip(df["category_name"], df["category_id"]))


def upload_fund_manager_tenure(df_clean: pd.DataFrame, resolutions: dict) -> None:
    """
    resolutions: dict keyed by NEW fund_name:
      {
        "ICICI Value Fund": {"old_name": "ICICI Value Discovery", "category_name": None},
        "Brand New Fund":   {"old_name": "", "category_name": "Flexi Cap Fund"},
      }
    """
    if df_clean.empty:
        return

    engine = get_engine()

    # Fresh master snapshots for validation
    fund_master = pd.read_sql("select fund_id, fund_name from fundlab.fund", engine)
    existing_names = set(fund_master["fund_name"].astype(str).str.strip())

    cat_map = _category_name_to_id(engine)
    category_options = set(cat_map.keys())

    with engine.begin() as conn:
        # 1) Apply rename / new-fund creation
        for new_name, info in (resolutions or {}).items():
            old_name = (info.get("old_name") or "").strip()
            category_name = info.get("category_name")

            if old_name:
                if old_name not in existing_names:
                    raise ValueError(f"Old fund name '{old_name}' not found in fund master for rename to '{new_name}'.")
                # Rename (fund_id unchanged)
                conn.execute(
                    text("UPDATE fundlab.fund SET fund_name = :new_name WHERE fund_name = :old_name"),
                    {"new_name": new_name, "old_name": old_name},
                )
                existing_names.discard(old_name)
                existing_names.add(new_name)
            else:
                if not category_name:
                    raise ValueError(f"Category is required to create new fund '{new_name}'.")
                if category_name not in category_options:
                    raise ValueError(f"Unknown category '{category_name}' for new fund '{new_name}'.")
                conn.execute(
                    text("""
                        INSERT INTO fundlab.fund (fund_name, category_id)
                        VALUES (:fund_name, :category_id)
                    """),
                    {"fund_name": new_name, "category_id": int(cat_map[category_name])},
                )
                existing_names.add(new_name)

        # 2) Resolve fund_id for all funds in this upload (post rename/insert)
        upload_funds = sorted(df_clean["fund_name"].unique().tolist())
        fund_rows = conn.execute(
            text("SELECT fund_id, fund_name FROM fundlab.fund WHERE fund_name = ANY(:names)"),
            {"names": upload_funds},
        ).fetchall()

        name_to_id = {r.fund_name: r.fund_id for r in fund_rows}
        missing_after = [n for n in upload_funds if n not in name_to_id]
        if missing_after:
            raise ValueError(f"Funds still missing in master after resolution: {missing_after}")

        df = df_clean.copy()
        df["fund_id"] = df["fund_name"].map(name_to_id).astype(int)

        affected_ids = sorted(df["fund_id"].unique().tolist())

        # 3) Delete existing tenure rows for affected funds
        conn.execute(
            text("DELETE FROM fundlab.fund_manager_tenure WHERE fund_id = ANY(:fund_ids)"),
            {"fund_ids": affected_ids},
        )

        # 4) Insert tenure rows (batch)
        # Convert NaN to None for to_date
        to_dates = []
        for x in df["to_date"].tolist():
            if pd.isna(x):
                to_dates.append(None)
            else:
                to_dates.append(x)

        insert_sql = text("""
            INSERT INTO fundlab.fund_manager_tenure
                (fund_id, inception_date, fund_manager, from_date, to_date)
            SELECT
                unnest(:fund_ids)        AS fund_id,
                unnest(:inceptions)      AS inception_date,
                unnest(:managers)        AS fund_manager,
                unnest(:from_dates)      AS from_date,
                unnest(:to_dates)        AS to_date
        """)

        conn.execute(
            insert_sql,
            {
                "fund_ids": list(df["fund_id"].astype(int)),
                "inceptions": [None if pd.isna(x) else x for x in df["inception_date"].tolist()],
                "managers": list(df["fund_manager"].astype(str)),
                "from_dates": list(df["from_date"]),
                "to_dates": to_dates,
            },
        )


def upload_stock_dividends(df: pd.DataFrame) -> None:
    """
    Upload dividend events into fundlab.stock_dividend.

    Expected columns in df:
      - isin (text)
      - ex_date (date or datetime)
      - dps (numeric)

    Table:
      fundlab.stock_dividend(isin text, ex_date date, dps numeric, PK(isin, ex_date))

    Behavior:
      - Upsert on (isin, ex_date): updates dps if already exists
      - Chunked inserts
    """
    import sqlalchemy as sa
    from sqlalchemy.exc import SQLAlchemyError

    if df is None or df.empty:
        raise ValueError("No dividend rows to upload.")

    required = {"isin", "ex_date", "dps"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dividend upload missing required columns: {sorted(missing)}")

    # Clean / normalize
    d = df.copy()
    d["isin"] = d["isin"].astype(str).str.strip()
    d["ex_date"] = pd.to_datetime(d["ex_date"], errors="coerce").dt.date
    d["dps"] = pd.to_numeric(d["dps"], errors="coerce")

    bad_isin = d["isin"].isna() | (d["isin"] == "") | (d["isin"].str.lower() == "nan")
    bad_date = d["ex_date"].isna()
    bad_dps = d["dps"].isna() | (d["dps"] < 0)

    if bad_isin.any() or bad_date.any() or bad_dps.any():
        n_bad = int((bad_isin | bad_date | bad_dps).sum())
        examples = d.loc[bad_isin | bad_date | bad_dps, ["isin", "ex_date", "dps"]].head(10)
        raise ValueError(
            f"Invalid dividend rows: {n_bad} rows have blank ISIN / invalid ex_date / invalid dps.\n"
            f"Examples:\n{examples.to_string(index=False)}"
        )

    # Drop exact duplicates within file (keep last)
    d = d.drop_duplicates(subset=["isin", "ex_date"], keep="last").reset_index(drop=True)

    engine = get_engine()

    upsert_sql = sa.text(
        """
        insert into fundlab.stock_dividend (isin, ex_date, dps)
        values (:isin, :ex_date, :dps)
        on conflict (isin, ex_date) do update
        set dps = excluded.dps
        """
    )

    # Chunked execute
    chunk_size = 5000
    rows = d[["isin", "ex_date", "dps"]].to_dict(orient="records")

    try:
        with engine.begin() as conn:
            for i in range(0, len(rows), chunk_size):
                conn.execute(upsert_sql, rows[i : i + chunk_size])
    except SQLAlchemyError as e:
        raise

def upload_corporate_actions(df: pd.DataFrame, source_file: str | None = None) -> dict:
    """Thin adapter retaining the dedicated corporate-actions writer."""
    return upload_stock_corporate_actions(df=df, source_file=source_file)
