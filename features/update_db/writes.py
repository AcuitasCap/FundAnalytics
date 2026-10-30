from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from core.db import get_engine
from .corporate_actions import upload_stock_corporate_actions
from .data import (
    CompanyNameAceConflictError, MissingPortfolioAceCodesError,
    find_company_name_ace_conflicts, find_conflicts_annual,
    find_missing_portfolio_ace_codes, load_ace_industry_flags,
)

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
    Upload cleaned fund portfolio data into fundlab.fund_portfolio_ace in batches.

    Expected canonical columns in df:
      - scheme_name
      - month_end      (python date)
      - holding_pct    (0–100)
      - asset_type
      - ace_code      (required for domestic equities, null otherwise)
      - instrument     (used for instrument_name in DB)
    """
    if df.empty:
        return
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    engine = get_engine()
    with engine.begin() as conn:
        # Check every domestic ACE code before inserting funds or holdings.
        missing = find_missing_portfolio_ace_codes(df, conn)
        if not missing.empty:
            raise MissingPortfolioAceCodesError(missing)

        # Ensure funds exist only after the complete ACE master check passes.
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

        # Batched insert into the ACE-keyed portfolio table.
        n = len(df)

        insert_sql = text("""
            INSERT INTO fundlab.fund_portfolio_ace (
                fund_id,
                month_end,
                instrument_name,
                holding_weight,
                asset_type,
                ace_code
            )
            SELECT
                f.fund_id,
                t.month_end,
                t.instrument_name,
                t.holding_weight,
                t.asset_type,
                t.ace_code
            FROM (
                SELECT
                    unnest(CAST(:scheme_names AS text[]))            AS scheme_name,
                    unnest(CAST(:month_ends AS date[]))              AS month_end,
                    unnest(CAST(:instrument_names AS text[]))        AS instrument_name,
                    unnest(CAST(:weights AS double precision[]))     AS holding_weight,
                    unnest(CAST(:asset_types AS text[]))             AS asset_type,
                    unnest(CAST(:ace_codes AS varchar[]))           AS ace_code
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
                "ace_codes":      [None if pd.isna(code) else str(code) for code in chunk["ace_code"]],
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


def upload_ace_stock_master(
    df_clean: pd.DataFrame, new_industry_flags: dict[str, bool]
) -> None:
    """Atomically update ACE stock master and company-name mapping."""
    if df_clean.empty:
        raise ValueError("The stock-master workbook has no rows to upload.")

    with get_engine().begin() as conn:
        existing_flags = load_ace_industry_flags(conn)
        needed = set(df_clean["industry_key"]) - set(existing_flags)
        unresolved = sorted(needed - set(new_industry_flags))
        if unresolved:
            raise ValueError(f"Classify these new industries before upload: {unresolved}")
        for key, flag in new_industry_flags.items():
            if type(flag) is not bool:
                raise ValueError(f"Industry classification must be confirmed as a boolean: {key}")
            if key in existing_flags and existing_flags[key] != flag:
                raise ValueError(f"Classification for '{key}' conflicts with existing stock master.")

        name_conflicts = find_company_name_ace_conflicts(df_clean, conn)
        if not name_conflicts.empty:
            raise CompanyNameAceConflictError(name_conflicts)

        master_sql = text("""
            INSERT INTO fundlab.stock_master_ace (ace_code, industry, is_financial)
            VALUES (:ace_code, :industry, :is_financial)
            ON CONFLICT (ace_code) DO UPDATE
            SET industry = EXCLUDED.industry,
                is_financial = EXCLUDED.is_financial
        """)
        for row in df_clean.drop_duplicates("ace_code").itertuples(index=False):
            conn.execute(master_sql, {
                "ace_code": row.ace_code,
                "industry": row.industry,
                "is_financial": existing_flags.get(
                    row.industry_key, new_industry_flags.get(row.industry_key)
                ),
            })

        name_sql = text("""
            INSERT INTO fundlab.company_name_acecode_map AS m (company_name, ace_code)
            VALUES (:company_name, :ace_code)
            ON CONFLICT (company_name) DO UPDATE
            SET ace_code = EXCLUDED.ace_code
            WHERE m.ace_code IS NULL OR m.ace_code = EXCLUDED.ace_code
        """)
        for row in df_clean.itertuples(index=False):
            result = conn.execute(name_sql, {
                "company_name": row.company_name,
                "ace_code": row.ace_code,
            })
            if result.rowcount != 1:
                raise ValueError(
                    f"Company name '{row.company_name}' was mapped to another ACE code "
                    "during upload. Transaction rolled back."
                )


def upload_roe_roce(df: pd.DataFrame):
    engine = get_engine()
    with engine.begin() as conn:
        ins = text("""
            INSERT INTO fundlab.stock_roe_roce_ace
                (ace_code, year_end_date, year_end, roe, roce)
            VALUES (:ace_code, :d, :year_end, :roe, :roce)
            ON CONFLICT (ace_code, year_end_date) DO UPDATE
            SET year_end = EXCLUDED.year_end,
                roe = COALESCE(EXCLUDED.roe, stock_roe_roce_ace.roe),
                roce = COALESCE(EXCLUDED.roce, stock_roe_roce_ace.roce)
        """)
        for _, r in df.iterrows():
            conn.execute(
                ins,
                {
                    "ace_code": r["ace_code"],
                    "d": r["year_end_date"],
                    "year_end": int(r["year_end"]),
                    "roe": None if pd.isna(r.get("roe")) else float(r["roe"]),
                    "roce": None if pd.isna(r.get("roce")) else float(r["roce"]),
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


def _upload_quarterly_metric(df_clean: pd.DataFrame, metric: str) -> None:
    """Upsert one ACE-keyed quarterly metric without changing the other."""
    if metric not in {"pat", "sales"}:
        raise ValueError(f"Unsupported quarterly metric: {metric}")
    if df_clean.empty:
        return
    if df_clean.duplicated(subset=["ace_code", "period_end"]).any():
        raise ValueError("Duplicate ACE code + period_end rows in quarterly upload.")

    insert_sql = text(
        f"""
        INSERT INTO fundlab.stock_quarterly_financials_ace AS q
            (ace_code, period_end, {metric})
        SELECT t.ace_code, t.period_end, t.metric_value
        FROM unnest(
            CAST(:ace_codes AS varchar[]),
            CAST(:period_ends AS date[]),
            CAST(:values AS numeric[])
        ) AS t(ace_code, period_end, metric_value)
        ON CONFLICT (ace_code, period_end) DO UPDATE
        SET {metric} = EXCLUDED.{metric}
        WHERE q.{metric} IS DISTINCT FROM EXCLUDED.{metric}
        """
    )

    n = len(df_clean)
    progress_bar, progress_text = _batch_progress()
    with get_engine().begin() as conn:
        for start in range(0, n, 10_000):
            end = min(start + 10_000, n)
            chunk = df_clean.iloc[start:end]
            conn.execute(insert_sql, {
                "ace_codes": chunk["ace_code"].astype(str).tolist(),
                "period_ends": [pd.Timestamp(value).date() for value in chunk["period_end"]],
                "values": [float(value) for value in chunk[metric]],
            })
            progress_bar.progress(end / n)
            progress_text.text(f"Uploading {metric.upper()}… {end} / {n} rows processed")

    progress_text.text(f"{metric.upper()} upload complete: {n} rows processed.")


def upload_quarterly_pat(df_clean: pd.DataFrame) -> None:
    _upload_quarterly_metric(df_clean, "pat")


def upload_quarterly_sales(df_clean: pd.DataFrame) -> None:
    _upload_quarterly_metric(df_clean, "sales")


def upload_annual_book_value(df_clean: pd.DataFrame) -> None:
    if df_clean.empty:
        return

    engine = get_engine()
    df_clean = df_clean.drop_duplicates(subset=["ace_code", "year_end"])

    conflicts = find_conflicts_annual(df_clean)
    if conflicts:
        sample = "; ".join([f"{r.ace_code} @ {r.year_end}" for r in conflicts])
        raise ValueError(
            "Duplicate book value data detected: These ACE code + year-end rows already exist.\n"
            f"Examples: {sample}\n\n"
            "Upload aborted. No rows inserted."
        )

    n = len(df_clean)
    BATCH_SIZE = 10000
    if n == 0:
        return

    insert_sql = text(
        """
        INSERT INTO fundlab.stock_annual_bv_ace (
            ace_code,
            year_end,
            book_value
        )
        SELECT
            t.ace_code,
            t.year_end,
            t.book_value
        FROM (
            SELECT
                unnest(CAST(:ace_codes AS varchar[])) AS ace_code,
                unnest(CAST(:year_ends AS date[])) AS year_end,
                unnest(CAST(:bvals AS numeric[])) AS book_value
        ) t
        """
    )

    progress_bar, progress_text = _batch_progress()

    with engine.begin() as conn:
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            chunk = df_clean.iloc[start:end]

            params = {
                "ace_codes":    list(chunk["ace_code"].astype(str)),
                "year_ends":    [value.date() for value in chunk["year_end"]],
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
