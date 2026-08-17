from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text

EXCEPTION_COLUMNS = [
    "isin",
    "action_type",
    "ex_date",
    "denominator",
    "numerator",
    "computed_multiplier",
    "mapped_month_start",
    "mapped_month_end_price_date",
    "reason",
]


def _empty_exceptions_df() -> pd.DataFrame:
    return pd.DataFrame(columns=EXCEPTION_COLUMNS)


def _to_month_start(value: dt.date | pd.Timestamp) -> dt.date:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    d = ts.date()
    return d.replace(day=1)


def compute_action_multiplier(row: pd.Series) -> float:
    action_type = str(row.get("action_type", "")).strip().upper()
    denominator = row.get("denominator")
    numerator = row.get("numerator")

    d = pd.to_numeric(pd.Series([denominator]), errors="coerce").iloc[0]
    n = pd.to_numeric(pd.Series([numerator]), errors="coerce").iloc[0]
    if pd.isna(d) or pd.isna(n) or d <= 0 or n <= 0:
        raise ValueError("invalid_ratio")

    if action_type == "BONUS":
        return float((d + n) / d)
    if action_type == "SPLIT":
        return float(n / d)
    raise ValueError(f"unsupported_action_type:{action_type}")


def build_month_end_map(price_df: pd.DataFrame) -> pd.DataFrame:
    p = price_df.copy()
    p["price_date"] = pd.to_datetime(p["price_date"], errors="coerce").dt.date
    p = p.dropna(subset=["price_date"])
    p["mapped_month_start"] = p["price_date"].apply(_to_month_start)
    map_df = (
        p.groupby(["isin", "mapped_month_start"], as_index=False)["price_date"]
        .max()
        .rename(columns={"price_date": "mapped_month_end_price_date"})
    )
    return map_df


def map_actions_to_month_end(actions_df: pd.DataFrame, month_end_map: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if actions_df is None or actions_df.empty:
        return pd.DataFrame(columns=["isin", "mapped_month_end_price_date", "computed_multiplier"]), _empty_exceptions_df()

    a = actions_df.copy()
    a["action_type"] = a["action_type"].astype(str).str.strip().str.upper()
    a["ex_date"] = pd.to_datetime(a["ex_date"], errors="coerce").dt.date
    a["mapped_month_start"] = a["ex_date"].apply(_to_month_start)

    multipliers: list[float | None] = []
    invalid_ratio = []
    for _, row in a.iterrows():
        try:
            multipliers.append(compute_action_multiplier(row))
            invalid_ratio.append(False)
        except ValueError:
            multipliers.append(None)
            invalid_ratio.append(True)
    a["computed_multiplier"] = multipliers
    a["invalid_ratio"] = invalid_ratio

    merged = a.merge(
        month_end_map[["isin", "mapped_month_start", "mapped_month_end_price_date"]],
        on=["isin", "mapped_month_start"],
        how="left",
    )
    merged["mapped_month_end_price_date"] = pd.to_datetime(
        merged["mapped_month_end_price_date"], errors="coerce"
    ).dt.date

    exc_invalid = merged[merged["invalid_ratio"]].copy()
    if not exc_invalid.empty:
        exc_invalid["reason"] = "invalid_ratio"

    exc_unmapped = merged[
        (~merged["invalid_ratio"]) & merged["mapped_month_end_price_date"].isna()
    ].copy()
    if not exc_unmapped.empty:
        exc_unmapped["reason"] = "missing_stock_price_month_row"

    exceptions = pd.concat([exc_invalid, exc_unmapped], ignore_index=True)
    if exceptions.empty:
        exceptions_df = _empty_exceptions_df()
    else:
        exceptions_df = exceptions[EXCEPTION_COLUMNS].copy()

    valid_mapped = merged[
        (~merged["invalid_ratio"]) & merged["mapped_month_end_price_date"].notna()
    ].copy()

    month_multiplier_df = (
        valid_mapped.groupby(["isin", "mapped_month_end_price_date"], as_index=False)["computed_multiplier"]
        .prod()
        .rename(columns={"computed_multiplier": "month_multiplier"})
    )
    return month_multiplier_df, exceptions_df


def compute_cumulative_multipliers(price_df: pd.DataFrame, month_multiplier_df: pd.DataFrame) -> pd.DataFrame:
    if price_df is None or price_df.empty:
        return pd.DataFrame(columns=["isin", "price_date", "price", "adj_multiplier", "adj_price"])

    p = price_df.copy()
    p["price_date"] = pd.to_datetime(p["price_date"], errors="coerce").dt.date
    p["price"] = pd.to_numeric(p["price"], errors="coerce")
    p = p.sort_values(["isin", "price_date"]).reset_index(drop=True)

    mm = month_multiplier_df.copy()
    if mm.empty:
        mm = pd.DataFrame(columns=["isin", "mapped_month_end_price_date", "month_multiplier"])
    mm["mapped_month_end_price_date"] = pd.to_datetime(mm["mapped_month_end_price_date"], errors="coerce").dt.date
    mm["month_multiplier"] = pd.to_numeric(mm["month_multiplier"], errors="coerce")

    month_mult_map: dict[tuple[str, dt.date], float] = {
        (str(r["isin"]), r["mapped_month_end_price_date"]): float(r["month_multiplier"])
        for _, r in mm.dropna(subset=["mapped_month_end_price_date", "month_multiplier"]).iterrows()
    }

    out_parts = []
    for isin, sub in p.groupby("isin", sort=False):
        running = 1.0
        rows = sub.copy()
        multipliers = []
        for d in rows["price_date"]:
            step = month_mult_map.get((str(isin), d))
            if step is not None:
                running *= float(step)
            multipliers.append(running)
        rows["adj_multiplier"] = multipliers
        rows["adj_price"] = rows["price"] * rows["adj_multiplier"]
        rows.loc[rows["price"].isna(), "adj_price"] = pd.NA
        out_parts.append(rows)

    out = pd.concat(out_parts, ignore_index=True)
    return out[["isin", "price_date", "price", "adj_multiplier", "adj_price"]]


def refresh_adjusted_prices(engine: Any) -> tuple[dict, pd.DataFrame]:
    price_sql = text(
        """
        select isin, price_date::date as price_date, price
        from fundlab.stock_price
        order by isin, price_date
        """
    )
    action_sql = text(
        """
        select isin, action_type, ex_date::date as ex_date, denominator, numerator
        from fundlab.stock_corporate_action
        order by isin, ex_date
        """
    )

    with engine.begin() as conn:
        price_df = pd.read_sql(price_sql, conn)
        actions_df = pd.read_sql(action_sql, conn)

    if price_df.empty:
        summary = {
            "actions_total": int(len(actions_df)),
            "actions_mapped": 0,
            "actions_unmapped": int(len(actions_df)),
            "unique_isins_actions": int(actions_df["isin"].nunique()) if not actions_df.empty else 0,
            "price_rows_total": 0,
            "price_rows_updated": 0,
            "min_price_date": None,
            "max_price_date": None,
            "exceptions": int(len(actions_df)),
        }
        if actions_df.empty:
            return summary, _empty_exceptions_df()
        exc = actions_df.copy()
        exc["computed_multiplier"] = pd.NA
        exc["mapped_month_start"] = pd.to_datetime(exc["ex_date"], errors="coerce").dt.date
        exc["mapped_month_end_price_date"] = pd.NA
        exc["reason"] = "missing_stock_price_month_row"
        return summary, exc[EXCEPTION_COLUMNS]

    month_end_map = build_month_end_map(price_df)
    month_multiplier_df, exceptions_df = map_actions_to_month_end(actions_df, month_end_map)
    adjusted_df = compute_cumulative_multipliers(price_df, month_multiplier_df)

    invariant_exceptions = []
    candidates = adjusted_df[adjusted_df["price"].notna()].copy()
    unique_isins = candidates["isin"].dropna().astype(str).unique().tolist()
    sample_size = min(5, len(unique_isins))
    if sample_size > 0:
        rng = pd.Series(unique_isins).sample(n=sample_size, random_state=42).tolist()
        sample_rows = candidates[candidates["isin"].astype(str).isin(rng)].copy()
        ratio = sample_rows["adj_price"] / sample_rows["price"]
        mismatch = (ratio - sample_rows["adj_multiplier"]).abs() > 1e-9
        bad_rows = sample_rows[mismatch].head(25)
        for _, r in bad_rows.iterrows():
            invariant_exceptions.append(
                {
                    "isin": r["isin"],
                    "action_type": "INVARIANT",
                    "ex_date": r["price_date"],
                    "denominator": pd.NA,
                    "numerator": pd.NA,
                    "computed_multiplier": pd.NA,
                    "mapped_month_start": _to_month_start(r["price_date"]),
                    "mapped_month_end_price_date": r["price_date"],
                    "reason": "invariant_mismatch",
                }
            )

    if invariant_exceptions:
        inv_df = pd.DataFrame(invariant_exceptions, columns=EXCEPTION_COLUMNS)
        if exceptions_df.empty:
            exceptions_df = inv_df
        else:
            exceptions_df = pd.concat([exceptions_df, inv_df], ignore_index=True)

    rows = adjusted_df[["isin", "price_date", "adj_multiplier", "adj_price"]].to_dict(orient="records")
    chunk_size = 3000
    updated_rows = 0

    with engine.begin() as conn:
        for i in range(0, len(rows), chunk_size):
            chunk = rows[i : i + chunk_size]
            if not chunk:
                continue
            params = {}
            values_sql = []
            for j, row in enumerate(chunk, start=1):
                values_sql.append(f"(:isin{j}, :price_date{j}, :adj_multiplier{j}, :adj_price{j})")
                params[f"isin{j}"] = row["isin"]
                params[f"price_date{j}"] = row["price_date"]
                params[f"adj_multiplier{j}"] = float(row["adj_multiplier"]) if pd.notna(row["adj_multiplier"]) else None
                params[f"adj_price{j}"] = float(row["adj_price"]) if pd.notna(row["adj_price"]) else None

            update_sql = sa.text(
                f"""
                update fundlab.stock_price sp
                set
                  adj_multiplier = v.adj_multiplier,
                  adj_price      = v.adj_price
                from (values {", ".join(values_sql)}) as v(isin, price_date, adj_multiplier, adj_price)
                where sp.isin = v.isin
                  and sp.price_date = v.price_date
                """
            )
            result = conn.execute(update_sql, params)
            if result.rowcount is not None and result.rowcount >= 0:
                updated_rows += int(result.rowcount)
            else:
                updated_rows += len(chunk)

    # actions_mapped should count only corporate actions successfully mapped.
    mapped_actions = int(
        len(actions_df)
        - len(exceptions_df[exceptions_df["reason"].isin(["missing_stock_price_month_row", "invalid_ratio"])])
    )
    if mapped_actions < 0:
        mapped_actions = 0

    summary = {
        "actions_total": int(len(actions_df)),
        "actions_mapped": mapped_actions,
        "actions_unmapped": int(len(actions_df) - mapped_actions),
        "unique_isins_actions": int(actions_df["isin"].nunique()) if not actions_df.empty else 0,
        "price_rows_total": int(len(price_df)),
        "price_rows_updated": int(updated_rows),
        "min_price_date": str(pd.to_datetime(price_df["price_date"], errors="coerce").min().date()),
        "max_price_date": str(pd.to_datetime(price_df["price_date"], errors="coerce").max().date()),
        "exceptions": int(len(exceptions_df)),
    }
    return summary, exceptions_df.reset_index(drop=True)
