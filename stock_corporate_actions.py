from __future__ import annotations

from typing import Any, Callable

import pandas as pd
from sqlalchemy import text

_ENGINE_GETTER: Callable[[], Any] | None = None


def set_engine_getter(getter: Callable[[], Any] | None) -> None:
    global _ENGINE_GETTER
    _ENGINE_GETTER = getter


def _resolve_engine():
    if _ENGINE_GETTER is not None:
        return _ENGINE_GETTER()
    try:
        import app13  # lazy import to avoid hard dependency during tests

        return app13.get_engine()
    except Exception as e:
        raise RuntimeError("Could not resolve database engine for corporate action upload.") from e


def _normalize_col_name(name: Any) -> str:
    s = str(name).strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = " ".join(s.split())
    return s


def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    alias_map = {
        "isin": {"isin"},
        "ex_date": {"ex date", "exdate"},
        "denominator": {"denominator", "denominator (d)", "denominator d", "denom", "d"},
        "numerator": {"numerator", "numerator (n)", "numerator n", "num", "n"},
    }

    normalized = {c: _normalize_col_name(c) for c in df.columns}
    rename: dict[Any, str] = {}
    used_physical: set[Any] = set()
    for logical, aliases in alias_map.items():
        for physical, normalized_name in normalized.items():
            if physical in used_physical:
                continue
            if normalized_name in aliases:
                rename[physical] = logical
                used_physical.add(physical)
                break

    out = df.rename(columns=rename)
    required = ["isin", "ex_date", "denominator", "numerator"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Expected logical columns: {required}")
    return out


def validate_stock_corporate_actions(
    df_raw: pd.DataFrame,
    action_type: str,
) -> tuple[pd.DataFrame, dict]:
    """
    Returns:
      df_clean with columns: isin, action_type, ex_date, denominator, numerator
      summary dict
    """
    if df_raw is None or df_raw.empty:
        raise ValueError("The uploaded file is empty.")

    action_type_clean = str(action_type).strip().upper()
    if action_type_clean not in {"BONUS", "SPLIT"}:
        raise ValueError("action_type must be one of: BONUS, SPLIT")

    df = _map_columns(df_raw.copy())
    out = df[["isin", "ex_date", "denominator", "numerator"]].copy()

    out["isin"] = out["isin"].astype(str).str.strip()
    out["ex_date"] = pd.to_datetime(out["ex_date"], errors="coerce").dt.date

    denom_num = pd.to_numeric(out["denominator"], errors="coerce")
    numer_num = pd.to_numeric(out["numerator"], errors="coerce")

    bad_isin = out["isin"].isna() | (out["isin"] == "") | (out["isin"].str.lower() == "nan")
    bad_ex_date = out["ex_date"].isna()
    bad_denom = denom_num.isna() | (denom_num <= 0) | ((denom_num % 1) != 0)
    bad_numer = numer_num.isna() | (numer_num <= 0) | ((numer_num % 1) != 0)

    invalid = bad_isin | bad_ex_date | bad_denom | bad_numer
    if invalid.any():
        examples = out.loc[invalid].copy().head(15)
        reasons = []
        for idx in examples.index:
            row_reasons = []
            if bool(bad_isin.loc[idx]):
                row_reasons.append("blank_isin")
            if bool(bad_ex_date.loc[idx]):
                row_reasons.append("invalid_ex_date")
            if bool(bad_denom.loc[idx]):
                row_reasons.append("invalid_denominator")
            if bool(bad_numer.loc[idx]):
                row_reasons.append("invalid_numerator")
            reasons.append(",".join(row_reasons))
        examples["reason"] = reasons
        raise ValueError(
            f"{int(invalid.sum())} invalid rows found (isin/ex_date/denominator/numerator checks failed).\n"
            f"Examples:\n{examples.to_string(index=False)}"
        )

    out["denominator"] = denom_num.astype(int)
    out["numerator"] = numer_num.astype(int)
    out["action_type"] = action_type_clean
    out = out[["isin", "action_type", "ex_date", "denominator", "numerator"]]

    before = len(out)
    out = out.drop_duplicates(subset=["isin", "ex_date"], keep="last").reset_index(drop=True)
    after = len(out)

    summary = {
        "rows_in_file": int(before),
        "rows_after_dedup": int(after),
        "action_type": action_type_clean,
        "unique_isins": int(out["isin"].nunique()),
        "min_ex_date": str(out["ex_date"].min()),
        "max_ex_date": str(out["ex_date"].max()),
        "rows_upserted": int(after),
    }
    return out, summary


def upload_stock_corporate_actions(
    df: pd.DataFrame,
    source_file: str | None = None,
) -> dict:
    """
    Upserts into fundlab.stock_corporate_action with overwrite-on-conflict.
    Returns summary dict.
    """
    if df is None or df.empty:
        raise ValueError("No corporate action rows to upload.")

    required = {"isin", "action_type", "ex_date", "denominator", "numerator"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Corporate action upload missing required columns: {sorted(missing)}")

    d = df.copy()
    d["isin"] = d["isin"].astype(str).str.strip()
    d["action_type"] = d["action_type"].astype(str).str.strip().str.upper()
    d["ex_date"] = pd.to_datetime(d["ex_date"], errors="coerce").dt.date
    d["denominator"] = pd.to_numeric(d["denominator"], errors="coerce")
    d["numerator"] = pd.to_numeric(d["numerator"], errors="coerce")

    bad_isin = d["isin"].isna() | (d["isin"] == "") | (d["isin"].str.lower() == "nan")
    bad_action = ~d["action_type"].isin(["BONUS", "SPLIT"])
    bad_date = d["ex_date"].isna()
    bad_denom = d["denominator"].isna() | (d["denominator"] <= 0) | ((d["denominator"] % 1) != 0)
    bad_numer = d["numerator"].isna() | (d["numerator"] <= 0) | ((d["numerator"] % 1) != 0)
    invalid = bad_isin | bad_action | bad_date | bad_denom | bad_numer
    if invalid.any():
        examples = d.loc[invalid, ["isin", "action_type", "ex_date", "denominator", "numerator"]].head(15)
        raise ValueError(
            f"{int(invalid.sum())} invalid corporate action rows found before upload.\n"
            f"Examples:\n{examples.to_string(index=False)}"
        )

    unique_action_types = sorted(d["action_type"].unique().tolist())
    if len(unique_action_types) != 1:
        raise ValueError(f"Expected one action_type per upload, found: {unique_action_types}")

    d["denominator"] = d["denominator"].astype(int)
    d["numerator"] = d["numerator"].astype(int)
    d = d.drop_duplicates(subset=["isin", "action_type", "ex_date"], keep="last").reset_index(drop=True)
    d["source_file"] = source_file

    rows = d[
        ["isin", "action_type", "ex_date", "denominator", "numerator", "source_file"]
    ].to_dict(orient="records")

    upsert_sql = text(
        """
        insert into fundlab.stock_corporate_action
          (isin, action_type, ex_date, denominator, numerator, source_file)
        values
          (:isin, :action_type, :ex_date, :denominator, :numerator, :source_file)
        on conflict (isin, action_type, ex_date) do update
        set
          denominator = excluded.denominator,
          numerator   = excluded.numerator,
          source_file = excluded.source_file
        """
    )

    engine = _resolve_engine()
    chunk_size = 5000
    with engine.begin() as conn:
        for i in range(0, len(rows), chunk_size):
            conn.execute(upsert_sql, rows[i : i + chunk_size])

    summary = {
        "rows_in_file": int(len(df)),
        "rows_after_dedup": int(len(d)),
        "action_type": unique_action_types[0],
        "unique_isins": int(d["isin"].nunique()),
        "min_ex_date": str(d["ex_date"].min()),
        "max_ex_date": str(d["ex_date"].max()),
        "rows_upserted": int(len(rows)),
        "source_file": source_file,
    }
    return summary
