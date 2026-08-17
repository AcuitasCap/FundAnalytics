"""Pure transformations used by Housekeeping maintenance jobs."""

import pandas as pd

ROLLING_WINDOWS_MONTHS = [12, 36]

def _compute_rolling_cagr_from_monthly_nav(nav_series: pd.Series, months: int) -> pd.Series:
    nav_series = pd.to_numeric(nav_series, errors="coerce").dropna().sort_index()
    if nav_series.empty:
        return pd.Series(dtype=float)
    return (nav_series / nav_series.shift(months)) ** (12.0 / months) - 1.0


def _prepare_monthly_nav_series(df: pd.DataFrame, id_col: str, date_col: str, value_col: str) -> dict[int, pd.Series]:
    if df.empty:
        return {}

    work = df[[id_col, date_col, value_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col]).dt.to_period("M").dt.to_timestamp("M")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[id_col, date_col, value_col])
    work = work.sort_values([id_col, date_col])
    work = work.drop_duplicates(subset=[id_col, date_col], keep="last")

    series_map: dict[int, pd.Series] = {}
    for entity_id, grp in work.groupby(id_col):
        series_map[int(entity_id)] = grp.set_index(date_col)[value_col].sort_index()
    return series_map


