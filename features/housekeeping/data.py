"""Read-only database helpers for Housekeeping jobs."""

import pandas as pd
from sqlalchemy import text

def _latest_rolling_asof_map(conn, table_name: str, id_col: str) -> dict[tuple[int, int], pd.Timestamp]:
    query = text(f"""
        SELECT {id_col} AS entity_id, window_months, MAX(asof_date)::date AS latest_asof
        FROM {table_name}
        GROUP BY {id_col}, window_months
    """)
    latest = pd.read_sql(query, conn, parse_dates=["latest_asof"])
    if latest.empty:
        return {}

    return {
        (int(row["entity_id"]), int(row["window_months"])): pd.Timestamp(row["latest_asof"])
        for _, row in latest.iterrows()
        if pd.notna(row["latest_asof"])
    }


