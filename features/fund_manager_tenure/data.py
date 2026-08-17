"""Database access for the fund manager tenure feature."""

import pandas as pd
import streamlit as st

from core.db import get_engine


@st.cache_data(ttl=600)
def fetch_fund_manager_tenure(fund_ids: list[int]) -> pd.DataFrame:
    """Return manager-tenure rows for the selected fund universe."""
    if not fund_ids:
        return pd.DataFrame()

    query = """
        SELECT fund_id, fund_manager, from_date, to_date
        FROM fundlab.fund_manager_tenure
        WHERE fund_id = ANY(%(fund_ids)s)
        ORDER BY fund_id, from_date
    """
    return pd.read_sql(query, get_engine(), params={"fund_ids": fund_ids})
