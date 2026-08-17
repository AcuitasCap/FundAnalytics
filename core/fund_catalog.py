"""Shared fund-category lookup queries."""

import pandas as pd
import streamlit as st
from sqlalchemy import text

from core.db import get_engine


FUND_COLUMNS = ["fund_id", "fund_name", "category_name"]


def fetch_categories() -> list[str]:
    """Return selectable fund categories, excluding the portfolio pseudo-category."""
    query = """
        SELECT DISTINCT category_name
        FROM fundlab.category
        WHERE LOWER(category_name) <> 'portfolio'
        ORDER BY category_name;
    """
    df = pd.read_sql(query, get_engine())
    if df.empty:
        return []
    return df["category_name"].dropna().tolist()


@st.cache_data(ttl=60)
def fetch_funds_for_categories(categories: list[str]) -> pd.DataFrame:
    """Return fund metadata for the supplied categories with a stable empty schema."""
    if not categories:
        return pd.DataFrame(columns=FUND_COLUMNS)

    query = text("""
        SELECT f.fund_id,
               f.fund_name,
               c.category_name
        FROM fundlab.fund f
        JOIN fundlab.category c
          ON f.category_id = c.category_id
        WHERE c.category_name = ANY(:cats)
        ORDER BY f.fund_name;
    """)
    return pd.read_sql(query, get_engine(), params={"cats": categories})
