import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text


def get_performance_engine():
    return create_engine(
        "postgresql+psycopg2://",
        connect_args={
            "host": st.secrets["pg"]["host"],
            "port": st.secrets["pg"]["port"],
            "user": st.secrets["pg"]["user"],
            "password": st.secrets["pg"]["password"],
            "dbname": st.secrets["pg"]["database"],
            "sslmode": "require",
        },
        pool_pre_ping=True,
    )


def load_fund_rolling(window_months: int, fund_names, start=None, end=None):
    if not fund_names:
        return pd.DataFrame([])

    fund_names = list(fund_names)
    date_filter = ""
    params = {"window": window_months, "fund_names": tuple(fund_names)}
    if start is not None:
        date_filter += " AND r.asof_date >= :start"
        params["start"] = start
    if end is not None:
        date_filter += " AND r.asof_date <= :end"
        params["end"] = end

    query = f"""
        SELECT
            f.fund_name,
            r.asof_date,
            r.rolling_cagr
        FROM fundlab.fund_rolling_return r
        JOIN fundlab.fund f
          ON f.fund_id = r.fund_id
        WHERE r.window_months = :window
          AND f.fund_name IN :fund_names
          {date_filter}
        ORDER BY r.asof_date, f.fund_name;
    """

    with get_performance_engine().begin() as conn:
        return pd.read_sql(text(query), conn, params=params, parse_dates=["asof_date"])


def load_bench_rolling(window_months: int, bench_name: str, start=None, end=None):
    if bench_name is None:
        return pd.DataFrame([])

    date_filter = ""
    params = {"window": window_months, "bench_name": bench_name}
    if start is not None:
        date_filter += " AND r.asof_date >= :start"
        params["start"] = start
    if end is not None:
        date_filter += " AND r.asof_date <= :end"
        params["end"] = end

    query = f"""
        SELECT
            b.bench_name,
            r.asof_date,
            r.rolling_cagr
        FROM fundlab.bench_rolling_return r
        JOIN fundlab.benchmark b
          ON b.bench_id = r.bench_id
        WHERE r.window_months = :window
          AND b.bench_name = :bench_name
          {date_filter}
        ORDER BY r.asof_date;
    """

    with get_performance_engine().begin() as conn:
        return pd.read_sql(text(query), conn, params=params, parse_dates=["asof_date"])


def load_funds_from_db():
    query = """
        SELECT
            f.fund_name      AS "Fund name",
            n.nav_date::date AS "month-end",
            n.nav_value::float AS "NAV",
            c.category_name  AS "market-cap",
            s.style_name     AS "style"
        FROM fundlab.fund_nav n
        JOIN fundlab.fund f
          ON f.fund_id = n.fund_id
        LEFT JOIN fundlab.category c
          ON c.category_id = f.category_id
        LEFT JOIN fundlab.style s
          ON s.style_id = f.style_id
        ORDER BY "Fund name", "month-end";
    """
    with get_performance_engine().begin() as conn:
        return pd.read_sql(query, conn, parse_dates=["month-end"])


def load_bench_from_db():
    query = """
        SELECT
            b.bench_name     AS "benchmark_name",
            n.nav_date::date AS "month-end",
            n.nav_value::float AS "NAV",
            c.category_name  AS "category_type",
            s.style_name     AS "category_value"
        FROM fundlab.bench_nav n
        JOIN fundlab.benchmark b
          ON b.bench_id = n.bench_id
        LEFT JOIN fundlab.category c
          ON c.category_id = b.category_id
        LEFT JOIN fundlab.style s
          ON s.style_id = b.style_id
        ORDER BY "benchmark_name", "month-end";
    """
    with get_performance_engine().begin() as conn:
        return pd.read_sql(query, conn, parse_dates=["month-end"])
