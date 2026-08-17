"""Lazy database-engine construction shared by dashboard features."""

import streamlit as st
from sqlalchemy import create_engine


_engine = None


def get_engine():
    """Create and return the database engine on first runtime use.

    Secrets are deliberately read only when this function is called, keeping
    application imports safe for tests and non-Streamlit tooling.
    """
    global _engine
    if _engine is None:
        _engine = create_engine(
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
    return _engine
