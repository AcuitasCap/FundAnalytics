"""Streamlit application bootstrap and page router for Fund Analytics Dashboard."""

import os

import streamlit as st

from core.navigation import home_button
from features.fund_attribution.page import fund_attribution_page
from features.fund_attribution.smoke import _run_attrib_smoke_test
from features.fund_manager_tenure.page import fund_manager_tenure_page
from features.housekeeping.page import housekeeping_page
from features.portfolio.page import portfolio_page
from features.portfolio_quality.page import portfolio_quality_page
from features.portfolio_valuations.page import portfolio_valuations_page
from features.update_db.page import update_db_page
from pages.performance_page import performance_page as performance_page_v2


def check_password() -> bool:
    """Render the password prompt until the current Streamlit session is authenticated."""

    def password_entered() -> None:
        if st.session_state["password"] == st.secrets["auth"]["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Do not retain plaintext in session state.
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Enter password", type="password", on_change=password_entered, key="password")
        return False

    if not st.session_state["password_correct"]:
        st.text_input("Enter password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False

    return True


def home_page() -> None:
    """Render the dashboard landing page and its direct-navigation controls."""
    st.subheader("Welcome to the Fund Analytics Dashboard!")
    st.markdown(
        """
        This app currently has these main sections:

        - **Performance** – NAV-based rolling returns, yearly returns, P2P, and PDF export.
        - **Portfolio quality** – RoE / RoCE and quality buckets based on portfolios.
        - **Portfolio valuations** – P/E, P/B, P/S for funds vs peers.
        - **Portfolio** – Holdings explorer, active share, look-through.
        - **Update DB** – Upload and refresh raw datasets.
        - **Housekeeping** – Rebuild precomputed tables and diagnostics.

        Use the buttons below to jump directly to a section.
        """
    )

    st.subheader("Navigation")

    if st.button("📈 Performance"):
        st.session_state["page"] = "Performance"
        st.rerun()
    if st.button("📊 Portfolio quality"):
        st.session_state["page"] = "Portfolio quality"
        st.rerun()
    if st.button("💹 Portfolio valuations"):
        st.session_state["page"] = "Portfolio valuations"
        st.rerun()
    if st.button("📉 Fund attribution"):
        st.session_state["page"] = "Fund attribution"
        st.rerun()
    if st.button("📂 Portfolio"):
        st.session_state["page"] = "Portfolio"
        st.rerun()
    if st.button("🧑‍💼 Fund manager tenure"):
        st.session_state["page"] = "Fund manager tenure"
        st.rerun()
    if st.button("🛠️ Update DB"):
        st.session_state["page"] = "Update DB"
        st.rerun()
    if st.button("🧹 Housekeeping"):
        st.session_state["page"] = "Housekeeping"
        st.rerun()


# The router is intentionally the only place that connects page modules to Streamlit startup.
PAGE_HANDLERS = {
    "Home": home_page,
    "Performance": lambda: performance_page_v2(home_button),
    "Portfolio quality": portfolio_quality_page,
    "Portfolio valuations": portfolio_valuations_page,
    "Fund attribution": fund_attribution_page,
    "Portfolio": portfolio_page,
    "Fund manager tenure": fund_manager_tenure_page,
    "Update DB": update_db_page,
    "Housekeeping": housekeeping_page,
}


def dispatch_page(page: str) -> None:
    """Render a registered page, falling back safely to Home for unknown names."""
    PAGE_HANDLERS.get(page, home_page)()


def main() -> None:
    """Initialise page state and render the selected feature page."""
    if "page" not in st.session_state:
        st.session_state["page"] = "Home"

    st.markdown("---")
    dispatch_page(st.session_state["page"])


def run_app() -> None:
    """Configure Streamlit, authenticate the session, and start the router."""
    st.set_page_config(page_title="", layout="wide")
    if not check_password():
        st.stop()
    main()


if __name__ == "__main__":
    if os.getenv("RUN_ATTRIB_SMOKE_TEST") == "1":
        _run_attrib_smoke_test()
    else:
        run_app()
