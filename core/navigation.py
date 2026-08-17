"""Shared Streamlit navigation widgets."""

import streamlit as st


def home_button():
    """Render a Home button that returns the user to the Home page."""
    if st.button("🏠 Home"):
        st.session_state["page"] = "Home"
        st.rerun()
