"""
Simple authentication module for Kuuma Booking Analyzer
"""

import streamlit as st
import hmac


def check_password():
    """Returns `True` if the user has entered a correct password."""

    def login_form():
        """Display login form."""
        with st.form("credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """Check whether a password entered by the user is correct."""
        try:
            # Get passwords from secrets
            passwords = st.secrets.get("passwords", {})

            username = st.session_state.get("username", "")
            password = st.session_state.get("password", "")

            if username in passwords and hmac.compare_digest(
                password,
                passwords[username],
            ):
                st.session_state["password_correct"] = True
                st.session_state["logged_in_user"] = username
                # Clear password from session state for security
                del st.session_state["password"]
                del st.session_state["username"]
            else:
                st.session_state["password_correct"] = False
        except Exception:
            st.session_state["password_correct"] = False

    # First run or logged out
    if st.session_state.get("password_correct", None) is None:
        st.image("https://kuuma.nl/wp-content/themes/kuuma/images/logo.svg", width=120)
        st.title("Kuuma Booking Analyzer")
        st.markdown("Please log in to access the dashboard.")
        login_form()
        return False

    # Password was entered but incorrect
    elif not st.session_state["password_correct"]:
        st.image("https://kuuma.nl/wp-content/themes/kuuma/images/logo.svg", width=120)
        st.title("Kuuma Booking Analyzer")
        st.markdown("Please log in to access the dashboard.")
        login_form()
        st.error("Incorrect username or password")
        return False

    # Password correct
    else:
        return True


def logout_button():
    """Display logout button in sidebar."""
    if st.session_state.get("password_correct"):
        user = st.session_state.get("logged_in_user", "User")
        st.sidebar.markdown(f"Logged in as: **{user}**")
        if st.sidebar.button("Logout"):
            st.session_state["password_correct"] = None
            st.session_state["logged_in_user"] = None
            st.rerun()
