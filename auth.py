"""App-wide access gate.

ACCESS_PASSWORD is required to use the app at all once deployed. If unset
(e.g. local dev, or before you've decided on one), the gate is skipped
entirely — nobody is locked out of their own local testing by accident.

Key rotation (e.g. if ANTHROPIC_API_KEY runs out mid-demo) is handled
through Streamlit Community Cloud's own "Manage app" -> Secrets screen,
not a custom panel here — see the project notes on what access that
requires.
"""

from __future__ import annotations

import hmac
import os

import streamlit as st

_MAX_ATTEMPTS = 5


def _matches(entered: str, expected: str) -> bool:
    return hmac.compare_digest(entered.strip().encode(), expected.strip().encode())


def require_access_password() -> None:
    """Block the whole app behind ACCESS_PASSWORD. Call this first, before
    rendering anything else. No-op if ACCESS_PASSWORD isn't configured."""
    expected = os.environ.get("ACCESS_PASSWORD")
    if not expected:
        return

    if st.session_state.get("access_granted"):
        return

    st.session_state.setdefault("access_attempts", 0)

    st.title("🔒 Access required")

    if st.session_state.access_attempts >= _MAX_ATTEMPTS:
        st.error("Too many failed attempts. Reload the page to try again.")
        st.stop()

    password = st.text_input("Password", type="password", key="access_password_input")
    if st.button("Enter"):
        if _matches(password, expected):
            st.session_state.access_granted = True
            st.rerun()
        else:
            st.session_state.access_attempts += 1
            st.error("Incorrect password.")

    st.stop()
