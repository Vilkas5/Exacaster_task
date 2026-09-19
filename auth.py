"""App-wide access gate, plus an admin panel for emergency API-key rotation.

Two independent secrets control this:
- ACCESS_PASSWORD: required to use the app at all once deployed. If unset
  (e.g. local dev, or before you've decided on one), the gate is skipped
  entirely — nobody is locked out of their own local testing by accident.
- ADMIN_PASSWORD: a separate, more privileged password that unlocks a
  sidebar panel for swapping ANTHROPIC_API_KEY / GOOGLE_API_KEY at runtime
  (e.g. "the key ran out mid-demo") without needing Streamlit Cloud or
  GitHub access. Also a no-op if unset.

Neither password is ever logged or echoed back.
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


def render_admin_panel() -> None:
    """Sidebar panel to rotate API keys at runtime. No-op if ADMIN_PASSWORD
    isn't configured, so the panel simply doesn't exist until you set one.

    Applies immediately for every visitor on this running instance — env
    vars are shared across the whole process, not per-browser-session —
    but only lasts until the app next restarts, sleeps, or redeploys, since
    that pulls fresh values from Streamlit Cloud's real Secrets again. This
    is an emergency stopgap for "the key ran out mid-demo," not a
    replacement for updating the real secret afterward.
    """
    admin_password = os.environ.get("ADMIN_PASSWORD")
    if not admin_password:
        return

    with st.sidebar.expander("🔒 Admin: rotate API keys"):
        if not st.session_state.get("admin_unlocked"):
            entered = st.text_input("Admin password", type="password", key="admin_pw_input")
            if st.button("Unlock", key="admin_unlock_btn"):
                if _matches(entered, admin_password):
                    st.session_state.admin_unlocked = True
                    st.rerun()
                else:
                    st.error("Incorrect admin password.")
            return

        st.caption(
            "Takes effect immediately for everyone on this running instance. "
            "Only lasts until it next restarts/redeploys — also update "
            "Streamlit Cloud → Settings → Secrets for a permanent fix."
        )
        new_anthropic_key = st.text_input(
            "New ANTHROPIC_API_KEY", type="password", key="new_anthropic_key"
        )
        new_google_key = st.text_input(
            "New GOOGLE_API_KEY", type="password", key="new_google_key"
        )
        if st.button("Apply", key="admin_apply_btn"):
            updated = []
            if new_anthropic_key:
                os.environ["ANTHROPIC_API_KEY"] = new_anthropic_key
                updated.append("ANTHROPIC_API_KEY")
            if new_google_key:
                os.environ["GOOGLE_API_KEY"] = new_google_key
                updated.append("GOOGLE_API_KEY")
            if updated:
                st.success(f"Updated: {', '.join(updated)}")
            else:
                st.warning("Enter at least one new key value first.")
