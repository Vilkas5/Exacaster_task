"""Resolves secret env vars (API keys), wherever they're stored.

Local dev:       .env file (gitignored) via python-dotenv
Streamlit Cloud: st.secrets, populated from the app's Secrets dashboard —
                 never committed to the repo

Call load() once at app startup, then read keys via os.environ as normal.
Nothing here ever logs or displays a key's value.
"""

import os

from dotenv import load_dotenv


def load() -> None:
    """Populate os.environ from .env (local) and/or secrets.toml (Cloud)."""
    load_dotenv()  # no-op if there's no .env file (e.g. on Streamlit Cloud)

    import streamlit as st

    # Parses secrets.toml if present and copies its keys into os.environ;
    # returns False (no exception) when no secrets file exists, e.g. local
    # dev without Streamlit Cloud secrets configured.
    st.secrets.load_if_toml_exists()


def has(name: str) -> bool:
    return bool(os.environ.get(name))
