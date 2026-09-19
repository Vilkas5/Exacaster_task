"""Resolves ANTHROPIC_API_KEY into the environment, wherever it's stored.

Local dev:      .env file (gitignored) via python-dotenv
Streamlit Cloud: st.secrets, populated from the app's Secrets dashboard —
                 never committed to the repo

Call load() once at app startup. Nothing here ever logs or displays the key.
"""

import os

from dotenv import load_dotenv


def load() -> bool:
    """Populate os.environ['ANTHROPIC_API_KEY'] from whichever source has it.

    Returns True if a key is available afterward, False otherwise.
    """
    load_dotenv()  # no-op if there's no .env file (e.g. on Streamlit Cloud)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        import streamlit as st

        # Parses secrets.toml if present and copies its keys into os.environ;
        # returns False (no exception) when no secrets file exists, e.g. local
        # dev without Streamlit Cloud secrets configured.
        st.secrets.load_if_toml_exists()

    return bool(os.environ.get("ANTHROPIC_API_KEY"))
