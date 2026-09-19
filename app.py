"""GenAI document analysis feasibility POC.

Identifies the primary topics across a document set and each document's own
perspective on them, via a three-stage pipeline (LLM per-document reading,
local embedding-based topic clustering, small LLM cleanup) designed to scale
to hundreds of documents (see claude_client.analyze_documents). Connects to
Claude via the Anthropic API (ANTHROPIC_API_KEY) and, optionally, to a public
Google Drive folder via the Drive API (GOOGLE_API_KEY) — both resolved
safely for local dev (.env) and Streamlit Cloud (secrets manager), see
api_key.py. Neither key ever appears in this file.
"""

import streamlit as st

import api_key
import auth
import tab_demo
import tab_feasibility
import tab_solution

st.set_page_config(page_title="Document Analysis POC", page_icon="📄", layout="wide")
api_key.load()
auth.require_access_password()

if not api_key.has("ANTHROPIC_API_KEY"):
    st.error(
        "No ANTHROPIC_API_KEY found. Locally: create a `.env` file with "
        "`ANTHROPIC_API_KEY=sk-ant-...`. On Streamlit Cloud: add it under "
        "your app's Settings → Secrets."
    )
    st.stop()

st.title("📄 GenAI Document Analysis — Feasibility POC")
st.caption(
    "First priority: identify the primary topics across the whole document "
    "set. Second priority: articulate each document's own perspective on "
    "those topics. Designed to scale to hundreds of documents, not just "
    "this sample set."
)

MODELS = {
    "Opus 5 (highest quality)": "claude-opus-5",
    "Sonnet 5 (balanced)": "claude-sonnet-5",
    "Haiku 4.5 (fastest/cheapest)": "claude-haiku-4-5",
}

with st.sidebar:
    st.header("Settings")
    model_label = st.selectbox("Model", options=list(MODELS.keys()), index=2)
    model = MODELS[model_label]
    st.caption(
        "Used to read each document (stage 1) and to clean up topic labels "
        "(stage 3). Topic clustering itself (stage 2) runs on local "
        "embeddings, not this model — see the Proposed Solution tab."
    )
    auth.render_admin_panel()

demo_tab, feasibility_tab, solution_tab = st.tabs(
    ["🔍 Live Demo", "✅ Feasibility Assessment", "🧩 Proposed Solution"]
)

with demo_tab:
    tab_demo.render(model=model)

with feasibility_tab:
    tab_feasibility.render()

with solution_tab:
    tab_solution.render()
