"""Step 1 proof of concept: upload documents and send them to Claude.

Connects to Claude via the Anthropic API (ANTHROPIC_API_KEY), resolved
safely for both local dev (.env) and Streamlit Cloud (secrets manager) —
see api_key.py. The key itself never appears in this file.
"""

import streamlit as st

import api_key
from claude_client import ask_claude
from document_utils import extract_text

st.set_page_config(page_title="Document Analysis POC", page_icon="📄", layout="wide")

if not api_key.load():
    st.error(
        "No ANTHROPIC_API_KEY found. Locally: create a `.env` file with "
        "`ANTHROPIC_API_KEY=sk-ant-...`. On Streamlit Cloud: add it under "
        "your app's Settings → Secrets."
    )
    st.stop()

st.title("📄 Document Analysis — Claude Connection POC")
st.caption(
    "Upload one or more documents, then send them to Claude. This is the first "
    "step: proving the upload → LLM → response pipeline works before adding "
    "topic extraction and perspective analysis."
)

MODELS = {
    "Opus 5 (highest quality)": "claude-opus-5",
    "Sonnet 5 (balanced)": "claude-sonnet-5",
    "Haiku 4.5 (fastest/cheapest)": "claude-haiku-4-5",
}

with st.sidebar:
    st.header("Settings")
    model_label = st.selectbox("Model", options=list(MODELS.keys()), index=0)
    model = MODELS[model_label]

uploaded_files = st.file_uploader(
    "Upload documents",
    type=["txt", "pdf", "docx"],
    accept_multiple_files=True,
)

documents: dict[str, str] = {}
if uploaded_files:
    for f in uploaded_files:
        try:
            documents[f.name] = extract_text(f.name, f.getvalue())
        except ValueError as e:
            st.error(str(e))

    st.subheader("Extracted text preview")
    for name, text in documents.items():
        with st.expander(f"{name} ({len(text):,} characters)"):
            st.text(text[:2000] + ("..." if len(text) > 2000 else ""))

default_prompt = (
    "Summarize the key topics discussed in the document(s) below, and briefly "
    "describe the perspective or stance each document takes on those topics."
)
instruction = st.text_area("Instruction / prompt sent to Claude", value=default_prompt, height=100)

send = st.button("Send to Claude", type="primary", disabled=not documents)

if send:
    combined = "\n\n".join(
        f"=== Document: {name} ===\n{text}" for name, text in documents.items()
    )
    full_prompt = f"{instruction}\n\n{combined}"

    with st.spinner("Waiting for Claude..."):
        response = ask_claude(full_prompt, model=model)

    if response.is_error:
        st.error(response.text)
    else:
        st.subheader("Response")
        st.markdown(response.text or "_(empty response)_")

        with st.expander("Run details"):
            st.write(
                {
                    "model": response.model,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "estimated_cost_usd": response.total_cost_usd,
                }
            )
