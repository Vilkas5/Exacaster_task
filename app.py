"""Step 1 proof of concept: pull documents (from upload or Google Drive) and
send them to Claude.

Connects to Claude via the Anthropic API (ANTHROPIC_API_KEY) and, optionally,
to a public Google Drive folder via the Drive API (GOOGLE_API_KEY) — both
resolved safely for local dev (.env) and Streamlit Cloud (secrets manager),
see api_key.py. Neither key ever appears in this file.
"""

import os

import streamlit as st

import api_key
import drive_source
from claude_client import ask_claude
from document_utils import extract_text

st.set_page_config(page_title="Document Analysis POC", page_icon="📄", layout="wide")
api_key.load()

if not api_key.has("ANTHROPIC_API_KEY"):
    st.error(
        "No ANTHROPIC_API_KEY found. Locally: create a `.env` file with "
        "`ANTHROPIC_API_KEY=sk-ant-...`. On Streamlit Cloud: add it under "
        "your app's Settings → Secrets."
    )
    st.stop()

st.title("📄 Document Analysis — Claude Connection POC")
st.caption(
    "Pull documents from a Google Drive folder (or upload manually), then send "
    "them to Claude. This is the first step: proving the ingestion → LLM → "
    "response pipeline works before adding topic extraction and perspective "
    "analysis."
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

st.session_state.setdefault("drive_documents", {})

st.subheader("Load from Google Drive")
DEFAULT_FOLDER_URL = "https://drive.google.com/drive/folders/16Q5mW_NbAO6DbeUO3zRfuXJypLV3H7ew"
folder_input = st.text_input("Drive folder URL or ID", value=DEFAULT_FOLDER_URL)

col1, col2 = st.columns([1, 1])
fetch_clicked = col1.button("Fetch from Drive")
clear_clicked = col2.button("Clear fetched documents")

if clear_clicked:
    st.session_state.drive_documents = {}

if fetch_clicked:
    if not api_key.has("GOOGLE_API_KEY"):
        st.error(
            "No GOOGLE_API_KEY found. Create one in Google Cloud Console "
            "(APIs & Services → Credentials, with the Drive API enabled) and "
            "add it to `.env` locally or Streamlit Cloud → Settings → Secrets."
        )
    else:
        folder_id = drive_source.extract_folder_id(folder_input)
        google_key = os.environ["GOOGLE_API_KEY"]
        try:
            with st.spinner("Listing files in Drive folder..."):
                files = drive_source.list_files(folder_id, google_key)

            fetched: dict[str, str] = {}
            skipped: list[str] = []
            progress = st.progress(0.0, text="Downloading...")
            for i, f in enumerate(files, start=1):
                if f["mimeType"] == drive_source.FOLDER_MIME_TYPE:
                    continue
                result = drive_source.download_file(f, google_key)
                if result is None:
                    skipped.append(f["name"])
                    continue
                name, content = result
                try:
                    fetched[name] = extract_text(name, content)
                except ValueError:
                    skipped.append(f["name"])
                progress.progress(i / max(len(files), 1), text=f"Downloaded {name}")
            progress.empty()

            st.session_state.drive_documents = fetched
            st.success(f"Fetched {len(fetched)} document(s) from Drive.")
            if skipped:
                st.caption(f"Skipped (unsupported type): {', '.join(skipped)}")
        except Exception as e:  # noqa: BLE001 — surface any Drive/network error to the user
            st.error(f"Failed to fetch from Drive: {e}")

st.subheader("Or upload manually")
uploaded_files = st.file_uploader(
    "Upload documents",
    type=["txt", "pdf", "docx"],
    accept_multiple_files=True,
)

uploaded_documents: dict[str, str] = {}
if uploaded_files:
    for f in uploaded_files:
        try:
            uploaded_documents[f.name] = extract_text(f.name, f.getvalue())
        except ValueError as e:
            st.error(str(e))

documents: dict[str, str] = {**st.session_state.drive_documents, **uploaded_documents}

if documents:
    st.subheader(f"Documents ready for analysis ({len(documents)})")
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
