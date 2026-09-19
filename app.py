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
from claude_client import analyze_documents
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
    "Pull documents from a Google Drive folder (or upload manually), then let "
    "Claude identify the primary topics across the whole set and each "
    "document's own perspective on them."
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

st.session_state.setdefault("drive_documents", {})
st.session_state.setdefault("analysis", None)

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

analyze_clicked = st.button("Analyze Documents", type="primary", disabled=not documents)

if analyze_clicked:
    with st.spinner("Analyzing with Claude..."):
        st.session_state.analysis = analyze_documents(documents, model=model)

analysis = st.session_state.get("analysis")

if analysis is not None:
    if analysis.is_error:
        st.error(analysis.error_message)
    else:
        result = analysis.result

        st.subheader("Overall Topics")
        for topic in result.overall_topics:
            st.markdown(f"- {topic}")

        with st.expander("📋 Per-Document Topics and Stance", expanded=False):
            for doc in result.documents:
                st.markdown(f"**{doc.name}**")
                st.markdown(f"Topics: {', '.join(doc.topics)}")
                st.markdown(f"Stance: {doc.stance}")
                st.divider()

        with st.expander("Run details"):
            st.write(
                {
                    "model": analysis.model,
                    "input_tokens": analysis.input_tokens,
                    "output_tokens": analysis.output_tokens,
                    "estimated_cost_usd": analysis.total_cost_usd,
                }
            )
