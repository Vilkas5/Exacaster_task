"""Live Demo tab: ingest documents (Drive or upload), run the three-stage
analysis (map, embedding-cluster, cleanup), and show the results."""

import os

import streamlit as st

import api_key
import drive_source
from claude_client import analyze_documents
from document_utils import extract_text

DEFAULT_FOLDER_URL = "https://drive.google.com/drive/folders/16Q5mW_NbAO6DbeUO3zRfuXJypLV3H7ew"


def render(model: str) -> None:
    st.session_state.setdefault("drive_documents", {})
    st.session_state.setdefault("analysis", None)

    st.warning(
        "⚠️ **Limited funds remaining on this API key (~$1).** Each "
        "analysis run costs roughly $0.01–0.02 per document, so please "
        "test sparingly — a handful of runs will exhaust it. If it runs "
        "out, the key needs to be replaced in Streamlit Cloud → Manage "
        "app → Secrets before the demo will work again."
    )

    st.subheader("Load from Google Drive")
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
                "(APIs & Services → Credentials, with the Drive API enabled) "
                "and add it to `.env` locally or Streamlit Cloud → Settings → "
                "Secrets."
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
            except Exception as e:  # noqa: BLE001 — surface any Drive/network error
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
        with st.spinner(
            "Stage 1: reading each document independently (in parallel)... "
            "Stage 2: clustering topics by embedding similarity... "
            "Stage 3: cleaning up topic labels..."
        ):
            st.session_state.analysis = analyze_documents(documents, model=model)

    analysis = st.session_state.get("analysis")

    if analysis is not None:
        if analysis.is_error:
            st.error(analysis.error_message)
        else:
            result = analysis.result

            st.subheader("Overall Topics — primary themes across the whole set")
            for topic in result.overall_topics:
                st.markdown(f"- {topic}")

            if result.minor_topics:
                with st.expander(
                    f"+ {len(result.minor_topics)} more topic(s) mentioned in only one document"
                ):
                    for topic in result.minor_topics:
                        st.markdown(f"- {topic}")

            with st.expander("📋 Per-Document Topics and Stance", expanded=False):
                for doc in result.documents:
                    st.markdown(f"**{doc.name}**")
                    st.markdown(f"Topics: {', '.join(doc.topics)}")
                    st.markdown(f"Stance: {doc.stance}")
                    st.divider()

            if analysis.warnings:
                with st.expander(f"⚠️ {len(analysis.warnings)} issue(s) during analysis"):
                    for w in analysis.warnings:
                        st.write(w)

            with st.expander("Run details"):
                st.write(
                    {
                        "documents_analyzed": analysis.documents_analyzed,
                        "documents_failed": analysis.documents_failed,
                        "model (stage 1 + stage 3)": analysis.model,
                        "raw_clusters_before_cleanup": analysis.raw_cluster_count,
                        "final_topics": len(result.overall_topics) + len(result.minor_topics),
                        "input_tokens": analysis.input_tokens,
                        "output_tokens": analysis.output_tokens,
                        "estimated_cost_usd": analysis.total_cost_usd,
                    }
                )
