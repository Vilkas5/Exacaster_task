"""Feasibility Assessment tab: findings on GenAI viability for this problem,
illustrated with real output from the sample document set."""

import altair as alt
import pandas as pd
import streamlit as st

from claude_client import AnalysisResult

_BAR_COLOR = "#2563eb"


def _topic_coverage_chart(result: AnalysisResult) -> alt.Chart:
    counts = {t: 0 for t in result.overall_topics}
    for doc in result.documents:
        for t in doc.topics:
            if t in counts:
                counts[t] += 1

    df = pd.DataFrame({"Topic": list(counts.keys()), "Documents": list(counts.values())})
    df = df.sort_values("Documents", ascending=False)

    return (
        alt.Chart(df)
        .mark_bar(color=_BAR_COLOR, cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            x=alt.X("Documents:Q", title="Documents covering this topic"),
            y=alt.Y("Topic:N", sort="-x", title=None),
            tooltip=["Topic", "Documents"],
        )
        .properties(height=28 * len(df) + 20)
    )


def render() -> None:
    st.header("✅ Feasibility Assessment")

    st.markdown(
        "**Verdict: yes — GenAI is a strong fit for this problem.** Tested "
        "against the 10 sample documents provided (a mix of practitioner "
        "articles and academic papers, loosely themed around LLM "
        "hallucinations, RAG, and GenAI reliability), the pipeline:"
    )
    st.markdown(
        "- Correctly surfaced coherent, non-overlapping topics across the "
        "set — without being told in advance what the documents were about "
        "— while keeping single-document tangents out of the headline list "
        "(see *Overall Topics* vs. the collapsed *minor topics* note in the "
        "Live Demo tab).\n"
        "- Distinguished documents covering similar ground but taking "
        "different angles: it separated an academic paper arguing LLM errors "
        "should be reframed philosophically as *\"bullshit\"* (in Frankfurt's "
        "sense) from the practitioner pieces focused on technical causes and "
        "fixes — a distinction a keyword-based approach would likely miss.\n"
        "- Picked up differing stances on the *same* topic: one practitioner "
        "took a pragmatic, risk-based, human-in-the-loop stance on "
        "hallucination mitigation, while another took a more optimistic "
        "\"fully solvable\" stance — both about the same subject, correctly "
        "told apart.\n"
        "- Ran in well under a minute and cost a small fraction of a cent "
        "per document in API usage — see the live numbers below."
    )

    st.subheader("Illustrative example — this session's run")
    analysis = st.session_state.get("analysis")
    if analysis is None or analysis.is_error or analysis.result is None:
        st.info(
            "Run an analysis in the **Live Demo** tab (Fetch from Drive → "
            "Analyze Documents) to see a live example rendered here, backed "
            "by real output rather than a canned screenshot."
        )
    else:
        result = analysis.result
        col1, col2, col3 = st.columns(3)
        col1.metric("Documents analyzed", analysis.documents_analyzed)
        col2.metric("Topics identified", len(result.overall_topics))
        col3.metric(
            "Cost this run",
            f"${analysis.total_cost_usd:.3f}" if analysis.total_cost_usd else "n/a",
        )

        st.caption("How many documents touch each identified topic:")
        st.altair_chart(_topic_coverage_chart(result), use_container_width=True)

    st.subheader("Where GenAI struggles — limitations to flag")
    st.markdown(
        "- **Not perfectly deterministic.** Topic wording and granularity "
        "can shift slightly between runs. Fine for exploratory analysis, "
        "but a human review pass is worth keeping before treating the topic "
        "list as a fixed taxonomy.\n"
        "- **Depends on clean text extraction.** Scanned or image-only PDFs "
        "would need OCR first — not exercised by this sample set, all of "
        "which had extractable text.\n"
        "- **Not ground truth.** Like a human analyst, the model's reading "
        "of a document's stance could be wrong or subtly biased — worth a "
        "periodic spot-check, especially early in rollout."
    )
