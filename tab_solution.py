"""Proposed Solution Overview tab: a plain-language explanation of the
algorithm's key steps and rationale — for a non-technical stakeholder."""

import streamlit as st


def render() -> None:
    st.header("🧩 Proposed Solution Overview")

    st.markdown(
        "**Goal:** turn a folder of documents into (1) the handful of "
        "topics that come up across the whole set, and (2) what each "
        "individual document has to say about them — automatically, and in "
        "a way that keeps working whether there are 10 documents or 500."
    )

    st.subheader("How it works, in three steps")
    st.markdown(
        "**1. Read each document on its own.** For every document, ask the "
        "AI to pull out a few broad topics and its overall stance. Each "
        "read doesn't need to know about any other document, so many "
        "documents can be read *at the same time* instead of one after "
        "another. Adding more documents mostly costs more parallel time, "
        "not more sequential time."
    )
    st.markdown(
        "**2. Group similar topics together automatically.** Every "
        "document now has its own short list of topic labels, but "
        "different documents will often describe the same idea in "
        "slightly different words. Rather than asking the AI to eyeball "
        "every label from every document and decide what matches — which "
        "gets slower and more expensive the more documents there are — "
        "this step converts each label into a numerical fingerprint that "
        "captures its meaning, and groups labels whose fingerprints are "
        "close together. This is standard, well-understood technology "
        "(the same idea used by search engines and recommendation systems "
        "to tell whether two pieces of text mean the same thing) — it "
        "runs in a fraction of a second even for hundreds of documents, "
        "and it doesn't cost anything per run."
    )
    st.markdown(
        "**3. A quick final polish.** The automatic grouping in step 2 "
        "occasionally leaves two labels for the same real-world topic "
        "un-grouped (worded just differently enough), or picks an odd "
        "phrase to represent a group. A small, one-time check tidies up "
        "the final topic names and merges any obvious duplicates. This "
        "check only ever looks at the *list of topics* (typically a "
        "few dozen at most), never at the documents themselves — so it "
        "stays quick and cheap no matter how large the document set gets."
    )

    st.subheader("Why this design, and not \"paste everything into one AI request\"")
    st.markdown(
        "- **It scales.** Reading documents one at a time means there's no "
        "hard ceiling on how much text the system can process — a folder "
        "of 500 documents works the same way as a folder of 10, just with "
        "more parallel work in step 1. And grouping topics by fingerprint "
        "(step 2) doesn't get more expensive as the document count grows — "
        "only the one-time polish (step 3) touches the AI again, and only "
        "over the short topic list, not the documents.\n"
        "- **It's cheaper where it matters.** The bulk of the AI cost is "
        "step 1's cheap, per-document reads; the two steps that look at "
        "the *whole* set — grouping and polishing — are either free "
        "(step 2) or tiny and one-time (step 3), instead of one expensive "
        "request that has to reread everything.\n"
        "- **It degrades gracefully.** If one document fails to process "
        "(a corrupted file, an unreadable scan), the rest of the batch "
        "isn't affected — the system reports which one failed and keeps "
        "going, rather than the whole analysis breaking.\n"
        "- **It tells 'primary' from 'minor.'** A topic only one document "
        "happens to mention isn't promoted to the headline topic list — "
        "it's kept, but set aside as a minor topic, so the main findings "
        "reflect what the *whole set* is actually about, per the brief."
    )

    st.subheader("What's demonstrated today vs. what comes next")
    st.markdown(
        "- **Demonstrated live in this app:** the full pipeline above, "
        "running against a real folder of sample documents pulled directly "
        "from Google Drive — no manual copy-pasting required.\n"
        "- **Natural next steps for a production rollout:** validating "
        "against a much larger, messier set of real client documents; "
        "adding OCR for scanned files; and a lightweight human review step "
        "for the first batch of results in a new domain."
    )
