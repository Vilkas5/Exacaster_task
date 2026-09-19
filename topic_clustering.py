"""Cluster per-document topic phrases into corpus-wide canonical topics using
local embeddings + agglomerative clustering — no LLM call, so this step is
free and effectively instant no matter how many documents are in the corpus.
This is what replaces the old "ask an LLM to eyeball-cluster everything"
reduce step, and is the main reason the pipeline scales to hundreds of
documents: embedding a few hundred short phrases and clustering them takes
milliseconds, versus a reduce prompt whose size still grows with corpus size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from embeddings import embed

# Cosine distance below which two topic phrases are considered the same
# underlying topic. Tuned empirically against real extracted topic phrases:
# kept conservative (favors under- over over-merging) because short-phrase
# embeddings sometimes place clear synonyms (e.g. "AI Hallucinations" vs
# "LLM Hallucinations") only moderately close together — claude_client.py's
# cleanup stage catches near-duplicates this threshold leaves unmerged.
DEFAULT_DISTANCE_THRESHOLD = 0.3


@dataclass
class ClusteredTopics:
    # Canonical topic labels, ordered by how many documents cover them (most first).
    overall_topics: list[str]
    # documents[doc_name] -> the canonical topics (from overall_topics) it covers.
    documents: dict[str, list[str]]


def cluster_topics(
    per_document_topics: dict[str, list[str]],
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
) -> ClusteredTopics:
    """per_document_topics: {doc_name: [local topic phrases from stage 1]}."""
    # Flatten to a (doc_name, phrase) pair per occurrence, embedding each
    # unique phrase once.
    pairs = [(name, phrase) for name, phrases in per_document_topics.items() for phrase in phrases]
    unique_phrases = sorted({phrase for _, phrase in pairs})

    if not unique_phrases:
        return ClusteredTopics(overall_topics=[], documents={name: [] for name in per_document_topics})

    # A short template gives the embedding model a bit more context than a
    # bare noun phrase, which measurably improves separation between
    # genuinely distinct short topic labels.
    vectors = embed([f"Topic: {phrase}" for phrase in unique_phrases])

    if len(unique_phrases) == 1:
        labels = np.array([0])
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(vectors)

    phrase_to_cluster = dict(zip(unique_phrases, labels))

    # Canonical label per cluster = the phrase closest to the cluster's mean
    # embedding (the "medoid") — a deterministic pick, no LLM call needed.
    cluster_label: dict[int, str] = {}
    for cluster_id in set(labels):
        members = [p for p in unique_phrases if phrase_to_cluster[p] == cluster_id]
        member_vecs = np.array([vectors[unique_phrases.index(p)] for p in members])
        centroid = member_vecs.mean(axis=0)
        similarities = member_vecs @ centroid
        cluster_label[cluster_id] = members[int(np.argmax(similarities))]

    # Documents per cluster, to order overall_topics by coverage (most-covered first).
    doc_names_per_cluster: dict[int, set[str]] = {}
    for name, phrase in pairs:
        cluster_id = phrase_to_cluster[phrase]
        doc_names_per_cluster.setdefault(cluster_id, set()).add(name)

    ordered_cluster_ids = sorted(
        doc_names_per_cluster, key=lambda c: len(doc_names_per_cluster[c]), reverse=True
    )
    overall_topics = [cluster_label[c] for c in ordered_cluster_ids]

    documents: dict[str, list[str]] = {name: [] for name in per_document_topics}
    for name, phrases in per_document_topics.items():
        covered_clusters = {phrase_to_cluster[p] for p in phrases}
        # Preserve overall_topics' coverage order for readability.
        documents[name] = [
            cluster_label[c] for c in ordered_cluster_ids if c in covered_clusters
        ]

    return ClusteredTopics(overall_topics=overall_topics, documents=documents)
