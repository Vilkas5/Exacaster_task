"""Three-stage document analysis pipeline, via the Anthropic API plus local
embeddings for topic clustering.

Reads ANTHROPIC_API_KEY from the environment only — see api_key.py for how
that variable gets populated (local .env vs. Streamlit Cloud secrets). This
module never receives or handles the raw key itself.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import anthropic
from pydantic import BaseModel

import topic_clustering

# Approximate USD per 1M tokens (input, output) — for the cost estimate shown
# in the UI only; not billing-accurate (ignores cache discounts).
_PRICING_PER_MTOK = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

_RETRYABLE_ERRORS = (
    anthropic.AuthenticationError,
    anthropic.RateLimitError,
    anthropic.APIStatusError,
    anthropic.APIConnectionError,
)


class DocumentTopics(BaseModel):
    """Stage 1 (map) output: one document's own local topics + stance,
    read in isolation — before we know what the other documents say."""

    topics: list[str]
    stance_summary: str


class DocumentAnalysis(BaseModel):
    name: str
    topics: list[str]
    stance: str


class AnalysisResult(BaseModel):
    # Topics covered by at least min_topic_coverage documents — the "primary
    # topics across the whole set" the task asks for, sorted by coverage.
    overall_topics: list[str]
    # Topics that survived clustering but are only covered by a single (or
    # otherwise below-threshold) document — real, but not "primary" for the
    # corpus as a whole. Kept, not discarded, so nothing is silently lost.
    minor_topics: list[str]
    documents: list[DocumentAnalysis]


class _TopicLabelMapping(BaseModel):
    raw_label: str
    final_label: str


class _TopicCleanupResult(BaseModel):
    mappings: list[_TopicLabelMapping]


@dataclass
class AnalysisResponse:
    result: AnalysisResult | None
    model: str
    total_cost_usd: float | None
    input_tokens: int | None
    output_tokens: int | None
    is_error: bool
    error_message: str | None = None
    warnings: list[str] = field(default_factory=list)
    documents_analyzed: int = 0
    documents_failed: int = 0
    raw_cluster_count: int = 0


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    rates = _PRICING_PER_MTOK.get(model)
    if rates is None:
        return None
    in_rate, out_rate = rates
    return (input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate


def _describe_error(e: Exception) -> str:
    """Map an SDK exception to a human-readable message (most-specific first)."""
    if isinstance(e, anthropic.AuthenticationError):
        return "Authentication failed — the API key is missing or invalid."
    if isinstance(e, anthropic.RateLimitError):
        retry_after = e.response.headers.get("retry-after", "a while")
        return f"Rate limited — please retry after {retry_after}s."
    if isinstance(e, anthropic.APIStatusError):
        return f"API error ({e.status_code}): {e.message}"
    return "Network error — could not reach the Anthropic API."


def _extract_document_topics(
    name: str, text: str, model: str
) -> tuple[DocumentTopics | None, int, int, str | None]:
    """Stage 1 (map): read ONE document in isolation and extract its local
    topics + stance. Called once per document — independent of every other
    document, so this call's size is bounded by a single document's length
    no matter how large the overall corpus is. That's what makes the map
    stage scale to hundreds of documents: cost and latency grow linearly,
    and every call fits in context regardless of corpus size.

    Returns (parsed_topics_or_None, input_tokens, output_tokens, error_or_None).
    """
    client = anthropic.Anthropic()
    prompt = (
        "Read the document below in isolation and extract:\n"
        "1. Its 2-4 BROAD subject-matter topics — general subject areas a "
        "reader would use to categorize this document on a shelf (e.g. "
        "'Retrieval-Augmented Generation', 'AI Hallucinations'), NOT narrow "
        "technical sub-points, specific techniques, or sentence-length "
        "descriptions. These will be clustered against other documents' "
        "topics later, so favor short, general, reusable phrases a different "
        "document on a similar subject would plausibly also produce.\n"
        "2. A concise 2-3 sentence summary of its perspective or stance.\n\n"
        f"Document: {name}\n\n{text}"
    )
    try:
        response = client.messages.parse(
            model=model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
            output_format=DocumentTopics,
        )
    except _RETRYABLE_ERRORS as e:
        return None, 0, 0, _describe_error(e)

    return (
        response.parsed_output,
        response.usage.input_tokens,
        response.usage.output_tokens,
        None,
    )


def _cleanup_topic_labels(
    overall_topics: list[str],
    model: str,
) -> tuple[dict[str, str] | None, int, int, str | None]:
    """Stage 3 (cleanup): a single small call over the RAW cluster labels
    from stage 2 — never over documents. Embedding clustering on short
    phrases sometimes leaves near-duplicates un-merged (e.g. "AI
    Hallucinations" vs "LLM Hallucinations" embed just far enough apart to
    land in separate clusters) and can pick an odd phrase as a cluster's
    representative label. This call fixes both: it merges any raw labels
    that are clearly the same underlying topic, and returns a clean final
    label for each.

    Input size here is the number of raw clusters, not the number of
    documents — and the number of distinct topics in a corpus stays roughly
    bounded even as document count grows into the hundreds, so this stays
    cheap and fast regardless of corpus size.
    """
    client = anthropic.Anthropic()
    listing = "\n".join(f"- {label}" for label in overall_topics)
    prompt = (
        "Below is a list of topic labels produced by an automated clustering "
        "step over short phrases. Some may be near-duplicates of each other "
        "(different wording for the same underlying topic) due to "
        "clustering noise on short text. For every label in the list, "
        "provide a clean, human-readable final label (usually the label "
        "itself, lightly tidied). Give two labels the exact same final "
        "label ONLY if they clearly describe the same underlying topic — "
        "not merely related, overlapping, or in the same general area. "
        "When in doubt, keep them separate. Each raw_label in your output "
        "must exactly match one of the labels below, verbatim, with no "
        "added text.\n\n"
        f"{listing}"
    )
    try:
        response = client.messages.parse(
            model=model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
            output_format=_TopicCleanupResult,
        )
    except _RETRYABLE_ERRORS as e:
        return None, 0, 0, _describe_error(e)

    parsed = response.parsed_output
    mapping = {m.raw_label: m.final_label for m in parsed.mappings} if parsed else None
    return mapping, response.usage.input_tokens, response.usage.output_tokens, None


def analyze_documents(
    documents: dict[str, str],
    model: str = "claude-haiku-4-5",
    max_workers: int = 5,
    distance_threshold: float = topic_clustering.DEFAULT_DISTANCE_THRESHOLD,
    min_topic_coverage: int = 2,
) -> AnalysisResponse:
    """Identify the primary topics across the whole document set, and each
    document's own perspective on those topics — via a three-stage pipeline:

    1. Map (LLM, parallel, once per document): read each document in
       isolation and extract its own local topics + stance. Bounded by a
       single document's length per call, however large the corpus is.
    2. Cluster (local embeddings, no LLM, no API key): embed every local
       topic phrase and group near-duplicates with agglomerative clustering.
       This is the step that used to be an LLM "reduce" call whose prompt
       size grew with corpus size — clustering short vectors is instead
       near-instant and free regardless of document count.
    3. Cleanup (LLM, once, tiny): polish cluster labels and merge any
       near-duplicates embedding noise left behind. Input size is the
       number of distinct topics, not the number of documents, so this
       stays cheap even at large corpus sizes.

    min_topic_coverage: a topic must be covered by at least this many
    documents to count as "primary" (AnalysisResult.overall_topics); topics
    below that land in minor_topics instead of inflating the headline list
    with single-document one-offs. Each document's own .topics still lists
    everything it covers, regardless of this threshold.
    """
    per_document: dict[str, DocumentTopics] = {}
    warnings: list[str] = []
    total_input = 0
    total_output = 0
    total_cost = 0.0

    # Stage 1 (map) — documents are independent, so run them concurrently.
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_extract_document_topics, name, text, model): name
            for name, text in documents.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            parsed, in_tok, out_tok, error = future.result()
            total_input += in_tok
            total_output += out_tok
            cost = _estimate_cost(model, in_tok, out_tok)
            if cost is not None:
                total_cost += cost
            if parsed is None:
                warnings.append(f"{name}: {error}")
            else:
                per_document[name] = parsed

    if not per_document:
        return AnalysisResponse(
            result=None,
            model=model,
            total_cost_usd=total_cost or None,
            input_tokens=total_input or None,
            output_tokens=total_output or None,
            is_error=True,
            error_message="No documents could be analyzed.",
            warnings=warnings,
            documents_analyzed=0,
            documents_failed=len(documents),
        )

    # Stage 2 (cluster) — local embeddings, no API call, no per-document cost.
    clustered = topic_clustering.cluster_topics(
        {name: dt.topics for name, dt in per_document.items()},
        distance_threshold=distance_threshold,
    )
    doc_counts = {
        topic: sum(1 for topics in clustered.documents.values() if topic in topics)
        for topic in clustered.overall_topics
    }

    # Stage 3 (cleanup) — one small call, sized by topic count, not document count.
    mapping, in_tok, out_tok, error = _cleanup_topic_labels(clustered.overall_topics, model)
    total_input += in_tok
    total_output += out_tok
    cost = _estimate_cost(model, in_tok, out_tok)
    if cost is not None:
        total_cost += cost

    if mapping is None:
        warnings.append(f"Topic label cleanup failed, using raw cluster labels: {error}")
        mapping = {topic: topic for topic in clustered.overall_topics}

    # Merge clusters that cleanup gave the same final label, preserving
    # coverage-descending order.
    final_doc_counts: dict[str, int] = {}
    for raw_topic in clustered.overall_topics:
        final_label = mapping.get(raw_topic, raw_topic)
        final_doc_counts[final_label] = final_doc_counts.get(final_label, 0) + doc_counts[raw_topic]

    ranked_topics = sorted(final_doc_counts, key=lambda t: final_doc_counts[t], reverse=True)
    overall_topics = [t for t in ranked_topics if final_doc_counts[t] >= min_topic_coverage]
    minor_topics = [t for t in ranked_topics if final_doc_counts[t] < min_topic_coverage]

    result_documents = [
        DocumentAnalysis(
            name=name,
            topics=list(dict.fromkeys(mapping.get(t, t) for t in clustered.documents[name])),
            stance=per_document[name].stance_summary,
        )
        for name in per_document
    ]

    return AnalysisResponse(
        result=AnalysisResult(
            overall_topics=overall_topics, minor_topics=minor_topics, documents=result_documents
        ),
        model=model,
        total_cost_usd=total_cost or None,
        input_tokens=total_input or None,
        output_tokens=total_output or None,
        is_error=False,
        warnings=warnings,
        documents_analyzed=len(per_document),
        documents_failed=len(documents) - len(per_document),
        raw_cluster_count=len(clustered.overall_topics),
    )
