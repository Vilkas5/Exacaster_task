"""Thin wrapper around the Anthropic API for one-shot text queries.

Reads ANTHROPIC_API_KEY from the environment only — see api_key.py for how
that variable gets populated (local .env vs. Streamlit Cloud secrets). This
module never receives or handles the raw key itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import anthropic
from pydantic import BaseModel

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


class DocumentAnalysis(BaseModel):
    name: str
    topics: list[str]
    stance: str


class AnalysisResult(BaseModel):
    overall_topics: list[str]
    documents: list[DocumentAnalysis]


@dataclass
class ClaudeResponse:
    text: str
    model: str | None
    total_cost_usd: float | None
    input_tokens: int | None
    output_tokens: int | None
    is_error: bool


@dataclass
class AnalysisResponse:
    result: AnalysisResult | None
    model: str | None
    total_cost_usd: float | None
    input_tokens: int | None
    output_tokens: int | None
    is_error: bool
    error_message: str | None = None


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


def ask_claude(
    prompt: str,
    model: str = "claude-opus-5",
    system_prompt: str | None = None,
    max_tokens: int = 16000,
) -> ClaudeResponse:
    """Send a single prompt to Claude and return the text response.

    Errors are caught and returned as a ClaudeResponse with is_error=True and
    a human-readable message in .text, so the caller can render it without a
    try/except at every call site.
    """
    client = anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY from env

    kwargs = {}
    if system_prompt:
        kwargs["system"] = system_prompt

    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
    except _RETRYABLE_ERRORS as e:
        return ClaudeResponse(
            text=_describe_error(e),
            model=model,
            total_cost_usd=None,
            input_tokens=None,
            output_tokens=None,
            is_error=True,
        )

    text = "\n".join(block.text for block in response.content if block.type == "text").strip()
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens

    return ClaudeResponse(
        text=text,
        model=response.model,
        total_cost_usd=_estimate_cost(model, input_tokens, output_tokens),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        is_error=False,
    )


def analyze_documents(
    documents: dict[str, str],
    model: str = "claude-haiku-4-5",
    max_tokens: int = 16000,
) -> AnalysisResponse:
    """Identify overall topics across all documents, plus each document's own
    topics and stance, as structured data (not freeform text).
    """
    client = anthropic.Anthropic()

    combined = "\n\n".join(
        f"=== Document: {name} ===\n{text}" for name, text in documents.items()
    )
    prompt = (
        "Identify the primary topics discussed across all of the documents "
        "below. Then, for each individual document, list which of those "
        "topics it covers and describe its perspective or stance on them.\n\n"
        f"{combined}"
    )

    try:
        response = client.messages.parse(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            output_format=AnalysisResult,
        )
    except _RETRYABLE_ERRORS as e:
        return AnalysisResponse(
            result=None,
            model=model,
            total_cost_usd=None,
            input_tokens=None,
            output_tokens=None,
            is_error=True,
            error_message=_describe_error(e),
        )

    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    parsed = response.parsed_output

    return AnalysisResponse(
        result=parsed,
        model=response.model,
        total_cost_usd=_estimate_cost(model, input_tokens, output_tokens),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        is_error=parsed is None,
        error_message=None if parsed is not None else "Claude did not return a parseable analysis.",
    )
