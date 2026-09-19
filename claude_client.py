"""Thin wrapper around the Anthropic API for one-shot text queries.

Reads ANTHROPIC_API_KEY from the environment only — see api_key.py for how
that variable gets populated (local .env vs. Streamlit Cloud secrets). This
module never receives or handles the raw key itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import anthropic

# Approximate USD per 1M tokens (input, output) — for the cost estimate shown
# in the UI only; not billing-accurate (ignores cache discounts).
_PRICING_PER_MTOK = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


@dataclass
class ClaudeResponse:
    text: str
    model: str | None
    total_cost_usd: float | None
    input_tokens: int | None
    output_tokens: int | None
    is_error: bool


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    rates = _PRICING_PER_MTOK.get(model)
    if rates is None:
        return None
    in_rate, out_rate = rates
    return (input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate


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
    except anthropic.AuthenticationError:
        return ClaudeResponse(
            text="Authentication failed — the API key is missing or invalid.",
            model=model,
            total_cost_usd=None,
            input_tokens=None,
            output_tokens=None,
            is_error=True,
        )
    except anthropic.RateLimitError as e:
        retry_after = e.response.headers.get("retry-after", "a while")
        return ClaudeResponse(
            text=f"Rate limited — please retry after {retry_after}s.",
            model=model,
            total_cost_usd=None,
            input_tokens=None,
            output_tokens=None,
            is_error=True,
        )
    except anthropic.APIStatusError as e:
        return ClaudeResponse(
            text=f"API error ({e.status_code}): {e.message}",
            model=model,
            total_cost_usd=None,
            input_tokens=None,
            output_tokens=None,
            is_error=True,
        )
    except anthropic.APIConnectionError:
        return ClaudeResponse(
            text="Network error — could not reach the Anthropic API.",
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
