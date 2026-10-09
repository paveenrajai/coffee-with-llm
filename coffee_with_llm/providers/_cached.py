"""Cache reads kept apart from the rest of a prompt."""

from __future__ import annotations

from typing import Optional


def uncached_input(prompt_tokens: int, cached_tokens: Optional[int]) -> int:
    """The part of ``prompt_tokens`` that was not read from cache.

    Google, OpenAI and Inception count cache reads inside their prompt count.
    :class:`~coffee_with_llm.types.TokenUsage` keeps the two apart, as
    Anthropic reports them, so no prompt token is counted or billed twice.
    """
    return max(0, prompt_tokens - (cached_tokens or 0))
