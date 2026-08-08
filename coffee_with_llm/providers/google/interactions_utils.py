"""Helpers for parsing Gemini Interactions API responses."""

from __future__ import annotations

from typing import Any

from ...types import TokenUsage


def _step_type(step: Any) -> str | None:
    return getattr(step, "type", None)


def _text_from_content_blocks(blocks: Any) -> str:
    parts: list[str] = []
    for block in blocks or []:
        if getattr(block, "type", None) != "text":
            continue
        text = getattr(block, "text", None)
        if isinstance(text, str) and text:
            parts.append(text)
    return "".join(parts)


def interaction_text(interaction: Any) -> str:
    """Extract model text from an Interaction (steps schema or legacy outputs)."""
    output_text = getattr(interaction, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    steps = getattr(interaction, "steps", None) or []
    parts: list[str] = []
    for step in steps:
        if _step_type(step) != "model_output":
            continue
        text = _text_from_content_blocks(getattr(step, "content", None))
        if text.strip():
            parts.append(text)

    if parts:
        return "".join(parts).strip()

    # Legacy outputs array (google-genai < 2.0)
    outputs = getattr(interaction, "outputs", None) or []
    legacy_parts: list[str] = []
    for block in outputs:
        if getattr(block, "type", None) == "text":
            text = getattr(block, "text", None)
            if isinstance(text, str) and text.strip():
                legacy_parts.append(text)
    return "".join(legacy_parts).strip()


def interaction_function_calls(interaction: Any) -> list[dict[str, Any]]:
    """Function calls requested by the model in this interaction."""
    steps = getattr(interaction, "steps", None) or []
    calls: list[dict[str, Any]] = []
    for step in steps:
        if _step_type(step) != "function_call":
            continue
        calls.append(
            {
                "id": getattr(step, "id", None),
                "name": getattr(step, "name", None),
                "arguments": getattr(step, "arguments", None) or {},
            }
        )
    if calls:
        return calls

    outputs = getattr(interaction, "outputs", None) or []
    for block in outputs:
        if getattr(block, "type", None) != "function_call":
            continue
        calls.append(
            {
                "id": getattr(block, "id", None),
                "name": getattr(block, "name", None),
                "arguments": getattr(block, "arguments", None) or {},
            }
        )
    return calls


def interaction_usage(interaction: Any) -> TokenUsage:
    """Map Interaction usage to coffee-with-llm :class:`TokenUsage`."""
    usage = getattr(interaction, "usage", None)
    if usage is None:
        return TokenUsage(0, 0, 0, None)
    input_tokens = int(getattr(usage, "total_input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "total_output_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0) or input_tokens + output_tokens
    cached = getattr(usage, "total_cached_tokens", None)
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_tokens=int(cached) if cached is not None else None,
    )
