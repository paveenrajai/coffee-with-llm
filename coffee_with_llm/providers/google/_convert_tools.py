"""Convert OpenAI-style tool schemas for Gemini Interactions API."""

from __future__ import annotations

from typing import Any

from .text_client import _convert_tools_to_gemini


def convert_openai_tools_to_interaction_functions(
    tools_schema: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """OpenAI tools → Interactions ``{"type": "function", ...}`` entries."""
    out: list[dict[str, Any]] = []
    for decl in _convert_tools_to_gemini(tools_schema):
        out.append({"type": "function", **decl})
    return out
