"""Two-pass Gemini flow: search grounding in prose, then JSON formatting.

Gemini often returns empty ``grounding_chunks`` when JSON output and Google Search
are requested in the same call. Research in plain text first, then format JSON
without the search tool, carrying citations forward.
"""

from __future__ import annotations

from typing import Any

import httpx

from coffee_with_llm.link_check import reachable_citation_urls
from coffee_with_llm.llm import AskLLM
from coffee_with_llm.providers.google.utils.citations import (
    async_resolve_urls,
    collect_research_citation_urls,
    restrict_json_hook_citations,
)
from coffee_with_llm.types import AskResult, StreamResult, TokenUsage


def _unwrap_ask_result(result: AskResult | StreamResult) -> AskResult:
    if isinstance(result, AskResult):
        return result
    raise TypeError("grounded flows do not support stream=True")


def _usage_or_zero(usage: TokenUsage | None) -> TokenUsage:
    return usage if usage is not None else TokenUsage(0, 0, 0, None)


def _merge_usage(first: TokenUsage | None, second: TokenUsage | None) -> TokenUsage:
    a = _usage_or_zero(first)
    b = _usage_or_zero(second)
    cached = None
    if a.cached_tokens is not None or b.cached_tokens is not None:
        cached = (a.cached_tokens or 0) + (b.cached_tokens or 0)
    return TokenUsage(
        input_tokens=a.input_tokens + b.input_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
        total_tokens=a.total_tokens + b.total_tokens,
        cached_tokens=cached,
        cost_usd=(a.cost_usd or 0.0) + (b.cost_usd or 0.0) or None,
    )


async def _resolve_allowed_urls(urls: list[str]) -> list[str]:
    if not urls:
        return []
    async with httpx.AsyncClient(follow_redirects=True, timeout=8) as client:
        resolved = await async_resolve_urls(set(urls), client)
    out: list[str] = []
    seen: set[str] = set()
    for url in urls:
        final = resolved.get(url, url)
        if final not in seen:
            out.append(final)
            seen.add(final)
    return out


def _build_json_prompt(
    json_prompt: str,
    *,
    research_text: str,
    allowed_urls: list[str],
) -> str:
    return _build_research_context_prompt(
        json_prompt,
        research_text=research_text,
        allowed_urls=allowed_urls,
    )


def _build_research_context_prompt(
    user_prompt: str,
    *,
    research_text: str,
    allowed_urls: list[str],
) -> str:
    lines = [user_prompt.strip(), "", "RESEARCH NOTES (facts and URLs from web search):"]
    lines.append(research_text.strip())
    lines.append("")
    if allowed_urls:
        lines.append("ALLOWED CITATION URLS — copy only from this list, exactly:")
        lines.extend(f"- {url}" for url in allowed_urls)
        lines.append("Never invent or guess a URL.")
    else:
        lines.append(
            "No verified source URLs were captured. Omit [cite:…] markers rather than invent URLs."
        )
    return "\n".join(lines)


async def ask_with_grounded_json(
    llm: AskLLM,
    *,
    research_prompt: str,
    json_prompt: str,
    research_system_instruct: str = "",
    json_system_instruct: str = "",
    **ask_kwargs: Any,
) -> AskResult:
    """Search-ground in prose, then format JSON without the search tool."""
    research = _unwrap_ask_result(
        await llm.ask(
            prompt=research_prompt,
            system_instruct=research_system_instruct,
            google_attach_search_tool=True,
            **ask_kwargs,
        )
    )
    allowed_urls = await _resolve_allowed_urls(
        collect_research_citation_urls(research.text)
    )
    allowed_urls = await reachable_citation_urls(allowed_urls)
    combined_prompt = _build_json_prompt(
        json_prompt,
        research_text=research.text,
        allowed_urls=allowed_urls,
    )
    json_result = _unwrap_ask_result(
        await llm.ask(
            prompt=combined_prompt,
            system_instruct=json_system_instruct,
            google_attach_search_tool=False,
            **ask_kwargs,
        )
    )
    text = restrict_json_hook_citations(json_result.text, allowed_urls)
    usage = _merge_usage(research.usage, json_result.usage)
    return AskResult(text=text, usage=usage)


def is_google_model(llm: AskLLM) -> bool:
    model = getattr(llm, "_model", "") or ""
    return model.startswith("google/") or "gemini" in model.lower()
