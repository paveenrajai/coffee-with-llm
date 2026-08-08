"""Orchestrator-style flow: web search answer, then markdown with citations.

Mirrors blackboard-agent ``fetch_web_context`` + pass-2 markdown delivery.
"""

from __future__ import annotations

from typing import Any

from coffee_with_llm.grounded_json import (
    _build_research_context_prompt,
    _merge_usage,
    _resolve_allowed_urls,
    _unwrap_ask_result,
)
from coffee_with_llm.link_check import check_citation_urls, reachable_citation_urls
from coffee_with_llm.llm import AskLLM
from coffee_with_llm.providers.google.utils.citations import (
    collect_research_citation_urls,
    extract_citation_urls_from_text,
    partition_citation_urls,
    restrict_inline_citations,
)
from coffee_with_llm.types import AskResult


async def ask_with_grounded_markdown(
    llm: AskLLM,
    *,
    web_query: str,
    markdown_prompt: str,
    web_system_instruct: str = "",
    markdown_system_instruct: str = "",
    **ask_kwargs: Any,
) -> tuple[AskResult, AskResult, list[str]]:
    """Web search (pass 1), then markdown reply (pass 2) with citation allowlist.

    Returns ``(web_result, markdown_result, allowed_citation_urls)``.
    """
    web = _unwrap_ask_result(
        await llm.ask(
            prompt=web_query,
            system_instruct=web_system_instruct,
            google_attach_search_tool=True,
            **ask_kwargs,
        )
    )
    allowed_urls = await _resolve_allowed_urls(
        collect_research_citation_urls(web.text)
    )
    allowed_urls = await reachable_citation_urls(allowed_urls)

    combined_prompt = _build_research_context_prompt(
        markdown_prompt,
        research_text=web.text,
        allowed_urls=allowed_urls,
    )
    markdown = _unwrap_ask_result(
        await llm.ask(
            prompt=combined_prompt,
            system_instruct=markdown_system_instruct,
            google_attach_search_tool=False,
            **ask_kwargs,
        )
    )
    reply = restrict_inline_citations(markdown.text, allowed_urls)
    usage = _merge_usage(web.usage, markdown.usage)
    return web, AskResult(text=reply, usage=usage), allowed_urls


async def verify_markdown_citations(
    reply: str,
    *,
    allowed_urls: list[str],
) -> dict[str, Any]:
    """Check markdown ``[cite:…]`` markers against the web-search allowlist."""
    cited = extract_citation_urls_from_text(reply)
    matched, hallucinated = partition_citation_urls(cited, allowed_urls)
    checks = await check_citation_urls(cited)
    dead = [c for c in checks if not c.ok]
    bot_blocked = [c for c in checks if c.ok and c.status in {401, 403}]
    return {
        "cited_urls": cited,
        "allowed_urls": allowed_urls,
        "matched_allowlist": matched,
        "hallucinated": hallucinated,
        "dead": dead,
        "bot_blocked": bot_blocked,
    }
