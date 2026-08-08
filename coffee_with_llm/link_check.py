"""Verify cited URLs resolve (smoke tests, grounded JSON allowlists).

A 403 after retry usually means a bot wall, not a missing page — those links
still work in a browser and are kept.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Iterable

import httpx

CHECK_TIMEOUT_SECONDS = 8.0
MAX_CONCURRENT_CHECKS = 6

_RETRY_WITH_GET = frozenset({400, 401, 403, 404, 405, 406, 501})
#: Server answered but blocks automated clients; treat as reachable for citations.
_BOT_WALL_STATUSES = frozenset({401, 403})

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    ),
    "Accept": "*/*",
}


@dataclass(frozen=True, slots=True)
class CitationLinkCheck:
    url: str
    ok: bool
    status: int | None = None
    resolved: str | None = None
    error: str | None = None


async def check_citation_url(client: httpx.AsyncClient, url: str) -> CitationLinkCheck:
    href = url.strip()
    if not href.startswith(("http://", "https://")):
        return CitationLinkCheck(url=url, ok=False, error="Not a usable link")

    try:
        response = await client.head(href)
        if response.status_code in _RETRY_WITH_GET:
            response = await client.get(href, headers={"Range": "bytes=0-2047"})
        status = response.status_code
        ok = status < 400 or status in _BOT_WALL_STATUSES
        return CitationLinkCheck(
            url=url,
            ok=ok,
            status=status,
            resolved=str(response.url),
        )
    except httpx.HTTPError as exc:
        return CitationLinkCheck(url=url, ok=False, error=type(exc).__name__)


async def check_citation_urls(urls: Iterable[str]) -> list[CitationLinkCheck]:
    ordered = list(dict.fromkeys(url.strip() for url in urls if url and url.strip()))
    if not ordered:
        return []

    limit = asyncio.Semaphore(MAX_CONCURRENT_CHECKS)
    async with httpx.AsyncClient(
        timeout=CHECK_TIMEOUT_SECONDS,
        follow_redirects=True,
        headers=_HEADERS,
    ) as client:

        async def guarded(url: str) -> CitationLinkCheck:
            async with limit:
                return await check_citation_url(client, url)

        return list(await asyncio.gather(*(guarded(url) for url in ordered)))


async def reachable_citation_urls(urls: Iterable[str]) -> list[str]:
    """Return input URLs that resolved, preserving order."""
    checks = await check_citation_urls(urls)
    out: list[str] = []
    for check in checks:
        if check.ok and check.url not in out:
            out.append(check.url)
    return out
