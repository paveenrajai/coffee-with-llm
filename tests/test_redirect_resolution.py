"""Google's grounding redirects are resolved from their own reply.

Live, 2026-10-04: the redirect answered in 0.13s with the page's address in
``Location``, and following it on failed at the page, a news site that broke
HTTP/2 on a HEAD. The redirect then stood as the citation.
"""

from __future__ import annotations

import httpx
import pytest

from coffee_with_llm.providers.google.utils.citations import (
    async_resolve_urls,
    resolve_vertex_redirect,
)

REDIRECT = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHtI3WQ"
PAGE = "https://example.it/en/news/20708/google-lincoln-data-center-redaction"


def _transport(seen: list[str], *, head: int = 302, get: int = 302, location: str = PAGE):
    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url.host}")
        if request.url.host != "vertexaisearch.cloud.google.com":
            raise httpx.RemoteProtocolError("the page broke HTTP/2", request=request)
        status = head if request.method == "HEAD" else get
        headers = {"location": location} if 300 <= status < 400 else {}
        return httpx.Response(status, headers=headers)

    return httpx.MockTransport(handle)


def test_the_address_is_read_off_the_redirect_and_the_page_is_never_asked() -> None:
    seen: list[str] = []
    with httpx.Client(transport=_transport(seen), follow_redirects=True) as client:
        assert resolve_vertex_redirect(REDIRECT, client, {}) == PAGE

    assert seen == ["HEAD vertexaisearch.cloud.google.com"]


def test_a_redirect_that_will_not_answer_a_head_is_asked_with_a_get() -> None:
    seen: list[str] = []
    with httpx.Client(transport=_transport(seen, head=405)) as client:
        assert resolve_vertex_redirect(REDIRECT, client, {}) == PAGE

    assert seen == ["HEAD vertexaisearch.cloud.google.com", "GET vertexaisearch.cloud.google.com"]


def test_a_relative_location_is_read_against_the_redirect() -> None:
    with httpx.Client(transport=_transport([], location="/elsewhere")) as client:
        assert (
            resolve_vertex_redirect(REDIRECT, client, {})
            == "https://vertexaisearch.cloud.google.com/elsewhere"
        )


def test_a_redirect_that_resolves_to_nothing_is_kept_as_it_was() -> None:
    with httpx.Client(transport=_transport([], head=404, get=404)) as client:
        assert resolve_vertex_redirect(REDIRECT, client, {}) == REDIRECT


def test_any_other_address_is_left_alone_and_never_requested() -> None:
    seen: list[str] = []
    with httpx.Client(transport=_transport(seen)) as client:
        assert resolve_vertex_redirect(PAGE, client, {}) == PAGE

    assert seen == []


@pytest.mark.asyncio
async def test_the_async_resolver_reads_the_redirect_too() -> None:
    seen: list[str] = []
    async with httpx.AsyncClient(transport=_transport(seen), follow_redirects=True) as client:
        resolved = await async_resolve_urls({REDIRECT, PAGE}, client)

    assert resolved == {REDIRECT: PAGE, PAGE: PAGE}
    assert seen == ["HEAD vertexaisearch.cloud.google.com"]


@pytest.mark.asyncio
async def test_the_async_resolver_keeps_a_redirect_it_cannot_read() -> None:
    async with httpx.AsyncClient(transport=_transport([], head=404, get=404)) as client:
        assert await async_resolve_urls({REDIRECT}, client) == {REDIRECT: REDIRECT}
