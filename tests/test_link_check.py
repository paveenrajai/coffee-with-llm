"""Tests for citation URL verification."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from coffee_with_llm.link_check import check_citation_url


@pytest.mark.asyncio
async def test_bot_wall_403_counts_as_ok():
    client = AsyncMock()
    head = MagicMock()
    head.status_code = 403
    get = MagicMock()
    get.status_code = 403
    get.url = "https://medium.com/example"
    client.head = AsyncMock(return_value=head)
    client.get = AsyncMock(return_value=get)

    result = await check_citation_url(client, "https://medium.com/example")
    assert result.ok is True
    assert result.status == 403


@pytest.mark.asyncio
async def test_missing_page_404_is_not_ok():
    client = AsyncMock()
    head = MagicMock()
    head.status_code = 404
    get = MagicMock()
    get.status_code = 404
    get.url = "https://example.com/missing"
    client.head = AsyncMock(return_value=head)
    client.get = AsyncMock(return_value=get)

    result = await check_citation_url(client, "https://example.com/missing")
    assert result.ok is False
    assert result.status == 404
