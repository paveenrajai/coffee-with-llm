"""ask() delivers attachments to the provider and validates them up front."""

from unittest.mock import AsyncMock, patch

import pytest

from coffee_with_llm import AskLLM, Attachment
from coffee_with_llm.exceptions import ValidationError
from coffee_with_llm.types import StreamTextDelta, TokenUsage

PDF = b"%PDF-1.4 fake"


def _llm(mock_google_api_key=None):
    with patch("coffee_with_llm.providers.google.text_client.genai.Client"):
        return AskLLM(model="gemini-flash-lite-latest", min_delay_between_calls=0)


@pytest.mark.asyncio
async def test_ask_passes_attachments_to_provider(mock_google_api_key):
    llm = _llm()
    llm._client.generate = AsyncMock(return_value=("ok", TokenUsage(1, 1, 2, None)))

    a = Attachment(data=PDF, mime_type="application/pdf", filename="doc.pdf")
    await llm.ask(prompt="summarize", attachments=[a])

    sent = llm._client.generate.call_args.kwargs["attachments"]
    assert sent == [a]


@pytest.mark.asyncio
async def test_ask_without_attachments_sends_none(mock_google_api_key):
    llm = _llm()
    llm._client.generate = AsyncMock(return_value=("ok", TokenUsage(1, 1, 2, None)))

    await llm.ask(prompt="hi")

    assert llm._client.generate.call_args.kwargs["attachments"] is None


@pytest.mark.asyncio
async def test_ask_validates_attachments_before_calling_provider(mock_google_api_key):
    llm = _llm()
    llm._client.generate = AsyncMock(return_value=("ok", TokenUsage(1, 1, 2, None)))

    with pytest.raises(ValidationError, match="must be an Attachment"):
        await llm.ask(prompt="hi", attachments=["/some/path.pdf"])

    llm._client.generate.assert_not_called()


@pytest.mark.asyncio
async def test_streaming_passes_attachments_to_provider(mock_google_api_key):
    """Attachments work with stream=True — the answer streams, the input is binary."""
    llm = _llm()
    captured = {}

    async def fake_stream(**kwargs):
        captured.update(kwargs)
        yield StreamTextDelta("chunk")
        yield TokenUsage(1, 1, 2, None)

    llm._client.generate_stream = fake_stream

    a = Attachment(data=PDF, mime_type="application/pdf")
    result = await llm.ask(prompt="summarize", attachments=[a], stream=True)
    chunks = [c async for c in result]

    assert [c.text for c in chunks] == ["chunk"]
    assert captured["attachments"] == [a]


@pytest.mark.asyncio
async def test_streaming_without_attachments_sends_none(mock_google_api_key):
    llm = _llm()
    captured = {}

    async def fake_stream(**kwargs):
        captured.update(kwargs)
        yield TokenUsage(0, 0, 0, None)

    llm._client.generate_stream = fake_stream

    result = await llm.ask(prompt="hi", stream=True)
    [c async for c in result]

    assert captured["attachments"] is None
