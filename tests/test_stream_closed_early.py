"""A Google stream stopped part way is closed, never read to its end.

It used to be drained for its usage on close: every chunk Google had still to
generate was read, billed and thrown away, and whoever closed it waited for
all of it (a caller retrying a bad layout waited tens of seconds before its
retry could start). Its usage is now the running total the chunks read so far
carried.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from coffee_with_llm import AskLLM, Config
from coffee_with_llm.types import StreamResult, StreamTextDelta

SERVED = "gemini-3.5-flash-lite"


def _chunk(text: str, output: int) -> SimpleNamespace:
    candidate = SimpleNamespace(
        finish_reason=None,
        content=SimpleNamespace(role="model", parts=[SimpleNamespace(text=text)]),
    )
    return SimpleNamespace(
        text=text,
        model_version=SERVED,
        candidates=[candidate],
        usage_metadata=SimpleNamespace(
            prompt_token_count=1_000,
            candidates_token_count=output,
            thoughts_token_count=None,
            cached_content_token_count=None,
        ),
    )


class _SdkStream:
    """google-genai's stream: counts what was read, and whether it was closed.

    ``hang_at`` makes that read wait on the network for good.
    """

    def __init__(self, *, hang_at: int | None = None) -> None:
        self.chunks = [_chunk("On 26 April", 40), _chunk(" 1986", 90), _chunk(" a test", 150)]
        self.hang_at = hang_at
        self.read = 0
        self.closed = False

    def __aiter__(self) -> _SdkStream:
        return self

    async def __anext__(self) -> SimpleNamespace:
        if self.read == self.hang_at:
            await asyncio.Event().wait()
        if self.read >= len(self.chunks):
            raise StopAsyncIteration
        self.read += 1
        return self.chunks[self.read - 1]

    async def aclose(self) -> None:
        self.closed = True


async def _open(sdk: _SdkStream) -> StreamResult:
    async def generate_content_stream(**_kwargs: Any) -> _SdkStream:
        return sdk

    with patch("coffee_with_llm.providers.google.text_client.genai.Client") as mock_genai:
        mock_genai.return_value.aio.models.generate_content_stream = generate_content_stream
        llm = AskLLM(model=f"google/{SERVED}", config=_config(), google_explicit_cache=False)
        stream = await llm.ask(prompt="Hi", stream=True)
    assert isinstance(stream, StreamResult)
    return stream


def _config() -> Config:
    return Config(google_api_key="test-key", request_timeout=60.0)


@pytest.mark.asyncio
async def test_a_consumer_stopping_early_reads_no_more_and_closes_the_stream() -> None:
    sdk = _SdkStream()
    stream = await _open(sdk)

    async for _event in stream:
        break
    await stream.aclose()

    assert sdk.read == 1, "the rest of the generation is never read"
    assert sdk.closed
    assert stream.usage is not None and stream.usage.output_tokens == 40


@pytest.mark.asyncio
async def test_a_cancel_while_paused_at_the_yield_reads_no_more() -> None:
    """The consumer is busy elsewhere (sending, storing) when it is cancelled."""
    sdk = _SdkStream()
    stream = await _open(sdk)
    holding_first = asyncio.Event()

    async def consume() -> None:
        events = stream.__aiter__()
        try:
            assert await events.__anext__() == StreamTextDelta("On 26 April")
            holding_first.set()
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await stream.aclose()
            raise

    task = asyncio.create_task(consume())
    await holding_first.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert (sdk.read, sdk.closed) == (1, True)
    assert stream.usage is not None and stream.usage.output_tokens == 40


@pytest.mark.asyncio
async def test_a_cancel_inside_the_network_read_closes_the_stream() -> None:
    sdk = _SdkStream(hang_at=1)
    stream = await _open(sdk)
    events = stream.__aiter__()
    await events.__anext__()

    reading = asyncio.create_task(events.__anext__())
    await asyncio.sleep(0)
    reading.cancel()
    async def settle() -> None:
        with pytest.raises(asyncio.CancelledError):
            await reading
        await stream.aclose()

    # Bounded: reading the rest would wait on the network for good.
    await asyncio.wait_for(settle(), 1)

    assert (sdk.read, sdk.closed) == (1, True)
    assert stream.usage is not None and stream.usage.output_tokens == 40
