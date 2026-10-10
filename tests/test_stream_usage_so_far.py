"""What a stream has been charged so far, read without touching the stream.

A caller that stops a stream part way (a reader cutting a board off) is billed
for what the provider generated up to the cut. Gemini repeats its running
total on every streamed chunk, so the latest one received is that bill.
Closing the stream to read it would be worse than wrong: a Google stream reads
the rest of itself on close, for its usage.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, AsyncIterator
from unittest.mock import patch

import pytest

from coffee_with_llm import AskLLM, Config, estimate_cost
from coffee_with_llm.types import StreamResult, StreamTextDelta, TokenUsage

SERVED = "gemini-3.5-flash-lite"


def _config() -> Config:
    return Config(google_api_key="test-key", request_timeout=60.0)


def _chunk(text: str, *, prompt: int, output: int | None) -> SimpleNamespace:
    """One streamed Gemini chunk; ``output=None`` carries no usage at all."""
    usage = (
        None
        if output is None
        else SimpleNamespace(
            prompt_token_count=prompt,
            candidates_token_count=output,
            thoughts_token_count=None,
            cached_content_token_count=None,
        )
    )
    candidate = SimpleNamespace(
        finish_reason=None,
        content=SimpleNamespace(role="model", parts=[SimpleNamespace(text=text)]),
    )
    return SimpleNamespace(
        text=text, model_version=SERVED, candidates=[candidate], usage_metadata=usage
    )


class _Gemini:
    """A Gemini stream that counts what was read from it."""

    def __init__(self, chunks: list[SimpleNamespace]) -> None:
        self.chunks = chunks
        self.opened = 0
        self.read = 0

    async def generate_content_stream(self, **_kwargs: Any) -> AsyncIterator[SimpleNamespace]:
        self.opened += 1

        async def gen() -> AsyncIterator[SimpleNamespace]:
            for chunk in self.chunks:
                self.read += 1
                yield chunk

        return gen()


async def _open(gemini: _Gemini) -> StreamResult:
    with patch("coffee_with_llm.providers.google.text_client.genai.Client") as mock_genai:
        mock_genai.return_value.aio.models.generate_content_stream = (
            gemini.generate_content_stream
        )
        llm = AskLLM(model=f"google/{SERVED}", config=_config(), google_explicit_cache=False)
        stream = await llm.ask(prompt="Hi", stream=True)
    assert isinstance(stream, StreamResult)
    return stream


def _priced(prompt: int, output: int) -> float | None:
    return estimate_cost(TokenUsage(prompt, output, prompt + output), SERVED)


@pytest.mark.asyncio
async def test_it_is_the_running_total_of_the_chunks_received() -> None:
    gemini = _Gemini(
        [
            _chunk("On 26 April", prompt=1_000, output=40),
            _chunk(" 1986", prompt=1_000, output=90),
            _chunk(" a test", prompt=1_000, output=150),
        ]
    )
    stream = await _open(gemini)
    events = stream.__aiter__()

    assert await events.__anext__() == StreamTextDelta("On 26 April")
    first = stream.usage_so_far
    assert await events.__anext__() == StreamTextDelta(" 1986")
    second = stream.usage_so_far

    assert first is not None and (first.input_tokens, first.output_tokens) == (1_000, 40)
    assert first.cost_usd == _priced(1_000, 40)
    assert second is not None and (second.input_tokens, second.output_tokens) == (1_000, 90)
    assert second.served_model == SERVED
    await stream.aclose()


@pytest.mark.asyncio
async def test_reading_it_reads_nothing_from_the_stream() -> None:
    gemini = _Gemini(
        [
            _chunk("On 26 April", prompt=1_000, output=40),
            _chunk(" 1986", prompt=1_000, output=90),
        ]
    )
    stream = await _open(gemini)
    events = stream.__aiter__()
    await events.__anext__()

    for _ in range(3):
        assert stream.usage_so_far is not None

    assert (gemini.opened, gemini.read) == (1, 1)
    assert stream.usage is None, "the stream is not settled by a look at it"
    await stream.aclose()


@pytest.mark.asyncio
async def test_it_is_none_while_no_chunk_has_carried_usage() -> None:
    gemini = _Gemini(
        [
            _chunk("On 26 April", prompt=0, output=None),
            _chunk(" 1986", prompt=1_000, output=90),
        ]
    )
    stream = await _open(gemini)
    events = stream.__aiter__()
    await events.__anext__()

    assert stream.usage_so_far is None
    await stream.aclose()


@pytest.mark.asyncio
async def test_it_is_the_usage_once_the_stream_has_ended() -> None:
    async def provider() -> AsyncIterator[object]:
        yield StreamTextDelta("ok")
        yield TokenUsage(10, 5, 15)

    stream = StreamResult(provider)
    assert stream.usage_so_far is None, "no sink: this provider says only at the end"

    _ = [event async for event in stream]

    assert stream.usage_so_far is stream.usage
