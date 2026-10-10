"""Every provider reports the prompt in disjoint buckets, and each is billed once.

``input_tokens`` is the uncached input, ``cached_tokens`` the cache reads and
``cache_creation_tokens`` the cache writes. Google, OpenAI and Inception count
cache reads inside their own input count. On 2026-10-09 a Gemini board
reported ``input_tokens=90116`` and ``cached_tokens=36841``, so its
``prompt_tokens`` came out 126957 while the real prompt was 90116. OpenAI's
cache reads were never read at all, and billed at the full input price.
Anthropic's buckets were always disjoint, and ``estimate_cost`` took its cache
read out of its input whenever the read was the smaller of the two.

Usage objects are the SDKs' own types, so a read of a field the provider does
not send fails here.
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from typing import Any, AsyncIterator, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic.types import Usage as AnthropicUsage
from google.genai import types as genai_types
from google.genai.interactions import Usage as InteractionUsage
from openai.types import CompletionUsage
from openai.types.completion_usage import PromptTokensDetails
from openai.types.responses import ResponseUsage

from coffee_with_llm import AskLLM, Config, estimate_cost
from coffee_with_llm.cost import _get_pricing
from coffee_with_llm.providers.inception.chat_client import InceptionChatClient
from coffee_with_llm.types import AskResult, StreamResult, TokenUsage

# The Gemini board of 2026-10-09.
BOARD_PROMPT = 90_116
BOARD_CACHED = 36_841
BOARD_OUTPUT = 12_988
BOARD_UNCACHED = BOARD_PROMPT - BOARD_CACHED


def _config() -> Config:
    return Config(
        openai_api_key="test-key",
        anthropic_api_key="test-key",
        google_api_key="test-key",
        inception_api_key="test-key",
        request_timeout=60.0,
    )


def _disjoint_cost(
    model: str, *, uncached: int, cached: int, output: int, cache_creation: int = 0
) -> float:
    """Each bucket billed once, at its own rate, at today's price for ``model``."""
    pricing = _get_pricing(model)
    assert pricing is not None
    input_rate, output_rate, cached_rate = pricing
    assert cached_rate is not None
    per_million = (
        uncached * input_rate
        + cached * cached_rate
        + cache_creation * input_rate * 1.25
        + output * output_rate
    )
    return round(per_million / 1_000_000, 6)


# --- estimate_cost -------------------------------------------------------------


def test_the_gemini_board_costs_what_it_did() -> None:
    """The subset guess priced it right; the disjoint buckets must too."""
    usage = TokenUsage(
        BOARD_UNCACHED, BOARD_OUTPUT, BOARD_UNCACHED + BOARD_OUTPUT, BOARD_CACHED
    )

    assert estimate_cost(usage, "gemini-3.8-flash", on=date(2026, 10, 9)) == 0.091424


def test_a_cache_read_smaller_than_input_is_not_taken_from_input() -> None:
    """Anthropic's buckets are disjoint: the read was subtracted from input."""
    usage = TokenUsage(10_000, 500, 10_500, cached_tokens=4_000)

    # 10,000 at $3 + 4,000 at $0.30 + 500 at $15 per 1M. The guess charged $0.0267.
    assert estimate_cost(usage, "claude-sonnet-4-6") == 0.0387


# --- Gemini generateContent ------------------------------------------------------


def _gemini_usage() -> genai_types.GenerateContentResponseUsageMetadata:
    return genai_types.GenerateContentResponseUsageMetadata(
        prompt_token_count=BOARD_PROMPT,
        cached_content_token_count=BOARD_CACHED,
        candidates_token_count=BOARD_OUTPUT,
    )


def _gemini_chunk(text: str, *, last: bool) -> SimpleNamespace:
    candidate = SimpleNamespace(
        finish_reason=SimpleNamespace(name="STOP"),
        content=SimpleNamespace(role="model", parts=[SimpleNamespace(text=text)]),
    )
    return SimpleNamespace(
        text=text,
        model_version="gemini-3.8-flash",
        candidates=[candidate] if last else [],
        usage_metadata=_gemini_usage() if last else None,
    )


def _assert_the_gemini_board(usage: TokenUsage | None) -> None:
    assert usage is not None
    assert usage.input_tokens == BOARD_UNCACHED
    assert usage.cached_tokens == BOARD_CACHED
    assert usage.prompt_tokens == BOARD_PROMPT
    assert usage.total_tokens == BOARD_UNCACHED + BOARD_OUTPUT
    assert usage.cost_usd == _disjoint_cost(
        "gemini-3.8-flash", uncached=BOARD_UNCACHED, cached=BOARD_CACHED, output=BOARD_OUTPUT
    )
    assert estimate_cost(usage, "gemini-3.8-flash", on=date(2026, 10, 9)) == 0.091424


@pytest.mark.asyncio
async def test_gemini_leaves_the_cache_out_of_input() -> None:
    async def generate_content(**_kwargs: Any) -> Any:
        return _gemini_chunk("ok", last=True)

    with patch("coffee_with_llm.providers.google.text_client.genai.Client") as mock_genai:
        mock_genai.return_value.aio.models.generate_content = generate_content
        llm = AskLLM(model="google/gemini-3.8-flash", config=_config(), google_explicit_cache=False)
        result = await llm.ask(prompt="Draw the board")

    assert isinstance(result, AskResult)
    _assert_the_gemini_board(result.usage)


async def _gemini_stream(**_kwargs: Any) -> AsyncIterator[Any]:
    async def gen() -> AsyncIterator[Any]:
        yield _gemini_chunk("Part one", last=False)
        yield _gemini_chunk(" and two", last=True)

    return gen()


@pytest.mark.asyncio
async def test_a_gemini_stream_leaves_the_cache_out_of_input() -> None:
    with patch("coffee_with_llm.providers.google.text_client.genai.Client") as mock_genai:
        mock_genai.return_value.aio.models.generate_content_stream = _gemini_stream
        llm = AskLLM(model="google/gemini-3.8-flash", config=_config(), google_explicit_cache=False)
        stream = await llm.ask(prompt="Draw the board", stream=True)
        assert isinstance(stream, StreamResult)
        _ = [event async for event in stream]

    _assert_the_gemini_board(stream.usage)


async def _gemini_stream_totalled_as_it_goes(**_kwargs: Any) -> AsyncIterator[Any]:
    """Gemini repeating its running total on every chunk."""

    async def gen() -> AsyncIterator[Any]:
        yield _gemini_chunk("Part one", last=True)
        yield _gemini_chunk(" and two", last=True)

    return gen()


@pytest.mark.asyncio
async def test_a_gemini_stream_closed_early_leaves_the_cache_out_of_input() -> None:
    """Usage then comes from the sink: the running total of the chunks read."""
    with patch("coffee_with_llm.providers.google.text_client.genai.Client") as mock_genai:
        mock_genai.return_value.aio.models.generate_content_stream = (
            _gemini_stream_totalled_as_it_goes
        )
        llm = AskLLM(model="google/gemini-3.8-flash", config=_config(), google_explicit_cache=False)
        stream = await llm.ask(prompt="Draw the board", stream=True)
        assert isinstance(stream, StreamResult)
        async for _event in stream:
            break
        await stream.aclose()

    _assert_the_gemini_board(stream.usage)


# --- Gemini Interactions -----------------------------------------------------------


@pytest.mark.asyncio
async def test_an_interaction_leaves_the_cache_out_of_input() -> None:
    interaction = SimpleNamespace(
        id="int-1",
        model="gemini-3.8-flash",
        output_text="ok",
        steps=[],
        outputs=[],
        status="completed",
        usage=InteractionUsage(
            total_input_tokens=50_000,
            total_cached_tokens=40_000,
            total_output_tokens=800,
            total_thought_tokens=200,
            total_tokens=51_000,
        ),
    )
    with patch("coffee_with_llm.providers.google.interactions_client.genai.Client") as mock_genai:
        mock_genai.return_value.aio.interactions.create = AsyncMock(return_value=interaction)
        llm = AskLLM(model="google/gemini-3.8-flash", config=_config())
        result = await llm.ask_interaction(prompt="Draw the board")

    usage = result.usage
    assert usage.input_tokens == 10_000
    assert usage.cached_tokens == 40_000
    assert usage.prompt_tokens == 50_000
    assert usage.total_tokens == 10_000 + 1_000
    assert usage.cost_usd == _disjoint_cost(
        "gemini-3.8-flash", uncached=10_000, cached=40_000, output=1_000
    )


# --- OpenAI Responses ------------------------------------------------------------------


def _openai_usage() -> ResponseUsage:
    # From a dict: newer SDKs require cache_write_tokens, which older ones do not know.
    return ResponseUsage.model_validate(
        {
            "input_tokens": 20_000,
            "input_tokens_details": {"cached_tokens": 16_000, "cache_write_tokens": 0},
            "output_tokens": 1_000,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 21_000,
        }
    )


def _openai_response() -> SimpleNamespace:
    return SimpleNamespace(
        id="resp_1",
        model="gpt-5.4-2026-03-05",
        output_text="ok",
        output=[],
        required_action=None,
        status="completed",
        incomplete_details=None,
        usage=_openai_usage(),
    )


def _assert_the_openai_reply(usage: TokenUsage | None) -> None:
    assert usage is not None
    assert usage.input_tokens == 4_000
    assert usage.cached_tokens == 16_000
    assert usage.prompt_tokens == 20_000
    assert usage.total_tokens == 4_000 + 1_000
    assert usage.cost_usd == _disjoint_cost("gpt-5.4", uncached=4_000, cached=16_000, output=1_000)
    # 4,000 at $2.50 + 16,000 at $0.25 + 1,000 at $15 per 1M. Unread, it was $0.065.
    assert usage.cost_usd == 0.029


@pytest.mark.asyncio
async def test_openai_reads_the_cache_from_the_input_details() -> None:
    with patch("openai.AsyncOpenAI") as mock_openai:
        mock_openai.return_value.responses.create = AsyncMock(return_value=_openai_response())
        llm = AskLLM(model="gpt-5.4", config=_config())
        result = await llm.ask(prompt="Draw the board")

    assert isinstance(result, AskResult)
    _assert_the_openai_reply(result.usage)


class _OpenAIStream:
    """``client.responses.stream()``: events, then the final response."""

    def __init__(self, events: List[Any], final: Any) -> None:
        self._events = events
        self._final = final

    async def __aenter__(self) -> _OpenAIStream:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def _iterate(self) -> AsyncIterator[Any]:
        for event in self._events:
            yield event

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._iterate()

    async def get_final_response(self) -> Any:
        return self._final


@pytest.mark.asyncio
async def test_an_openai_stream_reads_the_cache_from_the_input_details() -> None:
    final = _openai_response()
    events = [
        SimpleNamespace(type="response.output_text.delta", delta="ok"),
        SimpleNamespace(type="response.completed", response=final),
    ]
    with patch("openai.AsyncOpenAI") as mock_openai:
        mock_openai.return_value.responses.stream = MagicMock(
            return_value=_OpenAIStream(events, final)
        )
        llm = AskLLM(model="gpt-5.4", config=_config())
        stream = await llm.ask(prompt="Draw the board", stream=True)
        assert isinstance(stream, StreamResult)
        _ = [event async for event in stream]

    _assert_the_openai_reply(stream.usage)


# --- Inception ---------------------------------------------------------------------


def _inception_reply(content: str, *, prompt: int, cached: int, output: int) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=None, reasoning_summary=None)
    return SimpleNamespace(
        model="mercury-2",
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=CompletionUsage(
            prompt_tokens=prompt,
            completion_tokens=output,
            total_tokens=prompt + output,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=cached),
        ),
    )


@pytest.mark.asyncio
async def test_inception_leaves_the_cache_out_of_input() -> None:
    reply = _inception_reply("ok", prompt=100_000, cached=80_000, output=2_000)
    with patch("openai.AsyncOpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create = AsyncMock(return_value=reply)
        llm = AskLLM(model="mercury-2", config=_config())
        result = await llm.ask(prompt="Draw the board")

    assert isinstance(result, AskResult)
    usage = result.usage
    assert usage.input_tokens == 20_000
    assert usage.cached_tokens == 80_000
    assert usage.prompt_tokens == 100_000
    assert usage.total_tokens == 20_000 + 2_000
    assert usage.cost_usd == _disjoint_cost(
        "mercury-2", uncached=20_000, cached=80_000, output=2_000
    )


@pytest.mark.asyncio
async def test_inception_counts_the_cache_of_the_call_that_finalizes() -> None:
    """An empty answer is asked once more; that call's cache read was dropped."""
    empty = _inception_reply("", prompt=10_000, cached=6_000, output=100)
    final = _inception_reply("ok", prompt=10_050, cached=9_000, output=200)
    with patch("openai.AsyncOpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create = AsyncMock(side_effect=[empty, final])
        client = InceptionChatClient(config=_config())
        text, usage, _stop = await client.generate(prompt="Draw the board", model="mercury-2")

    assert text == "ok"
    assert usage.input_tokens == 4_000 + 1_050
    assert usage.cached_tokens == 6_000 + 9_000
    assert usage.prompt_tokens == 10_000 + 10_050
    assert usage.output_tokens == 100 + 200


# --- Anthropic ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_buckets_are_billed_once_each() -> None:
    """Already disjoint: none is taken out of another."""
    response = MagicMock()
    response.content = [{"type": "text", "text": "ok"}]
    response.stop_reason = "end_turn"
    response.model = "claude-sonnet-4-6"
    response.usage = AnthropicUsage(
        input_tokens=10_000,
        cache_read_input_tokens=4_000,
        cache_creation_input_tokens=2_000,
        output_tokens=500,
    )
    fake_anthropic = MagicMock()
    fake_anthropic.AsyncAnthropic.return_value.messages.create = AsyncMock(return_value=response)

    with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
        llm = AskLLM(model="claude-sonnet-4-6", config=_config())
        result = await llm.ask(prompt="Draw the board")

    assert isinstance(result, AskResult)
    usage = result.usage
    assert usage.input_tokens == 10_000
    assert usage.cached_tokens == 4_000
    assert usage.cache_creation_tokens == 2_000
    assert usage.prompt_tokens == 16_000
    assert usage.cost_usd == _disjoint_cost(
        "claude-sonnet-4-6", uncached=10_000, cached=4_000, output=500, cache_creation=2_000
    )
    # 10,000 at $3 + 4,000 at $0.30 + 2,000 at $3.75 + 500 at $15 per 1M.
    # Taking the read out of input charged $0.0342.
    assert usage.cost_usd == 0.0462
