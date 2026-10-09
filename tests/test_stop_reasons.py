"""Why a call stopped, and how much of its output was thinking, for every provider.

A stream cut off at ``max_tokens`` ends exactly like a finished one. On
2026-10-09 a Gemini board thought for two minutes, used the 16,384-token cap,
wrote one line and was stored as complete: nothing said it had been cut off,
and ``output_tokens`` left out the 15,000 tokens of thinking that used the cap.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, AsyncIterator, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coffee_with_llm import AskLLM, Config, estimate_cost
from coffee_with_llm.llm import _generated
from coffee_with_llm.providers._stop import stop_from
from coffee_with_llm.providers.anthropic.messages_client import AnthropicMessagesClient
from coffee_with_llm.providers.google import GoogleTextClient
from coffee_with_llm.providers.google.text_client import _gemini_stop
from coffee_with_llm.providers.inception.chat_client import InceptionChatClient
from coffee_with_llm.providers.openai.responses_client import OpenAIResponsesClient
from coffee_with_llm.types import (
    AskResult,
    Stop,
    StopReason,
    StreamResult,
    StreamTextDelta,
    TokenUsage,
)


def _config() -> Config:
    return Config(
        openai_api_key="test-key",
        anthropic_api_key="test-key",
        google_api_key="test-key",
        inception_api_key="test-key",
        request_timeout=60.0,
    )


# --- shared ------------------------------------------------------------------


def test_a_word_the_mapping_does_not_know_is_kept_as_other() -> None:
    """A provider adding a reason must not make a stop look like a finish."""
    assert stop_from("pause_turn", {"end_turn": StopReason.END}) == Stop(
        StopReason.OTHER, "pause_turn"
    )
    assert stop_from(SimpleNamespace(name="MAX_TOKENS"), {"MAX_TOKENS": "max_tokens"}) == Stop(
        StopReason.MAX_TOKENS, "MAX_TOKENS"
    )
    assert stop_from(None, {}) is None
    assert stop_from("", {}) is None


def test_only_max_tokens_is_truncated() -> None:
    assert Stop(StopReason.MAX_TOKENS, "length").truncated
    assert not Stop(StopReason.END, "STOP").truncated


def test_reasoning_tokens_survive_a_round_trip() -> None:
    usage = TokenUsage(10, 600, 610, reasoning_tokens=500)
    assert TokenUsage.from_mapping(usage.to_dict()) == usage
    assert TokenUsage.from_mapping({"input_tokens": 1}).reasoning_tokens is None


def test_a_provider_that_returns_text_and_usage_still_works() -> None:
    """A provider registered from outside may not say why it stopped."""
    usage = TokenUsage(1, 2, 3)
    assert _generated(("hi", usage)) == ("hi", usage, None)
    stop = Stop(StopReason.END, "STOP")
    assert _generated(("hi", usage, stop)) == ("hi", usage, stop)


@pytest.mark.asyncio
async def test_a_stream_keeps_its_stop_and_never_yields_it() -> None:
    """Consumers that know only the events they asked for never meet a Stop."""

    async def stream() -> AsyncIterator[object]:
        yield StreamTextDelta("On 26 April 1986")
        yield Stop(StopReason.MAX_TOKENS, "MAX_TOKENS")
        yield TokenUsage(10, 15_600, 15_610, reasoning_tokens=15_000)

    result = StreamResult(stream)
    events = [event async for event in result]

    assert events == [StreamTextDelta("On 26 April 1986")]
    assert result.stop is not None and result.stop.truncated
    assert result.usage is not None and result.usage.reasoning_tokens == 15_000


# --- Gemini ------------------------------------------------------------------


def _gemini_usage(*, output: int, thoughts: int | None) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_token_count=40_000,
        candidates_token_count=output,
        thoughts_token_count=thoughts,
        cached_content_token_count=None,
    )


def _gemini_response(text: str, finish: str, *, thoughts: int | None) -> SimpleNamespace:
    candidate = SimpleNamespace(
        finish_reason=SimpleNamespace(name=finish),
        content=SimpleNamespace(role="model", parts=[SimpleNamespace(text=text)]),
    )
    return SimpleNamespace(
        text=text,
        candidates=[candidate],
        usage_metadata=_gemini_usage(output=616, thoughts=thoughts),
    )


def _gemini_client(**models: Any) -> GoogleTextClient:
    with patch("coffee_with_llm.providers.google.text_client.genai.Client") as mock_genai:
        for name, fn in models.items():
            setattr(mock_genai.return_value.aio.models, name, fn)
        return GoogleTextClient(config=_config(), google_explicit_cache=False)


@pytest.mark.asyncio
async def test_gemini_cut_off_while_thinking_says_so_and_counts_the_thinking() -> None:
    async def generate_content(**_kwargs: Any) -> Any:
        return _gemini_response("On 26 April 1986…", "MAX_TOKENS", thoughts=15_768)

    client = _gemini_client(generate_content=generate_content)
    text, usage, stop = await client.generate(prompt="What happened?", model="gemini-flash")

    assert text == "On 26 April 1986…"
    assert stop == Stop(StopReason.MAX_TOKENS, "MAX_TOKENS")
    # Billed as output, and how much of it was thinking.
    assert usage.output_tokens == 616 + 15_768
    assert usage.reasoning_tokens == 15_768
    assert usage.total_tokens == 40_000 + 616 + 15_768


@pytest.mark.asyncio
async def test_gemini_with_no_thinking_reports_none() -> None:
    async def generate_content(**_kwargs: Any) -> Any:
        return _gemini_response("Done.", "STOP", thoughts=None)

    client = _gemini_client(generate_content=generate_content)
    _text, usage, stop = await client.generate(prompt="Hi", model="gemini-flash")

    assert stop == Stop(StopReason.END, "STOP")
    assert usage.output_tokens == 616
    assert usage.reasoning_tokens is None


@pytest.mark.asyncio
async def test_gemini_stream_ends_with_its_stop_before_its_usage() -> None:
    chunks = [
        SimpleNamespace(text="On 26 April", candidates=[], usage_metadata=None),
        SimpleNamespace(
            text=" 1986…",
            candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="MAX_TOKENS"))],
            usage_metadata=_gemini_usage(output=616, thoughts=15_768),
        ),
    ]

    async def generate_content_stream(**_kwargs: Any) -> Any:
        async def gen() -> AsyncIterator[Any]:
            for chunk in chunks:
                yield chunk

        return gen()

    client = _gemini_client(generate_content_stream=generate_content_stream)
    items: List[object] = [
        item async for item in client.generate_stream(prompt="What happened?", model="gemini")
    ]

    assert isinstance(items[-2], Stop) and items[-2].truncated
    usage = items[-1]
    assert isinstance(usage, TokenUsage)
    assert usage.output_tokens == 616 + 15_768
    assert usage.reasoning_tokens == 15_768


def test_gemini_still_asking_for_a_tool_is_tool_use_not_end() -> None:
    """A function call ends with STOP, like a finished answer."""
    resp = SimpleNamespace(candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))])
    assert _gemini_stop(resp, wants_tools=True) == Stop(StopReason.TOOL_USE, "STOP")
    assert _gemini_stop(resp) == Stop(StopReason.END, "STOP")


def test_gemini_prompt_blocked_before_any_answer_is_content_filter() -> None:
    resp = SimpleNamespace(
        candidates=[],
        prompt_feedback=SimpleNamespace(block_reason=SimpleNamespace(name="SAFETY")),
    )
    assert _gemini_stop(resp) == Stop(StopReason.CONTENT_FILTER, "SAFETY")


# --- Anthropic ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_cut_off_says_so() -> None:
    """Thinking is inside output_tokens on Anthropic, and not counted apart."""
    response = MagicMock()
    response.content = [{"type": "text", "text": "On 26 April 1986…"}]
    response.stop_reason = "max_tokens"
    response.usage = MagicMock(input_tokens=10, output_tokens=16_384)
    fake_anthropic = MagicMock()
    fake_anthropic.AsyncAnthropic.return_value.messages.create = AsyncMock(return_value=response)

    with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
        client = AnthropicMessagesClient(config=_config())
        _text, usage, stop = await client.generate(prompt="What happened?", model="claude")

    assert stop == Stop(StopReason.MAX_TOKENS, "max_tokens")
    assert usage.output_tokens == 16_384
    assert usage.reasoning_tokens is None


# --- OpenAI ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_incomplete_says_why_and_counts_the_reasoning() -> None:
    response = SimpleNamespace(
        id="resp_1",
        output_text="On 26 April 1986…",
        output=[],
        required_action=None,
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=1_000,
            total_tokens=1_010,
            cached_tokens=None,
            output_tokens_details=SimpleNamespace(reasoning_tokens=960),
        ),
    )
    with patch("openai.AsyncOpenAI") as mock_openai:
        mock_openai.return_value.responses.create = AsyncMock(return_value=response)
        client = OpenAIResponsesClient(config=_config())
        _text, usage, stop = await client.generate(prompt="What happened?", model="gpt-5")

    assert stop == Stop(StopReason.MAX_TOKENS, "max_output_tokens")
    # Already inside output_tokens on OpenAI: nothing is added.
    assert usage.output_tokens == 1_000
    assert usage.reasoning_tokens == 960


# --- Inception ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_inception_length_is_max_tokens() -> None:
    message = SimpleNamespace(content="On 26 April 1986…", tool_calls=None, reasoning_summary=None)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="length")],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=500,
            total_tokens=510,
            cached_tokens=None,
            prompt_tokens_details=None,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=420),
        ),
    )
    with patch("openai.AsyncOpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create = AsyncMock(return_value=response)
        client = InceptionChatClient(config=_config())
        _text, usage, stop = await client.generate(prompt="What happened?", model="mercury-2")

    assert stop == Stop(StopReason.MAX_TOKENS, "length")
    assert usage.output_tokens == 500
    assert usage.reasoning_tokens == 420


# --- through AskLLM ------------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_carries_the_stop_and_prices_the_thinking() -> None:
    """What a caller reads: AskResult.stop, and a cost that includes the thinking."""

    async def generate_content(**_kwargs: Any) -> Any:
        return _gemini_response("On 26 April 1986…", "MAX_TOKENS", thoughts=15_768)

    with patch("coffee_with_llm.providers.google.text_client.genai.Client") as mock_genai:
        mock_genai.return_value.aio.models.generate_content = generate_content
        llm = AskLLM(model="google/gemini-2.5-flash", config=_config(), google_explicit_cache=False)
        result = await llm.ask(prompt="What happened?")

    assert isinstance(result, AskResult)
    assert result.stop is not None and result.stop.truncated
    assert result.usage.reasoning_tokens == 15_768
    thinking_left_out = TokenUsage(40_000, 616, 40_616)
    assert result.usage.cost_usd is not None
    assert result.usage.cost_usd > (estimate_cost(thinking_left_out, "gemini-2.5-flash") or 0)


@pytest.mark.asyncio
async def test_a_streamed_ask_says_why_it_stopped_once_it_ends() -> None:
    async def generate_content_stream(**_kwargs: Any) -> Any:
        async def gen() -> AsyncIterator[Any]:
            yield _gemini_response("On 26 April 1986…", "MAX_TOKENS", thoughts=15_768)

        return gen()

    with patch("coffee_with_llm.providers.google.text_client.genai.Client") as mock_genai:
        mock_genai.return_value.aio.models.generate_content_stream = generate_content_stream
        llm = AskLLM(model="google/gemini-2.5-flash", config=_config(), google_explicit_cache=False)
        stream = await llm.ask(prompt="What happened?", stream=True)
        assert isinstance(stream, StreamResult)
        texts = [event.text async for event in stream if isinstance(event, StreamTextDelta)]

    assert texts == ["On 26 April 1986…"]
    assert stream.stop == Stop(StopReason.MAX_TOKENS, "MAX_TOKENS")
    assert stream.usage is not None and stream.usage.reasoning_tokens == 15_768
