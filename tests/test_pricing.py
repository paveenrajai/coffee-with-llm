"""A call is priced at the model that served it, at that day's price.

On 2026-10-09 deepgrasp recorded every board at $0: gemini-3.8-flash had no
price. Its router, on ``gemini-flash-lite-latest``, was priced as 2.5
Flash-Lite while Google served it 3.5 Flash-Lite, at three times the input
price and six times the output. An alias is swapped to a new model with every
release, so its price is the price of whatever served the call.
"""

from __future__ import annotations

import logging
from datetime import date
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coffee_with_llm import AskLLM, Config, estimate_cost
from coffee_with_llm.providers._served import served_model
from coffee_with_llm.providers.anthropic.messages_client import AnthropicMessagesClient
from coffee_with_llm.providers.inception.chat_client import InceptionChatClient
from coffee_with_llm.providers.openai.responses_client import OpenAIResponsesClient
from coffee_with_llm.types import AskResult, StreamResult, TokenUsage

MILLION = TokenUsage(1_000_000, 1_000_000, 2_000_000)


def _config() -> Config:
    return Config(
        openai_api_key="test-key",
        anthropic_api_key="test-key",
        google_api_key="test-key",
        inception_api_key="test-key",
        request_timeout=60.0,
    )


# --- the table ---------------------------------------------------------------


@pytest.mark.parametrize("model", ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash"])
def test_flash_is_at_its_launch_price_through_2026_and_doubles_after(model: str) -> None:
    assert estimate_cost(MILLION, model, on=date(2026, 12, 31)) == 0.75 + 3.75
    assert estimate_cost(MILLION, model, on=date(2027, 1, 1)) == 1.50 + 7.50


def test_the_other_gemini_3_models_are_priced() -> None:
    assert estimate_cost(MILLION, "gemini-3.5-flash") == 1.50 + 9.00
    assert estimate_cost(MILLION, "gemini-3.5-flash-lite") == 0.30 + 2.50
    assert estimate_cost(MILLION, "gemini-3.1-flash-lite") == 0.25 + 1.50


def test_flash_lite_is_not_priced_as_flash() -> None:
    """2.5 Flash-Lite came after 2.5 Flash in the table, so it matched Flash."""
    assert estimate_cost(MILLION, "gemini-2.5-flash-lite") == 0.10 + 0.40
    assert estimate_cost(MILLION, "gemini-2.5-flash") == 0.30 + 2.50


@pytest.mark.parametrize(
    "alias", ["gemini-flash-lite-latest", "gemini-flash-latest", "gemini-pro-latest"]
)
def test_an_alias_has_no_price_of_its_own(alias: str) -> None:
    assert estimate_cost(MILLION, alias) is None


def test_only_a_name_counts_as_the_served_model() -> None:
    assert served_model("models/gemini-3.5-flash-lite") == "gemini-3.5-flash-lite"
    assert served_model("  ") is None
    assert served_model(None) is None
    assert served_model(MagicMock()) is None


# --- AskLLM prices what served the call ----------------------------------------


def _gemini_reply(served: str | None) -> SimpleNamespace:
    candidate = SimpleNamespace(
        finish_reason=SimpleNamespace(name="STOP"),
        content=SimpleNamespace(role="model", parts=[SimpleNamespace(text="ok")]),
    )
    return SimpleNamespace(
        text="ok",
        model_version=served,
        candidates=[candidate],
        usage_metadata=SimpleNamespace(
            prompt_token_count=1_000_000,
            candidates_token_count=1_000_000,
            thoughts_token_count=None,
            cached_content_token_count=None,
        ),
    )


async def _ask_gemini(model: str, served: str | None) -> AskResult:
    async def generate_content(**_kwargs: Any) -> Any:
        return _gemini_reply(served)

    with patch("coffee_with_llm.providers.google.text_client.genai.Client") as mock_genai:
        mock_genai.return_value.aio.models.generate_content = generate_content
        llm = AskLLM(model=f"google/{model}", config=_config(), google_explicit_cache=False)
        result = await llm.ask(prompt="Hi")
    assert isinstance(result, AskResult)
    return result


@pytest.mark.asyncio
async def test_an_alias_is_priced_at_the_model_that_served_it() -> None:
    result = await _ask_gemini("gemini-flash-lite-latest", served="gemini-3.5-flash-lite")

    assert result.usage.served_model == "gemini-3.5-flash-lite"
    assert result.usage.cost_usd == 0.30 + 2.50


@pytest.mark.asyncio
async def test_a_model_that_does_not_say_is_priced_as_asked() -> None:
    result = await _ask_gemini("gemini-3.8-flash", served=None)

    assert result.usage.served_model is None
    assert result.usage.cost_usd == estimate_cost(MILLION, "gemini-3.8-flash")


@pytest.mark.asyncio
async def test_a_model_with_no_price_is_left_unpriced_and_said_so(caplog) -> None:
    """Never priced as some other model: a guess would hide the missing price."""
    caplog.set_level(logging.WARNING, logger="coffee_with_llm.llm")
    result = await _ask_gemini("gemini-flash-latest", served="gemini-9-flash")

    assert result.usage.cost_usd is None
    assert "gemini-9-flash" in caplog.text and "gemini-flash-latest" in caplog.text


# --- every provider says what served it ---------------------------------------


@pytest.mark.asyncio
async def test_anthropic_reports_the_model_that_answered() -> None:
    response = MagicMock()
    response.content = [{"type": "text", "text": "ok"}]
    response.stop_reason = "end_turn"
    response.model = "claude-sonnet-4-6-20260101"
    response.usage = MagicMock(input_tokens=10, output_tokens=5)
    fake_anthropic = MagicMock()
    fake_anthropic.AsyncAnthropic.return_value.messages.create = AsyncMock(return_value=response)

    with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
        client = AnthropicMessagesClient(config=_config())
        _text, usage, _stop = await client.generate(prompt="Hi", model="claude-sonnet-4-6")

    assert usage.served_model == "claude-sonnet-4-6-20260101"


@pytest.mark.asyncio
async def test_openai_reports_the_model_that_answered() -> None:
    response = SimpleNamespace(
        id="resp_1",
        model="gpt-5.4-2026-03-05",
        output_text="ok",
        output=[],
        required_action=None,
        status="completed",
        incomplete_details=None,
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            cached_tokens=None,
            output_tokens_details=None,
        ),
    )
    with patch("openai.AsyncOpenAI") as mock_openai:
        mock_openai.return_value.responses.create = AsyncMock(return_value=response)
        client = OpenAIResponsesClient(config=_config())
        _text, usage, _stop = await client.generate(prompt="Hi", model="gpt-5.4")

    assert usage.served_model == "gpt-5.4-2026-03-05"


@pytest.mark.asyncio
async def test_inception_reports_the_model_that_answered() -> None:
    message = SimpleNamespace(content="ok", tool_calls=None, reasoning_summary=None)
    response = SimpleNamespace(
        model="mercury-2",
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cached_tokens=None,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        ),
    )
    with patch("openai.AsyncOpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create = AsyncMock(return_value=response)
        client = InceptionChatClient(config=_config())
        _text, usage, _stop = await client.generate(prompt="Hi", model="mercury-2")

    assert usage.served_model == "mercury-2"


@pytest.mark.asyncio
async def test_a_gemini_stream_reports_the_model_that_served_it() -> None:
    async def generate_content_stream(**_kwargs: Any) -> Any:
        async def gen():
            yield _gemini_reply("gemini-3.5-flash-lite")

        return gen()

    with patch("coffee_with_llm.providers.google.text_client.genai.Client") as mock_genai:
        mock_genai.return_value.aio.models.generate_content_stream = generate_content_stream
        llm = AskLLM(
            model="google/gemini-flash-lite-latest", config=_config(), google_explicit_cache=False
        )
        stream = await llm.ask(prompt="Hi", stream=True)
        assert isinstance(stream, StreamResult)
        _ = [event async for event in stream]

    assert stream.usage is not None
    assert stream.usage.served_model == "gemini-3.5-flash-lite"
    assert stream.usage.cost_usd == 0.30 + 2.50
