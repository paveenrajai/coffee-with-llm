"""Tests for Inception provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coffee_with_llm import Config
from coffee_with_llm.attachments import Attachment
from coffee_with_llm.exceptions import ConfigurationError, ValidationError
from coffee_with_llm.providers.inception import InceptionChatClient
from coffee_with_llm.providers.inception.chat_client import (
    _convert_tools_to_openai,
    _normalize_response_format,
    _prepare_inception_request_params,
    normalize_inception_effort,
)
from coffee_with_llm.providers.inception.models import normalize_inception_model_id


def _config(inception_api_key="test-key"):
    return Config(
        openai_api_key=None,
        anthropic_api_key=None,
        google_api_key=None,
        inception_api_key=inception_api_key,
        request_timeout=60.0,
    )


class TestInceptionChatClientInitialization:
    def test_init_without_api_key(self):
        cfg = Config(
            openai_api_key=None,
            anthropic_api_key=None,
            google_api_key=None,
            inception_api_key=None,
            request_timeout=60.0,
        )
        with pytest.raises(ConfigurationError, match="Inception.*not configured"):
            InceptionChatClient(config=cfg)

    def test_init_with_api_key(self):
        with patch("openai.AsyncOpenAI"):
            client = InceptionChatClient(config=_config())
            assert client._api_key == "test-key"

    def test_init_with_missing_openai_package(self):
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "openai":
                raise ImportError("No module named 'openai'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ConfigurationError, match="OpenAI package not installed"):
                InceptionChatClient(config=_config())


class TestNormalizeInceptionModelId:
    def test_alias_mercury(self):
        assert normalize_inception_model_id("mercury") == "mercury-2"

    def test_mercury_2_passthrough(self):
        assert normalize_inception_model_id("mercury-2") == "mercury-2"

    def test_unknown_passthrough(self):
        assert normalize_inception_model_id("mercury-edit-2") == "mercury-edit-2"


class TestNormalizeInceptionEffort:
    def test_accepts_instant(self):
        assert normalize_inception_effort("instant") == "instant"

    def test_accepts_standard(self):
        assert normalize_inception_effort("HIGH") == "high"

    def test_unknown_returns_none(self, caplog):
        assert normalize_inception_effort("ultra") is None


class TestConvertToolsToOpenai:
    def test_nested_passthrough(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        result = _convert_tools_to_openai(tools)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_weather"

    def test_flat_to_nested(self):
        tools = [
            {
                "type": "function",
                "name": "ping",
                "description": "Ping",
                "parameters": {"type": "object"},
            }
        ]
        result = _convert_tools_to_openai(tools)
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "ping"


class TestPrepareInceptionRequestParams:
    def test_reasoning_effort_in_extra_body(self):
        params = _prepare_inception_request_params(
            model="mercury",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=None,
            top_p=None,
            temperature=None,
            presence_penalty=None,
            tools=[],
            force_tool_use=False,
            response_format=None,
            reasoning_effort="low",
        )
        assert params["model"] == "mercury-2"
        assert params["max_tokens"] == 8192
        assert params["extra_body"] == {"reasoning_effort": "low"}

    def test_force_tool_use_required(self):
        tools = _convert_tools_to_openai(
            [{"type": "function", "name": "x", "description": "", "parameters": {}}]
        )
        params = _prepare_inception_request_params(
            model="mercury-2",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
            top_p=None,
            temperature=0.75,
            presence_penalty=None,
            tools=tools,
            force_tool_use=True,
            response_format=None,
            reasoning_effort=None,
        )
        assert params["tool_choice"] == "required"
        assert params["temperature"] == 0.75
        assert "extra_body" not in params

    def test_response_format_json_string(self):
        assert _normalize_response_format("json") == {"type": "json_object"}


class TestInceptionGenerate:
    @pytest.mark.asyncio
    async def test_generate_basic(self):
        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_message = MagicMock()
            mock_message.content = "Hello from Mercury"
            mock_message.tool_calls = None
            mock_message.reasoning_summary = None
            mock_choice = MagicMock()
            mock_choice.message = mock_message
            mock_choice.finish_reason = "stop"
            mock_resp = MagicMock()
            mock_resp.choices = [mock_choice]
            mock_resp.usage = MagicMock(
                prompt_tokens=10, completion_tokens=5, total_tokens=15, cached_tokens=None
            )
            mock_resp.usage.prompt_tokens_details = None
            mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)
            mock_openai.return_value = mock_client

            client = InceptionChatClient(config=_config())
            text, usage = await client.generate(prompt="What is a dLLM?", model="mercury-2")
            assert text == "Hello from Mercury"
            assert usage.input_tokens == 10
            assert usage.output_tokens == 5
            mock_openai.assert_called()
            assert mock_openai.call_args.kwargs["base_url"] == "https://api.inceptionlabs.ai/v1"

    @pytest.mark.asyncio
    async def test_generate_with_tool_loop(self):
        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()

            fn = MagicMock()
            fn.name = "get_weather"
            fn.arguments = '{"location": "SF"}'
            tc = MagicMock()
            tc.id = "call_1"
            tc.function = fn

            tool_message = MagicMock()
            tool_message.content = None
            tool_message.tool_calls = [tc]
            tool_message.reasoning_summary = None
            tool_choice = MagicMock()
            tool_choice.message = tool_message
            tool_choice.finish_reason = "tool_calls"
            tool_resp = MagicMock()
            tool_resp.choices = [tool_choice]
            tool_resp.usage = MagicMock(
                prompt_tokens=8, completion_tokens=4, total_tokens=12, cached_tokens=None
            )
            tool_resp.usage.prompt_tokens_details = None

            final_message = MagicMock()
            final_message.content = "Sunny in SF"
            final_message.tool_calls = None
            final_message.reasoning_summary = None
            final_choice = MagicMock()
            final_choice.message = final_message
            final_choice.finish_reason = "stop"
            final_resp = MagicMock()
            final_resp.choices = [final_choice]
            final_resp.usage = MagicMock(
                prompt_tokens=20, completion_tokens=6, total_tokens=26, cached_tokens=None
            )
            final_resp.usage.prompt_tokens_details = None

            mock_client.chat.completions.create = AsyncMock(side_effect=[tool_resp, final_resp])
            mock_openai.return_value = mock_client

            async def execute(name, args):
                assert name == "get_weather"
                assert args["location"] == "SF"
                return {"ok": True, "result": {"temp": 72}}

            client = InceptionChatClient(config=_config())
            text, usage = await client.generate(
                prompt="Weather in SF?",
                model="mercury-2",
                tools_schema=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Weather",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                execute_tool_cb=execute,
                reasoning_effort="medium",
            )
            assert text == "Sunny in SF"
            assert usage.input_tokens == 28
            assert mock_client.chat.completions.create.await_count == 2
            first_kwargs = mock_client.chat.completions.create.await_args_list[0].kwargs
            assert first_kwargs["extra_body"] == {"reasoning_effort": "medium"}

    @pytest.mark.asyncio
    async def test_attachments_rejected(self):
        with patch("openai.AsyncOpenAI"):
            client = InceptionChatClient(config=_config())
            att = Attachment(data=b"%PDF-1.4", mime_type="application/pdf", filename="x.pdf")
            with pytest.raises(ValidationError, match="text-only"):
                await client.generate(
                    prompt="Read this",
                    model="mercury-2",
                    attachments=[att],
                )
