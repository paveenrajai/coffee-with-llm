"""Tests for Anthropic provider."""

from unittest.mock import MagicMock, patch

import pytest

from coffee_with_llm import Config
from coffee_with_llm.exceptions import ConfigurationError
from coffee_with_llm.providers.anthropic import AnthropicMessagesClient
from coffee_with_llm.providers.anthropic.messages_client import (
    _accumulate_anthropic_usage,
    _apply_prompt_cache,
    _apply_thinking,
    _convert_tools_to_anthropic,
    _prepare_anthropic_request_params,
    anthropic_uses_adaptive_thinking,
)
from coffee_with_llm.providers.tool_utils import normalize_tool_result


def _config(anthropic_api_key="test-key"):
    return Config(
        openai_api_key=None,
        anthropic_api_key=anthropic_api_key,
        google_api_key=None,
        request_timeout=60.0,
    )


class TestAnthropicMessagesClientInitialization:
    """Tests for AnthropicMessagesClient initialization."""

    def test_init_without_api_key(self):
        """Test that missing API key raises ConfigurationError."""
        cfg = Config(
            openai_api_key=None, anthropic_api_key=None, google_api_key=None, request_timeout=60.0
        )
        with pytest.raises(ConfigurationError, match="Anthropic.*not configured"):
            AnthropicMessagesClient(config=cfg)

    def test_init_with_api_key(self):
        """Test successful initialization with API key."""
        fake_anthropic = MagicMock()
        with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
            client = AnthropicMessagesClient(config=_config())
            assert client._api_key == "test-key"
            assert client._anthropic_prompt_cache is True

    def test_init_prompt_cache_disabled(self):
        fake_anthropic = MagicMock()
        with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
            client = AnthropicMessagesClient(config=_config(), anthropic_prompt_cache=False)
            assert client._anthropic_prompt_cache is False

    def test_init_with_missing_anthropic_package(self):
        """Test that missing Anthropic package raises ConfigurationError."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "anthropic":
                raise ImportError("No module named 'anthropic'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ConfigurationError, match="Anthropic package not installed"):
                AnthropicMessagesClient(config=_config())


class TestConvertToolsToAnthropic:
    """Tests for _convert_tools_to_anthropic."""

    def test_convert_openai_style_function(self):
        """Test converting OpenAI-style function tool."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ]
        result = _convert_tools_to_anthropic(tools)
        assert len(result) == 1
        assert result[0]["name"] == "get_weather"
        assert result[0]["description"] == "Get weather"
        assert result[0]["input_schema"]["type"] == "object"
        assert "location" in result[0]["input_schema"]["properties"]

    def test_convert_already_anthropic_format(self):
        """Test passing through already-Anthropic format."""
        tools = [
            {
                "name": "get_time",
                "description": "Get time",
                "input_schema": {"type": "object", "properties": {"tz": {"type": "string"}}},
            }
        ]
        result = _convert_tools_to_anthropic(tools)
        assert len(result) == 1
        assert result[0]["name"] == "get_time"
        assert result[0]["input_schema"]["type"] == "object"

    def test_convert_empty_list(self):
        """Test empty tools list."""
        assert _convert_tools_to_anthropic([]) == []


class TestAnthropicMessagesClientNormalizeToolResult:
    """Tests for normalize_tool_result (shared tool_utils)."""

    def test_normalize_with_ok_attribute(self):
        """Test normalization with object having ok attribute."""
        mock_result = MagicMock()
        mock_result.ok = True
        mock_result.result = {"data": "test"}
        mock_result.error = None

        normalized = normalize_tool_result(mock_result)
        assert normalized == {"ok": True, "result": {"data": "test"}, "error": None}

    def test_normalize_with_dict(self):
        """Test normalization with dict."""
        result_dict = {"ok": True, "result": {"data": "test"}, "error": None}
        normalized = normalize_tool_result(result_dict)
        assert normalized == result_dict

    def test_normalize_with_invalid_input(self):
        """Test normalization with invalid input."""
        normalized = normalize_tool_result("invalid")
        assert normalized == {"ok": False, "result": {}, "error": None}


class TestAnthropicUsesAdaptiveThinking:
    def test_opus_4_8_and_sonnet_4_6(self):
        assert anthropic_uses_adaptive_thinking("claude-opus-4-8")
        assert anthropic_uses_adaptive_thinking("claude-sonnet-4-6")
        assert anthropic_uses_adaptive_thinking("claude-mythos-preview")

    def test_gen5_models(self):
        assert anthropic_uses_adaptive_thinking("sonnet-5")
        assert anthropic_uses_adaptive_thinking("claude-sonnet-5")
        assert anthropic_uses_adaptive_thinking("claude-fable-5")

    def test_legacy_models(self):
        assert not anthropic_uses_adaptive_thinking("claude-sonnet-4-5")
        assert not anthropic_uses_adaptive_thinking("")
        assert not anthropic_uses_adaptive_thinking("gpt-4o")


class TestPrepareAnthropicRequestParams:
    def test_applies_thinking_and_cache_together(self):
        params = _prepare_anthropic_request_params(
            model="claude-opus-4-8",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=4096,
            system="You are helpful.",
            top_p=None,
            temperature=None,
            anthropic_tools=[],
            force_tool_use=False,
            response_format=None,
            reasoning_effort="high",
            prompt_cache=True,
        )
        assert params["thinking"] == {"type": "adaptive"}
        assert params["output_config"] == {"effort": "high"}
        assert params["cache_control"] == {"type": "ephemeral"}
        assert params["system"] == "You are helpful."

    def test_sonnet_5_without_reasoning_disables_thinking(self):
        params = _prepare_anthropic_request_params(
            model="sonnet-5",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=16384,
            system=None,
            top_p=0.9,
            temperature=0.7,
            anthropic_tools=[],
            force_tool_use=False,
            response_format=None,
            reasoning_effort=None,
            prompt_cache=False,
        )
        assert params["model"] == "claude-sonnet-5"
        assert params["thinking"] == {"type": "disabled"}
        assert "temperature" not in params
        assert "top_p" not in params
        assert params["max_tokens"] == 16384

    def test_sonnet_5_with_reasoning_uses_adaptive(self):
        params = _prepare_anthropic_request_params(
            model="sonnet-5",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=4096,
            system=None,
            top_p=None,
            temperature=None,
            anthropic_tools=[],
            force_tool_use=False,
            response_format=None,
            reasoning_effort="high",
            prompt_cache=False,
        )
        assert params["model"] == "claude-sonnet-5"
        assert params["thinking"] == {"type": "adaptive"}
        assert params["output_config"] == {"effort": "high"}

    def test_fable_5_without_reasoning_minimizes_thinking(self):
        params = _prepare_anthropic_request_params(
            model="claude-fable-5",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=16384,
            system=None,
            top_p=None,
            temperature=None,
            anthropic_tools=[],
            force_tool_use=False,
            response_format=None,
            reasoning_effort=None,
            prompt_cache=False,
        )
        assert params["thinking"] == {"type": "adaptive"}
        assert params["output_config"] == {"effort": "low"}


class TestApplyPromptCache:
    def test_adds_top_level_cache_control(self):
        params: dict = {"model": "claude-sonnet-4-6"}
        _apply_prompt_cache(params, True)
        assert params["cache_control"] == {"type": "ephemeral"}

    def test_no_op_when_disabled(self):
        params: dict = {"model": "claude-sonnet-4-6"}
        _apply_prompt_cache(params, False)
        assert "cache_control" not in params

    def test_no_op_when_already_set(self):
        params = {"cache_control": {"type": "ephemeral", "ttl": "1h"}}
        _apply_prompt_cache(params, True)
        assert params["cache_control"]["ttl"] == "1h"


class TestAccumulateAnthropicUsage:
    def test_sums_cache_read_and_creation_tokens(self):
        usage = MagicMock(
            input_tokens=100,
            output_tokens=20,
            cache_read_input_tokens=50,
            cache_creation_input_tokens=200,
        )
        inp, out, cached, created = _accumulate_anthropic_usage(
            usage,
            total_input=10,
            total_output=5,
            total_cached=0,
            total_cache_creation=0,
        )
        assert inp == 110
        assert out == 25
        assert cached == 50
        assert created == 200


class TestApplyThinking:
    """Tests for _apply_thinking — provider-agnostic reasoning_effort plumbing."""

    def test_no_op_when_effort_missing_on_legacy_model(self):
        params = {
            "model": "claude-sonnet-4-5",
            "max_tokens": 4096,
            "temperature": 0.7,
            "top_p": 0.9,
        }
        snapshot = dict(params)
        _apply_thinking(params, None)
        assert params == snapshot

    def test_sonnet_5_disables_thinking_when_effort_missing(self):
        params = {"model": "sonnet-5", "max_tokens": 16384}
        _apply_thinking(params, None)
        assert params["thinking"] == {"type": "disabled"}

    def test_fable_5_minimizes_thinking_when_effort_missing(self):
        params = {"model": "claude-fable-5", "max_tokens": 16384}
        _apply_thinking(params, None)
        assert params["thinking"] == {"type": "adaptive"}
        assert params["output_config"] == {"effort": "low"}

    def test_no_op_when_effort_unknown(self):
        params = {"max_tokens": 4096, "top_p": 0.9}
        snapshot = dict(params)
        _apply_thinking(params, "ultra")
        assert params == snapshot

    def test_high_effort_legacy_model_sets_budget_thinking(self):
        params = {
            "model": "claude-sonnet-4-5",
            "max_tokens": 4096,
            "temperature": 0.3,
            "top_p": 0.9,
            "top_k": 40,
        }
        _apply_thinking(params, "high")

        assert params["thinking"] == {"type": "enabled", "budget_tokens": 16384}
        assert params["temperature"] == 1
        assert "top_p" not in params
        assert "top_k" not in params
        assert params["max_tokens"] >= 16384 + 1024

    def test_high_effort_adaptive_model_sets_output_config(self):
        params = {"model": "claude-opus-4-8", "max_tokens": 4096, "temperature": 0.3, "top_p": 0.9}
        _apply_thinking(params, "High")

        assert params["thinking"] == {"type": "adaptive"}
        assert params["output_config"] == {"effort": "high"}
        assert params["temperature"] == 0.3
        assert params["top_p"] == 0.9
        assert params["max_tokens"] == 16_000

    def test_sonnet_5_high_effort_uses_adaptive_not_legacy(self):
        params = {"model": "sonnet-5", "max_tokens": 4096}
        _apply_thinking(params, "high")
        assert params["thinking"] == {"type": "adaptive"}
        assert params["output_config"] == {"effort": "high"}
        assert "budget_tokens" not in str(params.get("thinking", {}))

    def test_adaptive_merges_existing_output_config(self):
        params = {
            "model": "claude-opus-4-8",
            "max_tokens": 20_000,
            "output_config": {"format": {"type": "json_schema"}},
        }
        _apply_thinking(params, "medium")

        assert params["thinking"] == {"type": "adaptive"}
        assert params["output_config"]["effort"] == "medium"
        assert params["output_config"]["format"]["type"] == "json_schema"

    def test_low_effort_legacy_keeps_caller_max_tokens_when_sufficient(self):
        params = {"model": "claude-sonnet-4-5", "max_tokens": 32_000}
        _apply_thinking(params, "low")
        assert params["thinking"] == {"type": "enabled", "budget_tokens": 1024}
        assert params["max_tokens"] == 32_000

    def test_widens_max_tokens_when_too_small_legacy(self):
        params = {"model": "claude-sonnet-4-5", "max_tokens": 100}
        _apply_thinking(params, "medium")
        assert params["max_tokens"] >= 4096 + 1024


class TestAnthropicMessagesClientGenerate:
    """Tests for generate method."""

    @pytest.mark.asyncio
    async def test_generate_basic(self):
        """Test basic generation."""
        fake_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [{"type": "text", "text": "Test response"}]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)

        async def mock_create(*args, **kwargs):
            return mock_response

        mock_client.messages.create = mock_create
        fake_anthropic.AsyncAnthropic = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
            client = AnthropicMessagesClient(config=_config())
            text, usage = await client.generate(prompt="What is Python?", model="claude-sonnet-4-6")
            assert text == "Test response"
            assert usage is not None
            assert usage.input_tokens == 10
            assert usage.output_tokens == 5

    @pytest.mark.asyncio
    async def test_generate_with_system_instruction(self):
        """Test generation with system instruction."""
        fake_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [{"type": "text", "text": "Hello"}]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock(input_tokens=5, output_tokens=3)

        async def mock_create(*args, **kwargs):
            assert kwargs.get("system") == "You are helpful."
            return mock_response

        mock_client.messages.create = mock_create
        fake_anthropic.AsyncAnthropic = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
            client = AnthropicMessagesClient(config=_config())
            text, usage = await client.generate(
                prompt="Hi",
                model="claude-sonnet-4-6",
                instructions="You are helpful.",
            )
            assert text == "Hello"
