"""Tests for Gemini Interactions API client."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coffee_with_llm.exceptions import ConfigurationError, ValidationError
from coffee_with_llm.providers.google.interactions_client import GoogleInteractionsClient
from coffee_with_llm.providers.google.interactions_utils import (
    interaction_function_calls,
    interaction_text,
    interaction_usage,
)


def _config():
    cfg = MagicMock()
    cfg.require_google_key.return_value = "test-key"
    return cfg


def _text_block(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _function_call(call_id: str, name: str, arguments: dict) -> MagicMock:
    block = MagicMock()
    block.type = "function_call"
    block.id = call_id
    block.name = name
    block.arguments = arguments
    return block


class TestInteractionUtils:
    def test_interaction_text_joins_outputs(self):
        interaction = MagicMock()
        interaction.output_text = None
        interaction.steps = []
        interaction.outputs = [_text_block("Hello "), _text_block("world")]
        assert interaction_text(interaction) == "Hello world"

    def test_interaction_text_prefers_output_text(self):
        interaction = MagicMock()
        interaction.output_text = "From SDK sugar"
        interaction.steps = []
        interaction.outputs = [_text_block("ignored")]
        assert interaction_text(interaction) == "From SDK sugar"

    def test_interaction_text_joins_model_output_steps(self):
        content = MagicMock()
        content.type = "text"
        content.text = "Step text"
        model_output = MagicMock()
        model_output.type = "model_output"
        model_output.content = [content]
        interaction = MagicMock()
        interaction.output_text = None
        interaction.steps = [model_output]
        interaction.outputs = []
        assert interaction_text(interaction) == "Step text"

    def test_interaction_function_calls(self):
        interaction = MagicMock()
        interaction.steps = []
        interaction.outputs = [_function_call("c1", "add", {"a": 1, "b": 2})]
        calls = interaction_function_calls(interaction)
        assert calls == [{"id": "c1", "name": "add", "arguments": {"a": 1, "b": 2}}]

    def test_interaction_function_calls_from_steps(self):
        step = MagicMock()
        step.type = "function_call"
        step.id = "c2"
        step.name = "mul"
        step.arguments = {"a": 2, "b": 3}
        interaction = MagicMock()
        interaction.steps = [step]
        interaction.outputs = []
        calls = interaction_function_calls(interaction)
        assert calls == [{"id": "c2", "name": "mul", "arguments": {"a": 2, "b": 3}}]

    def test_interaction_usage_maps_totals(self):
        usage = MagicMock()
        usage.total_input_tokens = 10
        usage.total_output_tokens = 5
        usage.total_tokens = 15
        usage.total_cached_tokens = 2
        interaction = MagicMock()
        interaction.usage = usage
        mapped = interaction_usage(interaction)
        assert mapped.input_tokens == 10
        assert mapped.output_tokens == 5
        assert mapped.total_tokens == 15
        assert mapped.cached_tokens == 2


class TestGoogleInteractionsClient:
    def test_requires_interactions_on_client(self):
        with patch("coffee_with_llm.providers.google.interactions_client.genai.Client") as mock_cls:
            mock_cls.return_value = MagicMock(spec=[])  # no interactions attr
            with pytest.raises(ConfigurationError, match="does not expose interactions"):
                GoogleInteractionsClient(_config())

    @pytest.mark.asyncio
    async def test_create_interaction_returns_text_and_id(self):
        with patch("coffee_with_llm.providers.google.interactions_client.genai.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            interaction = MagicMock()
            interaction.id = "int-123"
            interaction.output_text = "Done."
            interaction.steps = []
            interaction.outputs = []
            interaction.usage = None
            mock_client.aio.interactions.create = AsyncMock(return_value=interaction)

            client = GoogleInteractionsClient(_config())
            text, usage, interaction_id = await client.create_interaction(
                prompt="Hi",
                model="gemini-flash-latest",
            )
            assert text == "Done."
            assert interaction_id == "int-123"
            assert usage.total_tokens == 0

    @pytest.mark.asyncio
    async def test_generate_rejects_attachments(self):
        with patch("coffee_with_llm.providers.google.interactions_client.genai.Client") as mock_cls:
            mock_cls.return_value = MagicMock(interactions=MagicMock())
            client = GoogleInteractionsClient(_config())
            with pytest.raises(ValidationError, match="attachments"):
                await client.generate(
                    prompt="Hi",
                    model="gemini-flash-latest",
                    attachments=[MagicMock()],
                )
