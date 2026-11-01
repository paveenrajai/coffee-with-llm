"""Tests for Anthropic model capability detection."""

import pytest

from coffee_with_llm.providers.anthropic.models import (
    anthropic_rejects_sampling_params,
    anthropic_supports_adaptive_thinking,
    anthropic_thinking_always_on,
    anthropic_thinking_can_disable,
    anthropic_thinking_defaults_on,
    anthropic_thinking_off,
    normalize_anthropic_model_id,
)


class TestNormalizeAnthropicModelId:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("sonnet-5", "claude-sonnet-5"),
            ("claude-sonnet-5", "claude-sonnet-5"),
            ("fable-5", "claude-fable-5"),
            ("claude-opus-4-8", "claude-opus-4-8"),
        ],
    )
    def test_aliases(self, raw: str, expected: str) -> None:
        assert normalize_anthropic_model_id(raw) == expected


class TestAdaptiveThinkingSupport:
    @pytest.mark.parametrize(
        "model",
        [
            "claude-opus-4-8",
            "claude-opus-4-7",
            "claude-sonnet-4-6",
            "claude-sonnet-5",
            "sonnet-5",
            "claude-fable-5",
            "claude-mythos-5",
            "claude-mythos-preview",
        ],
    )
    def test_adaptive_capable(self, model: str) -> None:
        assert anthropic_supports_adaptive_thinking(model)

    @pytest.mark.parametrize(
        "model",
        ["claude-sonnet-4-5", "claude-opus-4-5", "", "gpt-4o"],
    )
    def test_not_adaptive(self, model: str) -> None:
        assert not anthropic_supports_adaptive_thinking(model)


class TestThinkingDefaults:
    def test_sonnet_5_defaults_on_but_can_disable(self) -> None:
        assert anthropic_thinking_defaults_on("sonnet-5")
        assert anthropic_thinking_can_disable("sonnet-5")
        assert not anthropic_thinking_always_on("sonnet-5")

    def test_fable_5_always_on(self) -> None:
        assert anthropic_thinking_defaults_on("claude-fable-5")
        assert anthropic_thinking_always_on("claude-fable-5")
        assert not anthropic_thinking_can_disable("claude-fable-5")

    def test_opus_4_6_no_default(self) -> None:
        assert not anthropic_thinking_defaults_on("claude-opus-4-6")

    @pytest.mark.parametrize(
        "model", ["claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
    )
    def test_generation_5_models_are_adaptive_and_think_by_default(self, model: str) -> None:
        assert anthropic_supports_adaptive_thinking(model)
        assert anthropic_thinking_defaults_on(model)
        assert anthropic_rejects_sampling_params(model)

    def test_opus_5_5_cannot_turn_thinking_off(self) -> None:
        assert not anthropic_thinking_can_disable("claude-opus-5-5")
        assert anthropic_thinking_off("claude-opus-5-5") is None
        assert not anthropic_thinking_always_on("claude-opus-5-5")

    def test_sonnet_5_5_turns_thinking_off_between_tools(self) -> None:
        assert anthropic_thinking_off("claude-sonnet-5-5") == {"type": "between_tools"}

    @pytest.mark.parametrize("model", ["claude-opus-5", "claude-haiku-5-5", "claude-sonnet-5"])
    def test_disabled_turns_thinking_off(self, model: str) -> None:
        assert anthropic_thinking_off(model) == {"type": "disabled"}

    def test_haiku_4_5_is_not_generation_5(self) -> None:
        assert not anthropic_supports_adaptive_thinking("claude-haiku-4-5")
        assert not anthropic_thinking_defaults_on("claude-haiku-4-5")


class TestSamplingRestrictions:
    @pytest.mark.parametrize(
        "model",
        [
            "claude-sonnet-5",
            "sonnet-5",
            "claude-fable-5",
            "claude-opus-4-8",
            "claude-opus-4-7",
            "claude-mythos-preview",
        ],
    )
    def test_rejects_sampling(self, model: str) -> None:
        assert anthropic_rejects_sampling_params(model)

    def test_allows_sampling_on_4_6(self) -> None:
        assert not anthropic_rejects_sampling_params("claude-sonnet-4-6")
