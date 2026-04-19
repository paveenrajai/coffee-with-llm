"""Tests for the provider-agnostic reasoning_effort translation."""

from __future__ import annotations

import logging

from coffee_with_llm.providers._reasoning import (
    normalize_effort,
    thinking_budget_tokens,
)


class TestNormalizeEffort:
    def test_returns_none_for_none(self):
        assert normalize_effort(None) is None

    def test_returns_none_for_blank(self):
        assert normalize_effort("") is None
        assert normalize_effort("   ") is None

    def test_lowercases_and_trims(self):
        assert normalize_effort(" High ") == "high"
        assert normalize_effort("MEDIUM") == "medium"

    def test_unknown_value_returns_none_and_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            assert normalize_effort("ultra") is None
        assert any("Unknown reasoning_effort" in r.message for r in caplog.records)


class TestThinkingBudgetTokens:
    def test_known_values(self):
        assert thinking_budget_tokens("low") == 1024
        assert thinking_budget_tokens("medium") == 4096
        assert thinking_budget_tokens("high") == 16384

    def test_unknown_returns_none(self):
        assert thinking_budget_tokens(None) is None
        assert thinking_budget_tokens("ultra") is None

    def test_budgets_are_strictly_increasing(self):
        """Order must be monotonic so callers can reason about cost vs depth."""
        assert (
            thinking_budget_tokens("low")
            < thinking_budget_tokens("medium")
            < thinking_budget_tokens("high")
        )
