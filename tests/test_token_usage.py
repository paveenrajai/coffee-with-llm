"""Tests for TokenUsage helpers and observability export."""

from coffee_with_llm.cost import estimate_cost
from coffee_with_llm.types import TokenUsage


class TestTokenUsageHelpers:
    def test_prompt_tokens_sums_all_prompt_buckets(self):
        usage = TokenUsage(
            input_tokens=2,
            output_tokens=8158,
            total_tokens=8160,
            cache_creation_tokens=40_000,
        )
        assert usage.prompt_tokens == 40_002
        assert usage.billable_tokens == 48_160

    def test_prompt_tokens_includes_cache_read(self):
        usage = TokenUsage(
            input_tokens=200,
            output_tokens=50,
            total_tokens=250,
            cached_tokens=8000,
        )
        assert usage.prompt_tokens == 8200
        assert usage.billable_tokens == 8250

    def test_to_dict_includes_computed_fields(self):
        usage = TokenUsage(
            input_tokens=2,
            output_tokens=8158,
            total_tokens=8160,
            cache_creation_tokens=40_000,
            cost_usd=0.276617,
        )
        d = usage.to_dict()
        assert d["input_tokens"] == 2
        assert d["output_tokens"] == 8158
        assert d["total_tokens"] == 8160
        assert d["cache_creation_tokens"] == 40_000
        assert d["cached_tokens"] is None
        assert d["prompt_tokens"] == 40_002
        assert d["billable_tokens"] == 48_160
        assert d["cost_usd"] == 0.276617

    def test_from_mapping_round_trip(self):
        raw = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cached_tokens": 30,
            "cache_creation_tokens": 200,
            "cost_usd": 0.01,
        }
        usage = TokenUsage.from_mapping(raw)
        assert usage.total_tokens == 150
        assert usage.prompt_tokens == 330
        assert usage.to_dict()["billable_tokens"] == 380

    def test_anthropic_cache_creation_scenario_cost(self):
        """Regression: tiny input_tokens with large cache write is valid."""
        usage = TokenUsage(
            input_tokens=2,
            output_tokens=8158,
            total_tokens=8160,
            cache_creation_tokens=40_000,
        )
        cost = estimate_cost(usage, "claude-sonnet-5")
        assert cost is not None
        assert cost > 0.25
