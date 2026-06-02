"""Tests for cost module."""

from coffee_with_llm.cost import estimate_cost
from coffee_with_llm.types import TokenUsage


class TestEstimateCost:
    """Tests for estimate_cost."""

    def test_known_model_returns_cost(self):
        usage = TokenUsage(1000, 500, 1500, None)
        cost = estimate_cost(usage, "gpt-4o-mini")
        assert cost is not None
        assert cost > 0

    def test_unknown_model_returns_none(self):
        usage = TokenUsage(1000, 500, 1500, None)
        cost = estimate_cost(usage, "unknown-model-xyz")
        assert cost is None

    def test_cached_tokens_use_discounted_rate(self):
        usage_uncached = TokenUsage(1000, 100, 1100, None)
        usage_cached = TokenUsage(1000, 100, 1100, 500)
        cost_uncached = estimate_cost(usage_uncached, "gpt-4o")
        cost_cached = estimate_cost(usage_cached, "gpt-4o")
        assert cost_cached is not None
        assert cost_uncached is not None
        assert cost_cached < cost_uncached

    def test_anthropic_cached_tokens_use_discounted_rate(self):
        usage_uncached = TokenUsage(10_000, 100, 10_100, None)
        usage_cached = TokenUsage(10_000, 100, 10_100, 5000)
        cost_uncached = estimate_cost(usage_uncached, "claude-opus-4-8")
        cost_cached = estimate_cost(usage_cached, "claude-opus-4-8")
        assert cost_uncached is not None
        assert cost_cached is not None
        assert cost_cached < cost_uncached

    def test_anthropic_disjoint_cache_buckets_not_subtracted_from_input(self):
        """Anthropic bills input_tokens and cache_read separately."""
        usage = TokenUsage(input_tokens=200, output_tokens=50, total_tokens=250, cached_tokens=8000)
        cost = estimate_cost(usage, "claude-opus-4-8")
        assert cost is not None
        # 200 uncached @ $5 + 8000 cached @ $0.50 per MTok + output
        expected = (200 / 1_000_000) * 5.0 + (8000 / 1_000_000) * 0.50 + (50 / 1_000_000) * 25.0
        assert cost == round(expected, 6)

    def test_anthropic_cache_creation_billed_at_125_percent_input(self):
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=10,
            total_tokens=110,
            cache_creation_tokens=10_000,
        )
        cost = estimate_cost(usage, "claude-opus-4-8")
        assert cost is not None
        expected = (
            (100 / 1_000_000) * 5.0
            + (10_000 / 1_000_000) * 5.0 * 1.25
            + (10 / 1_000_000) * 25.0
        )
        assert cost == round(expected, 6)

    def test_zero_usage(self):
        usage = TokenUsage(0, 0, 0, None)
        cost = estimate_cost(usage, "gpt-4o-mini")
        assert cost == 0.0

    def test_model_prefix_match(self):
        usage = TokenUsage(1000, 100, 1100, None)
        cost = estimate_cost(usage, "gpt-4o-mini-2024-01")
        assert cost is not None
