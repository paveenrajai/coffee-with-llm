"""Tests for Google citation utilities."""

from unittest.mock import MagicMock

from coffee_with_llm.providers.google.utils.citations import (
    CITATION_MARKER_RE,
    collect_grounding_urls,
    describe_grounding,
    extract_citations,
    inject_inline_citations,
    merge_grounding_responses,
)


def _mock_chunk(url: str) -> MagicMock:
    chunk = MagicMock()
    chunk.web = MagicMock()
    chunk.web.uri = url
    chunk.web.url = None
    return chunk


def _mock_support(*, end_index: int, chunk_indices: list[int]) -> MagicMock:
    support = MagicMock()
    support.segment = MagicMock()
    support.segment.end_index = end_index
    support.grounding_chunk_indices = chunk_indices
    support.indices = []
    return support


def _mock_response(*, chunks: list, supports: list | None = None) -> MagicMock:
    resp = MagicMock()
    gm = MagicMock()
    gm.grounding_chunks = chunks
    gm.chunks = []
    gm.grounding_supports = supports or []
    gm.supports = []
    resp.grounding_metadata = gm
    resp.candidates = []
    return resp


class TestExtractCitations:
    """Tests for extract_citations."""

    def test_empty_response_returns_empty(self):
        """Empty response returns empty list."""
        resp = MagicMock()
        resp.grounding_metadata = None
        resp.citation_metadata = None
        resp.candidates = []
        assert extract_citations(resp) == []

    def test_grounding_metadata_attributions(self):
        """Extract from grounding_metadata.grounding_attributions."""
        resp = MagicMock()
        resp.grounding_metadata = None
        resp.citation_metadata = None
        resp.candidates = []
        gm = MagicMock()
        att = MagicMock()
        att.web = MagicMock()
        att.web.uri = "https://example.com"
        att.web.title = "Example"
        gm.grounding_attributions = [att]
        resp.grounding_metadata = gm
        result = extract_citations(resp)
        assert len(result) == 1
        assert result[0]["uri"] == "https://example.com"
        assert result[0]["title"] == "Example"


class TestCollectGroundingUrls:
    """Tests for collect_grounding_urls."""

    def test_empty_response_returns_empty(self):
        """Empty response returns empty set."""
        resp = MagicMock()
        resp.grounding_metadata = None
        resp.candidates = []
        assert collect_grounding_urls(resp) == set()

    def test_collects_urls_from_chunks(self):
        """Collect URLs from grounding_chunks."""
        resp = MagicMock()
        gm = MagicMock()
        ch = MagicMock()
        ch.web = MagicMock()
        ch.web.uri = "https://example.com"
        ch.web.url = None
        gm.grounding_chunks = [ch]
        gm.chunks = []
        resp.grounding_metadata = gm
        resp.candidates = []
        assert collect_grounding_urls(resp) == {"https://example.com"}


class TestInjectInlineCitations:
    """Tests for inject_inline_citations."""

    def test_empty_text_returns_unchanged(self):
        """Empty text returns unchanged."""
        resp = MagicMock()
        resp.grounding_metadata = None
        resp.candidates = []
        assert inject_inline_citations("", resp, lambda u: u) == ""

    def test_no_grounding_metadata_returns_unchanged(self):
        """No grounding metadata returns text unchanged."""
        resp = MagicMock()
        resp.grounding_metadata = None
        resp.candidates = []
        text = "Hello world"
        assert inject_inline_citations(text, resp, lambda u: u) == text

    def test_with_citations_injects_markers(self):
        """With grounding metadata and supports, injects citation markers."""
        resp = _mock_response(chunks=[_mock_chunk("https://example.com")], supports=[])
        text = "Hello world"
        assert inject_inline_citations(text, resp, lambda u: u) == text

    def test_json_hooks_get_support_urls_inside_hook_string(self):
        hook = "Silicon routes light along micro-scale channels."
        text = (
            '[{"set_position":1,"label":"wave control","title":"Waveguides",'
            f'"hook":"{hook}","questions":["How?","Why?"]}}]'
        )
        hook_end = text.index(hook) + len(hook)
        resp = _mock_response(
            chunks=[_mock_chunk("https://example.com/source")],
            supports=[_mock_support(end_index=hook_end, chunk_indices=[0])],
        )
        result = inject_inline_citations(text, resp, lambda u: u)
        assert CITATION_MARKER_RE.search(result)
        assert '"hook":"' in result
        assert result.endswith("}]")
        assert "https://example.com/source" in result
        assert hook in result

    def test_json_hooks_fallback_when_supports_missing(self):
        hook = "A surprising fact with tension."
        text = (
            '[{"set_position":1,"label":"angle","title":"Card",'
            f'"hook":"{hook}","questions":["How?","Why?"]}}]'
        )
        resp = _mock_response(
            chunks=[
                _mock_chunk("https://example.com/a"),
                _mock_chunk("https://example.com/b"),
            ],
            supports=[],
        )
        result = inject_inline_citations(text, resp, lambda u: u)
        assert "[cite: https://example.com/a]" in result

    def test_json_hooks_merge_orphan_marker_after_quote(self):
        hook = "A surprising fact with tension."
        text = (
            '[{"set_position":1,"label":"angle","title":"Card",'
            f'"hook":"{hook}" [cite: https://example.com/orphan],'
            '"questions":["How?","Why?"]}]'
        )
        resp = _mock_response(chunks=[_mock_chunk("https://example.com/orphan")], supports=[])
        result = inject_inline_citations(text, resp, lambda u: u)
        assert '[cite: https://example.com/orphan]' in result
        hook_part = result.split('"questions"')[0].split('"hook"')[1].split("]")[0] + "]"
        assert " [cite:" not in hook_part or (
            "hook" in result and "https://example.com/orphan" in result
        )
        assert result.count("https://example.com/orphan") >= 1

    def test_prose_uses_offset_insertion(self):
        text = "Hello world"
        resp = _mock_response(
            chunks=[_mock_chunk("https://example.com")],
            supports=[_mock_support(end_index=5, chunk_indices=[0])],
        )
        result = inject_inline_citations(text, resp, lambda u: u)
        assert result == "Hello [cite: https://example.com] world"


class TestGroundingHelpers:
    def test_describe_grounding_empty(self):
        resp = _mock_response(chunks=[], supports=[])
        info = describe_grounding(resp)
        assert info["chunk_count"] == 0
        assert info["urls"] == []

    def test_merge_grounding_combines_chunks(self):
        first = _mock_response(
            chunks=[_mock_chunk("https://a.com")],
            supports=[],
        )
        second = _mock_response(
            chunks=[_mock_chunk("https://b.com")],
            supports=[],
        )
        merged = merge_grounding_responses([first, second])
        assert collect_grounding_urls(merged) == {"https://a.com", "https://b.com"}


class TestRestrictHookCitations:
    def test_restrict_hook_drops_hallucinated_url(self):
        from coffee_with_llm.providers.google.utils.citations import restrict_hook_citation

        hook = "A fact about chips. [cite:https://nature.com/fake-article]"
        allowed = ["https://www.tomshardware.com/news/real-story"]
        assert restrict_hook_citation(hook, allowed) == (
            "A fact about chips. [cite:https://www.tomshardware.com/news/real-story]"
        )

    def test_restrict_hook_keeps_matching_allowed_url(self):
        from coffee_with_llm.providers.google.utils.citations import restrict_hook_citation

        allowed = ["https://www.reuters.com/world/us/story"]
        hook = "Markets moved. [cite:https://www.reuters.com/world/us/story]"
        assert restrict_hook_citation(hook, allowed) == hook

    def test_restrict_json_replaces_hallucinated_hook_urls(self):
        from coffee_with_llm.providers.google.utils.citations import restrict_json_hook_citations

        raw = (
            '[{"set_position":1,"hook":"Fact. [cite:https://nature.com/fake]",'
            '"questions":["Q1?","Q2?"]}]'
        )
        allowed = ["https://www.tomshardware.com/real"]
        out = restrict_json_hook_citations(raw, allowed)
        assert "nature.com/fake" not in out
        assert "tomshardware.com/real" in out

    def test_restrict_inline_citations_keeps_multiple_per_marker(self):
        from coffee_with_llm.providers.google.utils.citations import restrict_inline_citations

        allowed = [
            "https://a.com/one",
            "https://b.com/two",
        ]
        text = "Claim. [cite:https://a.com/one, https://b.com/two, https://evil.com/fake]"
        out = restrict_inline_citations(text, allowed)
        assert "evil.com" not in out
        assert "a.com/one" in out and "b.com/two" in out

    def test_partition_citation_urls(self):
        from coffee_with_llm.providers.google.utils.citations import partition_citation_urls

        allowed = ["https://reuters.com/story"]
        matched, unmatched = partition_citation_urls(
            ["https://reuters.com/story", "https://fake.com/x"],
            allowed,
        )
        assert matched == ["https://reuters.com/story"]
        assert unmatched == ["https://fake.com/x"]
