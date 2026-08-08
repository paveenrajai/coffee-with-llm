"""Shared types for coffee."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Dict, Mapping, Optional, Union, cast

from .rate_limit import retry_stream


@dataclass(frozen=True)
class TokenUsage:
    """Token usage for a generation session (aggregated across multi-step/tool loops).

    Field semantics vary by provider:

    - **OpenAI:** ``cached_tokens`` is often a subset of ``input_tokens``.
    - **Anthropic (prompt cache):** ``input_tokens``, ``cached_tokens`` (cache reads),
      and ``cache_creation_tokens`` (cache writes) are **disjoint** buckets. A large
      system prompt on the first turn can yield tiny ``input_tokens`` with most
      prompt tokens in ``cache_creation_tokens``.
    - **Google:** ``cached_tokens`` reflects context-cache reads when reported.

    ``total_tokens`` is ``input_tokens + output_tokens`` (legacy). It does **not**
    include cache read/write tokens. Use :meth:`prompt_tokens` or :meth:`billable_tokens`
    for observability and quota dashboards.
    """

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_tokens: Optional[int] = None
    # Anthropic cache_creation_input_tokens (prompt cache writes); optional elsewhere.
    cache_creation_tokens: Optional[int] = None
    cost_usd: Optional[float] = None

    @property
    def prompt_tokens(self) -> int:
        """All prompt-side tokens: uncached input + cache reads + cache writes."""
        return (
            self.input_tokens
            + (self.cached_tokens or 0)
            + (self.cache_creation_tokens or 0)
        )

    @property
    def billable_tokens(self) -> int:
        """All tokens that affect billing: prompt-side + output."""
        return self.prompt_tokens + self.output_tokens

    def to_dict(self) -> Dict[str, Any]:
        """Observability-friendly usage payload (includes computed prompt totals)."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "prompt_tokens": self.prompt_tokens,
            "billable_tokens": self.billable_tokens,
            "cost_usd": self.cost_usd,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> TokenUsage:
        """Build from a dict (e.g. aggregated session totals)."""
        input_tokens = int(raw.get("input_tokens") or 0)
        output_tokens = int(raw.get("output_tokens") or 0)
        total_raw = raw.get("total_tokens")
        total_tokens = (
            int(total_raw)
            if total_raw is not None
            else input_tokens + output_tokens
        )
        cached_raw = raw.get("cached_tokens")
        cached_tokens = int(cached_raw) if cached_raw is not None else None
        creation_raw = raw.get("cache_creation_tokens")
        cache_creation_tokens = (
            int(creation_raw) if creation_raw is not None else None
        )
        cost_raw = raw.get("cost_usd")
        cost_usd = float(cost_raw) if cost_raw is not None else None
        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
            cache_creation_tokens=cache_creation_tokens,
            cost_usd=cost_usd,
        )


@dataclass(frozen=True)
class StreamTextDelta:
    """Incremental model text from the provider (pass-through; not buffered)."""

    text: str


@dataclass(frozen=True)
class StreamToolCallStart:
    """A tool call has started (id and name known; arguments may stream next)."""

    id: str
    name: str


@dataclass(frozen=True)
class StreamToolArgumentsDelta:
    """Fragment of JSON arguments for a tool call (streaming providers)."""

    id: str
    fragment: str


@dataclass(frozen=True)
class StreamToolCallEnd:
    """Tool call is complete with parsed arguments."""

    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass(frozen=True)
class StreamStepBoundary:
    """Emitted between multi-step tool rounds (optional, for UI)."""

    step_index: int


StreamEvent = Union[
    StreamTextDelta,
    StreamToolCallStart,
    StreamToolArgumentsDelta,
    StreamToolCallEnd,
    StreamStepBoundary,
]

StreamChunk = Union[StreamEvent, TokenUsage]


@dataclass
class StreamUsageSink:
    """Best-effort token accumulation for early stream close; providers update while streaming."""

    _input: int = 0
    _output: int = 0
    _cached: Optional[int] = None
    _cache_creation: Optional[int] = None

    def merge(
        self,
        inp: int,
        out: int,
        cached: Optional[int] = None,
        *,
        cache_creation: Optional[int] = None,
    ) -> None:
        self._input += int(inp)
        self._output += int(out)
        if cached is not None:
            self._cached = (self._cached or 0) + int(cached)
        if cache_creation is not None:
            self._cache_creation = (self._cache_creation or 0) + int(cache_creation)

    def replace_with(self, usage: TokenUsage) -> None:
        self._input = usage.input_tokens
        self._output = usage.output_tokens
        self._cached = usage.cached_tokens
        self._cache_creation = usage.cache_creation_tokens

    def snapshot(self) -> TokenUsage:
        return TokenUsage(
            input_tokens=self._input,
            output_tokens=self._output,
            total_tokens=self._input + self._output,
            cached_tokens=self._cached,
            cache_creation_tokens=self._cache_creation,
        )


@dataclass
class AskResult:
    """Result of an LLM ask with token usage."""

    text: str
    usage: TokenUsage
    #: Set when the call used Gemini Interactions API (for multi-turn continuation).
    interaction_id: Optional[str] = None

    def __str__(self) -> str:
        return self.text


def _normalize_stream_item(item: object) -> object:
    """Allow bare str for backward compatibility; treat as StreamTextDelta."""
    if isinstance(item, str):
        return StreamTextDelta(item)
    return item


class StreamResult:
    """
    Result of streaming. Iterate for :class:`StreamEvent` chunks; a terminal
    :class:`TokenUsage` ends iteration (not delivered through ``__anext__``).

    ``usage`` is set when iteration completes or after :meth:`aclose` (e.g. early break),
    using final totals when available, otherwise :class:`StreamUsageSink` snapshot.

    Must be iterated via ``async for`` (``__aiter__`` before ``__anext__``).
    """

    def __init__(
        self,
        stream_factory: Callable[[], AsyncIterator[object]],
        usage_callback: Optional[Callable[[TokenUsage], TokenUsage]] = None,
        max_retries: int = 3,
        usage_sink: Optional[StreamUsageSink] = None,
    ) -> None:
        self._stream_factory = stream_factory
        self._usage_callback = usage_callback
        self._max_retries = max_retries
        self._usage_sink = usage_sink
        self._usage: Optional[TokenUsage] = None
        self._iter: Optional[AsyncIterator[object]] = None
        self._closed: bool = False

    def __aiter__(self) -> StreamResult:
        self._iter = cast(
            Optional[AsyncIterator[object]],
            retry_stream(
                self._stream_factory,
                max_retries=self._max_retries,
            ).__aiter__(),
        )
        self._closed = False
        return self

    async def __anext__(self) -> StreamEvent:
        if self._iter is None:
            raise RuntimeError(
                "StreamResult must be iterated via async for; __aiter__ was not called"
            )
        item = await self._iter.__anext__()
        item = _normalize_stream_item(item)
        if isinstance(item, TokenUsage):
            self._apply_usage(item)
            raise StopAsyncIteration
        return cast(StreamEvent, item)

    async def aclose(self) -> None:
        """Close the underlying stream and populate ``usage`` if iteration stopped early."""
        if self._closed:
            return
        self._closed = True
        if self._iter is not None and hasattr(self._iter, "aclose"):
            try:
                await self._iter.aclose()  # type: ignore[misc]
            except Exception:
                pass
        self._iter = None
        self._finalize_usage_if_needed()

    def _apply_usage(self, usage: TokenUsage) -> None:
        self._usage = self._usage_callback(usage) if self._usage_callback else usage

    def _finalize_usage_if_needed(self) -> None:
        if self._usage is not None:
            return
        if self._usage_sink is not None:
            self._apply_usage(self._usage_sink.snapshot())
            return
        self._apply_usage(TokenUsage(0, 0, 0, None))

    async def __aenter__(self) -> StreamResult:
        self.__aiter__()
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()

    @property
    def usage(self) -> Optional[TokenUsage]:
        return self._usage
