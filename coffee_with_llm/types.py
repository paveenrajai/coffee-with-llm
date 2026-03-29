"""Shared types for coffee."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Dict, Optional, Union, cast

from .rate_limit import retry_stream


@dataclass(frozen=True)
class TokenUsage:
    """Token usage for a generation session (aggregated across multi-step/tool loops)."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_tokens: Optional[int] = None
    cost_usd: Optional[float] = None


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

    def merge(self, inp: int, out: int, cached: Optional[int] = None) -> None:
        self._input += int(inp)
        self._output += int(out)
        if cached is not None:
            self._cached = (self._cached or 0) + int(cached)

    def replace_with(self, usage: TokenUsage) -> None:
        self._input = usage.input_tokens
        self._output = usage.output_tokens
        self._cached = usage.cached_tokens

    def snapshot(self) -> TokenUsage:
        return TokenUsage(
            input_tokens=self._input,
            output_tokens=self._output,
            total_tokens=self._input + self._output,
            cached_tokens=self._cached,
        )


@dataclass
class AskResult:
    """Result of an LLM ask with token usage."""

    text: str
    usage: TokenUsage

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
