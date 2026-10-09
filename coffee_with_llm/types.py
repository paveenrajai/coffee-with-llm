"""Shared types for coffee."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Dict, Mapping, Optional, Tuple, Union, cast

from .rate_limit import retry_stream


@dataclass(frozen=True)
class TokenUsage:
    """Token usage for a generation session (aggregated across multi-step/tool loops).

    The prompt is split into **disjoint** buckets, the same for every provider:
    ``input_tokens`` is the uncached input, billed at the full input rate;
    ``cached_tokens`` is cache reads; ``cache_creation_tokens`` is cache writes,
    which only Anthropic reports. Google, OpenAI and Inception count cache reads
    inside their own input count, and they are taken out of ``input_tokens``
    here. On Anthropic, a large system prompt on the first turn can yield tiny
    ``input_tokens`` with most prompt tokens in ``cache_creation_tokens``.

    ``reasoning_tokens`` are the model's thinking, and always a part of
    ``output_tokens``, which is what every provider bills them as. Google,
    OpenAI and Inception report them; Anthropic counts thinking in
    ``output_tokens`` without saying how much, so it is ``None`` there.

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
    #: Thinking tokens, counted within ``output_tokens``. ``None`` when the
    #: provider does not report them.
    reasoning_tokens: Optional[int] = None
    #: The model that served the call, as the provider named it in its reply:
    #: for an alias such as ``gemini-flash-lite-latest``, whatever it points to
    #: today. ``cost_usd`` is priced at this model. ``None`` when the provider
    #: did not say.
    served_model: Optional[str] = None

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
            "reasoning_tokens": self.reasoning_tokens,
            "served_model": self.served_model,
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
        reasoning_raw = raw.get("reasoning_tokens")
        reasoning_tokens = int(reasoning_raw) if reasoning_raw is not None else None
        served_raw = raw.get("served_model")
        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
            cache_creation_tokens=cache_creation_tokens,
            cost_usd=cost_usd,
            reasoning_tokens=reasoning_tokens,
            served_model=str(served_raw) if served_raw else None,
        )


class StopReason:
    """Why a model stopped writing, in the same words for every provider.

    Closed on purpose: each provider's own reasons fold into these few, and
    its own word for it is kept in :attr:`Stop.raw`.
    """

    #: It finished what it was writing.
    END = "end"
    #: It hit ``max_tokens`` and was cut off. Where a provider counts thinking
    #: against that limit (Gemini, OpenAI), a long think can use it up first.
    MAX_TOKENS = "max_tokens"
    #: The provider's safety or recitation filter stopped it, or it refused.
    CONTENT_FILTER = "content_filter"
    #: It ended asking for a tool, which a caller sees when a tool loop runs
    #: out of steps.
    TOOL_USE = "tool_use"
    #: Anything else. :attr:`Stop.raw` says what.
    OTHER = "other"


@dataclass(frozen=True)
class Stop:
    """Why the last step of a call stopped, and the provider's word for it."""

    #: One of :class:`StopReason`.
    reason: str
    #: As the provider gave it, e.g. ``MAX_TOKENS``, ``end_turn``, ``length``.
    raw: str

    @property
    def truncated(self) -> bool:
        """The answer was cut off at ``max_tokens``, not finished."""
        return self.reason == StopReason.MAX_TOKENS


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

StreamChunk = Union[StreamEvent, Stop, TokenUsage]


@dataclass
class StreamUsageSink:
    """Best-effort token accumulation for early stream close; providers update while streaming."""

    _input: int = 0
    _output: int = 0
    _cached: Optional[int] = None
    _cache_creation: Optional[int] = None
    _reasoning: Optional[int] = None
    _served_model: Optional[str] = None
    _has_usage: bool = False

    @property
    def has_usage(self) -> bool:
        """True once the provider has reported any usage into this sink."""
        return self._has_usage

    def merge(
        self,
        inp: int,
        out: int,
        cached: Optional[int] = None,
        *,
        cache_creation: Optional[int] = None,
        reasoning: Optional[int] = None,
    ) -> None:
        self._has_usage = True
        self._input += int(inp)
        self._output += int(out)
        if cached is not None:
            self._cached = (self._cached or 0) + int(cached)
        if cache_creation is not None:
            self._cache_creation = (self._cache_creation or 0) + int(cache_creation)
        if reasoning is not None:
            self._reasoning = (self._reasoning or 0) + int(reasoning)

    def replace_with(self, usage: TokenUsage) -> None:
        self._has_usage = True
        self._input = usage.input_tokens
        self._output = usage.output_tokens
        self._cached = usage.cached_tokens
        self._cache_creation = usage.cache_creation_tokens
        self._reasoning = usage.reasoning_tokens
        self._served_model = usage.served_model or self._served_model

    def snapshot(self) -> TokenUsage:
        return TokenUsage(
            input_tokens=self._input,
            output_tokens=self._output,
            total_tokens=self._input + self._output,
            cached_tokens=self._cached,
            cache_creation_tokens=self._cache_creation,
            reasoning_tokens=self._reasoning,
            served_model=self._served_model,
        )


@dataclass(frozen=True)
class UrlRetrieval:
    """One link Gemini's URL context tried to open, and whether it could."""

    url: str
    #: True only when the page was read. A refused fetch (a site that blocks
    #: it, a paywall, an unsafe page) is answered from search, if at all.
    ok: bool
    #: The provider's own status name, e.g. ``URL_RETRIEVAL_STATUS_ERROR``.
    status: str


@dataclass
class AskResult:
    """Result of an LLM ask with token usage."""

    text: str
    usage: TokenUsage
    #: Set when the call used Gemini Interactions API (for multi-turn continuation).
    interaction_id: Optional[str] = None
    #: Each link in the prompt that Gemini's URL context tried to open, once
    #: per link. Empty when there was no link, and for other providers.
    url_retrievals: Tuple[UrlRetrieval, ...] = ()
    #: Why the last step stopped. ``None`` only from a provider that does
    #: not say.
    stop: Optional[Stop] = None

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
    ``stop`` says why the last step stopped, once iteration completes: a
    stream cut off at ``max_tokens`` ends the same way as a finished one, and
    this is how a caller tells them apart.

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
        self._stop: Optional[Stop] = None
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
        item = _normalize_stream_item(await self._iter.__anext__())
        # A provider says why it stopped just before its usage. It is kept
        # here rather than passed on, so a consumer that knows only the
        # events it asked for never meets one it does not.
        while isinstance(item, Stop):
            self._stop = item
            item = _normalize_stream_item(await self._iter.__anext__())
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
        self._usage = self._priced(usage)

    def _priced(self, usage: TokenUsage) -> TokenUsage:
        return self._usage_callback(usage) if self._usage_callback else usage

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

    @property
    def usage_so_far(self) -> Optional[TokenUsage]:
        """The usage the provider has reported in the chunks received so far,
        priced like ``usage``.

        For a caller that stops reading mid-stream and must not wait: it reads
        only what has already arrived, and never closes, drains or awaits the
        stream. ``usage`` once that is set. ``None`` while no chunk has carried usage,
        and for a provider that reports it only at the end.
        """
        if self._usage is not None:
            return self._usage
        if self._usage_sink is None or not self._usage_sink.has_usage:
            return None
        return self._priced(self._usage_sink.snapshot())

    @property
    def stop(self) -> Optional[Stop]:
        """Why the last step stopped. ``None`` before the stream ends, after an
        early close, and from a provider that does not say."""
        return self._stop
