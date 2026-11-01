"""OpenTelemetry spans for model calls, when OpenTelemetry is installed.

One span per call, named and attributed by the GenAI semantic conventions
(https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/): the
provider, the model asked for and the model that answered, the tokens, and
why it stopped. The span starts in the caller's context, so a call made
inside a traced request joins that request's trace. Retries are inside the
one span: to the caller they are one call.

What was sent and what came back are left out unless
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` is ``true``, as the
conventions advise: a prompt holds whatever a person typed.

Without ``opentelemetry-api`` installed, a call gets a span that does nothing.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .types import Stop, TokenUsage

try:
    from opentelemetry import trace as _trace
    from opentelemetry.trace import SpanKind, StatusCode
except ImportError:  # tracing is optional
    _trace = None  # type: ignore[assignment]

#: The conventions' switch for recording prompts and answers on spans.
CAPTURE_CONTENT_ENV = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"


def captures_content() -> bool:
    return os.environ.get(CAPTURE_CONTENT_ENV, "").strip().lower() == "true"


class CallSpan:
    """One model call's span. Every method is safe when tracing is off, and ends it once."""

    def __init__(self, span: Any = None, *, captures: bool = False) -> None:
        self._span = span
        self.captures = captures and span is not None

    def finish(
        self,
        *,
        usage: Optional[TokenUsage] = None,
        stop: Optional[Stop] = None,
        text: Optional[str] = None,
    ) -> None:
        span, self._span = self._span, None
        if span is None:
            return
        if usage is not None:
            if usage.served_model:
                span.set_attribute("gen_ai.response.model", usage.served_model)
            # The conventions count cache reads and writes inside the input.
            span.set_attribute("gen_ai.usage.input_tokens", usage.prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", usage.output_tokens)
            if usage.cached_tokens is not None:
                span.set_attribute("gen_ai.usage.cache_read.input_tokens", usage.cached_tokens)
            if usage.cache_creation_tokens is not None:
                span.set_attribute(
                    "gen_ai.usage.cache_creation.input_tokens", usage.cache_creation_tokens
                )
            if usage.cost_usd is not None:
                span.set_attribute("coffee_with_llm.cost_usd", usage.cost_usd)
        if stop is not None:
            span.set_attribute("gen_ai.response.finish_reasons", [stop.raw or stop.reason])
        if self.captures and text is not None:
            answer: Dict[str, Any] = {
                "role": "assistant",
                "parts": [{"type": "text", "content": text}],
            }
            if stop is not None:
                answer["finish_reason"] = stop.reason
            span.set_attribute("gen_ai.output.messages", json.dumps([answer], default=str))
        span.end()

    def fail(self, error: BaseException) -> None:
        span, self._span = self._span, None
        if span is None:
            return
        span.record_exception(error)
        span.set_status(StatusCode.ERROR, str(error))
        span.set_attribute("error.type", type(error).__name__)
        span.end()


def start_call(
    *,
    provider: str,
    model: str,
    prompt: str,
    system_instruct: str = "",
    messages: Optional[List[Dict[str, Any]]] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> CallSpan:
    """Open a call's span in the caller's context; finish or fail it when the call ends."""
    if _trace is None:
        return CallSpan()
    from . import __version__

    attributes: Dict[str, Any] = {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": provider,
        "gen_ai.request.model": model,
    }
    if max_tokens is not None:
        attributes["gen_ai.request.max_tokens"] = max_tokens
    if temperature is not None:
        attributes["gen_ai.request.temperature"] = temperature
    captures = captures_content()
    if captures:
        attributes["gen_ai.input.messages"] = json.dumps(
            _input_messages(prompt, system_instruct, messages), default=str
        )
    span = _trace.get_tracer("coffee_with_llm", __version__).start_span(
        f"chat {model}", kind=SpanKind.CLIENT, attributes=attributes
    )
    return CallSpan(span, captures=captures)


def _input_messages(
    prompt: str, system_instruct: str, messages: Optional[List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    def message(role: str, content: Any) -> Dict[str, Any]:
        return {"role": role, "parts": [{"type": "text", "content": content}]}

    out = [message("system", system_instruct)] if system_instruct else []
    out += [message(str(m.get("role", "user")), m.get("content")) for m in messages or []]
    out.append(message("user", prompt))
    return out
