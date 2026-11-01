"""A span per model call, by the OpenTelemetry GenAI conventions.

Spans go to the real OpenTelemetry SDK and an in-memory exporter; the
provider is a stub, so nothing reaches a model.
"""

from unittest.mock import AsyncMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

from coffee_with_llm import AskLLM, Stop, StreamTextDelta, TokenUsage, tracing
from coffee_with_llm.tracing import CAPTURE_CONTENT_ENV

USAGE = TokenUsage(
    input_tokens=100,
    output_tokens=20,
    total_tokens=120,
    cached_tokens=900,
    served_model="gemini-x-001",
)
STOP = Stop(reason="end", raw="STOP")


@pytest.fixture
def spans(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracing._trace, "get_tracer", provider.get_tracer)
    monkeypatch.delenv(CAPTURE_CONTENT_ENV, raising=False)
    exporter.tracer = provider.get_tracer("test")
    return exporter


@pytest.fixture
def llm(mock_google_api_key):
    with patch("coffee_with_llm.providers.google.text_client.genai.Client"):
        llm = AskLLM(model="google/gemini-x", min_delay_between_calls=0)
    llm._client.generate = AsyncMock(return_value=("Blue light scatters most.", USAGE, STOP))
    return llm


def _streamed(*items):
    def generate_stream(**_kwargs):
        async def events():
            for item in items:
                if isinstance(item, Exception):
                    raise item
                yield item

        return events()

    return generate_stream


async def test_a_call_is_one_span_with_its_model_and_tokens(llm, spans):
    await llm.ask(prompt="Why is the sky blue?", max_tokens=500)

    (span,) = spans.get_finished_spans()
    assert span.name == "chat gemini-x"
    assert span.kind == SpanKind.CLIENT
    a = span.attributes
    assert a["gen_ai.operation.name"] == "chat"
    assert a["gen_ai.provider.name"] == "gcp.gemini"
    assert a["gen_ai.request.model"] == "gemini-x"
    assert a["gen_ai.request.max_tokens"] == 500
    assert a["gen_ai.response.model"] == "gemini-x-001"
    # The conventions count the cache inside the input; coffee keeps it apart.
    assert a["gen_ai.usage.input_tokens"] == 1000
    assert a["gen_ai.usage.cache_read.input_tokens"] == 900
    assert a["gen_ai.usage.output_tokens"] == 20
    assert a["gen_ai.response.finish_reasons"] == ("STOP",)
    assert "gen_ai.input.messages" not in a, "prompts stay off a span unless asked for"
    assert "gen_ai.output.messages" not in a


async def test_prompts_and_answers_are_recorded_only_when_asked_for(llm, spans, monkeypatch):
    monkeypatch.setenv(CAPTURE_CONTENT_ENV, "true")

    await llm.ask(prompt="Why is the sky blue?", system_instruct="Be brief.")

    (span,) = spans.get_finished_spans()
    assert span.attributes["gen_ai.input.messages"] == (
        '[{"role": "system", "parts": [{"type": "text", "content": "Be brief."}]}, '
        '{"role": "user", "parts": [{"type": "text", "content": "Why is the sky blue?"}]}]'
    )
    assert '"content": "Blue light scatters most."' in span.attributes["gen_ai.output.messages"]


async def test_a_call_joins_the_trace_it_is_made_in(llm, spans):
    with spans.tracer.start_as_current_span("turn") as turn:
        await llm.ask(prompt="Why is the sky blue?")

    call = next(s for s in spans.get_finished_spans() if s.name == "chat gemini-x")
    assert call.parent.span_id == turn.get_span_context().span_id


async def test_a_failed_call_ends_its_span_as_an_error(llm, spans):
    llm._max_retries = 1
    llm._client.generate = AsyncMock(side_effect=ValueError("bad request"))

    with pytest.raises(Exception):
        await llm.ask(prompt="Why is the sky blue?")

    (span,) = spans.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes["error.type"] == "ValueError"


async def test_a_stream_is_one_span_that_ends_with_the_stream(llm, spans, monkeypatch):
    monkeypatch.setenv(CAPTURE_CONTENT_ENV, "true")
    llm._client.generate_stream = _streamed(
        StreamTextDelta("Blue light "), StreamTextDelta("scatters."), STOP, USAGE
    )

    stream = await llm.ask(prompt="Why is the sky blue?", stream=True)
    assert not spans.get_finished_spans(), "the call is open while the stream is"
    async for _ in stream:
        pass

    (span,) = spans.get_finished_spans()
    assert span.attributes["gen_ai.usage.output_tokens"] == 20
    assert '"content": "Blue light scatters."' in span.attributes["gen_ai.output.messages"]


async def test_a_stream_closed_early_ends_its_span(llm, spans):
    llm._client.generate_stream = _streamed(StreamTextDelta("Blue"), StreamTextDelta("more"))

    stream = await llm.ask(prompt="Why is the sky blue?", stream=True)
    async for _ in stream:
        break
    await stream.aclose()

    (span,) = spans.get_finished_spans()
    assert span.status.status_code != StatusCode.ERROR


async def test_a_stream_that_fails_ends_its_span_as_an_error(llm, spans):
    llm._max_retries = 1
    llm._client.generate_stream = _streamed(StreamTextDelta("Blue"), ValueError("cut"))

    stream = await llm.ask(prompt="Why is the sky blue?", stream=True)
    with pytest.raises(ValueError):
        async for _ in stream:
            pass

    (span,) = spans.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR


async def test_without_opentelemetry_a_call_runs_untraced(llm, monkeypatch):
    monkeypatch.setattr(tracing, "_trace", None)

    result = await llm.ask(prompt="Why is the sky blue?")

    assert result.text == "Blue light scatters most."
