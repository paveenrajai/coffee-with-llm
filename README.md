# coffee_with_llm

A model-agnostic Python library providing a unified API for OpenAI, Anthropic Claude, Google Gemini, and Inception Mercury.

## Features

- **Model-agnostic**: Picks the provider from the model id, or from an explicit `provider/model` prefix (see below)
- **Unified API**: Same interface for OpenAI, Anthropic Claude, Google Gemini, and Inception Mercury
- **Tool Calling**: Full support for OpenAI's tool calling with multi-step execution
- **Structured Outputs**: Support for JSON schema and response formatting
- **Caching**: Google explicit context cache; Anthropic automatic prompt cache; OpenAI/Inception pass through `cached_tokens` when present
- **Attachments**: Send PDFs and images as input with one `Attachment` type; each provider translates it to its own native format (Inception Mercury is text-only)
- **Citations**: Automatic citation injection for Google Gemini responses
- **Extended thinking**: Single `reasoning_effort` knob (`"low" | "medium" | "high"`, plus `"instant"` on Inception) translates to OpenAI `reasoning.effort`, Anthropic adaptive `output_config.effort` (4.6+) or legacy `thinking.budget_tokens`, Google `thinking_config`, and Inception `reasoning_effort`

## Installation

```bash
pip install coffee_with_llm
```

## Quick Start

```python
import asyncio
from coffee_with_llm import AskLLM

async def main():
    # Initialize with any model (model is required)
    llm = AskLLM(model="gpt-5.4")
    
    # Simple question
    response = await llm.ask(
        prompt="What is Python?",
        system_instruct="You are a helpful assistant."
    )
    print(response.text)
    
    # Use Google Gemini
    llm_gemini = AskLLM(model="gemini-3.1-pro-preview")
    response = await llm_gemini.ask(
        prompt="Explain quantum computing",
        system_instruct="You are a physics expert."
    )
    print(response.text)

    # Google Gemma (or any Google API model id): prefix selects Google; only the part
    # after "/" is sent to the API
    llm_gemma = AskLLM(model="google/gemma-2-9b-it")
    response = await llm_gemma.ask(prompt="Say hi in five words or fewer.")
    print(response.text)

asyncio.run(main())
```

## Provider prefix (`provider/model`)

You can force the provider and pass the **exact API model id** with a single slash:

| Prefix       | Provider  | Example |
|-------------|-----------|---------|
| `google/`   | Google    | `google/gemma-2-9b-it` |
| `gemini/`   | Google    | `gemini/gemini-2.5-flash` |
| `openai/`   | OpenAI    | `openai/gpt-4o-mini` |
| `anthropic/`| Anthropic | `anthropic/claude-sonnet-4-6` |
| `claude/`   | Anthropic | `claude/claude-sonnet-4-6` |
| `inception/`| Inception | `inception/mercury-2` |

Only the segment **after** the first `/` is sent to the provider API (e.g. `google/gemma-2-9b-it` → API model `gemma-2-9b-it`). If the prefix is missing or unknown, the whole string is used for both routing and the API (legacy behavior: ids like `gpt-4o-mini`, `claude-…`, `gemini-…`, `google-…`, `mercury-…` still auto-route).

An empty id after the prefix (e.g. `google/`) is rejected with `ValidationError`.

## Configuration

Copy `.env.example` to `.env` and set the keys you need. `Config.from_env()` loads
`.env` automatically — no `export` required.

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
INCEPTION_API_KEY=...
```

Smoke-test Inception wiring (live API)::

```bash
python scripts/test_inception_wiring.py
```

## Usage Examples

### Basic Usage

```python
from coffee_with_llm import AskLLM

# Model parameter is required
llm = AskLLM(model="gpt-5.4")
response = await llm.ask(prompt="Hello, world!")
```

### With System Instructions

```python
response = await llm.ask(
    prompt="Write a haiku about coding",
    system_instruct="You are a creative poet."
)
```

### Structured Outputs (JSON Schema)

```python
response = await llm.ask(
    prompt="Extract key information from: 'John Doe, age 30, works at Acme Corp'",
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "person_info",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "number"},
                    "company": {"type": "string"}
                }
            }
        }
    }
)
```

### Tool Calling (OpenAI, Anthropic, Google, Inception)

```python
def get_weather(location: str) -> dict:
    # Your tool implementation
    return {"temperature": 72, "condition": "sunny"}

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
            }
        }
    }
]

async def execute_tool(name: str, args: dict) -> dict:
    if name == "get_weather":
        result = get_weather(args.get("location", ""))
        return {"ok": True, "result": result}
    return {"ok": False, "error": "Unknown tool"}

response = await llm.ask(
    prompt="What's the weather in San Francisco?",
    tools_schema=tools,
    execute_tool_cb=execute_tool
)
```

### Extended Thinking (provider-agnostic)

`reasoning_effort` is a single string — `"low"`, `"medium"`, or `"high"` (Inception
also accepts `"instant"`) — that each provider translates to its native
extended-thinking config. Pass `None` (or omit) to disable. Unknown values are
ignored with a warning.

| Effort   | Anthropic (4.6+)     | Anthropic (legacy) / Google | Inception |
|----------|----------------------|-----------------------------|-----------|
| `instant`| —                    | —                           | `reasoning_effort=instant` |
| `low`    | `output_config.effort` | 1,024 token budget          | `reasoning_effort=low` |
| `medium` | (adaptive thinking)  | 4,096 token budget          | `reasoning_effort=medium` |
| `high`   |                      | 16,384 token budget         | `reasoning_effort=high` |

```python
# Same call works on OpenAI, Anthropic, Google, or Inception
llm = AskLLM(model="anthropic/claude-sonnet-4-6")
response = await llm.ask(
    prompt="Solve this math problem: 2x + 5 = 15",
    reasoning_effort="high",
)
```

Provider-specific notes:

- **OpenAI** — passed through as `reasoning.effort`.
- **Anthropic** — on Claude Opus/Sonnet 4.6+, Gen 5, and Mythos, sets
  `thinking={"type": "adaptive"}` and `output_config={"effort": ...}` when
  `reasoning_effort` is set. On Sonnet 5 with no effort, explicitly disables
  thinking (API default is on). On Fable/Mythos 5 with no effort, uses adaptive
  at `low` effort (thinking cannot be disabled). On older models, sets
  `thinking={"type": "enabled", "budget_tokens": N}` (legacy); then `temperature`
  is forced to `1`, `top_p` / `top_k` are dropped, and `max_tokens` is widened
  to leave ~1024 tokens for the visible answer. Short model aliases such as
  `sonnet-5` normalize to `claude-sonnet-5`. Sampling params are omitted on
  Opus 4.7+, Sonnet 5, and Fable/Mythos 5 (API returns 400 otherwise).
  With `anthropic_prompt_cache=True` (default), adds top-level
  `cache_control={"type": "ephemeral"}` for automatic multi-turn caching;
  `TokenUsage.cached_tokens` reflects `cache_read_input_tokens`;
  `cache_creation_tokens` reflects cache writes; both are included in `cost_usd`.
- **Google (Gemini 2.5+ / 3.x)** — sets `thinking_config` with
  `include_thoughts=False` so only the final answer streams to the caller.
- **Inception (Mercury)** — passed as `reasoning_effort` on Chat Completions
  (`instant` | `low` | `medium` | `high`). Supports tool calling and structured
  outputs; text-only (no attachments).

### Attachments (PDFs and images)

Build one `Attachment` and it works on OpenAI, Anthropic, and Google — each
translates it into its own native content part (Anthropic `document`/`image`
blocks, OpenAI `input_file`/`input_image` parts, Google `inline_data` parts).
Inception Mercury is text-only and raises `ValidationError` if attachments are passed.

```python
from coffee_with_llm import AskLLM, Attachment

llm = AskLLM(model="claude-opus-4-8")  # or gpt-…, or gemini-…

result = await llm.ask(
    prompt="Which model wins on each benchmark? Read the charts.",
    attachments=[Attachment.from_path("model-card.pdf")],
)
print(result.text)
```

Construct from bytes when the file never touches disk:

```python
Attachment(data=pdf_bytes, mime_type="application/pdf", filename="report.pdf")
```

Notes:

- **Input-only.** Attachments are what the model *reads*; the response is always text.
- **Attached to the prompt turn**, never to `messages` history.
- Works with `stream=True` — the binary goes in, the answer streams out.
- Supported types: `application/pdf`, `image/png`, `image/jpeg`, `image/gif`, `image/webp`.
  Anything else raises `ValidationError` before any network call.
- Attachments are re-sent (and re-billed) on every request; providers do not index
  or persist them. For repeated questions about the same document, enable the
  provider's context/prompt cache.

### Streaming

```python
from coffee_with_llm import StreamTextDelta

llm = AskLLM(model="gpt-5.4")
result = await llm.ask(prompt="Explain recursion in programming.", stream=True)
async for chunk in result:
    if isinstance(chunk, StreamTextDelta):
        print(chunk.text, end="", flush=True)
print(f"\nUsage: {result.usage.input_tokens} in, {result.usage.output_tokens} out")
print(f"Prompt tokens (all buckets): {result.usage.prompt_tokens}")
print(f"Billable tokens: {result.usage.billable_tokens}")
print(result.usage.to_dict())
```

**Events:** Iteration yields `StreamTextDelta`, `StreamToolCallStart`, `StreamToolArgumentsDelta`, `StreamToolCallEnd`, and `StreamStepBoundary` (between tool rounds), then completes. Bare `str` from older mocks is accepted and normalized to `StreamTextDelta`.

**Tools and schema:** `stream=True` works with `tools_schema` and `response_format` when the provider supports them; you must pass `execute_tool_cb` whenever `tools_schema` is set.

**Gemini:** Streaming with custom tools uses function calling only (no Google Search grounding in the same streaming request).

**Usage and cost:** `result.usage` (including `cost_usd`) is set when the stream finishes normally. If you stop early (`break`), call `await result.aclose()` so usage can be filled from the best-effort `StreamUsageSink` when the provider reported partial usage.

### Understanding token usage

`TokenUsage` exposes provider-native buckets plus computed totals:

| Field | Meaning |
|-------|---------|
| `input_tokens` | Uncached input billed at the full input rate |
| `cached_tokens` | Cache **reads** (discounted on Anthropic/OpenAI) |
| `cache_creation_tokens` | Cache **writes** (Anthropic; billed at 125% of input) |
| `output_tokens` | Generated output |
| `total_tokens` | Legacy: `input_tokens + output_tokens` only |
| `prompt_tokens` | All prompt-side tokens (input + cache read + cache write) |
| `billable_tokens` | `prompt_tokens + output_tokens` |
| `cost_usd` | Estimated USD (includes cache buckets when priced) |

On Anthropic with prompt caching enabled (default), the **first** turn of a long system prompt often looks like `input_tokens=2` with most prompt tokens in `cache_creation_tokens` — not a metering bug. Use `usage.to_dict()` for logging and dashboards.

```python
payload = result.usage.to_dict()
# {"input_tokens": 2, "output_tokens": 8158, "cache_creation_tokens": 40000,
#  "prompt_tokens": 40002, "billable_tokens": 48160, "cost_usd": 0.28, ...}
```

Rate limits trigger retry before the first chunk.

## Supported Models

### OpenAI
- `gpt-5.4`, `gpt-5.4-pro` (flagship, reasoning)
- `gpt-5.3-instant`, `gpt-5-mini`, `gpt-5-nano` (fast, cost-effective)
- `gpt-4o`, `gpt-4o-mini`
- Any OpenAI model name

### Anthropic Claude
- `claude-sonnet-5`, `claude-opus-4-8`, `claude-sonnet-4-6` (latest)
- Short aliases after provider prefix: `anthropic/sonnet-5`, `anthropic/fable-5`
- `claude-haiku`, `claude-3-5-sonnet`
- Any Claude model name (claude-* prefix)

### Google (Gemini, Gemma, …)
- `gemini-3.1-pro-preview`, `gemini-3.1-flash` (latest)
- `gemini-2.5-pro`, `gemini-2.5-flash`
- Any Gemini id the API accepts (`gemini-…` or `google-…` still auto-routes to Google)
- **Gemma** and other non-`gemini` Google ids: use the prefix form, e.g. `google/gemma-2-9b-it`, so routing hits Google and the API receives `gemma-2-9b-it`

### Inception (Mercury)
- `mercury-2` (chat; also `inception/mercury-2`)
- Short alias: `mercury` → `mercury-2`
- `mercury-*` ids auto-route to Inception
- Text-only Chat Completions (tool calling + structured outputs + `reasoning_effort`)

## API Reference

### `AskLLM`

#### `__init__(*, model, config=None, ...)`

Initialize the LLM client.

**Parameters:**
- `model` (str): Model name (required). Optional `provider/model` form (see [Provider prefix](#provider-prefix-providermodel)); otherwise provider is inferred from the id (`gpt-…`, `claude-…`, `gemini-…`, `google-…`, `mercury-…`, etc.).
- `config` (Config, optional): Config instance. If None, uses `Config.from_env()` for API keys
- `min_delay_between_calls` (float, optional): Min delay between API calls in seconds (default: 1.0)
- `max_retries` (int, optional): Max retries for rate limit errors (default: 3)
- `request_timeout` (float, optional): Request timeout in seconds (default: 60)
- `google_explicit_cache` (bool, optional): Enable Google context caching (default: True)
- `anthropic_prompt_cache` (bool, optional): Enable Anthropic automatic prompt caching (default: True)
- `google_inline_citations` (bool, optional): Inject `[cite: url]` markers for Gemini grounding (default: True)
- `google_attach_search_tool` (bool, optional): When using Gemini with no custom tools, attach the Google Search tool (default: True). Ignored for non-Google models.

#### `ask(...)`

Generate a response from the LLM.

**Parameters:**
- `prompt` (str): User prompt/question
- `system_instruct` (str, optional): System instruction
- `messages` (list, optional): Conversation history
- `max_tokens` (int, optional): Maximum tokens to generate
- `temperature` (float, optional): Sampling temperature (0-2)
- `top_p` (float, optional): Nucleus sampling parameter
- `presence_penalty` (float, optional): Presence penalty (OpenAI only)
- `reasoning_effort` (str, optional): Extended-thinking effort, `"low" | "medium" | "high"` (Inception also: `"instant"`). Provider-agnostic — see [Extended Thinking](#extended-thinking-provider-agnostic).
- `tools_schema` (list, optional): Tool/function calling schema (OpenAI, Anthropic, Google, Inception)
- `response_format` (dict, optional): Response format specification
- `execute_tool_cb` (callable, optional): Tool execution callback (OpenAI, Anthropic, Google, Inception)
- `tool_error_callback` (callable, optional): Callback when tool returns ok=False
- `max_steps` (int, optional): Maximum tool-calling steps (default: 24)
- `max_effective_tool_steps` (int, optional): Maximum effective tool steps (default: 12)
- `force_tool_use` (bool, optional): Force at least one tool call when tools provided (default: False)
- `stream` (bool, optional): When True, return `StreamResult` (default: False)
- `attachments` (list[Attachment], optional): PDFs or images for the model to read alongside `prompt`. Provider-agnostic — see [Attachments](#attachments-pdfs-and-images).

**Returns:** `AskResult` – Object with `.text` (str) and `.usage` (TokenUsage). When `stream=True`, returns `StreamResult` – async iterable of stream events (see Streaming above); `.usage` after completion or `aclose()`.

**Raises:**
- `ValidationError`: If prompt is empty or invalid parameters provided
- `APIError`: If the API call fails
- `ConfigurationError`: If API keys are missing or client initialization fails

## Environment Variables

Loaded from `.env` (see `.env.example`) or the process environment:

- `OPENAI_API_KEY`: OpenAI API key (required for OpenAI models)
- `ANTHROPIC_API_KEY`: Anthropic API key (required for Claude models)
- `GOOGLE_API_KEY`: Google API key (required for Google models)
- `INCEPTION_API_KEY`: Inception API key (required for Mercury models)

Provider options (`google_explicit_cache`, `google_inline_citations`, `anthropic_prompt_cache`) are passed as constructor params to `AskLLM`; see docstring.

## License

MIT

## Contributing

Contributions welcome! Please open an issue or submit a pull request.
