# coffee_with_llm

A model-agnostic Python library providing a unified API for OpenAI, Anthropic Claude, and Google Gemini.

## Features

- **Model-agnostic**: Automatically selects the appropriate provider based on model name
- **Unified API**: Same interface for OpenAI, Anthropic Claude, and Google Gemini
- **Tool Calling**: Full support for OpenAI's tool calling with multi-step execution
- **Structured Outputs**: Support for JSON schema and response formatting
- **Caching**: Built-in prompt caching for both providers
- **Citations**: Automatic citation injection for Google Gemini responses
- **Reasoning**: Support for OpenAI's reasoning models with effort control

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

asyncio.run(main())
```

## Configuration

Set environment variables for API keys:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
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

### Tool Calling (OpenAI, Anthropic, Google)

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

### Reasoning Models (OpenAI)

```python
llm = AskLLM(model="gpt-5.4")
response = await llm.ask(
    prompt="Solve this math problem: 2x + 5 = 15",
    reasoning_effort="high"
)
```

### Streaming

```python
from coffee_with_llm import StreamTextDelta

llm = AskLLM(model="gpt-5.4")
result = await llm.ask(prompt="Explain recursion in programming.", stream=True)
async for chunk in result:
    if isinstance(chunk, StreamTextDelta):
        print(chunk.text, end="", flush=True)
print(f"\nUsage: {result.usage.input_tokens} in, {result.usage.output_tokens} out")
```

**Events:** Iteration yields `StreamTextDelta`, `StreamToolCallStart`, `StreamToolArgumentsDelta`, `StreamToolCallEnd`, and `StreamStepBoundary` (between tool rounds), then completes. Bare `str` from older mocks is accepted and normalized to `StreamTextDelta`.

**Tools and schema:** `stream=True` works with `tools_schema` and `response_format` when the provider supports them; you must pass `execute_tool_cb` whenever `tools_schema` is set.

**Gemini:** Streaming with custom tools uses function calling only (no Google Search grounding in the same streaming request).

**Usage and cost:** `result.usage` (including `cost_usd`) is set when the stream finishes normally. If you stop early (`break`), call `await result.aclose()` so usage can be filled from the best-effort `StreamUsageSink` when the provider reported partial usage.

Rate limits trigger retry before the first chunk.

## Supported Models

### OpenAI
- `gpt-5.4`, `gpt-5.4-pro` (flagship, reasoning)
- `gpt-5.3-instant`, `gpt-5-mini`, `gpt-5-nano` (fast, cost-effective)
- `gpt-4o`, `gpt-4o-mini`
- Any OpenAI model name

### Anthropic Claude
- `claude-opus-4-6`, `claude-sonnet-4-6` (latest)
- `claude-haiku`, `claude-3-5-sonnet`
- Any Claude model name (claude-* prefix)

### Google Gemini
- `gemini-3.1-pro-preview`, `gemini-3.1-flash` (latest)
- `gemini-2.5-pro`, `gemini-2.5-flash`
- Any Google Gemini model name

## API Reference

### `AskLLM`

#### `__init__(*, model, config=None, ...)`

Initialize the LLM client.

**Parameters:**
- `model` (str): Model name (provider auto-detected, required)
- `config` (Config, optional): Config instance. If None, uses `Config.from_env()` for API keys
- `min_delay_between_calls` (float, optional): Min delay between API calls in seconds (default: 1.0)
- `max_retries` (int, optional): Max retries for rate limit errors (default: 3)
- `request_timeout` (float, optional): Request timeout in seconds (default: 60)
- `google_explicit_cache` (bool, optional): Enable Google context caching (default: True)
- `google_inline_citations` (bool, optional): Inject `[cite: url]` markers for Gemini grounding (default: True)

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
- `reasoning_effort` (str, optional): Reasoning effort level (OpenAI only)
- `tools_schema` (list, optional): Tool/function calling schema (OpenAI, Anthropic, Google)
- `response_format` (dict, optional): Response format specification
- `execute_tool_cb` (callable, optional): Tool execution callback (OpenAI, Anthropic, Google)
- `tool_error_callback` (callable, optional): Callback when tool returns ok=False
- `max_steps` (int, optional): Maximum tool-calling steps (default: 24)
- `max_effective_tool_steps` (int, optional): Maximum effective tool steps (default: 12)
- `force_tool_use` (bool, optional): Force at least one tool call when tools provided (default: False)
- `stream` (bool, optional): When True, return `StreamResult` (default: False)

**Returns:** `AskResult` – Object with `.text` (str) and `.usage` (TokenUsage). When `stream=True`, returns `StreamResult` – async iterable of stream events (see Streaming above); `.usage` after completion or `aclose()`.

**Raises:**
- `ValidationError`: If prompt is empty or invalid parameters provided
- `APIError`: If the API call fails
- `ConfigurationError`: If API keys are missing or client initialization fails

## Environment Variables

- `OPENAI_API_KEY`: OpenAI API key (required for OpenAI models)
- `ANTHROPIC_API_KEY`: Anthropic API key (required for Claude models)
- `GOOGLE_API_KEY`: Google API key (required for Google models)

Google-specific options (`google_explicit_cache`, `google_inline_citations`) are passed as constructor params to `AskLLM`; see docstring.

## License

MIT

## Contributing

Contributions welcome! Please open an issue or submit a pull request.
