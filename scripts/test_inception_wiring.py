#!/usr/bin/env python3
"""Smoke-test Inception / Mercury wiring against the live API.

Reads keys from the repo-root ``.env`` (see ``.env.example``).

Usage (from repo root)::

    cp .env.example .env   # then set INCEPTION_API_KEY
    uv run scripts/test_inception_wiring.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coffee_with_llm import AskLLM, AskResult, Config  # noqa: E402
from coffee_with_llm.providers.inception import InceptionChatClient  # noqa: E402
from coffee_with_llm.providers.registry import get_provider  # noqa: E402

MODEL = "mercury-2"


def _check_registry(config: Config) -> None:
    client = get_provider(MODEL, config)
    assert isinstance(client, InceptionChatClient), (
        f"expected InceptionChatClient, got {type(client)}"
    )
    prefixed = get_provider(f"inception/{MODEL}", config)
    assert isinstance(prefixed, InceptionChatClient)
    print(f"[ok] registry routes {MODEL!r} and inception/{MODEL!r} → InceptionChatClient")


async def _check_basic_ask() -> None:
    llm = AskLLM(model=MODEL)
    result = await llm.ask(
        prompt="Reply with exactly one word: pong",
        reasoning_effort="instant",
        max_tokens=32,
        temperature=0.2,
    )
    assert isinstance(result, AskResult)
    print(f"[ok] basic ask: {result.text!r}")
    if result.usage:
        print(
            f"     usage: in={result.usage.input_tokens} out={result.usage.output_tokens} "
            f"cost_usd={result.usage.cost_usd}"
        )


async def _check_tool_loop() -> None:
    calls: list[tuple[str, dict]] = []

    async def execute_tool(name: str, args: dict) -> dict:
        calls.append((name, args))
        if name == "add":
            return {"ok": True, "result": {"sum": int(args["a"]) + int(args["b"])}}
        return {"ok": False, "result": {}, "error": f"unknown tool {name}"}

    tools = [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two integers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                },
            },
        }
    ]

    llm = AskLLM(model=MODEL)
    result = await llm.ask(
        prompt="Use the add tool to compute 17 + 25. Then state the sum.",
        tools_schema=tools,
        execute_tool_cb=execute_tool,
        force_tool_use=True,
        reasoning_effort="low",
        max_tokens=256,
    )
    assert isinstance(result, AskResult)
    assert calls, "expected at least one tool call"
    assert calls[0][0] == "add", calls
    print(f"[ok] tool loop: calls={calls}")
    print(f"     final: {result.text!r}")


async def main() -> int:
    config = Config.from_env()
    if not config.inception_api_key:
        print(
            "INCEPTION_API_KEY missing. Copy .env.example → .env and set the key.",
            file=sys.stderr,
        )
        return 1

    print(f"INCEPTION_API_KEY loaded (len={len(config.inception_api_key)})")
    _check_registry(config)

    print("--- live API ---")
    await _check_basic_ask()
    await _check_tool_loop()
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
