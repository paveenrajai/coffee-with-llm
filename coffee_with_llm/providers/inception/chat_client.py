"""Inception Mercury provider using OpenAI-compatible Chat Completions.

Uses ``AsyncOpenAI`` pointed at ``https://api.inceptionlabs.ai/v1``. Mercury is
text-only: attachments raise ``ValidationError``. Reasoning effort is passed as
``reasoning_effort`` (``instant`` | ``low`` | ``medium`` | ``high``) via
``extra_body`` so the OpenAI SDK forwards it to Inception.
"""

from __future__ import annotations

import inspect
import json
import logging
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union

from ...attachments import Attachment
from ...config import Config
from ...exceptions import APIError, ConfigurationError, ValidationError
from ...rate_limit import is_rate_limit_error
from ...types import (
    StreamStepBoundary,
    StreamTextDelta,
    StreamToolArgumentsDelta,
    StreamToolCallEnd,
    StreamToolCallStart,
    StreamUsageSink,
    TokenUsage,
)
from ..tool_utils import (
    extract_error_code,
    normalize_tool_result,
    should_break_loop,
    update_step_tracking,
)
from .models import normalize_inception_model_id

logger = logging.getLogger(__name__)

INCEPTION_BASE_URL = "https://api.inceptionlabs.ai/v1"
DEFAULT_MAX_TOKENS = 8192
_INCEPTION_EFFORTS = frozenset({"instant", "low", "medium", "high"})
REASONING_LOG_TOOL_NAME = "reasoning_log"


def normalize_inception_effort(effort: Optional[str]) -> Optional[str]:
    """Validate Inception ``reasoning_effort`` (includes ``instant``)."""
    if effort is None:
        return None
    key = str(effort).strip().lower()
    if not key:
        return None
    if key not in _INCEPTION_EFFORTS:
        logger.warning(
            "Unknown reasoning_effort=%r for Inception; expected one of %s. Ignoring.",
            effort,
            sorted(_INCEPTION_EFFORTS),
        )
        return None
    return key


def _convert_tools_to_openai(tools_schema: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize tools to OpenAI Chat Completions nested function format.

    Accepts:
    - Nested: ``{"type": "function", "function": {"name", "description", "parameters"}}``
    - Flat: ``{"type": "function", "name", "description", "parameters"}``
    """
    if not tools_schema:
        return []

    out: List[Dict[str, Any]] = []
    for t in tools_schema:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function" and isinstance(t.get("function"), dict):
            fn = t["function"]
            if "name" not in fn:
                continue
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": fn["name"],
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters")
                        if isinstance(fn.get("parameters"), dict)
                        else {},
                    },
                }
            )
            continue
        name = t.get("name")
        if not name:
            continue
        out.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters")
                    if isinstance(t.get("parameters"), dict)
                    else {},
                },
            }
        )
    return out


def _normalize_response_format(response_format: Any) -> Optional[Dict[str, Any]]:
    """Pass through Chat Completions ``response_format``; coerce common string aliases."""
    if not response_format:
        return None
    if isinstance(response_format, dict):
        return response_format
    if isinstance(response_format, str):
        fmt = response_format.strip().lower()
        if fmt in ("json", "json_object"):
            return {"type": "json_object"}
        if fmt in ("text", "markdown", "md"):
            return {"type": "text"}
    return None


def _extract_cached_tokens(usage: Any) -> Optional[int]:
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", None)
        if cached is not None:
            return int(cached)
    cached = getattr(usage, "cached_tokens", None)
    if cached is not None:
        return int(cached)
    return None


def _usage_from_chat(usage: Any) -> Optional[TokenUsage]:
    if usage is None:
        return None
    try:
        inp = int(getattr(usage, "prompt_tokens", 0) or 0)
        out = int(getattr(usage, "completion_tokens", 0) or 0)
        total = int(getattr(usage, "total_tokens", 0) or (inp + out))
        cached = _extract_cached_tokens(usage)
        return TokenUsage(
            input_tokens=inp,
            output_tokens=out,
            total_tokens=total,
            cached_tokens=cached,
        )
    except Exception:
        return None


def _message_to_dict(message: Any) -> Dict[str, Any]:
    """Serialize an assistant chat message for the next request turn."""
    if isinstance(message, dict):
        return dict(message)
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    content = getattr(message, "content", None)
    tool_calls = getattr(message, "tool_calls", None) or []
    out: Dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        serialized: List[Dict[str, Any]] = []
        for tc in tool_calls:
            if isinstance(tc, dict):
                serialized.append(tc)
                continue
            fn = getattr(tc, "function", None)
            serialized.append(
                {
                    "id": getattr(tc, "id", "") or "",
                    "type": getattr(tc, "type", None) or "function",
                    "function": {
                        "name": getattr(fn, "name", "") if fn is not None else "",
                        "arguments": getattr(fn, "arguments", "") if fn is not None else "",
                    },
                }
            )
        out["tool_calls"] = serialized
    return out


def _prepare_inception_request_params(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: Optional[int],
    top_p: Optional[float],
    temperature: Optional[float],
    presence_penalty: Optional[float],
    tools: List[Dict[str, Any]],
    force_tool_use: bool,
    response_format: Optional[Any],
    reasoning_effort: Optional[str],
) -> Dict[str, Any]:
    """Build Chat Completions kwargs for Inception."""
    api_model = normalize_inception_model_id(model)
    params: Dict[str, Any] = {
        "model": api_model,
        "messages": messages,
        "max_tokens": max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS,
    }
    if temperature is not None:
        params["temperature"] = temperature
    if top_p is not None:
        params["top_p"] = top_p
    if presence_penalty is not None:
        params["presence_penalty"] = presence_penalty
    if tools:
        params["tools"] = tools
        if force_tool_use:
            params["tool_choice"] = "required"
    fmt = _normalize_response_format(response_format)
    if fmt:
        params["response_format"] = fmt
    effort = normalize_inception_effort(reasoning_effort)
    if effort is not None:
        # OpenAI SDK may not type this for custom bases; extra_body always forwards.
        params["extra_body"] = {"reasoning_effort": effort}
    return params


class InceptionChatClient:
    """Inception Mercury client using OpenAI-compatible Chat Completions."""

    def __init__(self, config: Config, request_timeout: Optional[float] = None) -> None:
        self._api_key = config.require_inception_key()
        self._request_timeout = request_timeout

        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ConfigurationError(
                "OpenAI package not installed. Install with: pip install openai"
            ) from e
        except Exception as e:
            raise ConfigurationError(f"Failed to import OpenAI client: {e}") from e

        self._AsyncOpenAI = AsyncOpenAI

    def _make_client(self) -> Any:
        client_kwargs: Dict[str, Any] = {
            "api_key": self._api_key,
            "base_url": INCEPTION_BASE_URL,
        }
        if self._request_timeout is not None:
            client_kwargs["timeout"] = self._request_timeout
        return self._AsyncOpenAI(**client_kwargs)

    @staticmethod
    def _reject_attachments(attachments: Optional[List[Attachment]]) -> None:
        if attachments:
            raise ValidationError(
                "Inception Mercury models are text-only; attachments are not supported"
            )

    def _build_messages(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, Any]]],
        instructions: Optional[str],
        system_instruct: str = "",
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        system = (instructions or system_instruct or "").strip()
        if system:
            out.append({"role": "system", "content": system})
        if messages:
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                entry: Dict[str, Any] = {"role": role, "content": content}
                if role == "assistant" and msg.get("tool_calls"):
                    entry["tool_calls"] = msg["tool_calls"]
                if role == "tool" and msg.get("tool_call_id"):
                    entry["tool_call_id"] = msg["tool_call_id"]
                out.append(entry)
        out.append({"role": "user", "content": prompt})
        return out

    async def _execute_tool(
        self,
        name: Optional[str],
        args: Dict[str, Any],
        execute_tool_cb: Optional[Callable[[str, Dict[str, Any]], Any]],
    ) -> Dict[str, Any]:
        if execute_tool_cb is None:
            return {"ok": False, "result": {}, "error": "no executor provided"}
        try:
            maybe = execute_tool_cb(name, args)
            if inspect.isawaitable(maybe) or hasattr(maybe, "__await__"):
                result = await maybe
            else:
                result = maybe
            return normalize_tool_result(result)
        except Exception as e:
            logger.error(f"Tool execution failed for {name}: {e}")
            return {"ok": False, "result": {}, "error": str(e)}

    def _parse_tool_calls(self, message: Any) -> List[Dict[str, Any]]:
        raw = getattr(message, "tool_calls", None) or []
        parsed: List[Dict[str, Any]] = []
        for tc in raw:
            if isinstance(tc, dict):
                fn = tc.get("function") or {}
                args_raw = fn.get("arguments", "") if isinstance(fn, dict) else ""
                name = fn.get("name", "") if isinstance(fn, dict) else ""
                tc_id = tc.get("id", "")
            else:
                fn = getattr(tc, "function", None)
                args_raw = getattr(fn, "arguments", "") if fn is not None else ""
                name = getattr(fn, "name", "") if fn is not None else ""
                tc_id = getattr(tc, "id", "") or ""
            try:
                args = json.loads(args_raw) if args_raw else {}
            except json.JSONDecodeError:
                args = {}
            if not isinstance(args, dict):
                args = {}
            parsed.append(
                {
                    "id": tc_id,
                    "name": name or "",
                    "arguments": args,
                    "raw_arguments": args_raw,
                }
            )
        return parsed

    def _get_tool_error_retry_message(
        self,
        output_payloads: List[Dict[str, Any]],
        tool_error_callback: Optional[
            Callable[[str, Optional[str], Dict[str, Any]], Optional[str]]
        ],
    ) -> Optional[str]:
        if not tool_error_callback:
            return None
        for out in output_payloads:
            if out["payload"].get("ok"):
                continue
            msg = tool_error_callback(
                out["name"], extract_error_code(out["payload"]), out["payload"]
            )
            if msg:
                return msg
        return None

    def _log_reasoning_summary(self, message: Any) -> None:
        summary = getattr(message, "reasoning_summary", None)
        if summary and str(summary).strip():
            text = str(summary)
            preview = text[:200]
            suffix = "..." if len(text) > 200 else ""
            logger.debug(
                "Inception reasoning_summary (%d chars): %s%s",
                len(text),
                preview,
                suffix,
            )

    async def _finalize_empty_response(
        self,
        client: Any,
        params: Dict[str, Any],
        base_messages: List[Dict[str, Any]],
    ) -> tuple[str, int, int]:
        finalize_params = dict(params)
        finalize_params.pop("tools", None)
        finalize_params.pop("tool_choice", None)
        finalize_params["messages"] = base_messages + [
            {
                "role": "user",
                "content": "Finalize now. Return the final answer. No further tool calls.",
            }
        ]
        finalize_resp = await client.chat.completions.create(**finalize_params)
        choice = (finalize_resp.choices or [None])[0]
        text = ""
        if choice is not None:
            msg = getattr(choice, "message", None)
            text = (getattr(msg, "content", None) or "") if msg is not None else ""
        usage = getattr(finalize_resp, "usage", None)
        inp = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        out = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        return text, inp, out

    async def generate(
        self,
        *,
        prompt: str,
        model: str,
        messages: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        instructions: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        tools_schema: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Any] = None,
        execute_tool_cb: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
        tool_error_callback: Optional[
            Callable[[str, Optional[str], Dict[str, Any]], Optional[str]]
        ] = None,
        max_steps: int = 16,
        max_effective_tool_steps: int = 8,
        force_tool_use: bool = False,
        temperature: Optional[float] = None,
        system_instruct: str = "",
        attachments: Optional[List[Attachment]] = None,
    ) -> tuple[str, TokenUsage]:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        if not model or not model.strip():
            raise ValueError("Model name cannot be empty")
        self._reject_attachments(attachments)

        try:
            client = self._make_client()
        except Exception as e:
            raise APIError(f"Failed to initialize Inception client: {e}") from e

        tools = _convert_tools_to_openai(tools_schema or [])
        base_messages = self._build_messages(prompt, messages, instructions, system_instruct)
        params = _prepare_inception_request_params(
            model=model,
            messages=base_messages,
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
            presence_penalty=presence_penalty,
            tools=tools,
            force_tool_use=force_tool_use,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
        )
        if force_tool_use and tools:
            logger.info("[INCEPTION] tool_choice=required (force tool use)")

        last_resp: Optional[Any] = None
        last_nonempty_output = ""
        effective_steps = 0
        consecutive_reasoning_only = 0
        pending_resp: Optional[Any] = None
        total_input = 0
        total_output = 0
        total_cached = 0

        for step in range(max_steps):
            try:
                if pending_resp is not None:
                    resp = pending_resp
                    pending_resp = None
                else:
                    resp = await client.chat.completions.create(**params)
                last_resp = resp
            except Exception as e:
                if is_rate_limit_error(e):
                    logger.warning(f"Inception API rate limit hit at step {step + 1}: {e}")
                    raise
                logger.error(f"Inception API call failed at step {step + 1}: {e}")
                raise APIError(f"Inception API request failed: {e}") from e

            step_usage = _usage_from_chat(getattr(resp, "usage", None))
            if step_usage:
                total_input += step_usage.input_tokens
                total_output += step_usage.output_tokens
                if step_usage.cached_tokens:
                    total_cached += step_usage.cached_tokens

            choice = (getattr(resp, "choices", None) or [None])[0]
            if choice is None:
                break
            message = getattr(choice, "message", None)
            if message is None:
                break

            self._log_reasoning_summary(message)
            text = getattr(message, "content", None) or ""
            if text.strip():
                last_nonempty_output = text

            finish_reason = getattr(choice, "finish_reason", None) or "stop"
            tool_calls = self._parse_tool_calls(message)
            logger.info(
                "[INCEPTION] step=%d finish_reason=%s tool_calls=%d",
                step + 1,
                finish_reason,
                len(tool_calls),
            )

            if finish_reason != "tool_calls" and tools and force_tool_use and not tool_calls:
                logger.warning(
                    "Expected tool_calls but got finish_reason=%s (force_tool_use=True).",
                    finish_reason,
                )

            if finish_reason == "tool_calls" or tool_calls:
                if not tool_calls:
                    logger.warning(
                        "[INCEPTION] finish_reason=tool_calls but no tool_calls "
                        "on message. Breaking."
                    )
                    break

                tool_names = [t["name"] for t in tool_calls]
                logger.info("[INCEPTION] Executing tools: %s", tool_names)

                output_payloads: List[Dict[str, Any]] = []
                had_non_reasoning_tool = False
                for tc in tool_calls:
                    result_payload = await self._execute_tool(
                        tc["name"], tc["arguments"], execute_tool_cb
                    )
                    output_payloads.append(
                        {
                            "tool_call_id": tc["id"],
                            "name": tc["name"],
                            "payload": result_payload,
                        }
                    )
                    if tc["name"] and tc["name"] != REASONING_LOG_TOOL_NAME:
                        if "reasoning" not in tc["name"].lower():
                            had_non_reasoning_tool = True

                retry_message = self._get_tool_error_retry_message(
                    output_payloads, tool_error_callback
                )
                if retry_message is not None:
                    logger.info("[TOOL_ERROR] New session (callback returned retry message)")
                    new_messages = base_messages + [{"role": "user", "content": retry_message}]
                    params["messages"] = new_messages
                    pending_resp = await client.chat.completions.create(**params)
                    continue

                new_messages = list(base_messages)
                new_messages.append(_message_to_dict(message))
                for out in output_payloads:
                    new_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": out["tool_call_id"],
                            "content": json.dumps(out["payload"]),
                        }
                    )
                params["messages"] = new_messages
                base_messages = new_messages

                effective_steps, consecutive_reasoning_only = update_step_tracking(
                    had_non_reasoning_tool,
                    effective_steps,
                    consecutive_reasoning_only,
                    max_effective_tool_steps,
                )
                if should_break_loop(
                    effective_steps, consecutive_reasoning_only, max_effective_tool_steps
                ):
                    logger.info(
                        "[INCEPTION] Breaking loop: effective_steps=%d consecutive_reasoning=%d",
                        effective_steps,
                        consecutive_reasoning_only,
                    )
                    break
            else:
                break

        final_text = ""
        if last_resp is not None:
            choice = (getattr(last_resp, "choices", None) or [None])[0]
            if choice is not None:
                msg = getattr(choice, "message", None)
                final_text = (getattr(msg, "content", None) or "") if msg is not None else ""
        if not final_text.strip():
            final_text = last_nonempty_output or ""

        if not final_text.strip():
            try:
                final_text, inp_delta, out_delta = await self._finalize_empty_response(
                    client, params, base_messages
                )
                total_input += inp_delta
                total_output += out_delta
            except Exception as e:
                if is_rate_limit_error(e):
                    raise
                logger.warning(f"Failed to finalize response: {e}")
                if not final_text:
                    raise APIError(f"Failed to generate final response: {e}") from e

        if not final_text.strip():
            raise APIError("Empty response received from Inception API")

        return final_text, TokenUsage(
            input_tokens=total_input,
            output_tokens=total_output,
            total_tokens=total_input + total_output,
            cached_tokens=total_cached if total_cached else None,
        )

    async def generate_stream(
        self,
        *,
        prompt: str,
        model: str,
        messages: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        instructions: Optional[str] = None,
        system_instruct: str = "",
        presence_penalty: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
        tools_schema: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Any] = None,
        execute_tool_cb: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
        tool_error_callback: Optional[
            Callable[[str, Optional[str], Dict[str, Any]], Optional[str]]
        ] = None,
        max_steps: int = 16,
        max_effective_tool_steps: int = 8,
        force_tool_use: bool = False,
        usage_sink: Optional[StreamUsageSink] = None,
        attachments: Optional[List[Attachment]] = None,
    ) -> AsyncIterator[Union[object, TokenUsage]]:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        if not model or not model.strip():
            raise ValueError("Model name cannot be empty")
        self._reject_attachments(attachments)

        try:
            client = self._make_client()
        except Exception as e:
            raise APIError(f"Failed to initialize Inception client: {e}") from e

        tools = _convert_tools_to_openai(tools_schema or [])
        base_messages = self._build_messages(prompt, messages, instructions, system_instruct)
        use_tools = bool(tools and execute_tool_cb)

        total_input = 0
        total_output = 0
        total_cached = 0
        effective_steps = 0
        consecutive_reasoning_only = 0
        pending_resp: Optional[Any] = None

        def apply_usage(usage: Any) -> None:
            nonlocal total_input, total_output, total_cached
            step_usage = _usage_from_chat(usage)
            if step_usage is None:
                return
            total_input += step_usage.input_tokens
            total_output += step_usage.output_tokens
            if step_usage.cached_tokens:
                total_cached += step_usage.cached_tokens
            if usage_sink is not None:
                usage_sink.replace_with(
                    TokenUsage(
                        total_input,
                        total_output,
                        total_input + total_output,
                        total_cached if total_cached else None,
                    )
                )

        try:
            for step in range(max_steps):
                if use_tools and step > 0:
                    yield StreamStepBoundary(step)

                params = _prepare_inception_request_params(
                    model=model,
                    messages=base_messages,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    temperature=temperature,
                    presence_penalty=presence_penalty,
                    tools=tools,
                    force_tool_use=force_tool_use,
                    response_format=response_format,
                    reasoning_effort=reasoning_effort,
                )

                if pending_resp is not None:
                    resp = pending_resp
                    pending_resp = None
                    apply_usage(getattr(resp, "usage", None))
                    choice = (getattr(resp, "choices", None) or [None])[0]
                    message = getattr(choice, "message", None) if choice is not None else None
                    finish_reason = (
                        getattr(choice, "finish_reason", None) or "stop"
                        if choice is not None
                        else "stop"
                    )
                    tool_calls = self._parse_tool_calls(message) if message is not None else []
                else:
                    stream_params = dict(params)
                    stream_params["stream"] = True
                    stream_params["stream_options"] = {"include_usage": True}

                    content_parts: List[str] = []
                    tool_acc: Dict[int, Dict[str, str]] = {}
                    started_tools: set[int] = set()
                    finish_reason = "stop"
                    stream_usage: Any = None

                    stream = await client.chat.completions.create(**stream_params)
                    async for chunk in stream:
                        if getattr(chunk, "usage", None) is not None:
                            stream_usage = chunk.usage
                            if usage_sink is not None:
                                u = _usage_from_chat(stream_usage)
                                if u is not None:
                                    usage_sink.merge(
                                        u.input_tokens, u.output_tokens, u.cached_tokens
                                    )
                        choices = getattr(chunk, "choices", None) or []
                        if not choices:
                            continue
                        choice = choices[0]
                        if getattr(choice, "finish_reason", None):
                            finish_reason = choice.finish_reason
                        delta = getattr(choice, "delta", None)
                        if delta is None:
                            continue
                        piece = getattr(delta, "content", None) or ""
                        if piece:
                            content_parts.append(piece)
                            yield StreamTextDelta(piece)
                        for tc_delta in getattr(delta, "tool_calls", None) or []:
                            idx = int(getattr(tc_delta, "index", 0) or 0)
                            if idx not in tool_acc:
                                tool_acc[idx] = {"id": "", "name": "", "arguments": ""}
                            if getattr(tc_delta, "id", None):
                                tool_acc[idx]["id"] = str(tc_delta.id)
                            fn = getattr(tc_delta, "function", None)
                            if fn is not None:
                                name = getattr(fn, "name", None) or ""
                                if name:
                                    tool_acc[idx]["name"] = name
                                    if idx not in started_tools:
                                        started_tools.add(idx)
                                        yield StreamToolCallStart(
                                            tool_acc[idx]["id"] or f"tool-{idx}",
                                            name,
                                        )
                                args_frag = getattr(fn, "arguments", None) or ""
                                if args_frag:
                                    tool_acc[idx]["arguments"] += args_frag
                                    yield StreamToolArgumentsDelta(
                                        tool_acc[idx]["id"] or f"tool-{idx}",
                                        args_frag,
                                    )

                    apply_usage(stream_usage)

                    # Reconstruct message-like object for the tool loop.
                    class _Fn:
                        def __init__(self, name: str, arguments: str) -> None:
                            self.name = name
                            self.arguments = arguments

                    class _Tc:
                        def __init__(self, tc_id: str, name: str, arguments: str) -> None:
                            self.id = tc_id
                            self.type = "function"
                            self.function = _Fn(name, arguments)

                    class _Msg:
                        def __init__(self) -> None:
                            self.content = "".join(content_parts) or None
                            self.tool_calls = [
                                _Tc(v["id"] or f"tool-{i}", v["name"], v["arguments"])
                                for i, v in sorted(tool_acc.items())
                            ] or None
                            self.reasoning_summary = None

                    message = _Msg()
                    tool_calls = self._parse_tool_calls(message)

                if not use_tools:
                    break

                if finish_reason != "tool_calls" and not tool_calls:
                    break
                if not tool_calls:
                    break

                for tc in tool_calls:
                    yield StreamToolCallEnd(
                        id=str(tc["id"]),
                        name=str(tc["name"]),
                        arguments=dict(tc["arguments"]),
                    )

                output_payloads: List[Dict[str, Any]] = []
                had_non_reasoning_tool = False
                for tc in tool_calls:
                    result_payload = await self._execute_tool(
                        tc["name"], tc["arguments"], execute_tool_cb
                    )
                    output_payloads.append(
                        {
                            "tool_call_id": tc["id"],
                            "name": tc["name"],
                            "payload": result_payload,
                        }
                    )
                    if tc["name"] and tc["name"] != REASONING_LOG_TOOL_NAME:
                        if "reasoning" not in tc["name"].lower():
                            had_non_reasoning_tool = True

                retry_message = self._get_tool_error_retry_message(
                    output_payloads, tool_error_callback
                )
                if retry_message is not None:
                    base_messages = base_messages + [{"role": "user", "content": retry_message}]
                    params["messages"] = base_messages
                    pending_resp = await client.chat.completions.create(**params)
                    continue

                new_messages = list(base_messages)
                new_messages.append(_message_to_dict(message))
                for out in output_payloads:
                    new_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": out["tool_call_id"],
                            "content": json.dumps(out["payload"]),
                        }
                    )
                base_messages = new_messages

                effective_steps, consecutive_reasoning_only = update_step_tracking(
                    had_non_reasoning_tool,
                    effective_steps,
                    consecutive_reasoning_only,
                    max_effective_tool_steps,
                )
                if should_break_loop(
                    effective_steps, consecutive_reasoning_only, max_effective_tool_steps
                ):
                    break

            yield TokenUsage(
                input_tokens=total_input,
                output_tokens=total_output,
                total_tokens=total_input + total_output,
                cached_tokens=total_cached if total_cached else None,
            )
        except Exception as e:
            if is_rate_limit_error(e):
                raise
            raise APIError(f"Inception streaming failed: {e}") from e
