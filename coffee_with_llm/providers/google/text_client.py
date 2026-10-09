from __future__ import annotations

import hashlib
import inspect
import logging
import re
import time
from collections import OrderedDict
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union

import httpx
from google import genai
from google.genai import types

from ...attachments import Attachment
from ...config import Config
from ...exceptions import APIError, ConfigurationError
from ...rate_limit import is_rate_limit_error
from ...types import (
    Stop,
    StopReason,
    StreamStepBoundary,
    StreamTextDelta,
    StreamToolCallEnd,
    StreamUsageSink,
    TokenUsage,
    UrlRetrieval,
)
from .._cached import uncached_input
from .._reasoning import normalize_effort, thinking_budget_tokens
from .._served import served_model
from .._stop import stop_from
from ..tool_utils import (
    extract_error_code,
    normalize_tool_result,
    should_break_loop,
    update_step_tracking,
)
from .utils.citations import (
    _grounding_metadata,
    async_resolve_urls,
    collect_grounding_urls,
    inject_inline_citations,
    merge_grounding_responses,
)

logger = logging.getLogger(__name__)

#: Attached when search is on. Google Search finds pages in Google's index;
#: URL context opens a link in the prompt and reads it, so a page the index
#: does not hold (new, or small) is read instead of reported missing. URL
#: context does nothing when the prompt has no link.
SEARCH_TOOLS: List[Dict[str, Any]] = [{"google_search": {}}, {"url_context": {}}]

#: Sent after a tool loop that stopped with nothing written, with tool calls
#: switched off, so the loop ends in an answer from what the tools returned.
#: The same words the Anthropic client uses.
FINALIZE_PROMPT = "Finalize now. Return the final answer. No further tool calls."


def _url_retrievals(resp: Any) -> List[UrlRetrieval]:
    """The links URL context tried to open for this response, and how each went."""
    found: List[UrlRetrieval] = []
    for candidate in getattr(resp, "candidates", None) or []:
        meta = getattr(candidate, "url_context_metadata", None)
        for item in getattr(meta, "url_metadata", None) or []:
            url = getattr(item, "retrieved_url", None)
            if not isinstance(url, str) or not url:
                continue
            status = getattr(item, "url_retrieval_status", None)
            name = str(getattr(status, "name", None) or status or "")
            found.append(UrlRetrieval(url=url, ok=name.endswith("SUCCESS"), status=name))
    return found


def _attachment_part(attachment: Attachment) -> Dict[str, Any]:
    """Translate one attachment into a Gemini inline_data part.

    ``data`` stays raw bytes: the google-genai SDK base64-encodes it on the way
    out, so pre-encoding here would double-encode.
    """
    return {
        "inline_data": {
            "mime_type": attachment.mime_type,
            "data": attachment.data,
        }
    }


def _user_parts(
    text: str,
    attachments: Optional[List[Attachment]],
) -> List[Dict[str, Any]]:
    """Attachment parts first, then the prompt text."""
    parts: List[Dict[str, Any]] = []
    if attachments:
        parts.extend(_attachment_part(a) for a in attachments)
    parts.append({"text": text})
    return parts


MAX_CACHED_CONTEXTS = 10
CONTEXT_TTL_SECONDS = 3600  # 1 hour


#: Gemini's finish reasons, in the words every provider shares. A function
#: call ends with STOP; a reason not here is kept as OTHER.
_GEMINI_STOPS = {
    "STOP": StopReason.END,
    "MAX_TOKENS": StopReason.MAX_TOKENS,
    "SAFETY": StopReason.CONTENT_FILTER,
    "RECITATION": StopReason.CONTENT_FILTER,
    "BLOCKLIST": StopReason.CONTENT_FILTER,
    "PROHIBITED_CONTENT": StopReason.CONTENT_FILTER,
    "SPII": StopReason.CONTENT_FILTER,
    "IMAGE_SAFETY": StopReason.CONTENT_FILTER,
    "IMAGE_PROHIBITED_CONTENT": StopReason.CONTENT_FILTER,
    "IMAGE_RECITATION": StopReason.CONTENT_FILTER,
    "TOO_MANY_TOOL_CALLS": StopReason.TOOL_USE,
}


def _gemini_stop(resp: Any, *, wants_tools: bool = False) -> Optional[Stop]:
    """Why Gemini stopped writing ``resp``.

    A response that still asks for a tool says STOP like a finished one, so
    ``wants_tools`` marks it. A prompt blocked before any candidate has only
    its block reason.
    """
    if resp is None:
        return None
    candidates = getattr(resp, "candidates", None) or []
    if not candidates:
        blocked = getattr(getattr(resp, "prompt_feedback", None), "block_reason", None)
        stop = stop_from(blocked, {})
        return Stop(StopReason.CONTENT_FILTER, stop.raw) if stop else None
    stop = stop_from(getattr(candidates[0], "finish_reason", None), _GEMINI_STOPS)
    if stop is not None and wants_tools and stop.reason == StopReason.END:
        return Stop(StopReason.TOOL_USE, stop.raw)
    return stop


#: The Gemini generation a model name starts with: ``gemini-3.8-flash`` is 3.
_GEMINI_GENERATION = re.compile(r"^(?:models/)?gemini-(\d+)")


def _takes_thinking_level(model: str) -> bool:
    """Gemini 3 and later take ``thinking_level``; earlier models reject it.

    Read off the name. One with no generation in it, such as the alias
    ``gemini-flash-latest``, is not known to take a level, so it keeps the
    budget, which Gemini 3 still accepts for backward compatibility.
    """
    found = _GEMINI_GENERATION.match(model.strip().lower())
    return found is not None and int(found.group(1)) >= 3


def _thinking_config(model: str, reasoning_effort: Optional[str]) -> Optional[Any]:
    """``reasoning_effort`` as Gemini's thinking config, or ``None`` to send none.

    Gemini 3 and later are sent the level of the same name; Google recommends
    it over the budget, and a request carrying both is refused with a 400.
    Earlier models are sent a token budget.
    """
    effort = normalize_effort(reasoning_effort)
    if effort is None:
        return None
    if _takes_thinking_level(model):
        return types.ThinkingConfig(
            thinking_level=types.ThinkingLevel(effort.upper()),
            include_thoughts=False,
        )
    return types.ThinkingConfig(
        thinking_budget=thinking_budget_tokens(effort),
        include_thoughts=False,
    )


def _gemini_tokens(um: Any) -> tuple[int, int, int, Optional[int]]:
    """Uncached input, output, thinking and cached tokens from one usage_metadata.

    ``prompt_token_count`` includes the cached content, which is reported
    apart, so input here leaves it out. ``candidates_token_count`` leaves the
    thinking out, which Gemini bills as output, so output here is the two
    together.
    """
    thoughts = int(getattr(um, "thoughts_token_count", 0) or 0)
    output = int(getattr(um, "candidates_token_count", 0) or 0) + thoughts
    raw_cached = getattr(um, "cached_content_token_count", None)
    cached = int(raw_cached) if raw_cached is not None else None
    prompt = int(getattr(um, "prompt_token_count", 0) or 0)
    return (uncached_input(prompt, cached), output, thoughts, cached)


def _google_stream_chunk_to_usage_sink(
    chunk: Any,
    usage_sink: Optional[StreamUsageSink],
    base_input: int,
    base_output: int,
    base_cached: Optional[int],
    base_reasoning: int = 0,
) -> None:
    """Best-effort usage from a stream chunk (Gemini often fills this on the last chunks)."""
    um = getattr(chunk, "usage_metadata", None)
    if um is None or usage_sink is None:
        return
    pi, po, thoughts, cc = _gemini_tokens(um)
    cached_tokens: Optional[int] = base_cached
    if cc is not None:
        cached_tokens = (base_cached or 0) + cc
    usage_sink.replace_with(
        TokenUsage(
            base_input + pi,
            base_output + po,
            base_input + pi + base_output + po,
            cached_tokens if cached_tokens else None,
            reasoning_tokens=(base_reasoning + thoughts) or None,
            served_model=served_model(getattr(chunk, "model_version", None)),
        )
    )


# Gemini API rejects these JSON Schema keys; strip them when converting.
_GEMINI_REJECTED_KEYS = frozenset(
    {
        "$defs",
        "$ref",
        "additionalProperties",
        "additional_properties",
    }
)


def _inline_json_schema_refs(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Inline $ref, remove $defs and unsupported keys for Gemini compatibility.

    Gemini's API rejects: $ref, $defs, additionalProperties. This recursively
    resolves #/$defs/X references and strips unsupported fields.
    """
    if not isinstance(schema, dict):
        return schema

    defs_map: Dict[str, Any] = dict(schema.get("$defs", {}) or {})

    def resolve(obj: Any) -> Any:
        if isinstance(obj, dict):
            if "$ref" in obj and len(obj) == 1:
                ref = obj["$ref"]
                if isinstance(ref, str) and ref.startswith("#/$defs/"):
                    key = ref.split("/")[-1]
                    if key in defs_map:
                        return resolve(defs_map[key])
                return obj
            return {k: resolve(v) for k, v in obj.items() if k not in _GEMINI_REJECTED_KEYS}
        if isinstance(obj, list):
            return [resolve(v) for v in obj]
        return obj

    return resolve(schema)


def _convert_tools_to_gemini(tools_schema: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI-style tools to Gemini function_declarations format.

    OpenAI: {"type": "function", "function": {"name", "description", "parameters"}}
    Gemini: {"name", "description", "parameters"} (JSON Schema)
    """
    if not tools_schema:
        return []

    gemini_tools: List[Dict[str, Any]] = []
    for t in tools_schema:
        if t.get("type") == "function":
            fn = t.get("function") or t
            if not isinstance(fn, dict) or "name" not in fn:
                continue
            params = fn.get("parameters", {})
            params = _inline_json_schema_refs(params) if isinstance(params, dict) else {}
            gemini_tools.append(
                {
                    "name": fn.get("name", "unknown"),
                    "description": fn.get("description", ""),
                    "parameters": params,
                }
            )
        elif "name" in t and "parameters" in t:
            params = (
                _inline_json_schema_refs(t["parameters"])
                if isinstance(t.get("parameters"), dict)
                else t["parameters"]
            )
            gemini_tools.append(
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": params,
                }
            )
        elif "name" in t and "description" in t:
            raw_params = t.get("parameters", t.get("input_schema", {}))
            params = _inline_json_schema_refs(raw_params) if isinstance(raw_params, dict) else {}
            gemini_tools.append(
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": params,
                }
            )
    return gemini_tools


class GoogleTextClient:
    def __init__(
        self,
        config: Config,
        request_timeout: Optional[float] = None,
        google_explicit_cache: bool = True,
        google_inline_citations: bool = True,
        google_attach_search_tool: bool = True,
    ) -> None:
        self._api_key = config.require_google_key()
        self._google_explicit_cache = google_explicit_cache
        self._google_inline_citations = google_inline_citations
        self._google_attach_search_tool = google_attach_search_tool

        try:
            self._client = genai.Client(api_key=self._api_key)
        except ImportError as e:
            raise ConfigurationError(
                "Google GenAI package not installed. Install with: pip install google-genai"
            ) from e
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Google client: {e}") from e

        self._request_timeout = request_timeout
        self._cached_contexts: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._max_cached_contexts = MAX_CACHED_CONTEXTS
        self._context_ttl_seconds = CONTEXT_TTL_SECONDS

    def _build_config_dict(
        self,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
        tools_schema: Optional[List[Dict[str, Any]]] = None,
        include_google_search: bool = True,
        reasoning_effort: Optional[str] = None,
        model: str = "",
    ) -> Dict[str, Any]:
        """Build generation config as dict for Google Gemini API.

        ``model`` decides how ``reasoning_effort`` is sent: a level to Gemini 3
        and later, a budget to the rest and to a name that does not say.
        """
        config_dict: Dict[str, Any] = {}

        is_json_response = (
            response_format
            and isinstance(response_format, dict)
            and response_format.get("type") == "json_schema"
            and response_format.get("json_schema")
        )

        if not is_json_response:
            gemini_decls = _convert_tools_to_gemini(tools_schema or [])
            if gemini_decls:
                config_dict["tools"] = [types.Tool(function_declarations=gemini_decls)]
            elif include_google_search and not tools_schema:
                config_dict["tools"] = [dict(tool) for tool in SEARCH_TOOLS]

        if max_tokens is not None:
            config_dict["max_output_tokens"] = max_tokens
        if temperature is not None:
            config_dict["temperature"] = temperature
        if top_p is not None:
            config_dict["top_p"] = top_p

        if response_format and isinstance(response_format, dict):
            if response_format.get("type") == "json_schema":
                json_schema = response_format.get("json_schema")
                if json_schema:
                    config_dict["response_mime_type"] = "application/json"
                    config_dict["response_json_schema"] = json_schema

        thinking = _thinking_config(model, reasoning_effort)
        if thinking is not None:
            config_dict["thinking_config"] = thinking

        return config_dict

    async def _execute_tool(
        self,
        name: Optional[str],
        args: Dict[str, Any],
        execute_tool_cb: Optional[Callable[[str, Dict[str, Any]], Any]],
    ) -> Dict[str, Any]:
        """Execute a tool call via callback."""
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

    async def _finalize_empty_response(self, request_kwargs: Dict[str, Any]) -> Any:
        """Ask once more with tool calls off, after a tool loop wrote nothing.

        A loop stopped by its step cap stops right after a tool call: the
        results are in ``contents`` and no text was written. The Anthropic,
        OpenAI and Inception clients already ask once more; Gemini raised
        "Empty response" instead, and a caller lost every call the loop made.

        The tools stay declared and ``mode: NONE`` forbids calling them, so the
        function calls and results already in ``contents`` still read as valid.
        """
        config = dict(request_kwargs["config"])
        config["tool_config"] = {"function_calling_config": {"mode": "NONE"}}
        contents = list(request_kwargs["contents"]) + [
            {"role": "user", "parts": [{"text": FINALIZE_PROMPT}]}
        ]
        return await self._client.aio.models.generate_content(
            **{**request_kwargs, "config": config, "contents": contents}
        )

    def _get_system_prompt_hash(self, system_instruct: str) -> str:
        """Generate hash for system prompt to use as cache key."""
        return hashlib.sha256(system_instruct.encode()).hexdigest()

    async def _get_or_create_cached_context(
        self,
        system_instruct: str,
        model: str,
    ) -> Optional[str]:
        """Get or create cached context for static system prompt."""
        if not system_instruct or not system_instruct.strip():
            return None

        if not self._google_explicit_cache:
            return None

        model_lower = model.lower()
        if not ("2.5" in model_lower or "gemini-2" in model_lower):
            return None

        prompt_hash = self._get_system_prompt_hash(system_instruct)
        now = time.time()

        if prompt_hash in self._cached_contexts:
            context_name, created_at = self._cached_contexts[prompt_hash]
            if now - created_at < self._context_ttl_seconds:
                self._cached_contexts.move_to_end(prompt_hash)
                return context_name
            else:
                del self._cached_contexts[prompt_hash]

        try:
            static_content = system_instruct

            try:
                cached_context = await self._client.aio.cached_contents.create(
                    model=model,
                    contents=[static_content],
                    ttl=self._context_ttl_seconds,
                )
            except AttributeError:
                try:
                    cached_context = await self._client.aio.models.cached_contents.create(
                        model=model,
                        contents=[static_content],
                        ttl=self._context_ttl_seconds,
                    )
                except (AttributeError, Exception):
                    return None

            context_name = getattr(cached_context, "name", None)
            if context_name:
                if len(self._cached_contexts) >= self._max_cached_contexts:
                    self._cached_contexts.popitem(last=False)

                self._cached_contexts[prompt_hash] = (context_name, now)
                self._cached_contexts.move_to_end(prompt_hash)

                logger.debug(f"Google cache created: {context_name[:50]}...")
                return context_name
        except Exception as e:
            logger.debug(f"Google explicit caching unavailable: {e}")
            return None

        return None

    def _get_tool_error_retry_message(
        self,
        output_payloads: List[Dict[str, Any]],
        tool_error_callback: Optional[
            Callable[[str, Optional[str], Dict[str, Any]], Optional[str]]
        ],
    ) -> Optional[str]:
        """Check tool outputs for errors; return retry message if callback provides one."""
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

    def _build_initial_contents(
        self,
        cached_context_name: Optional[str],
        messages: Optional[List[Dict[str, Any]]],
        prompt: str,
        system_instruct: str,
        attachments: Optional[List[Attachment]] = None,
    ) -> List[Any]:
        """Build initial contents for Gemini API request."""
        out: List[Any] = []

        def final_user_turn(text: str) -> Any:
            """The prompt turn: a bare string normally, structured parts when attaching.

            Gemini accepts a bare string as a whole user turn, but an attachment
            has to ride in ``parts`` alongside the text, so attachments force the
            structured form.
            """
            if not attachments:
                return text
            return {"role": "user", "parts": _user_parts(text, attachments)}

        if cached_context_name:
            if messages:
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    google_role = "model" if role == "assistant" else "user"
                    out.append({"role": google_role, "parts": [{"text": content}]})
            out.append(final_user_turn(prompt))
        else:
            if messages:
                for i, msg in enumerate(messages):
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    google_role = "model" if role == "assistant" else "user"
                    if i == 0 and role == "user" and system_instruct:
                        content = f"{system_instruct}\n\n{content}"
                    out.append({"role": google_role, "parts": [{"text": content}]})
                out.append(
                    {"role": "user", "parts": _user_parts(prompt, attachments)}
                )
            else:
                merged = (
                    f"{system_instruct}\n\n{prompt}" if (system_instruct or "").strip() else prompt
                )
                out.append(final_user_turn(merged))
        return out

    def _extract_function_calls(self, resp: Any) -> List[Dict[str, Any]]:
        """Extract function calls from Gemini response."""
        calls: List[Dict[str, Any]] = []
        candidates = getattr(resp, "candidates", []) or []
        if not candidates:
            return calls
        content = getattr(candidates[0], "content", None)
        if not content:
            return calls
        parts = getattr(content, "parts", []) or []
        for part in parts:
            fc = getattr(part, "function_call", None)
            if fc:
                name = getattr(fc, "name", None)
                args = getattr(fc, "args", None) or {}
                if isinstance(args, dict):
                    calls.append({"name": name, "args": args, "part": part})
                else:
                    calls.append({"name": name, "args": {}, "part": part})
        return calls

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
        response_format: Optional[Dict[str, Any]] = None,
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
        include_google_search: Optional[bool] = None,
        url_retrievals: Optional[List[UrlRetrieval]] = None,
    ) -> tuple[str, TokenUsage, Optional[Stop]]:
        """Generate a reply, running the tool loop when tools are given.

        ``url_retrievals``, when passed, is filled with each link URL context
        tried to open, once per link, so a caller can tell a page that was read
        from one that was refused and answered from search instead.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        if not model or not model.strip():
            raise ValueError("Model name cannot be empty")

        system_instruct = system_instruct or (instructions or "")
        use_tools = bool(tools_schema and execute_tool_cb)
        search_enabled = (
            include_google_search
            if include_google_search is not None
            else self._google_attach_search_tool
        )
        cached_context_name = await self._get_or_create_cached_context(system_instruct, model)

        def build_initial_contents() -> List[Any]:
            return self._build_initial_contents(
                cached_context_name, messages, prompt, system_instruct, attachments
            )

        contents = build_initial_contents()
        config = self._build_config_dict(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            response_format=response_format,
            tools_schema=tools_schema if use_tools else None,
            include_google_search=search_enabled and not use_tools,
            reasoning_effort=reasoning_effort,
            model=model,
        )

        request_kwargs: Dict[str, Any] = {
            "model": model,
            "contents": contents,
            "config": config,
        }
        if cached_context_name:
            request_kwargs["cached_content"] = cached_context_name

        last_resp: Optional[Any] = None
        grounding_responses: List[Any] = []
        last_nonempty_output = ""
        effective_steps = 0
        consecutive_reasoning_only = 0
        total_input = 0
        total_output = 0
        total_reasoning = 0
        total_cached: Optional[int] = None
        served: Optional[str] = None
        retrieved: List[UrlRetrieval] = []

        def count(resp: Any) -> None:
            nonlocal total_input, total_output, total_reasoning, total_cached, served
            served = served_model(getattr(resp, "model_version", None)) or served
            um = getattr(resp, "usage_metadata", None)
            if um:
                pi, po, thoughts, cc = _gemini_tokens(um)
                total_input += pi
                total_output += po
                total_reasoning += thoughts
                if cc is not None:
                    total_cached = (total_cached or 0) + cc
            retrieved.extend(_url_retrievals(resp))

        for step in range(max_steps):
            try:
                resp = await self._client.aio.models.generate_content(**request_kwargs)
            except Exception as e:
                if is_rate_limit_error(e):
                    raise
                logger.error(f"Google API call failed: {e}")
                raise APIError(f"Google API request failed: {e}") from e

            last_resp = resp
            if _grounding_metadata(resp):
                grounding_responses.append(resp)
            text = str(getattr(resp, "text", None) or getattr(resp, "output_text", "") or "")
            if text.strip():
                last_nonempty_output = text

            count(resp)

            if not use_tools:
                break

            function_calls = self._extract_function_calls(resp)
            if not function_calls:
                break

            output_payloads: List[Dict[str, Any]] = []
            had_non_reasoning_tool = False

            for fc in function_calls:
                name = fc.get("name")
                args = fc.get("args") or {}
                result_payload = await self._execute_tool(name, args, execute_tool_cb)
                output_payloads.append({"name": name, "payload": result_payload})
                if name and "reasoning" not in (name or "").lower():
                    had_non_reasoning_tool = True

            retry_message = self._get_tool_error_retry_message(output_payloads, tool_error_callback)
            if retry_message is not None:
                contents = build_initial_contents() + [
                    {"role": "user", "parts": [{"text": retry_message}]}
                ]
                request_kwargs["contents"] = contents
                continue

            model_content = (
                getattr(last_resp.candidates[0], "content", None)
                if last_resp and getattr(last_resp, "candidates", None)
                else None
            )
            if model_content is not None:
                contents.append(model_content)

            response_parts: List[Any] = []
            for out in output_payloads:
                part = types.Part.from_function_response(
                    name=out["name"],
                    response=out["payload"],
                )
                response_parts.append(part)

            contents.append(types.Content(role="user", parts=response_parts))
            request_kwargs["contents"] = contents

            effective_steps, consecutive_reasoning_only = update_step_tracking(
                had_non_reasoning_tool,
                effective_steps,
                consecutive_reasoning_only,
                max_effective_tool_steps,
            )

            if should_break_loop(
                effective_steps,
                consecutive_reasoning_only,
                max_effective_tool_steps,
            ):
                break

        text = (
            str(getattr(last_resp, "text", None) or getattr(last_resp, "output_text", "") or "")
            if last_resp
            else ""
        )
        if not text.strip():
            text = last_nonempty_output or ""

        if not text.strip() and use_tools:
            try:
                final_resp = await self._finalize_empty_response(request_kwargs)
            except Exception as e:
                if is_rate_limit_error(e):
                    raise
                logger.warning(f"Failed to finalize response: {e}")
            else:
                count(final_resp)
                last_resp = final_resp
                text = str(getattr(final_resp, "text", None) or "")

        if not text.strip():
            raise APIError("Empty response received from Google API")

        if url_retrievals is not None:
            by_url = {r.url: r for r in retrieved}
            url_retrievals.extend(by_url.values())

        try:
            if self._google_inline_citations and last_resp:
                injection_resp = (
                    merge_grounding_responses(grounding_responses)
                    if grounding_responses
                    else last_resp
                )
                urls = collect_grounding_urls(injection_resp)
                if urls:
                    async with httpx.AsyncClient(follow_redirects=True, timeout=2) as http:
                        resolved = await async_resolve_urls(urls, http, max_concurrency=4)

                    def resolve_url(url: str) -> str:
                        return resolved.get(url, url)
                else:
                    def resolve_url(url: str) -> str:
                        return url
                text = inject_inline_citations(
                    text,
                    injection_resp,
                    resolve_url=resolve_url,
                )
        except Exception as e:
            logger.debug(f"Failed to inject citations: {e}")

        usage = TokenUsage(
            input_tokens=total_input,
            output_tokens=total_output,
            total_tokens=total_input + total_output,
            cached_tokens=total_cached if total_cached else None,
            reasoning_tokens=total_reasoning or None,
            served_model=served,
        )
        stop = _gemini_stop(
            last_resp, wants_tools=use_tools and bool(self._extract_function_calls(last_resp))
        )
        return text, usage, stop

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
        response_format: Optional[Dict[str, Any]] = None,
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
        del presence_penalty, force_tool_use

        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        if not model or not model.strip():
            raise ValueError("Model name cannot be empty")

        system_instruct = system_instruct or (instructions or "")
        use_tools = bool(tools_schema and execute_tool_cb)
        # Custom tools replace config tools; optional Google Search when no custom tools.
        include_search = self._google_attach_search_tool and not use_tools

        cached_context_name = await self._get_or_create_cached_context(system_instruct, model)

        def build_initial_contents() -> List[Any]:
            return self._build_initial_contents(
                cached_context_name, messages, prompt, system_instruct, attachments
            )

        contents = build_initial_contents()
        config = self._build_config_dict(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            response_format=response_format,
            tools_schema=tools_schema if use_tools else None,
            include_google_search=include_search,
            reasoning_effort=reasoning_effort,
            model=model,
        )

        request_kwargs: Dict[str, Any] = {
            "model": model,
            "contents": contents,
            "config": config,
        }
        if cached_context_name:
            request_kwargs["cached_content"] = cached_context_name

        total_input = 0
        total_output = 0
        total_reasoning = 0
        total_cached: Optional[int] = None
        effective_steps = 0
        consecutive_reasoning_only = 0
        stop: Optional[Stop] = None
        served: Optional[str] = None

        try:
            for step in range(max_steps):
                if use_tools and step > 0:
                    yield StreamStepBoundary(step)

                # google-genai: this API returns a coroutine whose result is the async iterator.
                stream = await self._client.aio.models.generate_content_stream(**request_kwargs)
                last_chunk: Optional[Any] = None
                try:
                    async for chunk in stream:
                        last_chunk = chunk
                        text = getattr(chunk, "text", None) or ""
                        if text:
                            yield StreamTextDelta(text)
                        _google_stream_chunk_to_usage_sink(
                            chunk,
                            usage_sink,
                            total_input,
                            total_output,
                            total_cached,
                            total_reasoning,
                        )
                finally:
                    # Drain remainder for usage_metadata only (consumer may have stopped early).
                    try:
                        async for chunk in stream:
                            last_chunk = chunk
                            _google_stream_chunk_to_usage_sink(
                                chunk,
                                usage_sink,
                                total_input,
                                total_output,
                                total_cached,
                                total_reasoning,
                            )
                    except Exception as e:
                        logger.debug("Google stream drain for usage: %s", e, exc_info=True)

                if last_chunk is None:
                    break

                served = served_model(getattr(last_chunk, "model_version", None)) or served
                um = getattr(last_chunk, "usage_metadata", None)
                if um:
                    pi, po, thoughts, cc = _gemini_tokens(um)
                    total_input += pi
                    total_output += po
                    total_reasoning += thoughts
                    if cc is not None:
                        total_cached = (total_cached or 0) + cc
                    if usage_sink is not None:
                        usage_sink.replace_with(
                            TokenUsage(
                                total_input,
                                total_output,
                                total_input + total_output,
                                total_cached if total_cached else None,
                                reasoning_tokens=total_reasoning or None,
                                served_model=served,
                            )
                        )

                function_calls = self._extract_function_calls(last_chunk) if use_tools else []
                stop = _gemini_stop(last_chunk, wants_tools=bool(function_calls))
                if not function_calls:
                    break

                output_payloads: List[Dict[str, Any]] = []
                had_non_reasoning_tool = False
                for fc in function_calls:
                    name = fc.get("name")
                    args = fc.get("args") or {}
                    yield StreamToolCallEnd(
                        id=str(name or ""),
                        name=str(name or ""),
                        arguments=dict(args) if isinstance(args, dict) else {},
                    )
                    result_payload = await self._execute_tool(name, args, execute_tool_cb)
                    output_payloads.append({"name": name, "payload": result_payload})
                    if name and "reasoning" not in (name or "").lower():
                        had_non_reasoning_tool = True

                retry_message = self._get_tool_error_retry_message(
                    output_payloads, tool_error_callback
                )
                if retry_message is not None:
                    contents = build_initial_contents() + [
                        {"role": "user", "parts": [{"text": retry_message}]}
                    ]
                    request_kwargs["contents"] = contents
                    continue

                model_content = (
                    getattr(last_chunk.candidates[0], "content", None)
                    if getattr(last_chunk, "candidates", None)
                    else None
                )
                if model_content is not None:
                    contents.append(model_content)

                response_parts: List[Any] = []
                for out in output_payloads:
                    response_parts.append(
                        types.Part.from_function_response(
                            name=out["name"],
                            response=out["payload"],
                        )
                    )
                contents.append(types.Content(role="user", parts=response_parts))
                request_kwargs["contents"] = contents

                effective_steps, consecutive_reasoning_only = update_step_tracking(
                    had_non_reasoning_tool,
                    effective_steps,
                    consecutive_reasoning_only,
                    max_effective_tool_steps,
                )
                if should_break_loop(
                    effective_steps,
                    consecutive_reasoning_only,
                    max_effective_tool_steps,
                ):
                    break

            if stop is not None:
                yield stop
            yield TokenUsage(
                input_tokens=total_input,
                output_tokens=total_output,
                total_tokens=total_input + total_output,
                cached_tokens=total_cached if total_cached else None,
                reasoning_tokens=total_reasoning or None,
                served_model=served,
            )
        except Exception as e:
            if is_rate_limit_error(e):
                raise
            raise APIError(f"Google streaming failed: {e}") from e
