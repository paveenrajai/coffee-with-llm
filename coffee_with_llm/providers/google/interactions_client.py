"""Google Gemini Interactions API client.

Use for multi-turn agent sessions with server-side state (``previous_interaction_id``).
The classic :class:`GoogleTextClient` remains the default for ``generateContent``.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union

from google import genai

from ...attachments import Attachment
from ...config import Config
from ...exceptions import APIError, ConfigurationError, ValidationError
from ...rate_limit import is_rate_limit_error
from ...types import StreamUsageSink, TokenUsage
from ..tool_utils import normalize_tool_result
from ._convert_tools import convert_openai_tools_to_interaction_functions
from .interactions_utils import (
    interaction_function_calls,
    interaction_text,
    interaction_usage,
)

logger = logging.getLogger(__name__)


class GoogleInteractionsClient:
    """Gemini via the Interactions API (``client.interactions.create``)."""

    def __init__(
        self,
        config: Config,
        *,
        request_timeout: Optional[float] = None,
        google_attach_search_tool: bool = True,
    ) -> None:
        self._api_key = config.require_google_key()
        self._google_attach_search_tool = google_attach_search_tool
        try:
            self._client = genai.Client(api_key=self._api_key)
        except ImportError as e:
            raise ConfigurationError(
                "Google GenAI package not installed. Install with: pip install google-genai"
            ) from e
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Google client: {e}") from e
        if not hasattr(self._client, "interactions"):
            raise ConfigurationError(
                "Installed google-genai does not expose interactions API. "
                "Upgrade with: pip install -U 'google-genai>=2.0.0'"
            )
        self._request_timeout = request_timeout

    def _build_tools(
        self,
        tools_schema: Optional[List[Dict[str, Any]]],
    ) -> Optional[List[Dict[str, Any]]]:
        tools: list[dict[str, Any]] = []
        if tools_schema:
            tools.extend(convert_openai_tools_to_interaction_functions(tools_schema))
        elif self._google_attach_search_tool:
            tools.append({"type": "google_search"})
        return tools or None

    async def _execute_tool(
        self,
        name: Optional[str],
        args: Dict[str, Any],
        execute_tool_cb: Optional[Callable[[str, Dict[str, Any]], Any]],
    ) -> Dict[str, Any]:
        if execute_tool_cb is None:
            return {"ok": False, "result": {}, "error": "no executor provided"}
        try:
            maybe = execute_tool_cb(name or "", args)
            if inspect.isawaitable(maybe) or hasattr(maybe, "__await__"):
                result = await maybe
            else:
                result = maybe
            return normalize_tool_result(result)
        except Exception as e:
            logger.error("Interactions tool %s failed: %s", name, e)
            return {"ok": False, "result": {}, "error": str(e)}

    async def create_interaction(
        self,
        *,
        prompt: str,
        model: str,
        system_instruct: str = "",
        previous_interaction_id: Optional[str] = None,
        tools_schema: Optional[List[Dict[str, Any]]] = None,
        execute_tool_cb: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
        max_steps: int = 16,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_mime_type: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        include_google_search: Optional[bool] = None,
    ) -> tuple[str, TokenUsage, str]:
        """Create an interaction, optionally looping on function calls.

        Returns ``(text, usage, interaction_id)``.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        if not model or not model.strip():
            raise ValueError("Model name cannot be empty")

        search_enabled = (
            include_google_search
            if include_google_search is not None
            else self._google_attach_search_tool
        )
        tools = self._build_tools(tools_schema) if search_enabled or tools_schema else (
            self._build_tools(tools_schema) if tools_schema else None
        )
        if not search_enabled and tools:
            tools = [tool for tool in tools if tool.get("type") != "google_search"] or None

        generation_config: dict[str, Any] = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_tokens is not None:
            generation_config["max_output_tokens"] = max_tokens

        create_kwargs: dict[str, Any] = {
            "model": model,
            "input": prompt,
        }
        if system_instruct.strip():
            create_kwargs["system_instruction"] = system_instruct.strip()
        if previous_interaction_id:
            create_kwargs["previous_interaction_id"] = previous_interaction_id
        if tools:
            create_kwargs["tools"] = tools
        if generation_config:
            create_kwargs["generation_config"] = generation_config
        if response_format and response_format.get("type") == "json_schema":
            create_kwargs["response_format"] = {
                "type": "text",
                "mime_type": "application/json",
                "schema": response_format.get("json_schema"),
            }
        elif response_mime_type:
            create_kwargs["response_format"] = {
                "type": "text",
                "mime_type": response_mime_type,
            }

        total_usage = TokenUsage(0, 0, 0, None)
        interaction_id = previous_interaction_id or ""
        text = ""

        for _step in range(max_steps):
            try:
                interaction = await self._client.aio.interactions.create(
                    **create_kwargs,
                    timeout=self._request_timeout,
                )
            except Exception as e:
                if is_rate_limit_error(e):
                    raise
                logger.error("Google Interactions API call failed: %s", e)
                raise APIError(f"Google Interactions request failed: {e}") from e

            step_usage = interaction_usage(interaction)
            total_usage = TokenUsage(
                input_tokens=total_usage.input_tokens + step_usage.input_tokens,
                output_tokens=total_usage.output_tokens + step_usage.output_tokens,
                total_tokens=total_usage.total_tokens + step_usage.total_tokens,
                cached_tokens=(
                    (total_usage.cached_tokens or 0) + (step_usage.cached_tokens or 0)
                )
                or None,
            )
            interaction_id = str(getattr(interaction, "id", "") or interaction_id)
            text = interaction_text(interaction)

            calls = interaction_function_calls(interaction)
            if not calls or execute_tool_cb is None:
                return text, total_usage, interaction_id

            result_steps: list[dict[str, Any]] = []
            for call in calls:
                payload = await self._execute_tool(
                    call.get("name"),
                    call.get("arguments") or {},
                    execute_tool_cb,
                )
                result_steps.append(
                    {
                        "type": "function_result",
                        "call_id": call.get("id"),
                        "name": call.get("name"),
                        "result": payload,
                    }
                )

            create_kwargs = {
                "model": model,
                "input": result_steps,
                "previous_interaction_id": interaction_id,
            }
            if system_instruct.strip():
                create_kwargs["system_instruction"] = system_instruct.strip()
            if tools:
                create_kwargs["tools"] = tools

        return text, total_usage, interaction_id

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
        previous_interaction_id: Optional[str] = None,
    ) -> tuple[str, TokenUsage]:
        if attachments:
            raise ValidationError("Interactions API does not support attachments yet.")
        if messages:
            raise ValidationError("Interactions API does not support messages history yet.")
        if tool_error_callback is not None:
            logger.debug("tool_error_callback is ignored for Interactions API")
        if force_tool_use:
            logger.debug("force_tool_use is ignored for Interactions API")
        if top_p is not None or presence_penalty is not None or reasoning_effort:
            logger.debug("top_p/presence_penalty/reasoning_effort ignored for Interactions API")

        system = system_instruct or (instructions or "")
        text, usage, _interaction_id = await self.create_interaction(
            prompt=prompt,
            model=model,
            system_instruct=system,
            previous_interaction_id=previous_interaction_id,
            tools_schema=tools_schema,
            execute_tool_cb=execute_tool_cb,
            max_steps=max_steps,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            include_google_search=include_google_search,
        )
        return text, usage

    def generate_stream(
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
        max_steps: int = 24,
        max_effective_tool_steps: int = 12,
        force_tool_use: bool = False,
        usage_sink: Optional[StreamUsageSink] = None,
        attachments: Optional[List[Attachment]] = None,
    ) -> AsyncIterator[Union[object, TokenUsage]]:
        raise NotImplementedError(
            "Interactions API streaming is not implemented yet. "
            "Use google_api_mode='generate_content' for streaming."
        )
