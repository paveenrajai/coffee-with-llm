from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union

from .attachments import Attachment, normalize_attachments
from .config import Config
from .cost import estimate_cost
from .exceptions import APIError, ConfigurationError, RateLimitError, ValidationError
from .providers.google.api_mode import DEFAULT_GOOGLE_API_MODE, GoogleApiMode
from .providers.google.interactions_client import GoogleInteractionsClient
from .providers.google.text_client import GoogleTextClient
from .providers.registry import get_google_interactions_client, get_provider, split_provider_model
from .rate_limit import is_rate_limit_error, with_retry
from .types import AskResult, StreamResult, StreamUsageSink, TokenUsage

logger = logging.getLogger(__name__)


class AskLLM:
    """
    Model-agnostic LLM interface supporting OpenAI, Anthropic Claude, and Google Gemini.

    Automatically selects the appropriate provider based on the model name.
    Provides a unified API for both providers with parameter normalization.

    Example:
        ```python
        from coffee import AskLLM

        llm = AskLLM(model="gpt-5.4")
        result = await llm.ask(
            prompt="What is Python?",
            system_instruct="You are a helpful assistant."
        )
        print(result.text)
        print(result.usage.input_tokens, result.usage.output_tokens)
        ```
    """

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        config: Optional[Config] = None,
        min_delay_between_calls: float = 1.0,
        max_retries: int = 3,
        request_timeout: Optional[float] = None,
        google_explicit_cache: bool = True,
        google_inline_citations: bool = True,
        google_attach_search_tool: bool = True,
        anthropic_prompt_cache: bool = True,
        google_api_mode: GoogleApiMode = DEFAULT_GOOGLE_API_MODE,
    ) -> None:
        """
        Initialize AskLLM with a model.

        Args:
            model: Model name, optionally ``provider/model`` (e.g. ``google/gemma-2-9b-it``,
                 ``openai/gpt-4o-mini``, ``anthropic/claude-sonnet-4-6``). The segment after
                 ``/`` is sent to the API; the provider segment selects the client. Legacy
                 ids without ``/`` still work (e.g. ``gpt-5.4``, ``claude-sonnet-4-6``).
                 Must be provided.
            config: Config instance. If None, uses Config.from_env() (API keys from env).
            min_delay_between_calls: Min delay in seconds between API calls (default: 1.0)
            max_retries: Maximum number of retries for rate limit errors (default: 3)
            request_timeout: Request timeout in seconds (default: 60)
            google_explicit_cache: Enable Google context caching for Gemini (default: True)
            google_inline_citations: Inject [cite: url] for Gemini grounding (default: True)
            google_attach_search_tool: When using Gemini with no custom tools, attach the
                Google Search tool (default: True). Ignored for non-Google models.
            google_api_mode: For Gemini only, ``"generate_content"`` (default) uses the
                classic Models API; ``"interactions"`` routes :meth:`ask` to the
                Interactions API. :meth:`ask_interaction` always uses Interactions.
            anthropic_prompt_cache: Enable Anthropic automatic prompt caching via top-level
                ``cache_control`` (default: True). Ignored for non-Anthropic models.

        Raises:
            ValidationError: If model is not provided.
            ConfigurationError: If API keys are missing or client initialization fails.
        """
        if not model or not str(model).strip():
            raise ValidationError("Model name is required")

        model_str = str(model).strip()
        api_model, _ = split_provider_model(model_str)
        if not api_model:
            raise ValidationError("Model name is required")

        self._model = api_model
        self._model_str = model_str
        self._min_delay = min_delay_between_calls
        self._max_retries = max_retries
        self._last_call_time: Optional[float] = None
        self._google_api_mode = google_api_mode
        self._google_attach_search_tool = google_attach_search_tool

        cfg = (config or Config.from_env()).with_request_timeout(request_timeout)
        self._config = cfg
        self._request_timeout = cfg.request_timeout

        try:
            self._client = get_provider(
                model_str,
                config=cfg,
                request_timeout=self._request_timeout,
                google_explicit_cache=google_explicit_cache,
                google_inline_citations=google_inline_citations,
                google_attach_search_tool=google_attach_search_tool,
                anthropic_prompt_cache=anthropic_prompt_cache,
                google_api_mode=google_api_mode,
            )
            self._interactions_client: GoogleInteractionsClient | None = None
        except Exception as e:
            raise ConfigurationError(
                f"Failed to initialize client for model '{model_str}': {e}"
            ) from e

    async def ask(
        self,
        *,
        prompt: str,
        system_instruct: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
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
        stream: bool = False,
        attachments: Optional[List[Attachment]] = None,
        google_attach_search_tool: Optional[bool] = None,
    ) -> Union[AskResult, StreamResult]:
        """
        Ask the LLM a question.

        Args:
            prompt: User prompt/question (appended to messages if provided)
            system_instruct: System instruction/prompt
            messages: Optional history (list of {"role": "user"|"assistant", "content": "..."})
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter
            presence_penalty: Presence penalty (OpenAI only)
            reasoning_effort: Extended-thinking effort, one of "low" | "medium" | "high".
                Provider-agnostic: maps to OpenAI ``reasoning.effort``, Anthropic
                Anthropic adaptive ``output_config.effort`` (4.6+) or legacy
                ``thinking.budget_tokens``, and Google ``thinking_config``. Unknown
                values are ignored. None disables extended thinking.
            tools_schema: Tool/function calling schema (OpenAI, Anthropic, Google)
            response_format: Response format specification (JSON schema, etc.)
            execute_tool_cb: Callback for executing tools (OpenAI, Anthropic, Google)
            tool_error_callback: When tool returns ok=False, (tool_name, error_code, payload).
                Return str to start new session; None to feed error back.
            max_steps: Maximum tool-calling steps (OpenAI, Anthropic, Google)
            max_effective_tool_steps: Maximum effective tool steps (OpenAI, Anthropic, Google)
            force_tool_use: When True, force at least one tool call (Anthropic tool_choice=any).
                Prevents text-only responses when tools are provided.
            stream: When True, return StreamResult (async iterable of stream events; usage
                after iteration or aclose). Supports tools_schema and response_format when
                the provider allows; requires execute_tool_cb if tools_schema is set.
            google_attach_search_tool: When set, overrides the constructor default for
                Gemini only (attach or omit the Google Search tool for this call).

        Returns:
            AskResult with text and token usage, or StreamResult when stream=True.

        Raises:
            ValidationError: If prompt is empty or invalid parameters provided.
            APIError: If the API call fails.
        """
        if not prompt or not prompt.strip():
            raise ValidationError("Prompt cannot be empty")

        if max_tokens is not None and max_tokens <= 0:
            raise ValidationError("max_tokens must be positive")

        if temperature is not None and (temperature < 0 or temperature > 2):
            raise ValidationError("temperature must be between 0 and 2")

        if top_p is not None and (top_p < 0 or top_p > 1):
            raise ValidationError("top_p must be between 0 and 1")

        if max_steps <= 0:
            raise ValidationError("max_steps must be positive")

        if max_effective_tool_steps <= 0:
            raise ValidationError("max_effective_tool_steps must be positive")

        if stream and tools_schema and not execute_tool_cb:
            raise ValidationError("execute_tool_cb is required when stream=True with tools_schema")

        resolved_attachments = normalize_attachments(attachments)

        # Rate limiting: wait if needed before making API call
        await self._wait_if_needed()

        if stream:
            return self._ask_stream(
                prompt=prompt,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                system_instruct=system_instruct,
                presence_penalty=presence_penalty,
                reasoning_effort=reasoning_effort,
                tools_schema=tools_schema,
                response_format=response_format,
                execute_tool_cb=execute_tool_cb,
                tool_error_callback=tool_error_callback,
                max_steps=max_steps,
                max_effective_tool_steps=max_effective_tool_steps,
                force_tool_use=force_tool_use,
                attachments=resolved_attachments,
            )

        async def _generate() -> AskResult:
            generate_kwargs: Dict[str, Any] = {}
            if google_attach_search_tool is not None and isinstance(
                self._client, (GoogleTextClient, GoogleInteractionsClient)
            ):
                generate_kwargs["include_google_search"] = google_attach_search_tool
            result = await self._client.generate(
                prompt=prompt,
                model=self._model,
                messages=messages,
                max_tokens=max_tokens,
                top_p=top_p,
                presence_penalty=presence_penalty,
                instructions=system_instruct if system_instruct else None,
                reasoning_effort=reasoning_effort,
                tools_schema=tools_schema,
                response_format=response_format,
                execute_tool_cb=execute_tool_cb,
                tool_error_callback=tool_error_callback,
                max_steps=max_steps,
                max_effective_tool_steps=max_effective_tool_steps,
                force_tool_use=force_tool_use,
                temperature=temperature,
                system_instruct=system_instruct or "",
                attachments=list(resolved_attachments) or None,
                **generate_kwargs,
            )
            text, usage = (
                result if isinstance(result, tuple) else (result, TokenUsage(0, 0, 0, None))
            )
            return AskResult(text=text, usage=self._usage_with_cost(usage))

        try:
            return await with_retry(
                _generate,
                max_retries=self._max_retries,
            )
        except Exception as e:
            if isinstance(e, (ValidationError, ConfigurationError, RateLimitError)):
                raise
            if is_rate_limit_error(e):
                logger.error(
                    "Rate limit exceeded for model '%s' after %d attempts",
                    self._model,
                    self._max_retries,
                )
                raise RateLimitError(
                    f"Rate limit exceeded after {self._max_retries} retries: {e}"
                ) from e
            logger.error(f"API call failed for model '{self._model}': {e}")
            raise APIError(f"Failed to generate response: {e}") from e

    def _get_interactions_client(self) -> GoogleInteractionsClient:
        if self._interactions_client is None:
            self._interactions_client = get_google_interactions_client(
                self._model_str,
                self._config,
                request_timeout=self._request_timeout,
                google_attach_search_tool=self._google_attach_search_tool,
            )
        return self._interactions_client

    async def ask_interaction(
        self,
        *,
        prompt: str,
        system_instruct: str = "",
        previous_interaction_id: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        tools_schema: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        execute_tool_cb: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
        max_steps: int = 16,
        google_attach_search_tool: Optional[bool] = None,
    ) -> AskResult:
        """Ask via Gemini Interactions API (server-side session state).

        Use ``previous_interaction_id`` from a prior :class:`AskResult` to continue
        a multi-turn agent session. Classic :meth:`ask` still uses ``generateContent``
        unless ``google_api_mode="interactions"``.
        """
        if not prompt or not prompt.strip():
            raise ValidationError("Prompt cannot be empty")

        await self._wait_if_needed()
        client = self._get_interactions_client()

        async def _create() -> AskResult:
            text, usage, interaction_id = await client.create_interaction(
                prompt=prompt,
                model=self._model,
                system_instruct=system_instruct,
                previous_interaction_id=previous_interaction_id,
                tools_schema=tools_schema,
                execute_tool_cb=execute_tool_cb,
                max_steps=max_steps,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                include_google_search=google_attach_search_tool,
            )
            return AskResult(
                text=text,
                usage=self._usage_with_cost(usage),
                interaction_id=interaction_id,
            )

        try:
            return await with_retry(_create, max_retries=self._max_retries)
        except Exception as e:
            if isinstance(e, (ValidationError, ConfigurationError, RateLimitError)):
                raise
            if is_rate_limit_error(e):
                raise RateLimitError(
                    f"Rate limit exceeded after {self._max_retries} retries: {e}"
                ) from e
            raise APIError(f"Failed to create interaction: {e}") from e

    def _ask_stream(
        self,
        *,
        prompt: str,
        messages: Optional[List[Dict[str, Any]]],
        max_tokens: Optional[int],
        temperature: Optional[float],
        top_p: Optional[float],
        system_instruct: str,
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
        attachments: tuple[Attachment, ...] = (),
    ) -> StreamResult:
        """Stream events with usage and rate-limit retry."""

        usage_sink = StreamUsageSink()

        def create_stream() -> AsyncIterator[object]:
            return self._client.generate_stream(
                prompt=prompt,
                model=self._model,
                messages=messages,
                max_tokens=max_tokens,
                top_p=top_p,
                temperature=temperature,
                instructions=system_instruct if system_instruct else None,
                system_instruct=system_instruct or "",
                presence_penalty=presence_penalty,
                reasoning_effort=reasoning_effort,
                tools_schema=tools_schema,
                response_format=response_format,
                execute_tool_cb=execute_tool_cb,
                tool_error_callback=tool_error_callback,
                max_steps=max_steps,
                max_effective_tool_steps=max_effective_tool_steps,
                force_tool_use=force_tool_use,
                usage_sink=usage_sink,
                attachments=list(attachments) or None,
            )

        return StreamResult(
            stream_factory=create_stream,
            usage_callback=lambda u: self._usage_with_cost(u),
            max_retries=self._max_retries,
            usage_sink=usage_sink,
        )

    def _usage_with_cost(self, usage: TokenUsage) -> TokenUsage:
        """Add cost_usd to usage."""
        cost = estimate_cost(usage, self._model)
        return TokenUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            cached_tokens=usage.cached_tokens,
            cache_creation_tokens=usage.cache_creation_tokens,
            cost_usd=cost,
        )

    async def _wait_if_needed(self) -> None:
        """Wait if needed to maintain minimum delay between calls."""
        if self._last_call_time is not None:
            elapsed = time.perf_counter() - self._last_call_time
            if elapsed < self._min_delay:
                wait_time = self._min_delay - elapsed
                await asyncio.sleep(wait_time)
        self._last_call_time = time.perf_counter()
